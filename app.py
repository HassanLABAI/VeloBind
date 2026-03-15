# app.py — VeloBind HF Spaces inference app
#
# Uses the exact 45 fold models that produced the reported R=0.8469 on CASF-2016.
# No retraining required. Upload output/models/ to HF model repo and set
# HF_MODEL_REPO below.
#
# HF model repo should contain:
#   fold_model_s{seed}_{type}_f{fold}.pkl   — 45 files (3 seeds × 3 types × 5 folds)
#   meta_type_casf16.pkl                    — Ridge meta-learner (from 06_eval_both.py)
#   target_scaler.pkl                       — TargetScaler (from 03_train.py)
#   ligand_scaler.pkl                       — from output/preprocessors/
#
# Free tier: 16GB RAM, 2 vCPU, 50GB disk — all 45 models fit easily (~2-3GB total).
# Cold start: ~30-40s to download + load models on first visit.

import os, json, warnings, time
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# ── Config ────────────────────────────────────────────────────────────
HF_MODEL_REPO = "YOUR-USERNAME/velobind-models"   # ← set this
MODEL_CACHE   = Path("/tmp/velobind_models")
SEEDS         = [42, 123, 456]
MODEL_TYPES   = ["lgbm", "cb", "xgb"]
N_FOLDS       = 5

# Best feature config — Step 9 winner from 03_train.py ablation
# MUST match what the fold models were trained on
import sys
sys.path.append(str(Path(__file__).parent))
from src.features.protein import load_esm, embed_batch, sequence_features
from src.features.ligand  import extract_ligand_features
from src.models.ensemble  import TargetScaler
from src.config import config


# ══════════════════════════════════════════════════════════════════════
# Model loading
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Downloading & loading VeloBind models (~30s first run)…")
def load_all_models():
    from huggingface_hub import hf_hub_download
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)

    # Build list of all files to fetch
    model_files = (
        [f"fold_model_s{s}_{t}_f{f}.pkl"
         for s in SEEDS for t in MODEL_TYPES for f in range(N_FOLDS)]
        + ["meta_type_casf16.pkl", "target_scaler.pkl", "ligand_scaler.pkl"]
    )

    progress = st.progress(0, text="Downloading models…")
    for i, fname in enumerate(model_files):
        local = MODEL_CACHE / fname
        if not local.exists():
            hf_hub_download(
                repo_id=HF_MODEL_REPO, filename=fname,
                local_dir=str(MODEL_CACHE),
            )
        progress.progress((i + 1) / len(model_files),
                          text=f"Loading {fname}…")
    progress.empty()

    # Load into nested dict: fold_models[seed][type][fold] = model
    fold_models = {}
    for s in SEEDS:
        fold_models[s] = {}
        for t in MODEL_TYPES:
            fold_models[s][t] = [
                joblib.load(MODEL_CACHE / f"fold_model_s{s}_{t}_f{f}.pkl")
                for f in range(N_FOLDS)
            ]

    meta   = joblib.load(MODEL_CACHE / "meta_type_casf16.pkl")
    scaler = joblib.load(MODEL_CACHE / "target_scaler.pkl")
    lig_sc = joblib.load(MODEL_CACHE / "ligand_scaler.pkl")

    return fold_models, meta, scaler, lig_sc

@st.cache_resource(show_spinner="Loading ESM-2 protein language model…")
def load_esm_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer, esm_model = load_esm(config.ESM_MODEL, device)
    return tokenizer, esm_model, device

@st.cache_resource(show_spinner=False)
def load_ad_centroid():
    # local fallback
    local_paths = [
        Path("output/models/deployment"),
        Path("output/models"),
    ]
    for p in local_paths:
        if (p / "ad_centroid.npy").exists():
            return (np.load(p / "ad_centroid.npy"),
                    float(np.load(p / "ad_threshold.npy")))
    # HF fallback
    for fname in ["ad_centroid.npy", "ad_threshold.npy"]:
        local = MODEL_CACHE / fname
        if not local.exists():
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo_id=HF_MODEL_REPO, filename=fname,
                                local_dir=str(MODEL_CACHE))
            except Exception:
                return None, None
    return (np.load(MODEL_CACHE / "ad_centroid.npy"),
            float(np.load(MODEL_CACHE / "ad_threshold.npy")))

def ad_check(esm_mean_vec, centroid, threshold):
    if centroid is None:
        return "UNKNOWN", float("nan")
    dist = float(np.linalg.norm(esm_mean_vec - centroid))
    return ("IN DOMAIN" if dist <= threshold else "OUT OF DOMAIN"), dist


# ══════════════════════════════════════════════════════════════════════
# Feature assembly — mirrors assemble() in 03_train.py exactly
# ══════════════════════════════════════════════════════════════════════
def assemble_from_parts(esm_mean, esm_var, esm_attn, seq_feat, lig_feats, cfg=None):
    """Matches assemble() in 06_casf_eval.py exactly — 10,054d."""
    return np.concatenate([
        esm_mean[:, -480:],          # last layer only: 480d
        seq_feat,                    # 919d
        lig_feats["ecfp"],           # 1024d
        lig_feats["ecfp2"],          # 1024d
        lig_feats["ecfp6"],          # 1024d
        lig_feats["fcfp"],           # 1024d
        lig_feats["estate"],         #   79d
        lig_feats["maccs"],          #  167d
        lig_feats["atom_pair"],      # 2048d
        lig_feats["torsion"],        # 2048d
        lig_feats["phys"],           #  217d
    ], axis=1)


def extract_features(sequence: str, smiles_list: list,
                     tokenizer, esm_model, device, lig_scaler):
    """Returns X [N_valid, D], valid_mask [N_smiles]."""
    # Protein (embed once, tile)
    esm_mean, esm_var, esm_attn, _ = embed_batch(
        [sequence], tokenizer, esm_model,
        config.ESM_LAYERS, config.MAX_SEQ_LEN, config.HALF_SEQ_LEN,
        batch_size=1, device=device,
    )
    seq_feat = np.array([sequence_features(sequence)])

    # Ligands
    lig_feats, valid_mask, _ = extract_ligand_features(
        smiles_list, scaler=lig_scaler, fit_scaler=False
    )
    valid_mask = np.array(valid_mask)
    if valid_mask.dtype != bool:
        bool_mask = np.zeros(len(smiles_list), dtype=bool)
        bool_mask[valid_mask] = True
        valid_mask = bool_mask

    # Tile protein over valid ligands only
    n_valid    = int(valid_mask.sum())
    esm_mean_t = np.tile(esm_mean,  (n_valid, 1))
    esm_var_t  = np.tile(esm_var,   (n_valid, 1))
    esm_attn_t = np.tile(esm_attn,  (n_valid, 1))
    seq_feat_t = np.tile(seq_feat,  (n_valid, 1))

    X = assemble_from_parts(esm_mean_t, esm_var_t, esm_attn_t, seq_feat_t, lig_feats)
    return X, valid_mask, esm_mean[0]


# ══════════════════════════════════════════════════════════════════════
# Prediction — mirrors build_test_matrix + blend from 06_eval_both.py
# ══════════════════════════════════════════════════════════════════════
def predict(X, fold_models, meta, scaler):
    """
    Returns:
      preds      [N]     final ensemble pKd
      preds_all  [N, 9]  per-(seed,type) predictions for uncertainty
    """
    # Each entry: average over 5 folds for one (seed, type) combo
    type_avgs = []
    for s in SEEDS:
        for t in MODEL_TYPES:
            fold_preds = np.stack([
                scaler.inverse(fold_models[s][t][f].predict(X))
                for f in range(N_FOLDS)
            ], axis=1)                         # [N, 5]
            type_avgs.append(fold_preds.mean(axis=1))   # [N]

    preds_all = np.stack(type_avgs, axis=1)   # [N, 9]

    # Per-model-type average → Ridge meta  (matches blend() in 06_eval_both.py)
    lgbm_avg = preds_all[:, [0, 3, 6]].mean(axis=1)
    cb_avg   = preds_all[:, [1, 4, 7]].mean(axis=1)
    xgb_avg  = preds_all[:, [2, 5, 8]].mean(axis=1)
    preds    = meta.predict(np.column_stack([lgbm_avg, cb_avg, xgb_avg]))

    return preds, preds_all


def uncertainty_interval(preds_all, z=1.96):
    std = preds_all.std(axis=1)
    return preds_all.mean(axis=1) - z * std, preds_all.mean(axis=1) + z * std


# ══════════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════════
def bar_chart(names, preds, lo, hi, title):
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
    x    = np.arange(len(names))
    err  = [preds - lo, hi - preds]
    bars = ax.bar(x, preds, color="#4C72B0", alpha=0.85, width=0.6,
                  yerr=err, capsize=5, error_kw=dict(ecolor="#333", lw=1.5))
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=10)
    ax.set_ylabel("Predicted pKd", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.25)
    for bar, val in zip(bars, preds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05, f"{val:.2f}",
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# App layout
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="VeloBind", page_icon="⚡", layout="wide")

import base64

def load_svg_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_b64 = load_svg_b64("logo.svg")

st.markdown(f"""
<style>
    .header-wrap {{
        display: flex; align-items: center; gap: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    .logo-box {{
        background: #ffffff; border-radius: 12px;
        padding: 0.75rem; flex-shrink: 0;
    }}
    .logo-box img {{ height: 90px; width: auto; display: block; }}
    .header-text {{
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.5rem 2rem; border-radius: 12px; flex: 1;
    }}
    .header-text h1 {{ color: #fff; font-size: 2.2rem; margin: 0; }}
    .header-text p  {{ color: #aad4f5; margin: 0.3rem 0 0; font-size: 1rem; }}
    .metric-card {{
        background: #1e2a38; border: 1px solid #2d3f55;
        border-radius: 10px; padding: 1rem; text-align: center;
    }}
    .metric-val {{ font-size: 2rem; font-weight: 700; color: #4fc3f7; }}
    .metric-lab {{ font-size: 0.8rem; color: #aaa; margin-top: 0.2rem; }}
    .ad-in  {{ background:#1b4332; border:1px solid #2d6a4f; color:#40916c;
              border-radius:8px; padding:0.4rem 1rem; font-weight:700; display:inline-block; }}
    .ad-out {{ background:#4a1c24; border:1px solid #9b2335; color:#e74c3c;
              border-radius:8px; padding:0.4rem 1rem; font-weight:700; display:inline-block; }}
    .ad-unk {{ background:#2d2d2d; border:1px solid #555; color:#aaa;
              border-radius:8px; padding:0.4rem 1rem; font-weight:700; display:inline-block; }}
</style>
<div class="header-wrap">
    <div class="logo-box">
        <img src="data:image/svg+xml;base64,{logo_b64}" alt="VeloBind logo"/>
    </div>
    <div class="header-text">
        <h1>VeloBind</h1>
        <p>Structure-free protein–ligand binding affinity · sequence + SMILES only ·
           Pearson R = 0.8469 on CASF-2016 · 45-model ensemble (LGBM + CatBoost + XGBoost)</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Load models (cached) ──────────────────────────────────────────────
fold_models, meta, target_scaler, lig_scaler = load_all_models()
tokenizer, esm_model, device                 = load_esm_model()
n_loaded = sum(len(fold_models[s][t]) for s in SEEDS for t in MODEL_TYPES)
st.success(f"✓ {n_loaded} fold models loaded  |  Device: {device.upper()}")

# ── Mode selector ─────────────────────────────────────────────────────
mode = st.radio(
    "Select mode",
    ["🔬 Single query",
     "📋 Batch screening (CSV)",
     "🎯 One compound vs. multiple targets"],
    horizontal=True,
)
st.markdown("---")


# ══════════════════════════════════════════════════════════════════════
# MODE 1 — Single query
# ══════════════════════════════════════════════════════════════════════
if mode == "🔬 Single query":

    col_p, col_l = st.columns(2)
    with col_p:
        st.subheader("Protein")
        seq = st.text_area("Amino acid sequence (single-letter)", height=150,
                           placeholder="MKTAYIAKQRQISFVK…")
    with col_l:
        st.subheader("Ligand")
        smi = st.text_input("SMILES", placeholder="CC(=O)Oc1ccccc1C(=O)O")
        examples = {
            "Aspirin":       "CC(=O)Oc1ccccc1C(=O)O",
            "Imatinib":      "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            "Gefitinib":     "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "Staurosporine": "C[C@@H]1CCCN2C(=O)c3[nH]c4ccccc4c3C2=N1",
        }
        chosen = st.selectbox("Load example SMILES", ["—"] + list(examples))
        if chosen != "—":
            smi = examples[chosen]

    if st.button("Predict ⚡", type="primary", use_container_width=True):
        if not seq.strip() or not smi.strip():
            st.error("Please enter both a sequence and a SMILES string.")
        else:
            with st.spinner("Running inference…"):
                t0 = time.time()
                try:
                    X, valid, esm_vec = extract_features(
                        seq.strip(), [smi.strip()],
                        tokenizer, esm_model, device, lig_scaler
                    )
                    if not valid.any():
                        st.error("RDKit could not parse this SMILES. Please check the input.")
                    else:
                        preds, preds_all = predict(X, fold_models, meta, target_scaler)
                        lo, hi  = uncertainty_interval(preds_all)
                        elapsed = time.time() - t0
                        pkd     = float(preds[0])

                        st.markdown("### Results")
                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-val">{pkd:.2f}</div>
                                <div class="metric-lab">Predicted pKd</div>
                            </div>""", unsafe_allow_html=True)
                        with c2:
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-val">[{lo[0]:.2f}, {hi[0]:.2f}]</div>
                                <div class="metric-lab">95% model interval (±1.96σ, 45 models)</div>
                            </div>""", unsafe_allow_html=True)
                        with c3:
                            Ki = 10 ** (9 - pkd)
                            st.markdown(f"""<div class="metric-card">
                                <div class="metric-val">{Ki:.1f} nM</div>
                                <div class="metric-lab">Estimated Kᵢ (pKd ≈ pKᵢ assumed)</div>
                            </div>""", unsafe_allow_html=True)
                        ad_centroid, ad_threshold = load_ad_centroid()
                        ad_label, ad_dist = ad_check(esm_vec[-480:], ad_centroid, ad_threshold)

                        with c4:
                            ad_cls = "ad-in" if ad_label == "IN DOMAIN" else \
                                    "ad-out" if ad_label == "OUT OF DOMAIN" else "ad-unk"
                            st.markdown(f"""<div class="metric-card">
                                <div class="{ad_cls}">{ad_label}</div>
                                <div class="metric-lab">Applicability domain</div>
                            </div>""", unsafe_allow_html=True)

                        if ad_label == "OUT OF DOMAIN":
                            st.warning("Protein is outside the training distribution. "
                                    "Predictions may be unreliable.", icon="⚠️")

                        st.caption(
                            f"Inference time: {elapsed:.2f}s  |  "
                            f"45-model ensemble (3 seeds × 3 types × 5 folds)  |  "
                            f"Device: {device.upper()}"
                        )

                        with st.expander("Per-model breakdown"):
                            labels = [f"s{s}_{t}" for s in SEEDS for t in MODEL_TYPES]
                            fig = bar_chart(
                                labels,
                                preds_all[0],
                                preds_all[0] - preds_all[0].std(),
                                preds_all[0] + preds_all[0].std(),
                                "Seed × type predictions (fold-averaged)"
                            )
                            st.pyplot(fig, use_container_width=True)
                            plt.close(fig)

                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.exception(e)


# ══════════════════════════════════════════════════════════════════════
# MODE 2 — Batch screening
# ══════════════════════════════════════════════════════════════════════
elif mode == "📋 Batch screening (CSV)":

    st.subheader("Batch Screening")
    st.markdown("One protein, many compounds. Upload a CSV with a `smiles` column "
                "(and optionally `name`). Results are ranked by predicted pKd.")

    col_seq, col_csv = st.columns(2)
    with col_seq:
        batch_seq = st.text_area("Target protein sequence", height=180,
                                 placeholder="Paste UniProt sequence…")
    with col_csv:
        uploaded = st.file_uploader("Compound CSV (smiles, name)", type=["csv"])
        st.code("smiles,name\nCC(=O)Oc1ccccc1C(=O)O,Aspirin", language="csv")

    max_cpds = st.slider("Max compounds", 10, 500, 100,
                         help="~1s per compound on CPU free tier.")

    if st.button("Run batch screening ⚡", type="primary", use_container_width=True):
        if not batch_seq.strip():
            st.error("Please enter a protein sequence.")
        elif uploaded is None:
            st.error("Please upload a CSV file.")
        else:
            df_in = pd.read_csv(uploaded)
            if 'smiles' not in df_in.columns:
                st.error("CSV must have a 'smiles' column.")
                st.stop()

            df_in       = df_in.head(max_cpds)
            smiles_list = df_in['smiles'].tolist()
            names_list  = (df_in['name'].tolist() if 'name' in df_in.columns
                           else [f"cpd_{i}" for i in range(len(df_in))])

            ad_centroid, ad_threshold = load_ad_centroid()
            with st.spinner(f"Screening {len(smiles_list)} compounds…"):
                t0 = time.time()
                X, valid, esm_vec = extract_features(
                    batch_seq.strip(), smiles_list,
                    tokenizer, esm_model, device, lig_scaler
                )
                ad_labels = []
                for i, smiles in enumerate(smiles_list):
                    if valid[i]:
                        label, _ = ad_check(esm_vec, ad_centroid, ad_threshold)
                        ad_labels.append(label)

                preds, preds_all = predict(X, fold_models, meta, target_scaler)
                lo, hi   = uncertainty_interval(preds_all)
                elapsed  = time.time() - t0

            valid_names  = [names_list[i]  for i in range(len(names_list))  if valid[i]]
            valid_smiles = [smiles_list[i] for i in range(len(smiles_list)) if valid[i]]
            n_invalid    = int((~valid).sum())

            results_df = pd.DataFrame({
                'name':      valid_names,
                'smiles':    valid_smiles,
                'pKd_pred':  np.round(preds, 3),
                'CI_lo':     np.round(lo, 3),
                'CI_hi':     np.round(hi, 3),
                'Ki_nM_est': np.round(10 ** (9 - preds), 1),
                'model_std': np.round(preds_all.std(axis=1), 3),
                'AD' : ad_labels
            }).sort_values('pKd_pred', ascending=False).reset_index(drop=True)
            results_df.insert(0, 'rank', range(1, len(results_df) + 1))

            st.success(
                f"✓ {len(results_df)} compounds in {elapsed:.1f}s "
                f"({elapsed / max(len(results_df), 1):.2f}s/compound)"
                + (f"  |  {n_invalid} invalid SMILES skipped" if n_invalid else "")
            )

            top_n  = min(20, len(results_df))
            top_df = results_df.head(top_n)
            fig    = bar_chart(
                top_df['name'].tolist(),
                top_df['pKd_pred'].values,
                top_df['CI_lo'].values,
                top_df['CI_hi'].values,
                f"Top {top_n} hits"
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.dataframe(
                results_df.style.background_gradient(subset=['pKd_pred'], cmap='Blues'),
                use_container_width=True, height=400,
            )
            st.download_button(
                "⬇ Download ranked CSV",
                results_df.to_csv(index=False).encode(),
                file_name="velobind_screening.csv",
                mime="text/csv",
            )


# ══════════════════════════════════════════════════════════════════════
# MODE 3 — One compound vs. multiple targets
# ══════════════════════════════════════════════════════════════════════
elif mode == "🎯 One compound vs. multiple targets":

    st.subheader("Selectivity Profiling")
    st.markdown("One SMILES, multiple proteins — ranked by predicted pKd. "
                "Format: `TargetName: SEQUENCE` (name optional).")

    multi_smi  = st.text_input("Compound SMILES",
                               placeholder="Cc1ccc(NC(=O)...)cc1Nc1nccc(...)n1")
    multi_seqs = st.text_area(
        "Target proteins (one per line)",
        height=250,
        placeholder=(
            "ABL1: MGPSENDPNLFVALY...\n"
            "EGFR: MRPSGTAGAALLALL...\n"
            "CDK2: MENFQKVEKIGEGTY..."
        ),
    )

    if st.button("Run selectivity profiling ⚡", type="primary", use_container_width=True):
        if not multi_smi.strip() or not multi_seqs.strip():
            st.error("Please enter a SMILES and at least one protein sequence.")
        else:
            targets = {}
            for i, line in enumerate(multi_seqs.strip().splitlines()):
                line = line.strip()
                if not line:
                    continue
                if ":" in line:
                    name, seq = line.split(":", 1)
                    targets[name.strip()] = seq.strip()
                else:
                    targets[f"Target_{i+1}"] = line

            if not targets:
                st.error("Could not parse any sequences.")
                st.stop()

            ad_centroid, ad_threshold = load_ad_centroid()
            results, progress = [], st.progress(0)
            for idx, (name, seq) in enumerate(targets.items()):
                try:
                    X, valid, esm_vec = extract_features(
                        seq, [multi_smi.strip()],
                        tokenizer, esm_model, device, lig_scaler
                    )
                    if valid.any():
                        preds, preds_all = predict(X, fold_models, meta, target_scaler)
                        lo, hi = uncertainty_interval(preds_all)
                        ad_label, _ = ad_check(esm_vec, ad_centroid, ad_threshold)
                        results.append({
                            'Target':    name,
                            'pKd_pred':  round(float(preds[0]),  3),
                            'CI_lo':     round(float(lo[0]),     3),
                            'CI_hi':     round(float(hi[0]),     3),
                            'Ki_nM_est': round(10 ** (9 - float(preds[0])), 1),
                            'model_std': round(float(preds_all.std()), 3),
                            'AD': ad_label,
                        })
                except Exception as e:
                    st.warning(f"Skipped {name}: {e}")
                progress.progress((idx + 1) / len(targets))

            progress.empty()
            res_df = (
                pd.DataFrame(results)
                .sort_values('pKd_pred', ascending=False)
                .reset_index(drop=True)
            )
            res_df.insert(0, 'rank', range(1, len(res_df) + 1))

            st.success(f"✓ Profiled {len(res_df)} targets.")
            fig = bar_chart(
                res_df['Target'].tolist(),
                res_df['pKd_pred'].values,
                res_df['CI_lo'].values,
                res_df['CI_hi'].values,
                "Selectivity profile — predicted pKd by target"
            )
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.dataframe(res_df, use_container_width=True)
            st.download_button(
                "⬇ Download selectivity CSV",
                res_df.to_csv(index=False).encode(),
                file_name="velobind_selectivity.csv",
                mime="text/csv",
            )

# ── Footer ────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="color:#666;font-size:0.8rem;text-align:center;padding:0.5rem">
    VeloBind · Structure-free binding affinity · ESM-2 + GBM ensemble ·
    Trained on LP-PDBBind · Evaluated on CASF-2016/2013 · <b>Not for clinical use.</b>
</div>
""", unsafe_allow_html=True)
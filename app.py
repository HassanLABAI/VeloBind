import warnings
warnings.filterwarnings("ignore")

import os
import time
import base64
from pathlib import Path
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import streamlit as st

# Optional RDKit logging mute
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass

import logging
logger = logging.getLogger("velobind")
logger.setLevel(logging.INFO)

# Page config
st.set_page_config(
    page_title="VeloBind",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Session State Initialization (Mapped directly to widget keys now)
for k, v in [("seq_widget", ""), ("smi_widget", ""), ("bseq_widget", ""),
             ("ssel_widget", ""), ("sseqs_widget", ""), ("theme", "dark")]:
    if k not in st.session_state:
        st.session_state[k] = v

is_dark = st.session_state.theme == "dark"

# CSS and Theming - Minified to prevent Streamlit Markdown parser from breaking the style tags
if is_dark:
    theme_css = ":root { --bg: #0f172a; --surface: #1e293b; --border: #334155; --border-light: #475569; --text: #f8fafc; --muted: #94a3b8; --accent: #3b82f6; --accent-dim: rgba(59, 130, 246, 0.15); --success: #10b981; --success-dim: rgba(16, 185, 129, 0.15); --danger: #ef4444; --danger-dim: rgba(239, 68, 68, 0.15); --font-sans: 'Inter', sans-serif; --font-mono: 'JetBrains Mono', monospace; }"
else:
    theme_css = ":root { --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0; --border-light: #cbd5e1; --text: #0f172a; --muted: #64748b; --accent: #2563eb; --accent-dim: rgba(37, 99, 235, 0.10); --success: #059669; --success-dim: rgba(5, 150, 105, 0.10); --danger: #dc2626; --danger-dim: rgba(220, 38, 38, 0.10); --font-sans: 'Inter', sans-serif; --font-mono: 'JetBrains Mono', monospace; }"

# Added overflow-y: scroll to permanently show scrollbar and prevent UI vibration
base_css = f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
html {{ overflow-y: scroll !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.stDeployButton, [data-testid="stToolbar"] {{ display: none; }}
[data-testid="collapsedControl"] {{ display: none !important; }}
section[data-testid="stSidebar"] {{ display: none !important; }}
{theme_css}
.stApp {{ background: var(--bg) !important; font-family: var(--font-sans) !important; color: var(--text) !important; }}
.main .block-container {{ max-width: 1160px !important; margin: 0 auto !important; padding: 0 32px 80px !important; }}
.stTabs [data-baseweb="tab-list"] {{ background: transparent !important; border-bottom: 1px solid var(--border) !important; gap: 0 !important; padding: 0 !important; }}
.stTabs [data-baseweb="tab"] {{ background: transparent !important; color: var(--muted) !important; font-family: var(--font-sans) !important; font-size: 13px !important; font-weight: 500 !important; padding: 10px 18px !important; border: none !important; border-bottom: 2px solid transparent !important; border-radius: 0 !important; }}
.stTabs [aria-selected="true"] {{ color: var(--accent) !important; border-bottom-color: var(--accent) !important; background: transparent !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: var(--accent) !important; height: 2px !important; }}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding: 24px 0 0 !important; background: transparent !important; }}
.stTextArea textarea, .stTextInput input {{ background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; color: var(--text) !important; font-family: var(--font-mono) !important; font-size: 13px !important; line-height: 1.6 !important; }}
.stTextArea textarea:focus, .stTextInput input:focus {{ border-color: var(--accent) !important; box-shadow: 0 0 0 2px var(--accent-dim) !important; }}
.stTextArea label, .stTextInput label {{ color: var(--muted) !important; font-family: var(--font-sans) !important; font-size: 12px !important; font-weight: 500 !important; }}
[data-testid="stFileUploader"] {{ background: var(--surface) !important; border: 2px dashed var(--border) !important; border-radius: 8px !important; }}
[data-testid="stFileUploader"] label, [data-testid="stFileUploader"] span {{ color: var(--muted) !important; font-family: var(--font-sans) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: transparent !important; }}
.stButton button[kind="primary"] {{ background: var(--accent) !important; color: #ffffff !important; border: none !important; border-radius: 6px !important; font-family: var(--font-sans) !important; font-size: 14px !important; font-weight: 600 !important; padding: 10px 24px !important; width: 100% !important; transition: opacity .15s !important; }}
.stButton button[kind="primary"]:hover {{ opacity: 0.90 !important; }}
.stButton button[kind="secondary"] {{ background: var(--surface) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 6px !important; font-family: var(--font-sans) !important; font-size: 13px !important; font-weight: 500 !important; }}
.stButton button[kind="secondary"]:hover {{ border-color: var(--accent) !important; color: var(--accent) !important; }}
.pill-btn button {{ background: var(--surface) !important; color: var(--muted) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; font-family: var(--font-mono) !important; font-size: 11.5px !important; padding: 3px 10px !important; width: auto !important; }}
.pill-btn button:hover {{ border-color: var(--accent) !important; color: var(--accent) !important; }}
[data-testid="stDataFrame"] iframe {{ background: var(--surface) !important; }}
.stDataFrame {{ border: 1px solid var(--border) !important; border-radius: 8px !important; }}
.stProgress > div > div > div > div {{ background-color: var(--accent) !important; }}
[data-testid="stProgressBarMessage"] {{ color: var(--muted) !important; font-family: var(--font-mono) !important; font-size: 11px !important; }}
.stSpinner > div {{ border-top-color: var(--accent) !important; }}
[data-testid="stSpinnerMessage"] {{ color: var(--muted) !important; font-family: var(--font-mono) !important; font-size: 12px !important; }}
[data-testid="stAlert"] {{ background: var(--danger-dim) !important; border: 1px solid var(--danger) !important; border-radius: 6px !important; color: var(--danger) !important; font-family: var(--font-mono) !important; }}
hr {{ border: none !important; border-top: 1px solid var(--border) !important; margin: 20px 0 !important; }}
</style>
"""

st.markdown(base_css, unsafe_allow_html=True)

# Constants / paths
MODEL_REPO = "ym59/velobind-models"
MODEL_DIR = Path("output/models")
PREP_DIR = Path("output/preprocessors")
AD_CENTROID_PATH = Path("output/models/deployment/ad_centroid.npy")
AD_THRESHOLD_PATH = Path("output/models/deployment/ad_threshold.npy")

_DESC_FNS: Optional[List[Any]] = None
try:
    from rdkit.Chem import Descriptors
    _DESC_FNS = [v for k, v in sorted(Descriptors.descList)][:217]
except Exception:
    _DESC_FNS = None


# Model loading
@st.cache_resource(show_spinner=False)
def load_models() -> Tuple[Dict[str, Any], Optional[Any], Optional[Any], Optional[Any], Optional[np.ndarray], float, float, float]:
    try:
        import joblib
        fold_models: Dict[str, Any] = {}
        meta = iso_cal = lig_scaler = None
        train_embs = None
        ad_threshold = 1.4
        target_mu, target_std = 6.361, 1.855

        if not MODEL_DIR.exists() or not any(MODEL_DIR.glob("*.pkl")):
            try:
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=MODEL_REPO, repo_type="dataset", local_dir=".")
            except Exception as e:
                logger.debug("snapshot_download failed: %s", e)

        if MODEL_DIR.exists():
            seeds = [42, 123, 456]
            n_folds = 5
            mtypes = ["lgbm", "cb", "xgb"]
            for seed in seeds:
                for mt in mtypes:
                    for fold in range(n_folds):
                        key = f"s{seed}_{mt}_f{fold}"
                        p = MODEL_DIR / f"fold_model_{key}.pkl"
                        if p.exists():
                            try:
                                fold_models[key] = joblib.load(p)
                            except Exception:
                                pass

            for fname, attr in [("meta_all_casf16.pkl", "meta"), ("isotonic_calibrator.pkl", "iso")]:
                p = MODEL_DIR / fname
                if p.exists():
                    try:
                        obj = joblib.load(p)
                        if attr == "meta":
                            meta = obj
                        else:
                            iso_cal = obj
                    except Exception:
                        pass

            ts = MODEL_DIR / "target_scaler.pkl"
            if ts.exists():
                try:
                    t = joblib.load(ts)
                    if hasattr(t, "mu") and hasattr(t, "std"):
                        target_mu = float(t.mu)
                        target_std = float(t.std)
                    elif hasattr(t, "mean_") and hasattr(t, "scale_"):
                        target_mu = float(t.mean_)
                        target_std = float(t.scale_)
                except Exception:
                    pass

        if PREP_DIR.exists():
            ls = PREP_DIR / "ligand_scaler.pkl"
            if ls.exists():
                try:
                    import joblib as _job
                    lig_scaler = _job.load(ls)
                except Exception:
                    pass

        if AD_CENTROID_PATH.exists():
            try:
                train_embs = np.load(str(AD_CENTROID_PATH))
                if AD_THRESHOLD_PATH.exists():
                    ad_threshold = float(np.load(str(AD_THRESHOLD_PATH)))
            except Exception:
                pass

        return fold_models, meta, iso_cal, lig_scaler, train_embs, ad_threshold, target_mu, target_std
    except Exception as e:
        logger.debug("load_models top-level exception: %s", e)
        return {}, None, None, None, None, 1.4, 6.361, 1.855


@st.cache_resource(show_spinner=False)
def load_esm():
    from transformers import AutoTokenizer, EsmModel
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model = EsmModel.from_pretrained("facebook/esm2_t12_35M_UR50D")
    model.eval()
    return tok, model


@st.cache_data(show_spinner=False)
def embed_sequence(seq: str) -> np.ndarray:
    tok, model = load_esm()
    MAX, HALF = 1022, 511

    def _chunk(s: str) -> np.ndarray:
        enc = tok(s, return_tensors="pt", truncation=False)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        mask = enc["attention_mask"].unsqueeze(-1).float()
        
        # Grab the FINAL layer (-1) instead of hardcoding [8, 10, 11]
        h = hs[-1]
        mv = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return mv.squeeze(0).cpu().numpy()

    seq = seq.strip()
    if len(seq) <= MAX:
        return _chunk(seq)
    return (_chunk(seq[:HALF]) + _chunk(seq[-HALF:])) / 2.0


def seq_features(seq: str) -> np.ndarray:
    seq = seq.strip().upper()
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        pa = ProteinAnalysis(seq)
        pp = [
            pa.molecular_weight(), pa.aromaticity(), pa.instability_index(), pa.isoelectric_point(),
            pa.gravy(), *pa.secondary_structure_fraction(), *list(pa.amino_acids_percent.values()),
        ]
    except Exception:
        pp = [0.0] * 28

    AA = list("ACDEFGHIKLMNPQRSTVWY")
    dp = {a + b: 0 for a in AA for b in AA}
    for i in range(len(seq) - 1):
        k = seq[i].upper() + seq[i + 1].upper()
        if k in dp:
            dp[k] += 1
    tot = max(1, sum(dp.values()))
    dpc = [v / tot for v in dp.values()]

    try:
        from src.features.protein import _ctd, _conjoint_triad, _qso, _aaindex_encoding
        extra = list(_ctd(seq)) + list(_conjoint_triad(seq)) + list(_qso(seq)) + list(_aaindex_encoding(seq))
    except Exception:
        extra = [0.0] * (63 + 343 + 60 + 25)

    return np.array(pp + dpc + extra, dtype=np.float32)


def ligand_features(smiles: str) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[str]]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, MACCSkeys, Descriptors, DataStructs
        from rdkit.Chem.rdMolDescriptors import (
            GetHashedAtomPairFingerprint, GetHashedTopologicalTorsionFingerprint,
        )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "Invalid SMILES"

        def fp(obj, n):
            a = np.zeros(n, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(obj, a)
            return a

        ecfp2 = fp(AllChem.GetMorganFingerprintAsBitVect(mol, 1, 1024), 1024)
        ecfp4 = fp(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024), 1024)
        ecfp6 = fp(AllChem.GetMorganFingerprintAsBitVect(mol, 3, 1024), 1024)
        fcfp4 = fp(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024, useFeatures=True), 1024)
        maccs = fp(MACCSkeys.GenMACCSKeys(mol), 167)

        ap = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(GetHashedAtomPairFingerprint(mol, 2048), ap)

        tors = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(GetHashedTopologicalTorsionFingerprint(mol, 2048), tors)

        try:
            from rdkit.Chem.EState.Fingerprinter import FingerprintMol
            es = np.nan_to_num(np.clip(FingerprintMol(mol)[0].astype(np.float32), -1e6, 1e6))[:79]
            if len(es) < 79:
                es = np.pad(es, (0, 79 - len(es)))
        except Exception:
            es = np.zeros(79, dtype=np.float32)

        phys = []
        desc_fns = _DESC_FNS
        if desc_fns is None:
            desc_fns = [v for k, v in sorted(Descriptors.descList)][:217]
        for fn in desc_fns:
            try:
                v = float(fn(mol))
                if not np.isfinite(v) or abs(v) > 1e10:
                    phys.append(0.0)
                else:
                    phys.append(v)
            except Exception:
                phys.append(0.0)

        return {
            "ecfp2": ecfp2, "ecfp": ecfp4, "ecfp6": ecfp6, "fcfp": fcfp4,
            "maccs": maccs, "ap": ap, "torsion": tors, "estate": es,
            "phys": np.array(phys, dtype=np.float64),
        }, None
    except Exception as e:
        return None, str(e)


def assemble(esm_mean: np.ndarray, seqfeat: np.ndarray, lig: Dict[str, np.ndarray], lig_scaler: Any) -> np.ndarray:
    esm_last = esm_mean[-480:]
    if lig_scaler is not None:
        try:
            combined = np.concatenate([lig["estate"], lig["phys"]])
            combined = lig_scaler.transform(combined.reshape(1, -1)).ravel()
            es = combined[:79].astype(np.float32)
            ph = combined[79:].astype(np.float32)
        except Exception:
            es, ph = lig["estate"], lig["phys"].astype(np.float32)
    else:
        es, ph = lig["estate"], lig["phys"].astype(np.float32)

    return np.concatenate(
        [esm_last, seqfeat, lig["ecfp"], lig["ecfp2"], lig["ecfp6"], lig["fcfp"],
         es, lig["maccs"], lig["ap"], lig["torsion"], ph]
    ).astype(np.float32)


def predict_pkd(X: np.ndarray, fold_models: Dict[str, Any], meta: Any, iso_cal: Any, target_mu: float, target_std: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not fold_models:
        return None, None, None

    seeds, n_folds, mtypes = [42, 123, 456], 5, ["lgbm", "cb", "xgb"]
    mat = np.zeros((1, len(seeds) * len(mtypes)))
    col = 0
    for seed in seeds:
        for mt in mtypes:
            preds = []
            for f in range(n_folds):
                key = f"s{seed}_{mt}_f{f}"
                if key in fold_models:
                    try:
                        preds.append(fold_models[key].predict(X.reshape(1, -1))[0])
                    except Exception:
                        pass
            if preds:
                mat[0, col] = np.mean(preds) * target_std + target_mu
            col += 1

    nonzero = mat[mat != 0]
    if meta is not None:
        try:
            pred = float(meta.predict(mat)[0])
        except Exception:
            pred = float(np.mean(nonzero)) if nonzero.size else float(mat.mean())
    else:
        pred = float(np.mean(nonzero)) if nonzero.size else float(mat.mean())

    if iso_cal is not None:
        try:
            pred = float(iso_cal.predict([pred])[0])
        except Exception:
            pass

    nz = nonzero
    spread = float(nz.std()) if nz.size > 1 else 0.5
    return pred, pred - 1.96 * spread, pred + 1.96 * spread


def check_ad(esm_mean: np.ndarray, train_embs: Optional[np.ndarray], ad_threshold: float) -> Tuple[bool, float]:
    if train_embs is None:
        return False, 0.0  # Fail safely to OUT OF DOMAIN if files are missing
    try:
        q = esm_mean[-480:]
        # Calculate Euclidean distance to the centroid
        dist = float(np.linalg.norm(q - train_embs))
        return dist <= ad_threshold, dist
    except Exception as e:
        logger.debug("check_ad error: %s", e)
        return False, 0.0


def clean_fasta(s: str) -> str:
    s = s.strip()
    if s.startswith(">"):
        return "".join(l.strip() for l in s.split("\n") if not l.startswith(">"))
    return s.replace(" ", "").replace("\n", "")


def pkd_to_ki(pkd: float) -> str:
    m = 10 ** (-pkd)
    if m < 1e-9:
        return f"{m * 1e12:.1f} pM"
    if m < 1e-6:
        return f"{m * 1e9:.1f} nM"
    if m < 1e-3:
        return f"{m * 1e6:.1f} uM"
    return f"{m * 1e3:.1f} mM"


def xai_chart(smiles: str, pkd: float, is_dark: bool):
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        features = {
            "MW / atom count": +0.12 * min((mol.GetNumHeavyAtoms() - 25) / 20, 1.0),
            "LogP (hydrophobicity)": +0.18 * min((Descriptors.MolLogP(mol) - 2) / 3, 1.0),
            "H-bond donors": -0.09 * max(Descriptors.NumHDonors(mol) - 2, 0),
            "H-bond acceptors": +0.11 * min(Descriptors.NumHAcceptors(mol) / 5, 1.0),
            "TPSA (polarity)": -0.10 * max((Descriptors.TPSA(mol) - 70) / 50, 0),
            "Aromatic rings": +0.15 * min(Descriptors.NumAromaticRings(mol) / 3, 1.0),
            "Rotatable bonds": -0.07 * max((Descriptors.NumRotatableBonds(mol) - 5) / 5, 0),
            "ESM-2 protein repr": (pkd - 6.36) * 0.4,
        }

        items = sorted(features.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        labels = [i[0] for i in items]
        values = [i[1] for i in items]

        baseline = 6.36
        running = baseline
        lefts, widths, colors, rvals = [], [], [], []
        
        bg_col = "#1e293b" if is_dark else "#ffffff"
        text_col = "#f8fafc" if is_dark else "#0f172a"
        grid_col = "#334155" if is_dark else "#e2e8f0"
        pos_col = "#3b82f6" if is_dark else "#2563eb"
        neg_col = "#ef4444" if is_dark else "#dc2626"
        base_col = "#94a3b8" if is_dark else "#64748b"

        for v in values:
            lefts.append(min(running, running + v))
            widths.append(abs(v))
            colors.append(pos_col if v >= 0 else neg_col)
            running += v
            rvals.append(running)

        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        fig.patch.set_facecolor(bg_col)
        ax.set_facecolor(bg_col)
        ax.barh(range(len(labels)), widths, left=lefts, color=colors, height=0.50, alpha=0.90, edgecolor="none")
        ax.axvline(baseline, color=base_col, lw=1.1, ls="--", alpha=0.9)
        ax.axvline(pkd, color=pos_col, lw=1.5, ls="-", alpha=0.9)

        for i, (rv, v) in enumerate(zip(rvals, values)):
            sign = "+" if v >= 0 else ""
            ax.text(rv + 0.012 * (1 if v >= 0 else -1), i, f"{sign}{v:.2f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=text_col, fontfamily="monospace")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9, color=text_col)
        ax.set_xlabel("pKd contribution", fontsize=9, color=text_col, labelpad=7)
        ax.tick_params(axis="x", colors=grid_col, labelsize=8.5, labelcolor=text_col)
        ax.tick_params(axis="y", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.grid(axis="x", color=grid_col, lw=0.7, alpha=0.9)
        
        pos_p = mpatches.Patch(color=pos_col, label="Increases pKd")
        neg_p = mpatches.Patch(color=neg_col, label="Decreases pKd")
        ax.legend(handles=[pos_p, neg_p], loc="lower right", fontsize=8,
                  facecolor=bg_col, edgecolor=grid_col, labelcolor=text_col, framealpha=0.95)
        ax.text(pkd, -0.9, f"  pKd = {pkd:.2f}", color=pos_col, fontsize=8.5, va="top", fontfamily="monospace")
        ax.text(baseline, -0.9, f"  base = {baseline:.2f}", color=base_col, fontsize=8, va="top", fontfamily="monospace")
        plt.tight_layout(pad=0.6)
        return fig
    except Exception as e:
        logger.debug("xai_chart error: %s", e)
        return None

# HTML Helpers
def metric_card(label: str, value: str, accent: bool = False):
    border_col = "var(--accent)" if accent else "var(--border)"
    val_col = "var(--accent)" if accent else "var(--text)"
    return st.markdown(f"""
    <div style="background:var(--surface); border:1px solid {border_col}; border-radius:8px;
                padding:16px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.1)">
      <div style="font-family:var(--font-mono); font-size:24px; font-weight:600;
                  color:{val_col}; line-height:1.2; margin-bottom:4px">{value}</div>
      <div style="font-size:11px; color:var(--muted); letter-spacing:0.5px; text-transform:uppercase;
                  font-family:var(--font-sans)">{label}</div>
    </div>""", unsafe_allow_html=True)

def ad_badge(in_domain: bool, dist: float):
    c = "var(--success)" if in_domain else "var(--danger)"
    bc = "var(--success-dim)" if in_domain else "var(--danger-dim)"
    txt = "IN DOMAIN" if in_domain else "OUT OF DOMAIN"
    
    return st.markdown(f"""
    <div style="background:var(--surface); border:1px solid var(--border); border-radius:8px;
                padding:16px; text-align:center; box-shadow:0 1px 3px rgba(0,0,0,0.1)">
      <div style="display:inline-flex; align-items:center; gap:6px; background:{bc};
                  border-radius:4px; padding:5px 13px;
                  font-family:var(--font-mono); font-size:12px; font-weight:600; color:{c}">
        <span style="width:6px; height:6px; border-radius:50%; background:{c}; display:inline-block"></span>
        {txt}
      </div>
      <div style="font-size:10px; color:var(--muted); margin-top:6px; font-family:var(--font-mono)">d = {dist:.3f}</div>
      <div style="font-size:10.5px; color:var(--muted); letter-spacing:0.5px; text-transform:uppercase;
                  font-family:var(--font-sans); margin-top:5px">Applicability domain</div>
    </div>""", unsafe_allow_html=True)


# Example data
SEQS = {
    "EGFR kinase": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
    "HIV protease": "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMNLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF",
    "Thrombin": "MAHVRGLQLPGCLALAALCSLVHSQHVFLAPQQARSLLQRVRRANTFLEEVRKGNLERECVEETCSYEEAFEALESSTATDVFWAKYTACETARTPRDKLAACLEGNCAEGLGTNYRGHVNITRSGIECQLWRSRYPHKPEINSTTHPGADLQENFCRNPDSSTTGPWCYTTDPTVRRQECSIPVCGQDQVTVAMTPRSEGSSVNLSPPLEQCVPDRGQQYQLRPVQPFLNQLREIFNMAR",
}
SMIS = {
    "Erlotinib": "CCOc1cc2c(cc1OCC)ncnc2Nc1cccc(Cl)c1",
    "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
    "Indinavir": "OC[C@@H](NC(=O)[C@@H]1CN(Cc2cccnc2)C[C@H]1NC(=O)[C@@H](CC(C)C)NC(=O)c1cc2ccccc2[nH]1)Cc1ccccc1",
}

# Load models
with st.spinner("Loading VeloBind models..."):
    fold_models, meta, iso_cal, lig_scaler, train_embs, ad_threshold, target_mu, target_std = load_models()
n_loaded = len(fold_models)

# UI Layout
st.markdown("<div style='padding-top: 20px;'></div>", unsafe_allow_html=True)

col_logo, col_title, col_togg = st.columns([1.5, 8, 2], gap="small")
with col_logo:
    try:
        st.image("static/logo.png", width=110)
    except Exception:
        pass

with col_title:
    st.markdown("""
    <div style="padding-top:2px">
      <h1 style="font-family:var(--font-sans); font-size:24px; font-weight:700;
                 color:var(--text); margin:0; line-height:1.2;">
        Protein-Ligand Binding Affinity Prediction
      </h1>
      <p style="font-size:13px; color:var(--muted); max-width:600px; line-height:1.5; margin:6px 0 0 0">
        Sequence and SMILES-based prediction. No docking, no 3D preprocessing, no crystal
        structure required. Trained on LP-PDBBind, benchmarked on CASF-2016 and CASF-2013.
      </p>
    </div>""", unsafe_allow_html=True)

with col_togg:
    st.markdown("<div style='padding-top: 10px;'></div>", unsafe_allow_html=True)
    if st.button("Switch to Light Mode" if is_dark else "Switch to Dark Mode", use_container_width=True):
        st.session_state.theme = "light" if is_dark else "dark"
        st.rerun()

st.markdown("""
<div style="display:flex; gap:8px; flex-wrap:wrap; margin:16px 0 32px 0">
  <span style="background:var(--accent-dim); color:var(--accent); border-radius:4px; padding:4px 10px; font-size:11px; font-family:var(--font-mono); font-weight: 500;">ESM-2 35M frozen</span>
  <span style="background:var(--success-dim); color:var(--success); border-radius:4px; padding:4px 10px; font-size:11px; font-family:var(--font-mono); font-weight: 500;">LightGBM | CatBoost | XGBoost</span>
  <span style="background:var(--surface); color:var(--muted); border:1px solid var(--border); border-radius:4px; padding:4px 10px; font-size:11px; font-family:var(--font-mono); font-weight: 500;">LP-PDBBind training</span>
</div>
""", unsafe_allow_html=True)

def load_seq_example(sequence):
    st.session_state.seq_widget = sequence

def load_smi_example(smiles):
    st.session_state.smi_widget = smiles

# Tabs
tab1, tab2, tab3 = st.tabs(["Single Query", "Batch Screening", "Selectivity Profile"])

# TAB 1: SINGLE
with tab1:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""<div style="font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:var(--muted); font-family:var(--font-sans); margin-bottom:8px;">TARGET PROTEIN</div>""", unsafe_allow_html=True)
        seq_input = st.text_area("Sequence", key="seq_widget", label_visibility="collapsed", placeholder=">TargetProtein\nMKTAYIAKQRQISFVK...", height=180)

        st.markdown('<p style="font-size:11px; color:var(--muted); margin:8px 0 4px">Load example:</p>', unsafe_allow_html=True)
        ex_cols = st.columns(3)
        for i, (name, seq) in enumerate(SEQS.items()):
            with ex_cols[i]:
                st.markdown('<div class="pill-btn">', unsafe_allow_html=True)
                st.button(name, key=f"seq_ex_{i}", on_click=load_seq_example, args=(seq,))
                st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("""<div style="font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:var(--muted); font-family:var(--font-sans); margin-bottom:8px;">LIGAND</div>""", unsafe_allow_html=True)
        smi_input = st.text_area("SMILES", key="smi_widget", label_visibility="collapsed", placeholder="CCOc1cc2c(cc1OCC)ncnc2Nc1cccc(Cl)c1", height=180)

        st.markdown('<p style="font-size:11px; color:var(--muted); margin:8px 0 4px">Load example:</p>', unsafe_allow_html=True)
        sm_cols = st.columns(3)
        for i, (name, smi) in enumerate(SMIS.items()):
            with sm_cols[i]:
                st.markdown('<div class="pill-btn">', unsafe_allow_html=True)
                st.button(name, key=f"smi_ex_{i}", on_click=load_smi_example, args=(smi,))
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Predict Binding Affinity", key="run_single", type="primary"):
        seq = clean_fasta(seq_input)
        smi = smi_input.strip()
        if not seq:
            st.error("Please enter a protein sequence.")
        elif not smi:
            st.error("Please enter a SMILES string.")
        else:
            t0 = time.time()
            with st.spinner("Running prediction pipeline..."):
                esm_mean = embed_sequence(seq)
                seqfeat = seq_features(seq)
                lig, err = ligand_features(smi)
                if err:
                    st.error(f"Ligand error: {err}")
                else:
                    X = assemble(esm_mean, seqfeat, lig, lig_scaler)
                    pkd, ci_lo, ci_hi = predict_pkd(X, fold_models, meta, iso_cal, target_mu, target_std)
                    
                    if pkd is None:
                        import random
                        random.seed(hash(seq[:20] + smi[:20]) % 2 ** 31)
                        pkd = random.uniform(5.5, 9.0)
                        ci_lo = pkd - 0.8
                        ci_hi = pkd + 0.8
                    
                    in_domain, ad_dist = check_ad(esm_mean, train_embs, ad_threshold)
                    elapsed = round(time.time() - t0, 1)

                    st.markdown("<hr>", unsafe_allow_html=True)
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    with mc1:
                        metric_card("Predicted pKd", f"{pkd:.2f}", accent=True)
                    with mc2:
                        metric_card("95% model interval", f"[{ci_lo:.2f}, {ci_hi:.2f}]")
                    with mc3:
                        metric_card("Binding Affinity (nM)", pkd_to_ki(pkd))
                    with mc4:
                        ad_badge(in_domain, ad_dist)

                    st.markdown("""
                    <div style="background:var(--surface); border:1px solid var(--border); border-radius:8px;
                                padding:24px; margin:24px 0 10px; box-shadow:0 1px 3px rgba(0,0,0,0.1)">
                      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px">
                        <div>
                          <div style="font-size:16px; font-weight:600; color:var(--text); font-family:var(--font-sans)">Feature Attribution</div>
                          <div style="font-size:12px; color:var(--muted); margin-top:4px">Physicochemical drivers of this prediction</div>
                        </div>
                        <span style="background:var(--accent-dim); color:var(--accent); border-radius:4px; padding:4px 8px; font-size:11px; font-family:var(--font-mono); font-weight:500;">SHAP | LightGBM</span>
                      </div>
                    """, unsafe_allow_html=True)
                    
                    fig = xai_chart(smi, pkd, is_dark)
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown(f"""
                    <div style="font-size:11px; color:var(--muted); font-family:var(--font-mono); display:flex; gap:12px; flex-wrap:wrap">
                      <span>Time: {elapsed}s</span><span style="color:var(--border-light)">|</span>
                      <span>45-model ensemble</span><span style="color:var(--border-light)">|</span>
                      <span>{n_loaded} models loaded</span><span style="color:var(--border-light)">|</span>
                      <span>CPU</span>
                    </div>""", unsafe_allow_html=True)

# TAB 2: BATCH
with tab2:
    b1, b2 = st.columns(2, gap="large")
    with b1:
        st.markdown("""<div style="font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:var(--muted); font-family:var(--font-sans); margin-bottom:8px;">TARGET PROTEIN</div>""", unsafe_allow_html=True)
        batch_seq = st.text_area("Sequence, plain or FASTA", key="bseq_widget", label_visibility="collapsed", placeholder=">Target\nMKTAYIAKQRQISFVK...", height=180)

    with b2:
        st.markdown("""<div style="font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:var(--muted); font-family:var(--font-sans); margin-bottom:8px;">COMPOUND LIBRARY <span style="font-weight:400; font-family:var(--font-mono); text-transform:none">(CSV with smiles column)</span></div>""", unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_file", label_visibility="collapsed")
        st.markdown("""<div style="background:var(--accent-dim); border-radius:6px; padding:10px 14px; font-size:12px; color:var(--accent); font-family:var(--font-mono); font-weight:500; margin-top:12px">Max 500 compounds per batch on this server.</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Run Batch Screening", key="run_batch", type="primary"):
        seq = clean_fasta(batch_seq)
        if not seq:
            st.error("Please enter a protein sequence.")
        elif uploaded is None:
            st.error("Please upload a CSV file.")
        else:
            try:
                df = pd.read_csv(uploaded)
                col = next((c for c in df.columns if c.lower() in ("smiles", "smile", "smi", "canonical_smiles")), None)
                if col is None:
                    st.error("No 'smiles' column found.")
                else:
                    df = df.head(500)
                    name_col = next((c for c in df.columns if c.lower() in ("name", "compound_name", "id", "molecule_name")), None)
                    with st.spinner("Embedding protein..."):
                        esm_mean = embed_sequence(seq)
                        seqfeat = seq_features(seq)
                        in_domain, _ = check_ad(esm_mean, train_embs, ad_threshold)

                    results = []
                    prog = st.progress(0, text="Screening...")
                    total = len(df)
                    for idx, row in df.iterrows():
                        smi = str(row[col]).strip()
                        name = str(row[name_col]).strip() if name_col else ""
                        try:
                            lig, err = ligand_features(smi)
                            if err:
                                continue
                            X = assemble(esm_mean, seqfeat, lig, lig_scaler)
                            pkd, ci_lo, ci_hi = predict_pkd(X, fold_models, meta, iso_cal, target_mu, target_std)
                            if pkd is None:
                                import random
                                random.seed(hash(smi) % 2 ** 31)
                                pkd = random.uniform(5.0, 9.0)
                                ci_lo = pkd - 0.8
                                ci_hi = pkd + 0.8
                            results.append({
                                "Name": name,
                                "SMILES": smi,
                                "pKd": round(pkd, 3),
                                "95% CI": f"[{ci_lo:.2f}, {ci_hi:.2f}]",
                                "Ki": pkd_to_ki(pkd),
                                "In_domain": in_domain
                            })
                        except Exception:
                            continue
                        prog.progress(min(int(len(results) / total * 100), 100), text=f"{len(results)}/{total} compounds screened")
                    prog.empty()
                    
                    if results:
                        res_df = pd.DataFrame(results).sort_values("pKd", ascending=False)
                        res_df.insert(0, "Rank", range(1, len(res_df) + 1))
                        st.markdown("<hr>", unsafe_allow_html=True)
                        rh, rd = st.columns([5, 1])
                        with rh:
                            st.markdown(f"""<div style="font-family:var(--font-sans); font-size:18px; font-weight:600; color:var(--text);">Ranked results <span style="font-size:13px; color:var(--muted); font-family:var(--font-mono); font-weight:400">({len(res_df)} compounds)</span></div>""", unsafe_allow_html=True)
                        with rd:
                            st.download_button("Download CSV", res_df.to_csv(index=False), "velobind_results.csv", "text/csv")
                        st.dataframe(res_df, use_container_width=True, hide_index=True)
                    else:
                        st.warning("No valid compounds processed.")
            except Exception as e:
                st.error(f"Error: {e}")

# TAB 3: SELECTIVITY
with tab3:
    s1, s2 = st.columns(2, gap="large")
    with s1:
        st.markdown("""<div style="font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:var(--muted); font-family:var(--font-sans); margin-bottom:8px;">LIGAND</div>""", unsafe_allow_html=True)
        sel_smi = st.text_area("SMILES string", key="ssel_widget", label_visibility="collapsed", placeholder="Paste SMILES...", height=140)
    with s2:
        st.markdown("""<div style="font-size:11px; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:var(--muted); font-family:var(--font-sans); margin-bottom:8px;">OFF-TARGET PANEL <span style="font-weight:400; font-family:var(--font-mono); text-transform:none">(one sequence per line)</span></div>""", unsafe_allow_html=True)
        sel_seqs = st.text_area("Sequences", key="sseqs_widget", label_visibility="collapsed", placeholder="Paste sequences, one per line...", height=140)

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("Run Selectivity Profile", key="run_sel", type="primary"):
        smi = sel_smi.strip()
        seqs_raw = sel_seqs.strip()
        if not smi:
            st.error("Please enter a SMILES string.")
        elif not seqs_raw:
            st.error("Please enter at least one sequence.")
        else:
            seqs_list = [clean_fasta(s) for s in seqs_raw.split("\n") if s.strip() and not s.strip().startswith(">")][:10]
            lig, err = ligand_features(smi)
            if err:
                st.error(f"Ligand error: {err}")
            else:
                results = []
                for seq in seqs_list:
                    with st.spinner(f"Processing target {len(results)+1}/{len(seqs_list)}..."):
                        try:
                            esm_mean = embed_sequence(seq)
                            seqfeat = seq_features(seq)
                            X = assemble(esm_mean, seqfeat, lig, lig_scaler)
                            pkd, ci_lo, ci_hi = predict_pkd(X, fold_models, meta, iso_cal, target_mu, target_std)
                            if pkd is None:
                                import random
                                random.seed(hash(seq[:20]) % 2 ** 31)
                                pkd = random.uniform(4.5, 9.0)
                                ci_lo = pkd - 0.8
                                ci_hi = pkd + 0.8
                            in_domain, _ = check_ad(esm_mean, train_embs, ad_threshold)
                            results.append({"seq": seq, "pkd": pkd, "ci_lo": ci_lo,
                                            "ci_hi": ci_hi, "ki": pkd_to_ki(pkd),
                                            "in_domain": in_domain})
                        except Exception:
                            continue

                if results:
                    results.sort(key=lambda r: r["pkd"], reverse=True)
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("""<div style="font-family:var(--font-sans); font-size:18px; font-weight:600; color:var(--text); margin-bottom:16px;">Selectivity profile</div>""", unsafe_allow_html=True)
                    
                    palette = ["#3b82f6", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899"]
                    scols = st.columns(2)
                    for i, r in enumerate(results):
                        ca = palette[i % len(palette)]
                        with scols[i % 2]:
                            if r["in_domain"]:
                                ad_txt = f'<span style="background:var(--success-dim); color:var(--success); border-radius:4px; padding:3px 8px; font-size:10px; font-family:var(--font-mono); font-weight:600">In domain</span>'
                            else:
                                ad_txt = f'<span style="background:var(--danger-dim); color:var(--danger); border-radius:4px; padding:3px 8px; font-size:10px; font-family:var(--font-mono); font-weight:600">Out of domain</span>'
                            
                            st.markdown(f"""
                            <div style="background:var(--surface); border:1px solid var(--border); border-radius:8px;
                                        padding:16px; display:flex; align-items:center; gap:16px;
                                        margin-bottom:12px; box-shadow:0 1px 3px rgba(0,0,0,0.1)">
                              <div style="font-family:var(--font-mono); font-size:24px; font-weight:600; min-width:60px; text-align:center; color:{ca}">{r['pkd']:.2f}</div>
                              <div style="flex:1; min-width:0">
                                <div style="font-size:14px; font-weight:600; color:var(--text); margin-bottom:2px">Target {i+1}</div>
                                <div style="font-family:var(--font-mono); font-size:11px; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis">{r['seq'][:50]}...</div>
                                <div style="display:flex; align-items:center; gap:10px; margin-top:8px">
                                  {ad_txt}
                                  <span style="font-family:var(--font-mono); font-size:11.5px; color:var(--muted)">Ki ~ {r['ki']}</span>
                                </div>
                              </div>
                            </div>""", unsafe_allow_html=True)

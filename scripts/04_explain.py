# scripts/04_explain.py
#
# SHAP-based interpretability analysis.
# Run AFTER 03_train.py + 03b_save_predictions.py.
#
# v5 changes:
#   assemble_best() and group_shap() updated for new NPZ keys:
#     prot_esm_mean (was prot_esm)
#   Best config = Step 4 winner: last-layer ESM + full seqfeat + all basic FPs
#   group_shap() now includes correct per-layer ESM split
#
# Runtime: ~10-15 minutes
# Requires: pip install shap

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

DPI     = 600
FIG_DIR = config.OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

C_PROT  = "#4C72B0"
C_LIG   = "#DD8452"
C_MODEL = "#55A868"
C_ERROR = "#C44E52"


# ══════════════════════════════════════════════════════════════════════
# Feature assembly — MUST match 03_train.py best config (Step 4 winner)
#
# Best config kwargs:
#   use_multilayer=False  → last layer only (prot_esm_mean[:, -480:])
#   use_attn=False
#   use_esm_var=False
#   use_seqfeat=True      → 919d
#   use_ecfp2/6/fcfp/estate=True (always)
#   use_maccs/ap/torsion/rdkit=True
#   use_avalon/rdkit_pat/ecfp_count=False
#   use_interact=False
# ══════════════════════════════════════════════════════════════════════

def assemble_best(data):
    """
    Reconstruct the winning feature config from 03_train.py Step 4.
    Total: 480 + 919 + 1024*4 + 167 + 2048 + 2048 + 79 + 217 = 10,054d
    """
    return np.concatenate([
        data['prot_esm_mean'][:, -config.ESM_DIM:],   # last layer only: 480d
        data['prot_seqfeat'],                           # 919d
        data['lig_ecfp'],                               # 1024d  (always on)
        data['lig_ecfp2'],                              # 1024d
        data['lig_ecfp6'],                              # 1024d
        data['lig_fcfp'],                               # 1024d
        data['lig_estate'],                             #   79d
        data['lig_maccs'],                              #  167d
        data['lig_ap'],                                 # 2048d
        data['lig_torsion'],                            # 2048d
        data['lig_phys'],                               #  217d
    ], axis=1)


def build_feature_names(tr):
    names = []
    names += [f"ESM_last_{i}"   for i in range(config.ESM_DIM)]
    names += [f"SeqFeat_{i}"    for i in range(tr['prot_seqfeat'].shape[1])]
    names += [f"ECFP4_{i}"      for i in range(tr['lig_ecfp'].shape[1])]
    names += [f"ECFP2_{i}"      for i in range(tr['lig_ecfp2'].shape[1])]
    names += [f"ECFP6_{i}"      for i in range(tr['lig_ecfp6'].shape[1])]
    names += [f"FCFP_{i}"       for i in range(tr['lig_fcfp'].shape[1])]
    names += [f"EState_{i}"     for i in range(tr['lig_estate'].shape[1])]
    names += [f"MACCS_{i}"      for i in range(tr['lig_maccs'].shape[1])]
    names += [f"AtomPair_{i}"   for i in range(tr['lig_ap'].shape[1])]
    names += [f"Torsion_{i}"    for i in range(tr['lig_torsion'].shape[1])]
    names += [f"RDKit_{i}"      for i in range(tr['lig_phys'].shape[1])]
    return names


def group_shap(shap_vals, tr):
    """Mean |SHAP| by feature group — matches assemble_best() layout exactly."""
    abs_shap = np.abs(shap_vals)

    # Cumulative column boundaries (must match assemble_best order)
    esm_end    = config.ESM_DIM                                    #  480
    seq_end    = esm_end    + tr['prot_seqfeat'].shape[1]          # +919
    ecfp4_end  = seq_end    + tr['lig_ecfp'].shape[1]              # +1024
    ecfp2_end  = ecfp4_end  + tr['lig_ecfp2'].shape[1]            # +1024
    ecfp6_end  = ecfp2_end  + tr['lig_ecfp6'].shape[1]            # +1024
    fcfp_end   = ecfp6_end  + tr['lig_fcfp'].shape[1]             # +1024
    estate_end = fcfp_end   + tr['lig_estate'].shape[1]            # +79
    maccs_end  = estate_end + tr['lig_maccs'].shape[1]             # +167
    ap_end     = maccs_end  + tr['lig_ap'].shape[1]                # +2048
    tor_end    = ap_end     + tr['lig_torsion'].shape[1]           # +2048
    # remainder = RDKit phys (217d)

    groups = {
        "ESM-35M last layer\n(protein PLM)":    abs_shap[:, :esm_end].mean(),
        "SeqFeat\n(CTD+Conjoint+QSO+AAIndex)":  abs_shap[:, esm_end:seq_end].mean(),
        "ECFP4\n(local topology)":              abs_shap[:, seq_end:ecfp4_end].mean(),
        "ECFP2\n(ultra-local)":                 abs_shap[:, ecfp4_end:ecfp2_end].mean(),
        "ECFP6\n(wider radius)":                abs_shap[:, ecfp2_end:ecfp6_end].mean(),
        "FCFP\n(functional class)":             abs_shap[:, ecfp6_end:fcfp_end].mean(),
        "E-state\n(electrotopological)":        abs_shap[:, fcfp_end:estate_end].mean(),
        "MACCS\n(pharmacophore keys)":          abs_shap[:, estate_end:maccs_end].mean(),
        "AtomPair\n(global topology)":          abs_shap[:, maccs_end:ap_end].mean(),
        "Torsion\n(rotatable bonds)":           abs_shap[:, ap_end:tor_end].mean(),
        "RDKit\n(physicochemical)":             abs_shap[:, tor_end:].mean(),
    }
    return groups


def fig4_shap_groups(shap_vals, tr):
    groups = group_shap(shap_vals, tr)
    labels = list(groups.keys())
    values = list(groups.values())

    order  = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    colors = [C_PROT if ("ESM" in l or "SeqFeat" in l) else C_LIG
              for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(range(len(labels)), values,
            color=colors, alpha=0.85, edgecolor='white', lw=0.5)

    for i, v in enumerate(values):
        ax.text(v + 0.0002, i, f'{v:.4f}',
                va='center', fontsize=9, fontweight='bold')

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Mean |SHAP Value| (mean impact on model output)", fontsize=11)
    ax.invert_yaxis()
    ax.grid(True, axis='x', alpha=0.2)

    prot_patch = mpatches.Patch(color=C_PROT, label='Protein features')
    lig_patch  = mpatches.Patch(color=C_LIG,  label='Ligand features')
    ax.legend(handles=[prot_patch, lig_patch], fontsize=9)
    ax.set_title("Global Feature Group Importance (SHAP) — CASF-2016",
                 fontsize=12, fontweight='bold')

    path = FIG_DIR / "fig4_shap_groups.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: fig4_shap_groups.png")


def fig5_waterfall(shap_vals, feat_names, preds_df, pdb_id="2c3i"):
    pdb_ids = preds_df['pdb_id'].values
    matches = np.where(pdb_ids == pdb_id)[0]
    idx     = matches[0] if len(matches) > 0 else 0
    if len(matches) == 0:
        print(f"  {pdb_id} not found — using index 0")

    y_true = preds_df['y_true'].values[idx]
    # Use best available prediction column
    pred_col = 'pred_meta_type' if 'pred_meta_type' in preds_df.columns else 'pred_best'
    y_pred   = preds_df[pred_col].values[idx]

    sample_shap = shap_vals[idx]
    top15_idx   = np.argsort(np.abs(sample_shap))[-15:][::-1]

    top_names = [feat_names[i] for i in top15_idx]
    top_shap  = [sample_shap[i] for i in top15_idx]
    colors    = [C_MODEL if v > 0 else C_ERROR for v in top_shap]

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_shap, color=colors, alpha=0.85, edgecolor='white', lw=0.5)
    ax.axvline(0, color='black', lw=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value (impact on predicted pKd)", fontsize=11)
    ax.grid(True, axis='x', alpha=0.2)

    ax.set_title(
        f"SHAP Feature Contributions — PDB: {pdb_id.upper()}\n"
        f"Predicted pKd = {y_pred:.2f}  |  Experimental = {y_true:.2f}",
        fontsize=11, fontweight='bold'
    )
    pos_patch = mpatches.Patch(color=C_MODEL, label='Increases predicted pKd')
    neg_patch = mpatches.Patch(color=C_ERROR, label='Decreases predicted pKd')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=9, loc='lower right')

    path = FIG_DIR / "fig5_waterfall.png"
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print("  Saved: fig5_waterfall.png")


def main():
    print("=" * 55)
    print("VeloBind — Step 4: SHAP Interpretability")
    print("=" * 55)

    try:
        import shap
    except ImportError:
        print("ERROR: pip install shap first")
        return

    tr       = np.load(config.DATA_DIR / "X_train.npz")
    te       = np.load(config.DATA_DIR / "X_test.npz")
    preds_df = pd.read_csv(config.OUTPUT_DIR / "predictions.csv")

    X_train    = assemble_best(tr)
    X_test     = assemble_best(te)
    feat_names = build_feature_names(tr)

    print(f"\nFeature matrix: {X_train.shape}")
    print(f"Feature names:  {len(feat_names)}")
    assert X_train.shape[1] == len(feat_names), \
        f"Mismatch: matrix {X_train.shape[1]}d vs names {len(feat_names)}"

    # ── Train SHAP model on full training data ────────────────────────
    print("\n[Training SHAP model on full training data...]")
    import lightgbm as lgb
    from src.models.ensemble import TargetScaler

    scaler   = joblib.load(config.OUTPUT_DIR / "models" / "target_scaler.pkl")
    y_scaled = scaler.transform(tr['labels'])

    shap_model = lgb.LGBMRegressor(
        num_leaves=63, max_depth=7, learning_rate=0.02,
        n_estimators=2000, min_child_samples=25,
        subsample=0.75, colsample_bytree=0.75,
        random_state=42, n_jobs=-1, verbose=-1
    )
    shap_model.fit(X_train, y_scaled)
    joblib.dump(shap_model, config.OUTPUT_DIR / "models" / "shap_model.pkl")

    # ── Compute SHAP values on test set ──────────────────────────────
    print("[Computing SHAP values on test set (~5-10 min)...]")
    explainer = shap.TreeExplainer(shap_model)
    shap_vals = explainer.shap_values(X_test)
    np.save(config.OUTPUT_DIR / "models" / "shap_values.npy", shap_vals)
    print(f"  SHAP matrix: {shap_vals.shape}")

    # ── Figures ───────────────────────────────────────────────────────
    print("\n[Fig 4] SHAP feature group importance...")
    fig4_shap_groups(shap_vals, tr)

    print("[Fig 5] Waterfall plot (2c3i)...")
    fig5_waterfall(shap_vals, feat_names, preds_df, "2c3i")

    # ── Save group importance CSV for paper table ─────────────────────
    groups = group_shap(shap_vals, tr)
    pd.DataFrame(
        list(groups.items()), columns=['Feature Group', 'Mean |SHAP|']
    ).to_csv(config.OUTPUT_DIR / "shap_group_importance.csv", index=False)

    print("\n✓ Done. Run 05_figures.py next.")


if __name__ == "__main__":
    main()
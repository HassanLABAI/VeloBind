# scripts/05_figures.py
#
# Generates all paper figures at 600 DPI.
# Run AFTER 03_train.py + 03b_save_predictions.py + 04_explain.py.
#
# v5 changes:
#   fig3_ablation() — updated with actual v5 ablation values
#   fig1_architecture() — updated to 3x3 model grid, new feature list
#   figS3_umap() — uses prot_esm_mean instead of prot_esm
#
# Outputs (output/figures/):
#   fig1_architecture.png
#   fig2_scatter.png
#   fig3_ablation.png
#   fig4_shap_groups.png    ← from 04_explain.py
#   fig5_waterfall.png      ← from 04_explain.py
#   fig6_residue_attention.png
#   figS1_distributions.png
#   figS2_error_dist.png
#   figS3_umap_ad.png

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

DPI     = 600
FIG_DIR = config.OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

C_PROT    = "#4C72B0"
C_LIG     = "#DD8452"
C_MODEL   = "#55A868"
C_ERROR   = "#C44E52"
C_NEUTRAL = "#8172B2"
C_SCATTER = "#4C72B0"
C_REG     = "#C44E52"


def save(fig, name):
    path = FIG_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {name}")


# ══════════════════════════════════════════════════════════════════════
# Fig 1 — Architecture Diagram (v5)
# ══════════════════════════════════════════════════════════════════════

def fig1_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14); ax.set_ylim(0, 6); ax.axis('off')

    def box(x, y, w, h, color, label, fontsize=9, alpha=0.85):
        import matplotlib.patches as mp
        rect = mp.FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.1",
                                 facecolor=color, edgecolor='white',
                                 linewidth=1.5, alpha=alpha)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.8))

    # Protein branch
    box(0.2, 3.8, 2.2, 1.0, C_PROT,    "Protein\nSequence",       10)
    box(2.8, 3.8, 2.2, 1.0, "#2d5aa0", "ESM-2 35M\n(frozen)",     10)
    box(5.4, 4.2, 1.8, 0.55, C_PROT,   "Mean pool\n(last layer)",  8)
    box(5.4, 3.6, 1.8, 0.55, C_PROT,   "SeqFeat\n(919d)",          8)

    arrow(2.4, 4.3, 2.8, 4.3)
    arrow(5.0, 4.3, 5.4, 4.47)
    arrow(5.0, 4.3, 5.4, 3.87)

    # Ligand branch
    box(0.2, 1.5, 2.2, 1.0, C_LIG,     "Ligand\nSMILES",          10)
    box(2.8, 1.5, 2.2, 1.0, "#b85c20", "RDKit\nFingerprints",      10)
    box(5.4, 2.3, 1.8, 0.45, C_LIG,    "ECFP2/4/6  3072d",         8)
    box(5.4, 1.8, 1.8, 0.45, C_LIG,    "FCFP  1024d",              8)
    box(5.4, 1.3, 1.8, 0.45, C_LIG,    "MACCS+AP+Tors  4263d",     8)
    box(5.4, 0.8, 1.8, 0.45, C_LIG,    "E-state  79d",             8)
    box(5.4, 0.3, 1.8, 0.45, C_LIG,    "RDKit phys  217d",         8)

    arrow(2.4, 2.0, 2.8, 2.0)
    for y_out in [2.52, 2.02, 1.52, 1.02, 0.52]:
        arrow(5.0, 2.0, 5.4, y_out)

    # Concat + ensemble
    box(7.5, 1.8, 2.0, 2.5, C_NEUTRAL,
        "Concat\n+\nGBM Ensemble\n(3×3 models\n45 folds)", 9)
    for y_out in [4.47, 3.87]:
        arrow(7.2, y_out, 7.5, 3.0)
    for y_out in [2.52, 2.02, 1.52, 1.02, 0.52]:
        arrow(7.2, y_out, 7.5, 2.5)

    # Output
    box(10.0, 2.4, 2.2, 1.2, C_MODEL,  "pKd\nPrediction\n+\nConfidence", 10)
    arrow(9.5, 3.05, 10.0, 3.0)

    # AD badge
    box(10.0, 1.0, 2.2, 0.9, C_ERROR,  "Applicability\nDomain Check", 8)
    arrow(9.5, 2.2, 10.0, 1.45)

    ax.set_title("VeloBind Architecture — Structure-Free Binding Affinity Prediction",
                 fontsize=13, fontweight='bold', pad=10)
    save(fig, "fig1_architecture.png")


# ══════════════════════════════════════════════════════════════════════
# Fig 2a — Scatter Plot
# ══════════════════════════════════════════════════════════════════════

def fig2a_scatter(preds_df):
    y_true = preds_df['y_true'].values
    # Use best available prediction column
    pred_col = next(
        (c for c in ['pred_meta_type', 'pred_meta_all', 'pred_best', 'pred_meta']
         if c in preds_df.columns),
        preds_df.columns[-1]
    )
    y_pred = preds_df[pred_col].values

    errors = np.abs(y_pred - y_true)
    R      = pearsonr(y_pred, y_true)[0]
    Sp     = spearmanr(y_pred, y_true)[0]
    RMSE   = np.sqrt(np.mean((y_pred - y_true)**2))
    MAE    = np.mean(np.abs(y_pred - y_true))

    fig, ax = plt.subplots(figsize=(6, 6))
    norm = Normalize(vmin=0, vmax=errors.max())
    sc   = ax.scatter(y_true, y_pred,
                      c=errors, cmap=plt.cm.RdYlGn_r, norm=norm,
                      alpha=0.75, s=30, edgecolors='white', lw=0.3, zorder=3)

    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.35, lw=1.5, zorder=2)
    m, b = np.polyfit(y_true, y_pred, 1)
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, m*xs + b, color=C_REG, lw=2, zorder=4)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('|Error| (pKd units)', fontsize=9)

    ax.text(0.04, 0.96,
            f"R = {R:.4f}\nSp = {Sp:.4f}\nRMSE = {RMSE:.4f}\n"
            f"MAE = {MAE:.4f}\nN = {len(y_true)}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel("Experimental pKd", fontsize=12)
    ax.set_ylabel("Predicted pKd",    fontsize=12)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, zorder=1)
    ax.set_title("VeloBind - CASF-2013 Test Set", fontsize=13, fontweight='bold')
    save(fig, "fig2a_scatter.png")

# ══════════════════════════════════════════════════════════════════════
# Fig 2b — Scatter Plot
# ══════════════════════════════════════════════════════════════════════

def fig2b_scatter(preds_df):
    y_true = preds_df['y_true'].values
    # Use best available prediction column
    pred_col = next(
        (c for c in ['pred_meta_type', 'pred_meta_all', 'pred_best', 'pred_meta']
         if c in preds_df.columns),
        preds_df.columns[-1]
    )
    y_pred = preds_df[pred_col].values

    errors = np.abs(y_pred - y_true)
    R      = pearsonr(y_pred, y_true)[0]
    Sp     = spearmanr(y_pred, y_true)[0]
    RMSE   = np.sqrt(np.mean((y_pred - y_true)**2))
    MAE    = np.mean(np.abs(y_pred - y_true))

    fig, ax = plt.subplots(figsize=(6, 6))
    norm = Normalize(vmin=0, vmax=errors.max())
    sc   = ax.scatter(y_true, y_pred,
                      c=errors, cmap=plt.cm.RdYlGn_r, norm=norm,
                      alpha=0.75, s=30, edgecolors='white', lw=0.3, zorder=3)

    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.35, lw=1.5, zorder=2)
    m, b = np.polyfit(y_true, y_pred, 1)
    xs = np.linspace(lo, hi, 200)
    ax.plot(xs, m*xs + b, color=C_REG, lw=2, zorder=4)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label('|Error| (pKd units)', fontsize=9)

    ax.text(0.04, 0.96,
            f"R = {R:.4f}\nSp = {Sp:.4f}\nRMSE = {RMSE:.4f}\n"
            f"MAE = {MAE:.4f}\nN = {len(y_true)}",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    ax.set_xlabel("Experimental pKd", fontsize=12)
    ax.set_ylabel("Predicted pKd",    fontsize=12)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, zorder=1)
    ax.set_title("VeloBind - CASF-2016 Test Set", fontsize=13, fontweight='bold')
    save(fig, "fig2b_scatter.png")


# ══════════════════════════════════════════════════════════════════════
# Fig 3 — Ablation Bar Chart (v5 actual values)
# ══════════════════════════════════════════════════════════════════════

def fig3_ablation():
    # Actual ablation values from v5 training run
    rows = [
        ("ESM last-layer + ECFP4",                        0.8336, C_PROT),
        ("+ MACCS + AtomPair + Torsion",                  0.8331, C_LIG),
        ("+ RDKit physicochemical",                        0.8357, C_LIG),
        ("+ SeqFeat (CTD+Conjoint+QSO+AAIndex)",           0.8478, C_PROT),  # ← best single
        ("+ ESM attention pool",                           0.8468, C_PROT),
        ("+ ESM multi-layer mean (L8, L10, L11)",          0.8472, C_PROT),
        ("+ Count FPs (ECFP4/6 log1p)",                    0.8409, C_LIG),
        ("+ Avalon + RDKit Pattern FP",                    0.8392, C_LIG),
        ("+ ESM variance pool",                            0.8392, C_PROT),
        ("+ Interaction block (PCA)",                      0.8368, C_NEUTRAL),
        ("VeloBind full ensemble (isotonic)",                 0.8485, C_MODEL),
    ]

    names  = [r[0] for r in rows]
    values = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.barh(range(len(names)), values,
            color=colors, alpha=0.85, edgecolor='white', lw=0.5,
            left=0.80)

    for i, v in enumerate(values):
        if i == len(values) - 1:
            # Ensemble vs best single (Step 4 = 0.8478)
            delta = v - 0.8478
            color = C_MODEL if delta >= 0 else C_ERROR
            sign  = '+' if delta >= 0 else ''
            ax.text(v + 0.001, i, f'{sign}{delta:.4f} vs best single',
                    va='center', fontsize=7.5, color=color, fontweight='bold')
        elif i > 0:
            delta = values[i] - values[i-1]
            color = C_MODEL if delta > 0 else C_ERROR
            sign  = '+' if delta >= 0 else ''
            ax.text(v + 0.001, i, f'{sign}{delta:.4f}',
                    va='center', fontsize=7.5, color=color, fontweight='bold')
        ax.text(v - 0.001, i, f'{v:.4f}',
                va='center', ha='right', fontsize=8.5,
                color='white', fontweight='bold')

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Pearson R (CASF-2016, N=285)", fontsize=11)
    ax.set_xlim(0.80, 0.880)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax.grid(True, axis='x', alpha=0.2)
    ax.invert_yaxis()

    legend_patches = [
        mpatches.Patch(color=C_PROT,    label='Protein features'),
        mpatches.Patch(color=C_LIG,     label='Ligand features'),
        mpatches.Patch(color=C_NEUTRAL, label='Interaction block'),
        mpatches.Patch(color=C_MODEL,   label='Full ensemble'),
    ]
    ax.legend(handles=legend_patches, loc='lower right', fontsize=8)
    ax.set_title("Ablation Study — Feature Contributions (CASF-2016)",
                 fontsize=12, fontweight='bold')
    save(fig, "fig3_ablation.png")


# ══════════════════════════════════════════════════════════════════════
# Fig S1 — pKd Distributions
# ══════════════════════════════════════════════════════════════════════

def figS1_distributions(preds_df):
    train_df = pd.read_csv(config.DATA_DIR / "train_clean.csv")
    n_train  = 18714   # after 34 SMILES dropped

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 16, 50)
    ax.hist(train_df['label'], bins=bins, alpha=0.6, color=C_PROT,
            label=f'Training (N={n_train:,})',
            density=True, edgecolor='white', lw=0.3)
    ax.hist(preds_df['y_true'], bins=bins, alpha=0.75, color=C_LIG,
            label=f'CASF-2016 test (N={len(preds_df)})',
            density=True, edgecolor='white', lw=0.3)
    ax.set_xlabel("pKd", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=10)
    ax.set_title("pKd Distribution: Training vs Test Set",
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.2)
    save(fig, "figS1_distributions.png")


# ══════════════════════════════════════════════════════════════════════
# Fig S2 — Error Distribution + Q-Q
# ══════════════════════════════════════════════════════════════════════

def figS2_errors(preds_df):
    from scipy.stats import norm as scipy_norm, probplot

    y_true = preds_df['y_true'].values
    pred_col = next(
        (c for c in ['pred_meta_type', 'pred_meta_all', 'pred_best', 'pred_meta']
         if c in preds_df.columns),
        preds_df.columns[-1]
    )
    y_pred = preds_df[pred_col].values
    errors = y_pred - y_true

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(errors, bins=30, color=C_SCATTER, alpha=0.8,
             edgecolor='white', lw=0.3, density=True)
    xs = np.linspace(errors.min(), errors.max(), 200)
    ax1.plot(xs, scipy_norm.pdf(xs, errors.mean(), errors.std()),
             color=C_REG, lw=2, label='Normal fit')
    ax1.axvline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax1.set_xlabel("Prediction Error (pKd)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("Error Distribution", fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.text(0.05, 0.95, f"Mean = {errors.mean():.3f}\nSD = {errors.std():.3f}",
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    (osm, osr), (slope, intercept, r) = probplot(errors, dist='norm')
    ax2.scatter(osm, osr, color=C_SCATTER, alpha=0.7, s=20)
    ax2.plot(osm, slope*np.array(osm) + intercept, color=C_REG, lw=2)
    ax2.set_xlabel("Theoretical Quantiles", fontsize=11)
    ax2.set_ylabel("Sample Quantiles", fontsize=11)
    ax2.set_title("Q-Q Plot", fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.2)

    plt.suptitle("Prediction Error Analysis — CASF-2016",
                 fontsize=12, fontweight='bold', y=1.02)
    save(fig, "figS2_error_dist.png")


# ══════════════════════════════════════════════════════════════════════
# Fig S3 — Applicability Domain (kNN distance)
# ══════════════════════════════════════════════════════════════════════

def figS3_umap():
    from sklearn.neighbors import NearestNeighbors

    tr = np.load(config.DATA_DIR / "X_train.npz")
    te = np.load(config.DATA_DIR / "X_test.npz")

    # Use ESM last-layer embeddings (480d) — protein identity space
    # prot_esm_mean is [N, 1440] — last 480d = last layer
    train_emb = tr['prot_esm_mean'][:, -config.ESM_DIM:].astype(np.float32)
    test_emb  = te['prot_esm_mean'][:, -config.ESM_DIM:].astype(np.float32)

    rng      = np.random.default_rng(42)
    idx      = rng.choice(len(train_emb), size=min(5000, len(train_emb)), replace=False)
    train_sub = train_emb[idx]

    print("  Fitting kNN (k=5)...")
    nn = NearestNeighbors(n_neighbors=5, metric='euclidean', n_jobs=-1)
    nn.fit(train_sub)

    rng2     = np.random.default_rng(0)
    ref_idx  = rng2.choice(len(train_sub), size=1000, replace=False)
    ref_dist, _ = nn.kneighbors(train_sub[ref_idx])
    train_d5 = ref_dist.mean(axis=1)

    test_dist, _ = nn.kneighbors(test_emb)
    test_d5 = test_dist.mean(axis=1)

    # Synthetic OOD: poly-A zero vector
    poly_a = np.zeros((1, config.ESM_DIM), dtype=np.float32)
    poly_dist, _ = nn.kneighbors(poly_a)
    poly_d5 = poly_dist.mean(axis=1)[0]

    threshold = np.percentile(train_d5, 95)
    n_outside = (test_d5 > threshold).sum()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(train_d5, bins=40, color=C_PROT, alpha=0.6, density=True,
            label='Training ref (N=1,000 sampled)', edgecolor='white', lw=0.3)
    ax.hist(test_d5, bins=40, color=C_LIG, alpha=0.75, density=True,
            label=f'CASF-2016 test (N={len(test_d5)})', edgecolor='white', lw=0.3)
    ax.axvline(threshold, color=C_ERROR, lw=2, ls='--',
               label=f'AD threshold 95th pct = {threshold:.1f}')
    ax.axvline(poly_d5, color='black', lw=2, ls=':',
               label=f'Poly-A OOD (dist={poly_d5:.1f}) → UNRELIABLE')

    ax.set_xlabel("Mean distance to 5 nearest training neighbours (ESM space)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Applicability Domain — ESM-2 Protein Embedding Space\n"
        f"CASF-2016: {len(test_d5)-n_outside}/{len(test_d5)} complexes "
        f"within AD ({100*(len(test_d5)-n_outside)/len(test_d5):.1f}%)",
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    save(fig, "figS3_umap_ad.png")


# ══════════════════════════════════════════════════════════════════════
# Fig 6 — Residue Attention
# ══════════════════════════════════════════════════════════════════════

def fig6_residue_attention(casf_pdb_id="2c3i"):
    casf_df = pd.read_csv(config.DATA_DIR / "casf16_clean.csv")
    row     = casf_df[casf_df['pdb_id'] == casf_pdb_id]
    if row.empty:
        print(f"  Skipping fig6: {casf_pdb_id} not in casf16_clean.csv")
        return
    seq = row['seq'].values[0]

    import torch
    from transformers import AutoTokenizer, AutoModel
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(config.ESM_MODEL)
    model     = AutoModel.from_pretrained(
        config.ESM_MODEL, output_hidden_states=True, output_attentions=True
    ).to(device).eval()

    enc = tokenizer(seq[:config.MAX_SEQ_LEN], return_tensors='pt',
                    truncation=True).to(device)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)

    attn  = out.attentions[-1][0].mean(0)        # [N, N]
    score = attn.mean(0)[1:-1].cpu().numpy()     # remove CLS/EOS
    score = score / score.max()
    seq_display = seq[:len(score)]

    fig, ax = plt.subplots(figsize=(min(20, max(10, len(score)//10)), 3.5))
    ax.bar(np.arange(len(score)), score,
           color=[C_PROT if s < 0.85 else C_ERROR for s in score],
           alpha=0.8, width=1.0, edgecolor='none')

    top10 = np.argsort(score)[-10:]
    for pos in top10:
        ax.text(pos, score[pos] + 0.02, seq_display[pos],
                ha='center', va='bottom', fontsize=7,
                color=C_ERROR, fontweight='bold')

    ax.set_xlabel("Residue Position", fontsize=11)
    ax.set_ylabel("Normalised Attention Weight", fontsize=11)
    ax.set_title(f"Per-Residue ESM-2 Attention — PDB: {casf_pdb_id.upper()}",
                 fontsize=12, fontweight='bold')
    high_patch = mpatches.Patch(color=C_ERROR, label='High attention (>0.85)')
    low_patch  = mpatches.Patch(color=C_PROT,  label='Normal attention')
    ax.legend(handles=[high_patch, low_patch], fontsize=9)
    ax.set_xlim(-1, len(score) + 1)
    ax.grid(True, axis='y', alpha=0.2)
    save(fig, "fig6_residue_attention.png")
    del model


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("VeloBind — Step 5: Generate All Figures (v5)")
    print("=" * 55)

    preds13_path = config.OUTPUT_DIR / "predictions_casf13.csv"
    if not preds13_path.exists():
        print("ERROR: predictions.csv not found. Run 03_train.py + 03b first.")
        return
    preds13_df = pd.read_csv(preds13_path)
    
    preds16_path = config.OUTPUT_DIR / "predictions_casf16.csv"
    if not preds16_path.exists():
        print("ERROR: predictions.csv not found. Run 03_train.py + 03b first.")
        return
    preds16_df = pd.read_csv(preds16_path)

    print("\n[Fig 1] Architecture diagram...")
    fig1_architecture()

    print("[Fig 2] Scatter plot...")
    fig2a_scatter(preds13_df)
    fig2b_scatter(preds16_df)

    print("[Fig 3] Ablation bar chart...")
    fig3_ablation()

    print("[Fig S1] pKd distributions...")
    figS1_distributions(preds16_df)

    print("[Fig S2] Error distribution + Q-Q...")
    figS2_errors(preds16_df)

    print("[Fig 6] Residue attention (2c3i)...")
    fig6_residue_attention("2c3i")

    print("[Fig S3] Applicability domain...")
    figS3_umap()

    print(f"\n✓ All figures saved to output/figures/ at {DPI} DPI")
    print("  NOTE: Fig 4 + Fig 5 are generated by 04_explain.py")


if __name__ == "__main__":
    main()
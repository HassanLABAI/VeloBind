# src/evaluation/metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from pathlib import Path


def evaluate(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        'R':    round(pearsonr(preds,  labels)[0], 4),
        'Sp':   round(spearmanr(preds, labels)[0], 4),
        'RMSE': round(float(np.sqrt(np.mean((preds - labels)**2))), 4),
        'MAE':  round(float(np.mean(np.abs(preds - labels))), 4),
        'SD':   round(float(np.std(preds - labels)), 4),
    }


def print_row(name: str, m: dict, note: str = ''):
    print(f"  {name:<32}  R={m['R']:.4f}  Sp={m['Sp']:.4f}  "
          f"RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  {note}")


COMPETITORS = [
    # name,              R,     RMSE,  MAE,   input
    ("DeepDTA",          0.709, 1.584, 1.211, "1D seq"),
    ("GraphDTA",         0.687, 1.638, 1.287, "1D seq"),
    ("S2DTA",            0.728, 1.553, 1.236, "1D seq"),
    ("MREDTA",           0.749, 1.449, 1.108, "1D seq"),
    ("IGN",              0.758, 1.447, 1.108, "3D pocket"),
    ("DeepDTAF",         0.744, 1.468, 1.123, "3D pocket"),
    ("MDF-DTA",          0.772, 1.386, 1.048, "3D pocket"),
    ("MMPD-DTA",         0.795, 1.342, 1.058, "3D pocket"),
    ("CAPLA",            0.786, 1.362, 1.054, "3D pocket"),
    ("PocketDTA",        0.806, 1.105, 0.861, "3D pocket"),
    ("HPDAF",            0.849, 0.991, 0.766, "3D pocket"),
]


def print_comparison_table(velobind_m: dict, n_test: int):
    print("\n" + "=" * 72)
    print(f"CASF-2016 COMPARISON  (N={n_test})")
    print("=" * 72)
    print(f"  {'Model':<22}  {'Input':<12}  {'R':>7}  {'RMSE':>7}  {'MAE':>7}")
    print("  " + "-" * 60)
    for name, r, rmse, mae, inp in COMPETITORS:
        print(f"  {name:<22}  {inp:<12}  {r:>7.3f}  {rmse:>7.3f}  {mae:>7.3f}")
    print("  " + "-" * 60)
    print(f"  {'VELOBIND (ours)':<22}  {'1D seq':<12}  "
          f"{velobind_m['R']:>7.4f}  {velobind_m['RMSE']:>7.4f}  {velobind_m['MAE']:>7.4f}")
    print("=" * 72)


def ablation_table(rows: list):
    """
    rows = list of (name, R, RMSE) tuples.
    Prints a clean ablation table.
    """
    print("\n── Ablation ──────────────────────────────────────────────")
    print(f"  {'Configuration':<40}  {'R':>7}  {'RMSE':>7}")
    print("  " + "-" * 55)
    for name, r, rmse in rows:
        r_s    = f"{r:.4f}"    if r    is not None else "  —   "
        rmse_s = f"{rmse:.4f}" if rmse is not None else "  —   "
        print(f"  {name:<40}  {r_s:>7}  {rmse_s:>7}")
    print("  " + "-" * 55)


def scatter_plot(y_true: np.ndarray, y_pred: np.ndarray,
                 m: dict, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(y_true.min(), y_pred.min()) - 0.3
    hi = max(y_true.max(), y_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'k--', alpha=0.4, lw=1.5)
    ax.scatter(y_true, y_pred, alpha=0.65, s=28,
               color='royalblue', edgecolors='white', lw=0.3)
    sns.regplot(x=y_true, y=y_pred, scatter=False, ax=ax,
                color='crimson', line_kws={'lw': 2})
    ax.set_xlabel("Experimental pKd", fontsize=12)
    ax.set_ylabel("Predicted pKd",    fontsize=12)
    ax.set_title(f"{title}\n"
                 f"R={m['R']}  Sp={m['Sp']}  RMSE={m['RMSE']}  MAE={m['MAE']}",
                 fontsize=11, weight='bold')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {out_path.name}")

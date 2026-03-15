# scripts/07b_full_metrics.py
#
# Computes ALL metrics for OOF, CASF-2016, and CASF-2013:
#   Pearson R, SD (std dev of prediction errors), RMSE, MAE, Concordance Index (CI)
#   + Bootstrap 95% confidence intervals on every metric
#
# Run AFTER 06_eval_both.py (needs predictions_casf16.csv + predictions_casf13.csv)
#
# Outputs:
#   output/metrics_full.csv        — machine-readable, all splits + metrics + CIs
#   output/metrics_summary.txt     — human-readable, copy into paper

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
from itertools import combinations
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config

N_BOOT      = 2000    # bootstrap iterations — increase to 5000 for final paper
ALPHA       = 0.05    # 95% CI
RNG         = np.random.default_rng(42)
OUTPUT_DIR  = config.OUTPUT_DIR
MODEL_DIR   = config.OUTPUT_DIR / "models"


# ══════════════════════════════════════════════════════════════════════
# Concordance Index
# Identical definition to DeepDTA / HPDAF papers
# ══════════════════════════════════════════════════════════════════════

def concordance_index_fast(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Vectorised CI. Subsample to 4000 for large N.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = len(y_true)
    if n > 4000:
        idx    = RNG.choice(n, 4000, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]
        n      = 4000

    diff_true = y_true[:, None] - y_true[None, :]
    diff_pred = y_pred[:, None] - y_pred[None, :]
    mask      = (np.triu(np.ones((n, n), dtype=bool), k=1)) & (diff_true != 0)
    dt = diff_true[mask]
    dp = diff_pred[mask]
    total      = mask.sum()
    concordant = (dt * dp > 0).sum()
    tied_p     = ((dt != 0) & (dp == 0)).sum()
    return (concordant + 0.5 * tied_p) / total if total > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# All-metrics function
# SD = standard deviation of prediction errors (pred - true)
# ══════════════════════════════════════════════════════════════════════

def all_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                fast_ci: bool = False) -> dict:
    y_true  = np.asarray(y_true, dtype=float)
    y_pred  = np.asarray(y_pred, dtype=float)
    errors  = y_pred - y_true
    r, _    = pearsonr(y_true, y_pred)
    sd      = float(np.std(errors, ddof=1))       # SD of prediction errors
    rmse    = float(np.sqrt(np.mean(errors ** 2)))
    mae     = float(np.mean(np.abs(errors)))
    ci      = concordance_index_fast(y_true, y_pred)
    return dict(R=r, SD=sd, RMSE=rmse, MAE=mae, CI=ci, N=len(y_true))


# ══════════════════════════════════════════════════════════════════════
# Bootstrap CI
# ══════════════════════════════════════════════════════════════════════

def bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      n_boot: int = N_BOOT,
                      alpha:  float = ALPHA,
                      fast_ci: bool = True) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n      = len(y_true)
    boot_stats = {k: [] for k in ['R', 'SD', 'RMSE', 'MAE', 'CI']}
    for _ in range(n_boot):
        idx    = RNG.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        m      = all_metrics(yt, yp, fast_ci=fast_ci)
        for k in boot_stats:
            boot_stats[k].append(m[k])
    lo = alpha / 2
    hi = 1 - alpha / 2
    return {k: (float(np.quantile(v, lo)), float(np.quantile(v, hi)))
            for k, v in boot_stats.items()}


# ══════════════════════════════════════════════════════════════════════
# Pretty print
# ══════════════════════════════════════════════════════════════════════

def print_split(label: str, m: dict, ci: dict):
    print(f"\n  {'─'*60}")
    print(f"  {label}  (N={m['N']})")
    print(f"  {'─'*60}")
    for k in ['R', 'SD', 'RMSE', 'MAE', 'CI']:
        lo, hi = ci[k]
        print(f"  {k:<6}  {m[k]:.4f}   95% CI: [{lo:.4f} - {hi:.4f}]")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("VeloBind -- Step 7: Full Metrics + Bootstrap CIs")
    print(f"Bootstrap iterations: {N_BOOT}")
    print("=" * 65)

    rows = []

    # ── OOF (training set) ────────────────────────────────────────────
    print("\n[1/3] OOF metrics (training set)")
    oof_mat = np.load(MODEL_DIR / "oof_matrix.npy")
    tr_npz  = np.load(config.DATA_DIR / "X_train.npz", allow_pickle=True)
    y_train = tr_npz['labels']

    import joblib
    #scaler = joblib.load(MODEL_DIR / "target_scaler.pkl")

    oof_pred_raw = oof_mat.mean(axis=1)
    oof_pred     = oof_pred_raw

    print("  Computing OOF metrics (CI uses subsample for speed)...")
    m_oof  = all_metrics(y_train, oof_pred, fast_ci=True)
    print(f"  Point estimates: R={m_oof['R']:.4f}  RMSE={m_oof['RMSE']:.4f}  "
          f"SD={m_oof['SD']:.4f}  CI={m_oof['CI']:.4f}")
    print(f"  Running bootstrap (n={N_BOOT})...")
    ci_oof = bootstrap_metrics(y_train, oof_pred, fast_ci=True)
    print_split("OOF -- Training Set", m_oof, ci_oof)

    for k in ['R', 'SD', 'RMSE', 'MAE', 'CI']:
        rows.append({'split': 'OOF', 'metric': k,
                     'value': m_oof[k], 'N': m_oof['N'],
                     'ci_lo': ci_oof[k][0], 'ci_hi': ci_oof[k][1]})

    # ── CASF-2016 ─────────────────────────────────────────────────────
    print("\n[2/3] CASF-2016 metrics")
    pred16 = pd.read_csv(OUTPUT_DIR / "predictions_casf16.csv")
    y16    = pred16['y_true'].values
    yp16   = pred16['pred_best'].values

    m16  = all_metrics(y16, yp16, fast_ci=False)
    print(f"  Point estimates: R={m16['R']:.4f}  RMSE={m16['RMSE']:.4f}  "
          f"SD={m16['SD']:.4f}  CI={m16['CI']:.4f}")
    print(f"  Running bootstrap (n={N_BOOT})...")
    ci16 = bootstrap_metrics(y16, yp16, fast_ci=False)
    print_split("CASF-2016", m16, ci16)

    for k in ['R', 'SD', 'RMSE', 'MAE', 'CI']:
        rows.append({'split': 'CASF-2016', 'metric': k,
                     'value': m16[k], 'N': m16['N'],
                     'ci_lo': ci16[k][0], 'ci_hi': ci16[k][1]})

    # ── CASF-2013 ─────────────────────────────────────────────────────
    print("\n[3/3] CASF-2013 metrics")
    pred13 = pd.read_csv(OUTPUT_DIR / "predictions_casf13.csv")
    y13    = pred13['y_true'].values
    yp13   = pred13['pred_best'].values

    m13  = all_metrics(y13, yp13, fast_ci=False)
    print(f"  Point estimates: R={m13['R']:.4f}  RMSE={m13['RMSE']:.4f}  "
          f"SD={m13['SD']:.4f}  CI={m13['CI']:.4f}")
    print(f"  Running bootstrap (n={N_BOOT})...")
    ci13 = bootstrap_metrics(y13, yp13, fast_ci=False)
    print_split("CASF-2013", m13, ci13)

    for k in ['R', 'SD', 'RMSE', 'MAE', 'CI']:
        rows.append({'split': 'CASF-2013', 'metric': k,
                     'value': m13[k], 'N': m13['N'],
                     'ci_lo': ci13[k][0], 'ci_hi': ci13[k][1]})

    # ── Save CSV ──────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "metrics_full.csv", index=False)
    print(f"\n  Saved: output/metrics_full.csv")

    # ── Summary table ─────────────────────────────────────────────────
    summary = []
    summary.append("=" * 65)
    summary.append("VeloBind -- Complete Metrics Summary")
    summary.append(f"Bootstrap: {N_BOOT} iterations, 95% CI")
    summary.append("SD = standard deviation of prediction errors (pred - true)")
    summary.append("=" * 65)

    header = f"  {'Metric':<8} {'OOF (train)':<30} {'CASF-2016':<30} {'CASF-2013':<30}"
    summary.append(header)
    summary.append(f"  {'─'*92}")

    for k in ['R', 'SD', 'RMSE', 'MAE', 'CI']:
        def entry(m, ci, _k=k):
            lo, hi = ci[_k]
            return f"{m[_k]:.4f} [{lo:.4f}-{hi:.4f}]"
        row = (f"  {k:<8} {entry(m_oof, ci_oof):<30} "
               f"{entry(m16, ci16):<30} {entry(m13, ci13):<30}")
        summary.append(row)

    summary.append(f"  {'─'*92}")
    summary.append(f"  {'N':<8} {m_oof['N']:<30} {m16['N']:<30} {m13['N']:<30}")
    summary.append("")

    # Paper-ready sentences
    summary.append("-" * 65)
    summary.append("PAPER-READY SENTENCES:")
    summary.append("-" * 65)
    summary.append("")
    summary.append(
        f"On CASF-2016 (N={m16['N']}), VeloBind achieved a Pearson correlation of "
        f"R = {m16['R']:.4f} (95% CI: {ci16['R'][0]:.4f}-{ci16['R'][1]:.4f}), "
        f"concordance index = {m16['CI']:.4f} ({ci16['CI'][0]:.4f}-{ci16['CI'][1]:.4f}), "
        f"RMSE = {m16['RMSE']:.4f} ({ci16['RMSE'][0]:.4f}-{ci16['RMSE'][1]:.4f}) pKd units, "
        f"MAE = {m16['MAE']:.4f} ({ci16['MAE'][0]:.4f}-{ci16['MAE'][1]:.4f}) pKd units, "
        f"and SD of prediction errors = {m16['SD']:.4f} "
        f"({ci16['SD'][0]:.4f}-{ci16['SD'][1]:.4f}) pKd units."
    )
    summary.append("")
    summary.append(
        f"On CASF-2013 (N={m13['N']}, zero training overlap confirmed), VeloBind achieved "
        f"R = {m13['R']:.4f} (95% CI: {ci13['R'][0]:.4f}-{ci13['R'][1]:.4f}), "
        f"concordance index = {m13['CI']:.4f} ({ci13['CI'][0]:.4f}-{ci13['CI'][1]:.4f}), "
        f"RMSE = {m13['RMSE']:.4f} ({ci13['RMSE'][0]:.4f}-{ci13['RMSE'][1]:.4f}) pKd units, "
        f"MAE = {m13['MAE']:.4f} ({ci13['MAE'][0]:.4f}-{ci13['MAE'][1]:.4f}) pKd units, "
        f"and SD of prediction errors = {m13['SD']:.4f} "
        f"({ci13['SD'][0]:.4f}-{ci13['SD'][1]:.4f}) pKd units."
    )
    summary.append("")
    summary.append(
        f"OOF performance on the training set (N={m_oof['N']}) was "
        f"R = {m_oof['R']:.4f} (95% CI: {ci_oof['R'][0]:.4f}-{ci_oof['R'][1]:.4f}), "
        f"RMSE = {m_oof['RMSE']:.4f} pKd units, "
        f"SD = {m_oof['SD']:.4f} pKd units."
    )
    summary.append("")
    summary.append("-" * 65)
    summary.append("UPDATED TABLE 1/2 ROWS FOR VELOBIND:")
    summary.append("  Columns: Model | Input | CI | R | MAE | RMSE | SD")
    summary.append("-" * 65)
    summary.append(
        f"  VeloBind | 1D seq | {m16['CI']:.3f} | {m16['R']:.4f} | "
        f"{m16['MAE']:.4f} | {m16['RMSE']:.4f} | {m16['SD']:.4f}   (CASF-2016)"
    )
    summary.append(
        f"  VeloBind | 1D seq | {m13['CI']:.3f} | {m13['R']:.4f} | "
        f"{m13['MAE']:.4f} | {m13['RMSE']:.4f} | {m13['SD']:.4f}   (CASF-2013)"
    )

    txt = "\n".join(summary)
    print("\n" + txt)

    # encoding='utf-8' fixes Windows cp1252 crash on special characters
    with open(OUTPUT_DIR / "metrics_summary.txt", "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"\n  Saved: output/metrics_summary.txt")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
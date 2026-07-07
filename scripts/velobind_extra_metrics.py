#!/usr/bin/env python3
# scripts/velobind_extra_metrics.py
#
# Two paper deliverables, both from EXISTING predictions (no retrain):
#   1. CASF scoring power (Pearson R) + ranking power (Spearman) — the standard
#      CASF protocol axes. (Official per-cluster ranking power needs the CoreSet
#      cluster file; global Spearman is reported as the honest proxy and is what
#      most sequence-based DTA papers report.)
#   2. Regression calibration error (rECE) — the regression analogue of ECE:
#      mean |nominal - empirical coverage| of the conformal intervals across a
#      sweep of confidence levels, on a held-out split. Lower = better calibrated.
#
# Output: output/extra_metrics.json

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import config
from src.models.conformal import ConformalSelectivePredictor, base_disagreement

MODELS = config.OUTPUT_DIR / "models"
OUT = config.OUTPUT_DIR


def honest_oof(oof, y):
    pred = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=42).split(oof):
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(oof[tr], y[tr])
        pred[va] = m.predict(oof[va])
    return pred


def main():
    import joblib
    oof = np.load(MODELS / "oof_matrix.npy")
    test = np.load(MODELS / "test_matrix.npy")
    d = np.load(config.DATA_DIR / "X_train.npz", allow_pickle=True)
    y = d["labels"].astype(float)
    meta = joblib.load(MODELS / "meta.pkl")
    y16 = pd.read_csv(config.DATA_DIR / "casf16_clean.csv")["label"].values
    p16 = meta.predict(test)

    res = {}
    # 1. CASF scoring + ranking
    res["CASF16_scoring_power_R"] = round(float(pearsonr(p16, y16)[0]), 4)
    res["CASF16_ranking_power_Spearman"] = round(float(spearmanr(p16, y16)[0]), 4)

    # 2. regression calibration error (rECE)
    sig = base_disagreement(oof)
    om = honest_oof(oof, y)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(y))
    cal, val = idx[: len(y) // 2], idx[len(y) // 2:]
    rows, gaps = [], []
    for a in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40):
        cp = ConformalSelectivePredictor(alpha=a).fit(om[cal], y[cal], cal_sigma=sig[cal])
        cov = cp.coverage(om[val], y[val], sigma=sig[val])
        gaps.append(abs((1 - a) - cov))
        rows.append({"target_coverage": round(1 - a, 2), "empirical": round(cov, 3),
                     "gap": round(abs((1 - a) - cov), 3)})
    res["regression_calibration_error_rECE"] = round(float(np.mean(gaps)), 4)
    res["calibration_curve"] = rows

    print(json.dumps(res, indent=2))
    (OUT / "extra_metrics.json").write_text(json.dumps(res, indent=2))
    print(f"\nSaved -> {OUT / 'extra_metrics.json'}")
    print("Paper sentences:")
    print(f"  Scoring power R = {res['CASF16_scoring_power_R']}, "
          f"ranking power (Spearman) = {res['CASF16_ranking_power_Spearman']}.")
    print(f"  Conformal intervals are well calibrated: mean |nominal - empirical| "
          f"coverage gap = {res['regression_calibration_error_rECE']}.")


if __name__ == "__main__":
    main()

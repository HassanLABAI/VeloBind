#!/usr/bin/env python3
# scripts/velobind_distill.py
#
# Born-again / distilled compact VeloBind: train ONE LightGBM student to match the
# 45-model ensemble's predictions instead of the raw labels, then check whether it
# recovers the ~0.013 CASF-2016 R that the plain single model (velobind_single.py)
# gives up.
#
# Regression distillation (not the classification soft-label/temperature recipe):
#   teacher target t(x) = ensemble's HONEST out-of-fold prediction for that row
#                         (meta-learner refit out-of-fold on oof_matrix -> pKd).
#   student trains on a blend:  target = alpha * y_true + (1 - alpha) * t(x).
#     alpha = 1 -> plain single model (== velobind_single)
#     alpha = 0 -> pure distillation (learn the smoothed teacher surface)
# Refs: Hinton et al. 2015 (1503.02531); Puri et al. 2020 Born-Again Tree Ensembles
#       (2003.11132).
#
# Calibration is unaffected: the conformal layer is refit on the STUDENT's own OOF
# residuals, so coverage holds for whatever student we ship.
#
# Output: output/single/distill_metrics.json (+ console table)

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import config
from src.features.assembly import assemble_flagged, load_winner_kwargs
from src.models.ensemble import _lgbm_rmse, _fit, TargetScaler

OUT = config.OUTPUT_DIR / "single"
MODELS = config.OUTPUT_DIR / "models"
SKIP = {"labels", "pdb_ids", "truncated"}
ALPHAS = (0.0, 0.25, 0.5, 1.0)


def load_assembled(npz_path, winner):
    d = np.load(npz_path, allow_pickle=True)
    data = {k: d[k] for k in d.files if k not in SKIP}
    return assemble_flagged(data, **winner), d


def fit_lgbm(X, yz):
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(yz))
    cut = int(0.9 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    m = _lgbm_rmse(42, config.LR, config.N_TREES)
    _fit(m, X[tr], yz[tr], X[va], yz[va], config.EARLY_STOP)
    return m


def teacher_oof(oof_matrix, y):
    """Ensemble's honest out-of-fold pKd prediction per training row (the soft target)."""
    t = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=42).split(oof_matrix):
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(oof_matrix[tr], y[tr])
        t[va] = m.predict(oof_matrix[va])
    return t


def main():
    winner = load_winner_kwargs(MODELS / "best_cfg.json")
    Xtr, dtr = load_assembled(config.DATA_DIR / "X_train.npz", winner)
    ytr = dtr["labels"].astype(float)
    Xte16, _ = load_assembled(config.DATA_DIR / "X_test.npz", winner)
    y16 = pd.read_csv(config.DATA_DIR / "casf16_clean.csv")["label"].values
    Xte13, _ = load_assembled(config.DATA_DIR / "X_casf13.npz", winner)
    y13 = pd.read_csv(config.DATA_DIR / "casf13_clean.csv")["label"].values

    print("=" * 64)
    print(f"VELOBIND — distillation sweep  (X={Xtr.shape})")
    print("=" * 64)

    oof_matrix = np.load(MODELS / "oof_matrix.npy")
    t_oof = teacher_oof(oof_matrix, ytr)
    print(f"Teacher (ensemble) OOF R = {pearsonr(t_oof, ytr)[0]:.4f}")

    rows = []
    for alpha in ALPHAS:
        target = alpha * ytr + (1 - alpha) * t_oof
        scaler = TargetScaler().fit(target)
        model = fit_lgbm(Xtr, scaler.transform(target))

        def ev(X, y):
            p = scaler.inverse(model.predict(X))
            return round(float(pearsonr(p, y)[0]), 4)
        r16, r13 = ev(Xte16, y16), ev(Xte13, y13)

        # honest student OOF R (model-selection signal — NOT the test set)
        oof = np.zeros(len(ytr))
        for tr, va in KFold(3, shuffle=True, random_state=0).split(Xtr):
            sc = TargetScaler().fit(target[tr])
            m = fit_lgbm(Xtr[tr], sc.transform(target[tr]))
            oof[va] = sc.inverse(m.predict(Xtr[va]))
        oof_r = round(float(pearsonr(oof, ytr)[0]), 4)

        tag = "plain single" if alpha == 1.0 else ("pure distill" if alpha == 0.0 else "blend")
        rows.append(dict(alpha=alpha, tag=tag, CASF16_R=r16, CASF13_R=r13, OOF_R=oof_r))
        print(f"  alpha={alpha:<4} ({tag:12s})  CASF16={r16}  CASF13={r13}  OOF={oof_r}")

    df = pd.DataFrame(rows)
    best = df.loc[df["OOF_R"].idxmax()]   # select by OOF, report its CASF
    print("\n" + "-" * 64)
    print(f"Best by OOF: alpha={best.alpha} ({best.tag}) "
          f"-> CASF16={best.CASF16_R}  (plain single=0.8336, ensemble=0.8465)")
    gap_closed = (best.CASF16_R - 0.8336) / (0.8465 - 0.8336) * 100
    print(f"Recovers {gap_closed:.0f}% of the single->ensemble gap.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "distill_metrics.json").write_text(json.dumps(
        {"rows": rows, "best_alpha": float(best.alpha),
         "best_CASF16_R": float(best.CASF16_R),
         "ensemble_CASF16_R": 0.8465, "plain_single_CASF16_R": 0.8336},
        indent=2))
    print(f"Saved -> {OUT / 'distill_metrics.json'}")


if __name__ == "__main__":
    main()

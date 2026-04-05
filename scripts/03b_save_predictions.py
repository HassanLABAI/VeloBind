# scripts/03b_save_predictions.py
#
# Recovery + prediction optimisation script.
# Loads already-saved OOF/test matrices and finds the best blend.
# No retraining required. Run AFTER 03_train.py.
#
# v5 changes:
#   n_models = 3 (LGBM, CatBoost, XGBoost) — LGBM-Quantile dropped
#   Column layout: [s0_lgbm, s0_cb, s0_xgb, s1_lgbm, s1_cb, s1_xgb, ...]
#
# Tries five blending strategies and picks best by RMSE:
#   1. Equal blend (all 9 columns)
#   2. LGBM only (cols 0, 3, 6)
#   3. Ridge meta on all 9 columns
#   4. Ridge meta on LGBM-only columns
#   5. Ridge meta on per-type blends (3 → 1 prediction per model type)

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.evaluation.metrics import evaluate, print_row, print_comparison_table, scatter_plot


def main():
    print("=" * 60)
    print("VELOBIND — Recovery + Prediction Optimisation (v5)")
    print("=" * 60)

    # ── Load artefacts ────────────────────────────────────────────────
    te      = np.load(config.DATA_DIR / "X_test.npz",  allow_pickle=True)
    tr      = np.load(config.DATA_DIR / "X_train.npz", allow_pickle=True)
    y_test  = te['labels']
    y_train = tr['labels']

    oof_mat  = np.load(config.OUTPUT_DIR / "models" / "oof_matrix.npy")
    test_mat = np.load(config.OUTPUT_DIR / "models" / "test_matrix.npy")

    n_seeds  = len(config.SEEDS)   # 3
    n_models = 3                   # LGBM, CatBoost, XGBoost
    print(f"\nOOF matrix:  {oof_mat.shape}  "
          f"({n_seeds} seeds × {n_models} models)")
    print(f"Test matrix: {test_mat.shape}")

    # Column layout: [s0_lgbm, s0_cb, s0_xgb, s1_lgbm, s1_cb, s1_xgb, ...]
    lgbm_cols = [i * n_models + 0 for i in range(n_seeds)]   # [0, 3, 6]
    cb_cols   = [i * n_models + 1 for i in range(n_seeds)]   # [1, 4, 7]
    xgb_cols  = [i * n_models + 2 for i in range(n_seeds)]   # [2, 5, 8]

    print(f"\nLGBM columns:     {lgbm_cols}")
    print(f"CatBoost columns: {cb_cols}")
    print(f"XGBoost columns:  {xgb_cols}")

    # ── Per-model-type OOF performance ────────────────────────────────
    print("\n[Per-Model-Type OOF R on Training Set]")
    for name, cols in [("LGBM",     lgbm_cols),
                       ("CatBoost", cb_cols),
                       ("XGBoost",  xgb_cols)]:
        blend = oof_mat[:, cols].mean(axis=1)
        r = pearsonr(blend, y_train)[0]
        print(f"  {name:<12}  OOF R={r:.4f}")

    # ── Five blending strategies on test set ──────────────────────────
    print("\n[Blending Strategies — CASF-2016 Test]")

    results = {}

    # 1. Equal blend — all columns
    results['Equal blend (all)']      = test_mat.mean(axis=1)

    # 2. LGBM only
    results['LGBM only']              = test_mat[:, lgbm_cols].mean(axis=1)

    # 3. Ridge meta — all 9 columns (fit on OOF)
    meta_all = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    meta_all.fit(oof_mat, y_train)
    results['Ridge meta (all cols)']  = meta_all.predict(test_mat)

    # 4. Ridge meta — LGBM only
    meta_lgbm = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    meta_lgbm.fit(oof_mat[:, lgbm_cols], y_train)
    results['Ridge meta (LGBM only)'] = meta_lgbm.predict(test_mat[:, lgbm_cols])

    # 5. Ridge meta — per-type blends (3 → 3 cols → weighted)
    type_oof  = np.column_stack([
        oof_mat[:, lgbm_cols].mean(1),
        oof_mat[:, cb_cols].mean(1),
        oof_mat[:, xgb_cols].mean(1),
    ])
    type_test = np.column_stack([
        test_mat[:, lgbm_cols].mean(1),
        test_mat[:, cb_cols].mean(1),
        test_mat[:, xgb_cols].mean(1),
    ])
    meta_type = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    meta_type.fit(type_oof, y_train)
    results['Ridge meta (per-type)']  = meta_type.predict(type_test)
    print(f"  Meta type weights: LGBM={meta_type.coef_[0]:.3f}  "
          f"CB={meta_type.coef_[1]:.3f}  XGB={meta_type.coef_[2]:.3f}")

    # ── Print all + pick best ─────────────────────────────────────────
    best_name, best_preds, best_rmse = None, None, 999
    for name, preds in results.items():
        m = evaluate(preds, y_test)
        print_row(name, m)
        if m['RMSE'] < best_rmse:
            best_rmse  = m['RMSE']
            best_name  = name
            best_preds = preds

    best_m = evaluate(best_preds, y_test)
    print(f"\n  Best strategy: {best_name}")
    print_comparison_table(best_m, len(y_test))

    # ── Save models ───────────────────────────────────────────────────
    model_dir = config.OUTPUT_DIR / "models"
    joblib.dump(meta_all,  model_dir / "meta_all.pkl")
    joblib.dump(meta_lgbm, model_dir / "meta_lgbm_only.pkl")
    joblib.dump(meta_type, model_dir / "meta_type_blend.pkl")

    # ── Save predictions CSV ──────────────────────────────────────────
    pred_df = pd.DataFrame({
        'pdb_id':              te['pdb_ids'],
        'y_true':              y_test,
        'pred_equal':          results['Equal blend (all)'],
        'pred_lgbm':           results['LGBM only'],
        'pred_meta_all':       results['Ridge meta (all cols)'],
        'pred_meta_lgbm':      results['Ridge meta (LGBM only)'],
        'pred_meta_type':      results['Ridge meta (per-type)'],
        'pred_best':           best_preds,
    })
    pred_df.to_csv(config.OUTPUT_DIR / "predictions.csv", index=False)
    print(f"\n✓ predictions.csv saved ({len(pred_df)} rows)")

    # Also write a clean predictions_casf16.csv for 06_eval_both.py
    pred_df.to_csv(config.OUTPUT_DIR / "predictions_casf16.csv", index=False)

    # ── Scatter plot of best ──────────────────────────────────────────
    fig_dir = config.OUTPUT_DIR / "figures"
    fig_dir.mkdir(exist_ok=True)
    scatter_plot(
        y_test, best_preds, best_m,
        title=f"VELOBIND — CASF-2016  [{best_name}]",
        out_path=fig_dir / "velobind_final_scatter.png"
    )

    print(f"\nFinal result: R={best_m['R']:.4f}  RMSE={best_m['RMSE']:.4f}  "
          f"MAE={best_m['MAE']:.4f}  Sp={best_m['Sp']:.4f}")
    print("\n✓ Done. Run 06_eval_both.py next.")


if __name__ == "__main__":
    main()

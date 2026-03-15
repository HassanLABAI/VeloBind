# src/models/ensemble.py
#
# Multi-seed OOF ensemble.
# 3 seeds × 3 model types × 5 folds = 45 GBM runs.
# Models: LGBM-RMSE, CatBoost, XGBoost.
# All trained on z-scored targets, predictions inverse-transformed.

import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import joblib
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.model_selection import KFold


class TargetScaler:
    """Z-score normalization on training targets only."""
    def fit(self, y):
        self.mu  = y.mean()
        self.std = y.std()
        return self
    def transform(self, y):   return (y - self.mu) / self.std
    def inverse(self, y):     return y * self.std + self.mu


def _lgbm_rmse(seed, lr, n_trees):
    return lgb.LGBMRegressor(
        objective='regression', num_leaves=63, max_depth=7,
        learning_rate=lr, n_estimators=n_trees,
        min_child_samples=25, subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.2, reg_lambda=2.0,
        random_state=seed, n_jobs=4, verbose=-1,
    )

def _catboost(seed, lr, n_trees, early_stop):
    return cb.CatBoostRegressor(
        depth=7, learning_rate=lr, iterations=n_trees,
        l2_leaf_reg=5.0, subsample=0.75, min_data_in_leaf=25,
        loss_function='RMSE', eval_metric='RMSE',
        early_stopping_rounds=early_stop,
        random_seed=seed, verbose=0, task_type='CPU',
    )

def _xgboost(seed, lr, n_trees, early_stop):
    return xgb.XGBRegressor(
        n_estimators=n_trees, max_depth=6, learning_rate=lr,
        subsample=0.75, colsample_bytree=0.75,
        reg_alpha=0.2, reg_lambda=2.0, min_child_weight=6,
        early_stopping_rounds=early_stop, eval_metric='rmse',
        random_state=seed, n_jobs=4, verbosity=0,
    )


def _fit(model, Xtr, ytr, Xval, yval, early_stop):
    if isinstance(model, lgb.LGBMRegressor):
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)],
                  callbacks=[lgb.early_stopping(early_stop, verbose=False),
                              lgb.log_evaluation(-1)])
    elif isinstance(model, cb.CatBoostRegressor):
        model.fit(Xtr, ytr, eval_set=(Xval, yval), use_best_model=True)
    else:
        model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    return model


def run_oof(X_train, y_train_raw, X_test,
            seeds, n_folds, lr, n_trees, early_stop,
            models_dir: Path = None) -> tuple:
    """
    Multi-seed OOF stacking.
    3 model types: LGBM, CatBoost, XGBoost.

    Args:
        models_dir: if provided, saves each fold model as
                    fold_model_s{seed}_{type}_f{fold}.pkl
                    Required for CASF-2013 zero-shot evaluation.

    Returns:
        oof_matrix   [N_train, n_seeds * 3]
        test_matrix  [N_test,  n_seeds * 3]
        scaler       fitted TargetScaler
    """
    scaler  = TargetScaler().fit(y_train_raw)
    y_train = scaler.transform(y_train_raw)
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    n_cols   = len(seeds) * 3
    oof_mat  = np.zeros((len(X_train), n_cols))
    test_mat = np.zeros((len(X_test),  n_cols))

    for si, seed in enumerate(seeds):
        print(f"\n  Seed {seed}  ({si+1}/{len(seeds)})")

        oof_lr = np.zeros(len(X_train))
        oof_cb = np.zeros(len(X_train))
        oof_xb = np.zeros(len(X_train))

        t_lr = np.zeros((len(X_test), n_folds))
        t_cb = np.zeros((len(X_test), n_folds))
        t_xb = np.zeros((len(X_test), n_folds))

        for fold, (tri, vali) in enumerate(kf.split(X_train)):
            Xtr, Xval = X_train[tri], X_train[vali]
            ytr, yval = y_train[tri], y_train[vali]

            mlr = _fit(_lgbm_rmse(seed, lr, n_trees),           Xtr, ytr, Xval, yval, early_stop)
            mcb = _fit(_catboost(seed, lr, n_trees, early_stop), Xtr, ytr, Xval, yval, early_stop)
            mxb = _fit(_xgboost(seed, lr, n_trees, early_stop), Xtr, ytr, Xval, yval, early_stop)

            # Save fold models for zero-shot evaluation on new test sets
            if models_dir is not None:
                models_dir = Path(models_dir)
                models_dir.mkdir(exist_ok=True)
                joblib.dump(mlr, models_dir / f"fold_model_s{seed}_lgbm_f{fold}.pkl")
                joblib.dump(mcb, models_dir / f"fold_model_s{seed}_cb_f{fold}.pkl")
                joblib.dump(mxb, models_dir / f"fold_model_s{seed}_xgb_f{fold}.pkl")

            oof_lr[vali] = mlr.predict(Xval)
            oof_cb[vali] = mcb.predict(Xval)
            oof_xb[vali] = mxb.predict(Xval)

            t_lr[:, fold] = mlr.predict(X_test)
            t_cb[:, fold] = mcb.predict(X_test)
            t_xb[:, fold] = mxb.predict(X_test)

        base = si * 3
        oof_mat[:, base+0] = scaler.inverse(oof_lr)
        oof_mat[:, base+1] = scaler.inverse(oof_cb)
        oof_mat[:, base+2] = scaler.inverse(oof_xb)

        test_mat[:, base+0] = scaler.inverse(t_lr.mean(1))
        test_mat[:, base+1] = scaler.inverse(t_cb.mean(1))
        test_mat[:, base+2] = scaler.inverse(t_xb.mean(1))

        p = pearsonr(oof_mat[:, base:base+3].mean(1), y_train_raw)[0]
        print(f"    OOF Pearson (seed {seed}): {p:.4f}")

    return oof_mat, test_mat, scaler

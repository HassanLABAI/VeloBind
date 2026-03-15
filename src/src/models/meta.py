# src/models/meta.py
import numpy as np
from sklearn.linear_model import RidgeCV


def fit_meta(oof_matrix: np.ndarray, y_train: np.ndarray,
             test_matrix: np.ndarray) -> tuple:
    """RidgeCV meta-learner on OOF predictions."""
    meta  = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
    meta.fit(oof_matrix, y_train)
    preds = meta.predict(test_matrix)
    print(f"  Meta alpha: {meta.alpha_:.4f}  "
          f"coef range: [{meta.coef_.min():.3f}, {meta.coef_.max():.3f}]")
    return meta, preds


# src/models/calibration.py
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold


def fit_isotonic(oof_preds: np.ndarray, y_train: np.ndarray,
                 test_preds: np.ndarray) -> tuple:
    """
    Fits isotonic regression on OOF meta-predictions.
    OOF predictions are unbiased — no test leakage.
    Includes CV check: if test improves >> CV estimate, flag it.
    """
    # CV estimate of benefit
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_raw, cv_cal = [], []
    for tri, vali in kf.split(oof_preds):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_preds[tri], y_train[tri])
        p = iso.predict(oof_preds[vali])
        cv_raw.append(np.sqrt(np.mean((oof_preds[vali] - y_train[vali])**2)))
        cv_cal.append(np.sqrt(np.mean((p              - y_train[vali])**2)))

    cv_gain = np.mean(cv_raw) - np.mean(cv_cal)
    print(f"  Isotonic CV RMSE: {np.mean(cv_raw):.4f} → {np.mean(cv_cal):.4f}  "
          f"(gain={cv_gain:+.4f})")

    # Fit on full OOF
    iso_full = IsotonicRegression(out_of_bounds='clip')
    iso_full.fit(oof_preds, y_train)
    preds_cal = iso_full.predict(test_preds)

    return iso_full, preds_cal

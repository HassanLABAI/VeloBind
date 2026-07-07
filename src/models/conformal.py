# Conformal selective prediction for triage virtual screening.
#
# Wraps the existing RidgeCV meta-learner. Turns a point pKd prediction into:
#   (1) a calibrated prediction interval with finite-sample marginal coverage
#   (2) a per-compound uncertainty score (interval half-width)

import numpy as np

EPS = 1e-6


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1-alpha) quantile of calibration scores."""
    n = len(scores)
    if n == 0:
        return np.inf
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def base_disagreement(base_matrix: np.ndarray) -> np.ndarray:
    """Per-row std across the base-model columns (oof_matrix / test_matrix).

    Used as the difficulty estimate sigma(x). A small floor avoids divide-by-zero
    when all base models agree exactly.
    """
    sigma = base_matrix.std(axis=1)
    return np.maximum(sigma, np.median(sigma) * 0.1 + EPS)


class ConformalSelectivePredictor:
    """Locally-adaptive split-conformal predictor with an optional Mondrian mode.

    Parameters
    ----------
    alpha : float
        Target miscoverage. Coverage target is 1 - alpha (e.g. alpha=0.1 -> 90%).
    normalize : bool
        If True, use normalized nonconformity |y-yhat|/sigma (recommended; needed
        for per-point triage). If False, plain absolute-residual conformal.
    """

    def __init__(self, alpha: float = 0.1, normalize: bool = True):
        self.alpha = alpha
        self.normalize = normalize
        self.q_ = None            # global quantile
        self.group_q_ = None      # dict {group_id: quantile} in Mondrian mode

    def fit(self, cal_preds, cal_y, cal_sigma=None, cal_groups=None):
        cal_preds = np.asarray(cal_preds, float)
        cal_y = np.asarray(cal_y, float)
        resid = np.abs(cal_preds - cal_y)

        if self.normalize:
            if cal_sigma is None:
                raise ValueError("normalize=True requires cal_sigma")
            scores = resid / np.asarray(cal_sigma, float)
        else:
            scores = resid
        self._scores = scores

        if cal_groups is None:
            self.q_ = _conformal_quantile(scores, self.alpha)
            self.group_q_ = None
        else:
            cal_groups = np.asarray(cal_groups)
            self.group_q_ = {
                g: _conformal_quantile(scores[cal_groups == g], self.alpha)
                for g in np.unique(cal_groups)
            }
            # fallback for groups unseen at calibration time
            self.q_ = _conformal_quantile(scores, self.alpha)
        return self

    def _q_for(self, groups, n):
        if self.group_q_ is None or groups is None:
            return np.full(n, self.q_)
        groups = np.asarray(groups)
        return np.array([self.group_q_.get(g, self.q_) for g in groups])

    def half_width(self, preds, sigma=None, groups=None):
        """Interval half-width per compound = q * sigma(x). This is the triage score."""
        preds = np.asarray(preds, float)
        q = self._q_for(groups, len(preds))
        if self.normalize:
            if sigma is None:
                raise ValueError("normalize=True requires sigma")
            return q * np.asarray(sigma, float)
        return q

    def interval(self, preds, sigma=None, groups=None):
        preds = np.asarray(preds, float)
        hw = self.half_width(preds, sigma, groups)
        return preds - hw, preds + hw

    def coverage(self, preds, y, sigma=None, groups=None):
        """Empirical coverage: fraction of true values inside the interval."""
        lo, hi = self.interval(preds, sigma, groups)
        y = np.asarray(y, float)
        return float(np.mean((y >= lo) & (y <= hi)))

    def selective_mask(self, preds, keep_fraction, sigma=None, groups=None):
        """Boolean keep-mask retaining the `keep_fraction` most-confident compounds
        (smallest half-width). The discarded remainder is what you DON'T spend
        docking/MD compute on.
        """
        hw = self.half_width(preds, sigma, groups)
        if keep_fraction >= 1.0:
            return np.ones(len(hw), bool)
        thr = np.quantile(hw, keep_fraction)
        return hw <= thr

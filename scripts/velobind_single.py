# Compact single-model VeloBind: ONE LightGBM instead of the 45-model ensemble, with per-protein-family (Mondrian) conformal calibration. Zero-config deployment.

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import config
from src.features.assembly import assemble_flagged, load_winner_kwargs
from src.models.ensemble import _lgbm_rmse, _fit, TargetScaler
from src.models.conformal import ConformalSelectivePredictor

OUT = config.OUTPUT_DIR / "single"
OUT.mkdir(parents=True, exist_ok=True)
SKIP = {"labels", "pdb_ids", "truncated"}


def load_assembled(npz_path, winner):
    d = np.load(npz_path, allow_pickle=True)
    data = {k: d[k] for k in d.files if k not in SKIP}
    return assemble_flagged(data, **winner), d


def fit_lgbm(X, yz):
    """One LightGBM with a 90/10 internal split for early stopping."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(yz))
    cut = int(0.9 * len(idx))
    tr, va = idx[:cut], idx[cut:]
    m = _lgbm_rmse(42, config.LR, config.N_TREES)
    _fit(m, X[tr], yz[tr], X[va], yz[va], config.EARLY_STOP)
    return m


def main():
    winner = load_winner_kwargs(config.OUTPUT_DIR / "models" / "best_cfg.json")
    Xtr, dtr = load_assembled(config.DATA_DIR / "X_train.npz", winner)
    ytr = dtr["labels"].astype(float)
    Xte16, _ = load_assembled(config.DATA_DIR / "X_test.npz", winner)
    y16 = pd.read_csv(config.DATA_DIR / "casf16_clean.csv")["label"].values
    Xte13, _ = load_assembled(config.DATA_DIR / "X_casf13.npz", winner)
    y13 = pd.read_csv(config.DATA_DIR / "casf13_clean.csv")["label"].values

    print("=" * 60)
    print(f"VELOBIND -- single-model  (X={Xtr.shape})")
    print("=" * 60)

    scaler = TargetScaler().fit(ytr)
    yz = scaler.transform(ytr)

    print("Training single LightGBM on all training data...")
    model = fit_lgbm(Xtr, yz)

    def ev(X, y, name):
        p = scaler.inverse(model.predict(X))
        r, rmse = pearsonr(p, y)[0], float(np.sqrt(np.mean((p - y) ** 2)))
        print(f"  {name}: R={r:.4f}  RMSE={rmse:.4f}")
        return round(r, 4), round(rmse, 4)
    r16, rmse16 = ev(Xte16, y16, "CASF-2016")
    r13, rmse13 = ev(Xte13, y13, "CASF-2013")

    print("5-fold OOF for conformal calibration...")
    oof = np.zeros(len(ytr))
    for tr, va in KFold(config.N_FOLDS, shuffle=True, random_state=42).split(Xtr):
        m = fit_lgbm(Xtr[tr], yz[tr])
        oof[va] = scaler.inverse(m.predict(Xtr[va]))
    oof_r = round(pearsonr(oof, ytr)[0], 4)
    print(f"  OOF R={oof_r}")

    clusters = MiniBatchKMeans(12, random_state=42, n_init=3).fit_predict(dtr["prot_esm_mean"])
    cp = ConformalSelectivePredictor(alpha=0.1, normalize=False).fit(oof, ytr, cal_groups=clusters)
    cov = cp.coverage(oof, ytr, groups=clusters)
    print(f"  Conformal per-family coverage @90%: {cov:.3f}")

    model.booster_.save_model(str(OUT / "velobind_single.txt"))
    joblib.dump(scaler, OUT / "target_scaler.pkl")
    joblib.dump(cp, OUT / "conformal.pkl")
    metrics = {"CASF16_R": r16, "CASF16_RMSE": rmse16, "CASF13_R": r13,
               "CASF13_RMSE": rmse13, "OOF_R": oof_r, "conformal_cov90": round(cov, 3)}
    (OUT / "single_metrics.json").write_text(json.dumps(metrics, indent=2))

    sz = (OUT / "velobind_single.txt").stat().st_size / 1e6
    print(f"\nSingle model: {sz:.1f} MB  (vs ~450 MB for the 45-model ensemble)")
    print(f"Compare: ensemble CASF-2016 R=0.8465  ->  single R={r16}")
    print(f"Saved -> {OUT}/")


if __name__ == "__main__":
    main()

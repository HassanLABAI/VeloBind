# Protein-family-stratified evaluation

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "output" / "models"
PROC = ROOT / "data" / "processed"
OUT = ROOT / "output" / "conformal"
OUT.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 12
SEED = 42


def _metrics(pred, y):
    if len(y) < 3 or np.std(y) < 1e-6:
        return float("nan"), float("nan")
    return pearsonr(pred, y)[0], float(np.sqrt(np.mean((pred - y) ** 2)))


def main():
    import joblib
    oof = np.load(MODELS / "oof_matrix.npy")
    test = np.load(MODELS / "test_matrix.npy")
    dtr = np.load(PROC / "X_train.npz", allow_pickle=True)
    dte = np.load(PROC / "X_test.npz", allow_pickle=True)
    y = dtr["labels"].astype(float)
    emb_tr = dtr["prot_esm_mean"]
    emb_te = dte["prot_esm_mean"]
    c16 = pd.read_csv(PROC / "casf16_clean.csv")
    y16 = c16["label"].values
    meta = joblib.load(MODELS / "meta.pkl")
    p16 = meta.predict(test)

    oof_meta = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(oof):
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(oof[tr], y[tr])
        oof_meta[va] = m.predict(oof[va])

    km = MiniBatchKMeans(N_CLUSTERS, random_state=SEED, n_init=3).fit(emb_tr)
    cl_tr = km.labels_
    cl_te = km.predict(emb_te)

    rows = []
    for g in range(N_CLUSTERS):
        mtr, mte = cl_tr == g, cl_te == g
        r_tr, rmse_tr = _metrics(oof_meta[mtr], y[mtr])
        r_te, rmse_te = _metrics(p16[mte], y16[mte])
        rows.append({
            "cluster": g,
            "n_train": int(mtr.sum()),
            "n_casf16": int(mte.sum()),
            "R_train_oof": round(r_tr, 4),
            "RMSE_train_oof": round(rmse_tr, 4),
            "R_casf16": round(r_te, 4) if not np.isnan(r_te) else None,
            "RMSE_casf16": round(rmse_te, 4) if not np.isnan(rmse_te) else None,
        })

    df = pd.DataFrame(rows).sort_values("n_train", ascending=False)
    df.to_csv(OUT / "family_stratified.csv", index=False)

    print(f"Per-cluster performance (K={N_CLUSTERS} ESM-embedding clusters)\n")
    print(df.to_string(index=False))
    overall_r, overall_rmse = _metrics(p16, y16)
    print(f"\nOverall CASF-2016: R={overall_r:.4f}  RMSE={overall_rmse:.4f}")
    small = df.nsmallest(3, "n_train")
    print("\nUnder-represented clusters (smallest n_train) — the generalization stress test:")
    print(small[["cluster", "n_train", "n_casf16", "R_casf16", "RMSE_casf16"]].to_string(index=False))
    print(f"\nSaved -> {OUT / 'family_stratified.csv'}")


if __name__ == "__main__":
    main()

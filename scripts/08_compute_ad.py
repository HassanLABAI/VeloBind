import sys
import joblib
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.applicability_domain import EmbeddingAD


def main():
    print("=" * 55)
    print("VELOBIND — Step 8: Fit Applicability Domain (kNN)")
    print("=" * 55)

    npz_path = config.DATA_DIR / "X_train.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found. Run 02_extract_features.py first.")
        return

    tr  = np.load(npz_path, allow_pickle=True)
    esm = tr['prot_esm_mean'][:, -config.ESM_DIM:]
    print(f"  Training proteins: {esm.shape[0]}  |  ESM dim: {esm.shape[1]}")

    ad = EmbeddingAD(k=config.AD_KNN_K, percentile=config.AD_PERCENTILE)
    ad.fit(esm)

    out_dir = config.OUTPUT_DIR / "models" / "deployment"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ad_model.pkl"
    joblib.dump(ad, out_path)

    print(f"\n  Saved: {out_path}")
    print(f"  AD threshold ({config.AD_PERCENTILE}th pct): {ad.threshold:.4f}")
    print(f"\n  NOTE: ad_centroid.npy / ad_threshold.npy are superseded.")
    print(f"        Load ad_model.pkl and call ad.score(embedding) instead.")


if __name__ == "__main__":
    main()

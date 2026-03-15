# scripts/09_compute_ad.py
import numpy as np
from pathlib import Path

tr = np.load("data/processed/X_train.npz", allow_pickle=True)
# Use last-layer ESM mean (480d) — matches what 06_casf_eval uses
esm = tr['prot_esm_mean'][:, -480:]

centroid  = esm.mean(axis=0)
dists     = np.linalg.norm(esm - centroid, axis=1)
threshold = float(np.percentile(dists, 95))   # 95th percentile of training distances

out = Path("output/models/deployment")
out.mkdir(parents=True, exist_ok=True)
np.save(out / "ad_centroid.npy",  centroid)
np.save(out / "ad_threshold.npy", np.array(threshold))

print(f"Centroid shape: {centroid.shape}")
print(f"AD threshold (95th pct): {threshold:.4f}")
print(f"Poly-alanine would be flagged if its distance > {threshold:.4f}")
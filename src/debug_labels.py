# Quick debug script — run this before anything else
import numpy as np
import pandas as pd
from pathlib import Path

tr = np.load("data/processed/X_train.npz")
te = np.load("data/processed/X_test.npz")

print("=== LABEL CHECK ===")
print(f"y_train: min={tr['labels'].min():.3f}  max={tr['labels'].max():.3f}  "
      f"mean={tr['labels'].mean():.3f}  std={tr['labels'].std():.3f}")
print(f"y_test:  min={te['labels'].min():.3f}  max={te['labels'].max():.3f}  "
      f"mean={te['labels'].mean():.3f}  std={te['labels'].std():.3f}")

print("\n=== FEATURE CHECK ===")
print(f"X_train shape: {np.concatenate([tr['prot_esm'], tr['prot_phys'], tr['lig_ecfp'], tr['lig_maccs'], tr['lig_physical'], tr['interaction']], axis=1).shape}")
print(f"X_test  shape: {np.concatenate([te['prot_esm'], te['prot_phys'], te['lig_ecfp'], te['lig_maccs'], te['lig_physical'], te['interaction']], axis=1).shape}")

print("\n=== SAMPLE LABELS (first 10 train) ===")
print(tr['labels'][:10])

print("\n=== SAMPLE LABELS (first 10 test) ===")
print(te['labels'][:10])

# Check clean CSVs too
train_csv = pd.read_csv("data/processed/train_clean.csv")
casf_csv  = pd.read_csv("data/processed/casf_clean.csv")
print(f"\n=== CSV LABEL CHECK ===")
print(f"train_clean label: min={train_csv['label'].min():.3f}  "
      f"max={train_csv['label'].max():.3f}  mean={train_csv['label'].mean():.3f}")
print(f"casf_clean  label: min={casf_csv['label'].min():.3f}  "
      f"max={casf_csv['label'].max():.3f}  mean={casf_csv['label'].mean():.3f}")
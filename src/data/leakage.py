# src/data/leakage.py
#
# Verifies zero overlap between training PDB IDs and CASF-2016.
# Saves a supplementary CSV confirming each CASF complex was not in training.
# This CSV goes directly into the paper as Supplementary Table S1.

import pandas as pd
from pathlib import Path


def check_leakage(train_df: pd.DataFrame,
                  casf_df:  pd.DataFrame,
                  out_path: Path) -> int:
    """
    Compares PDB IDs between training and test sets.

    Returns number of overlapping complexes (should be 0).
    Saves a report CSV at out_path.
    """
    train_ids = set(train_df['pdb_id'].str.lower())
    casf_ids  = set(casf_df['pdb_id'].str.lower())

    overlap = train_ids & casf_ids

    # Build supplementary table
    rows = []
    for pid in sorted(casf_ids):
        rows.append({
            'PDB_ID':       pid.upper(),
            'In_Training':  'Yes' if pid in train_ids else 'No',
        })

    report = pd.DataFrame(rows)
    report.to_csv(out_path, index=False)

    print(f"\nLeakage Check:")
    print(f"  Training complexes:  {len(train_ids)}")
    print(f"  CASF complexes:      {len(casf_ids)}")
    print(f"  Overlap:             {len(overlap)}")

    if overlap:
        print(f"\n  WARNING — overlapping PDB IDs:")
        for pid in sorted(overlap):
            print(f"    {pid}")
    else:
        print(f"  Result: CLEAN — zero overlap confirmed")

    print(f"  Report saved: {out_path}")
    return len(overlap)

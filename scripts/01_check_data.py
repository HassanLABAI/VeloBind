# scripts/01_check_data.py
#
# Run this FIRST before anything else.
# Verifies all paths exist, runs leakage check, reports dropped CASF IDs.
# Takes ~2 minutes. Saves a leakage report CSV for Supplementary Table S1.

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.config import config
from src.data.loader import load_lppdb, load_casf, load_casf2013
from src.data.leakage import check_leakage


def main():
    print("=" * 55)
    print("VELOBIND — Step 1: Data Check")
    print("=" * 55)

    # ── Path checks ──
    print("\n[Paths]")
    lp_csv = config.RAW_DIR / config.LPPDB_CSV
    checks = {
        "LP-PDBBind CSV":    lp_csv,
        "CASF-2016 dir":     config.CASF_DIR,
        "CASF-2016 dat":     config.CASF_DIR / "power_scoring" / "CoreSet.dat",
        "CASF-2016 coreset": config.CASF_DIR / "coreset",
        "CASF-2013 dir":     config.CASF13_DIR,
        "CASF-2013 dat":     config.CASF13_DIR / "power_scoring" / "CoreSet.dat",
        "CASF-2013 coreset": config.CASF13_DIR / "coreset",
    }
    all_ok = True
    for name, path in checks.items():
        ok = path.exists()
        print(f"  {'✓' if ok else '✗'} {name}: {path}")
        if not ok and 'CASF-2013' not in name:
            all_ok = False   # CASF-2013 missing is fatal
        elif not ok:
            all_ok = False

    if not all_ok:
        print("\nFix missing paths before continuing.")
        sys.exit(1)

    # ── Load data ──
    print("\n[LP-PDBBind]")
    train_df = load_lppdb(lp_csv)
    pmin = train_df['label'].min()
    pmax = train_df['label'].max()
    pmean = train_df['label'].mean()
    pstd = train_df['label'].std()
    print(f"  pKd range: {pmin:.2f} – {pmax:.2f}")
    print(f"  Mean pKd:  {pmean:.3f} ± {pstd:.3f} (std dev)")
    n_long = (train_df['seq'].str.len() > config.MAX_SEQ_LEN).sum()
    print(f"  Sequences > {config.MAX_SEQ_LEN} residues: {n_long} "
          f"({n_long/len(train_df)*100:.1f}%) → will use N+C chunking")

    print("\n[CASF-2016]")
    casf16_df, dropped16 = load_casf(config.CASF_DIR)
    print(f"  Usable: {len(casf16_df)}  Dropped: {len(dropped16)}")
    for pid, reason in dropped16:
        print(f"    dropped {pid}: {reason}")

    print("\n[CASF-2013]")
    casf13_df, dropped13 = load_casf2013(config.CASF13_DIR)
    print(f"  Usable: {len(casf13_df)}  Dropped: {len(dropped13)}")
    for pid, reason in dropped13:
        print(f"    dropped {pid}: {reason}")

    # ── Leakage check + removal (BOTH test sets) ──
    print("\n[Leakage Check — removing BOTH CASF-2016 and CASF-2013 from training]")

    casf16_ids = set(casf16_df['pdb_id'].str.lower())
    casf13_ids = set(casf13_df['pdb_id'].str.lower())
    all_exclude = casf16_ids | casf13_ids

    before   = len(train_df)
    train_df = train_df[~train_df['pdb_id'].str.lower().isin(all_exclude)]
    train_df = train_df.reset_index(drop=True)
    removed  = before - len(train_df)

    print(f"  CASF-2016 IDs to exclude: {len(casf16_ids)}")
    print(f"  CASF-2013 IDs to exclude: {len(casf13_ids)}")
    print(f"  Union (unique):           {len(all_exclude)}")
    print(f"  Removed from training:    {removed}")
    print(f"  Training set:             {before} → {len(train_df)}")

    # Save leakage report for supplementary
    out_csv = config.OUTPUT_DIR / "supplementary_leakage_check.csv"
    check_leakage(train_df, casf16_df, out_csv)

    # ── Save clean DataFrames ──
    train_df.to_csv(config.DATA_DIR  / "train_clean.csv",  index=False)
    casf16_df.to_csv(config.DATA_DIR / "casf16_clean.csv",   index=False)
    casf13_df.to_csv(config.DATA_DIR / "casf13_clean.csv", index=False)

    print("\n[Summary]")
    # Show final training count from extract_features output if available
    npz_path = config.DATA_DIR / "X_train.npz"
    final_training_msg = ""
    if npz_path.exists():
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if 'labels' in data:
                    final_n = data['labels'].shape[0]
                elif 'pdb_ids' in data:
                    final_n = len(data['pdb_ids'])
                else:
                    final_n = None
            if final_n is not None:
                final_training_msg = f"  Final training (after feature extraction): {final_n} complexes (from {npz_path.name})\n"
        except Exception as e:
            final_training_msg = f"  (Could not read {npz_path.name}: {e})\n"

    # original training count (after CASF removal)
    print(f"  Training (after CASF removal): {len(train_df)} complexes  ({before} → {len(train_df)})")
    # print final count if available
    if final_training_msg:
        print(final_training_msg, end="")
    print(f"  CASF-2016:   {len(casf16_df)} complexes (test set 1)")
    print(f"  CASF-2013:   {len(casf13_df)} complexes (test set 2)")
    print(f"  Saved: train_clean.csv, casf16_clean.csv, casf13_clean.csv")
    print("\n✓ All checks passed. Run 02_extract_features.py next.")


if __name__ == "__main__":
    main()

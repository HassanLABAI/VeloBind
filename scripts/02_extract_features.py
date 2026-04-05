# scripts/02_extract_features.py
#
# Feature extraction pipeline — v4
#
# Changes from v3:
#   protein: embed_batch now returns 4 values (added var_arr)
#   protein: sequence_features now 919d (added AAIndex-25)
#   ligand:  added avalon, rdkit_pat, ecfp_count, ecfp6_count
#
# NPZ keys (train + test):
#   prot_esm_mean  [N, 1440]   ESM multi-layer mean pool
#   prot_esm_var   [N, 1440]   ESM multi-layer variance pool  NEW
#   prot_esm_attn  [N,  480]   ESM attention-weighted pool
#   prot_seqfeat   [N,  919]   ProtParam+Dipeptide+CTD+ConjTriad+QSO+AAIndex
#   lig_ecfp2      [N, 1024]   Morgan r=1 binary
#   lig_ecfp       [N, 1024]   Morgan r=2 binary (ECFP4)
#   lig_ecfp6      [N, 1024]   Morgan r=3 binary
#   lig_fcfp       [N, 1024]   Functional class Morgan binary
#   lig_maccs      [N,  167]   MACCS keys
#   lig_ap         [N, 2048]   AtomPair binary
#   lig_torsion    [N, 2048]   Torsion binary
#   lig_avalon     [N,  512]   Avalon binary  NEW
#   lig_rdkit_pat  [N, 2048]   RDKit pattern  NEW
#   lig_ecfp_cnt   [N, 1024]   ECFP4 count (log1p)  NEW
#   lig_ecfp6_cnt  [N, 1024]   ECFP6 count (log1p)  NEW
#   lig_estate     [N,   79]   EState indices (scaled)
#   lig_phys       [N,  217]   RDKit descriptors (scaled)
#   interaction    [N,  512]   Cross-modal PCA interaction block
#   truncated      [N]         1.0 if sequence was chunked
#   labels         [N]         pKd values
#   pdb_ids        [N]         PDB identifiers

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.features.protein import load_esm, embed_batch, sequence_features
from src.features.ligand import extract_ligand_features
from src.features.interaction import build_interaction_features, save_pcas


def main():
    print("=" * 60)
    print("VELOBIND — Step 2: Feature Extraction v4")
    print("=" * 60)

    train_df = pd.read_csv(config.DATA_DIR / "train_clean.csv")
    casf_df  = pd.read_csv(config.DATA_DIR / "casf16_clean.csv")
    print(f"\nTrain: {len(train_df)} | Test (CASF-2016): {len(casf_df)}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # ── ESM embeddings ────────────────────────────────────────────────
    print(f"\n[Protein — ESM {config.ESM_MODEL}]")
    tokenizer, esm_model = load_esm(config.ESM_MODEL, device)

    print("  Embedding training sequences...")
    tr_mean, tr_var, tr_attn, tr_trunc = embed_batch(
        train_df['seq'].tolist(), tokenizer, esm_model,
        config.ESM_LAYERS, config.MAX_SEQ_LEN, config.HALF_SEQ_LEN,
        batch_size=8, device=device
    )
    print(f"  Train ESM: mean={tr_mean.shape} var={tr_var.shape} "
          f"attn={tr_attn.shape} truncated={int(tr_trunc.sum())}")

    print("  Embedding test sequences (CASF-2016)...")
    te_mean, te_var, te_attn, te_trunc = embed_batch(
        casf_df['seq'].tolist(), tokenizer, esm_model,
        config.ESM_LAYERS, config.MAX_SEQ_LEN, config.HALF_SEQ_LEN,
        batch_size=8, device=device
    )
    print(f"  Test  ESM: mean={te_mean.shape} var={te_var.shape}")

    del esm_model
    if device == 'cuda':
        torch.cuda.empty_cache()

    # ── Sequence features ─────────────────────────────────────────────
    print("\n[Protein — Sequence Features]")
    from tqdm import tqdm
    tr_seqfeat = np.array([sequence_features(s)
                            for s in tqdm(train_df['seq'], desc="  Train seqfeat", ncols=70)])
    te_seqfeat = np.array([sequence_features(s)
                            for s in tqdm(casf_df['seq'], desc="  Test  seqfeat", ncols=70)])
    print(f"  Shape: {tr_seqfeat.shape}  "
          f"(28 ProtParam + 400 dipeptide + 63 CTD + 343 ConjTriad + 60 QSO + 25 AAIndex)")

    # ── Ligand features ───────────────────────────────────────────────
    print("\n[Ligand — Full Panel]")
    tr_lig, tr_valid, scaler = extract_ligand_features(
        train_df['smiles'].tolist(), fit_scaler=True
    )
    te_lig, te_valid, _ = extract_ligand_features(
        casf_df['smiles'].tolist(), scaler=scaler, fit_scaler=False
    )

    # ── Align after dropped SMILES ────────────────────────────────────
    tr_mean    = tr_mean[tr_valid];    te_mean    = te_mean[te_valid]
    tr_var     = tr_var[tr_valid];     te_var     = te_var[te_valid]
    tr_attn    = tr_attn[tr_valid];    te_attn    = te_attn[te_valid]
    tr_seqfeat = tr_seqfeat[tr_valid]; te_seqfeat = te_seqfeat[te_valid]
    tr_trunc   = tr_trunc[tr_valid];   te_trunc   = te_trunc[te_valid]
    y_train    = train_df['label'].values[tr_valid]
    y_test     = casf_df['label'].values[te_valid]
    tr_ids     = train_df['pdb_id'].values[tr_valid]
    te_ids     = casf_df['pdb_id'].values[te_valid]

    # ESM combined for interaction block = mean + attn (no var to keep it clean)
    tr_esm_comb = np.concatenate([tr_mean, tr_attn], axis=1)
    te_esm_comb = np.concatenate([te_mean, te_attn], axis=1)

    # ── Interaction block ─────────────────────────────────────────────
    print("\n[Interaction Block]")
    tr_lig_fp = np.concatenate([
        tr_lig['ecfp'], tr_lig['maccs'], tr_lig['atom_pair'], tr_lig['torsion']
    ], axis=1)
    te_lig_fp = np.concatenate([
        te_lig['ecfp'], te_lig['maccs'], te_lig['atom_pair'], te_lig['torsion']
    ], axis=1)
    tr_interact, prot_pca, lig_pca = build_interaction_features(
        tr_esm_comb, tr_lig_fp, dim=config.INTERACT_DIM, fit=True)
    te_interact, _, _ = build_interaction_features(
        te_esm_comb, te_lig_fp, dim=config.INTERACT_DIM,
        prot_pca=prot_pca, lig_pca=lig_pca, fit=False)
    print(f"  Interaction: {tr_interact.shape}")

    # ── Save preprocessors ────────────────────────────────────────────
    preproc_dir = config.OUTPUT_DIR / "preprocessors"
    preproc_dir.mkdir(exist_ok=True)
    save_pcas(prot_pca, lig_pca, preproc_dir)
    joblib.dump(scaler, preproc_dir / "ligand_scaler.pkl")

    # ── Save NPZ ──────────────────────────────────────────────────────
    print(f"\n[Saving NPZ]")

    common_train = dict(
        prot_esm_mean = tr_mean,
        prot_esm_var  = tr_var,
        prot_esm_attn = tr_attn,
        prot_seqfeat  = tr_seqfeat,
        lig_ecfp2     = tr_lig['ecfp2'],
        lig_ecfp      = tr_lig['ecfp'],
        lig_ecfp6     = tr_lig['ecfp6'],
        lig_fcfp      = tr_lig['fcfp'],
        lig_maccs     = tr_lig['maccs'],
        lig_ap        = tr_lig['atom_pair'],
        lig_torsion   = tr_lig['torsion'],
        lig_avalon    = tr_lig['avalon'],
        lig_rdkit_pat = tr_lig['rdkit_pat'],
        lig_ecfp_cnt  = tr_lig['ecfp_count'],
        lig_ecfp6_cnt = tr_lig['ecfp6_count'],
        lig_estate    = tr_lig['estate'],
        lig_phys      = tr_lig['phys'],
        interaction   = tr_interact,
        truncated     = tr_trunc,
        labels        = y_train,
        pdb_ids       = tr_ids,
    )
    common_test = dict(
        prot_esm_mean = te_mean,
        prot_esm_var  = te_var,
        prot_esm_attn = te_attn,
        prot_seqfeat  = te_seqfeat,
        lig_ecfp2     = te_lig['ecfp2'],
        lig_ecfp      = te_lig['ecfp'],
        lig_ecfp6     = te_lig['ecfp6'],
        lig_fcfp      = te_lig['fcfp'],
        lig_maccs     = te_lig['maccs'],
        lig_ap        = te_lig['atom_pair'],
        lig_torsion   = te_lig['torsion'],
        lig_avalon    = te_lig['avalon'],
        lig_rdkit_pat = te_lig['rdkit_pat'],
        lig_ecfp_cnt  = te_lig['ecfp_count'],
        lig_ecfp6_cnt = te_lig['ecfp6_count'],
        lig_estate    = te_lig['estate'],
        lig_phys      = te_lig['phys'],
        interaction   = te_interact,
        truncated     = te_trunc,
        labels        = y_test,
        pdb_ids       = te_ids,
    )

    np.savez_compressed(config.DATA_DIR / "X_train.npz", **common_train)
    np.savez_compressed(config.DATA_DIR / "X_test.npz",  **common_test)
    print(f"  Saved X_train.npz and X_test.npz")

    # ── Dimension summary ─────────────────────────────────────────────
    prot_dim  = (tr_mean.shape[1] + tr_var.shape[1] +
                 tr_attn.shape[1] + tr_seqfeat.shape[1])
    lig_dim   = sum(v.shape[1] for k, v in common_train.items()
                    if k.startswith('lig_'))
    total_dim = prot_dim + lig_dim + tr_interact.shape[1]

    print(f"\n[Feature Dimensions — v4]")
    print(f"  PROTEIN ({prot_dim}d):")
    print(f"    ESM mean pool:   {tr_mean.shape[1]}d  ({len(config.ESM_LAYERS)} layers x {config.ESM_DIM}d)")
    print(f"    ESM var pool:    {tr_var.shape[1]}d  (NEW — heterogeneity signal)")
    print(f"    ESM attn pool:   {tr_attn.shape[1]}d")
    print(f"    SeqFeat:         {tr_seqfeat.shape[1]}d  (ProtParam+Dipeptide+CTD+ConjTriad+QSO+AAIndex)")
    print(f"  LIGAND ({lig_dim}d):")
    for k in ['lig_ecfp2','lig_ecfp','lig_ecfp6','lig_fcfp','lig_maccs',
              'lig_ap','lig_torsion','lig_avalon','lig_rdkit_pat',
              'lig_ecfp_cnt','lig_ecfp6_cnt','lig_estate','lig_phys']:
        arr = common_train[k]
        tag = " NEW" if k in ('lig_avalon','lig_rdkit_pat','lig_ecfp_cnt','lig_ecfp6_cnt') else ""
        print(f"    {k:<16} {arr.shape[1]:>5}d{tag}")
    print(f"  INTERACTION:     {tr_interact.shape[1]}d")
    print(f"  {'─'*40}")
    print(f"  TOTAL:           {total_dim}d")
    print(f"\n  Train: {len(y_train)} x {total_dim}")
    print(f"  Test:  {len(y_test)} x {total_dim}")
    print(f"\n✓ Done. Run 03_train.py next.")


if __name__ == "__main__":
    main()

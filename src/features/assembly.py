#   Shared feature assembly — single source of truth.
#
#   Protein — ESM
#   prot_esm_mean  [1440d]  3-layer mean pool (layers 8,10,11 of ESM-35M)
#   prot_esm_attn  [ 480d]  attention-weighted pool (last layer)
#   prot_esm_var   [1440d]  3-layer variance pool  (heterogeneity signal)
#
#   Protein — sequence descriptors
#   prot_seqfeat   [ 919d]  ProtParam+Dipep+CTD+ConjTriad+QSO+AAIndex
#
#   Ligand — binary fingerprints
#   lig_ecfp       [1024d]  Morgan r=2   (always on, unconditional baseline)
#   lig_ecfp2      [1024d]  Morgan r=1
#   lig_ecfp6      [1024d]  Morgan r=3
#   lig_fcfp       [1024d]  Functional-class Morgan r=2
#   lig_maccs      [ 167d]  MACCS keys
#   lig_ap         [2048d]  AtomPair binary
#   lig_torsion    [2048d]  TopologicalTorsion binary
#   lig_avalon     [ 512d]  Avalon (different path-based algorithm)
#   lig_rdkit_pat  [2048d]  RDKit layered (ring+aromaticity+bond-order)
#
#   Ligand — count fingerprints
#   lig_ecfp_cnt   [1024d]  Morgan r=2 counts, log1p
#   lig_ecfp6_cnt  [1024d]  Morgan r=3 counts, log1p
#
#   Ligand — dense continuous
#   lig_estate     [  79d]  EState sum indices
#   lig_phys       [ 217d]  RDKit physicochemical descriptors
#
# Total: 1440+480+1440+919+1024+1024+1024+1024+167+2048+2048+512+2048
#        +1024+1024+79+217 = 17,542d

import json
import numpy as np

ESM_DIM = 480  # hidden dim of ESM-35M last layer


def assemble(data: dict) -> np.ndarray:
    """
    Build the v5 feature matrix from a dict/npz of named arrays.

    Args:
        data: dict-like with keys matching the NPZ keys written by
              scripts/02_extract_features.py.  Accepts both numpy npz
              objects and plain dicts.

    Returns:
        X: float32 array of shape [N, 18542]
    """
    parts = [
        data['prot_esm_mean'],    # 1440d — 3-layer mean pool
        data['prot_esm_attn'],    #  480d — attention-weighted pool
        data['prot_esm_var'],     # 1440d — 3-layer variance pool
        data['prot_seqfeat'],     #  919d — classical sequence descriptors

        data['lig_ecfp'],         # 1024d — ECFP4 binary (baseline; always on)
        data['lig_ecfp2'],        # 1024d — ECFP2 binary
        data['lig_ecfp6'],        # 1024d — ECFP6 binary
        data['lig_fcfp'],         # 1024d — FCFP binary
        data['lig_maccs'],        #  167d — MACCS keys
        data['lig_ap'],           # 2048d — AtomPair binary
        data['lig_torsion'],      # 2048d — Torsion binary
        data['lig_avalon'],       #  512d — Avalon
        data['lig_rdkit_pat'],    # 2048d — RDKit pattern/layered

        data['lig_ecfp_cnt'],     # 1024d — ECFP4 counts (log1p)
        data['lig_ecfp6_cnt'],    # 1024d — ECFP6 counts (log1p)

        data['lig_estate'],       #   79d — EState indices
        data['lig_phys'],         #  217d — RDKit physicochemical
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


EXPECTED_DIM = 17_542   # sanity-check against this after assemble()


def assemble_flagged(data, **kwargs) -> np.ndarray:
    """
    Assemble features using only the flags in kwargs (from best_cfg.json).
    ECFP4 is always included. All other features toggled by use_* flags.
    Use this everywhere models need to be applied — keeps dims consistent.
    """
    parts = []

    if kwargs.get('use_multilayer', True):
        parts.append(data['prot_esm_mean'])
    else:
        parts.append(data['prot_esm_mean'][:, -ESM_DIM:])

    if kwargs.get('use_attn', True):
        parts.append(data['prot_esm_attn'])
    if kwargs.get('use_esm_var', True):
        parts.append(data['prot_esm_var'])
    if kwargs.get('use_seqfeat', True):
        parts.append(data['prot_seqfeat'])

    parts.append(data['lig_ecfp'])  # always

    if kwargs.get('use_ecfp2', True):   parts.append(data['lig_ecfp2'])
    if kwargs.get('use_ecfp6', True):   parts.append(data['lig_ecfp6'])
    if kwargs.get('use_fcfp', True):    parts.append(data['lig_fcfp'])
    if kwargs.get('use_maccs', True):   parts.append(data['lig_maccs'])
    if kwargs.get('use_ap', True):      parts.append(data['lig_ap'])
    if kwargs.get('use_torsion', True): parts.append(data['lig_torsion'])
    if kwargs.get('use_avalon', True):  parts.append(data['lig_avalon'])
    if kwargs.get('use_rdkit_pat', True): parts.append(data['lig_rdkit_pat'])

    if kwargs.get('use_ecfp_count', True):
        parts.append(data['lig_ecfp_cnt'])
        parts.append(data['lig_ecfp6_cnt'])

    if kwargs.get('use_estate', True):  parts.append(data['lig_estate'])
    if kwargs.get('use_rdkit', True):   parts.append(data['lig_phys'])

    return np.concatenate(parts, axis=1).astype(np.float32)


def load_winner_kwargs(cfg_path) -> dict:
    """Load the ablation winner kwargs from best_cfg.json."""
    with open(cfg_path) as f:
        return json.load(f)['kwargs']

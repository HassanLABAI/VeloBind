# src/features/ligand.py
#
# Four orthogonal fingerprint types:
#   ECFP4     1024d — radial atom neighbourhood (local topology)
#   MACCS      167d — SMARTS pharmacophore patterns
#   AtomPair  2048d — all atom pairs + graph distance (global topology)
#   Torsion   2048d — 4-atom linear paths (rotatable bond context)
#   RDKit      ~217d — bulk physicochemical descriptors
#
# ECFP vs AtomPair vs Torsion are genuinely orthogonal:
#   ECFP:      "what atoms are within radius 2 of atom X"  (local)
#   AtomPair:  "which atoms are N bonds apart globally"    (global)
#   Torsion:   "what 4-atom paths exist"                   (conformational)
# Inference cost: ~0.002 sec total per SMILES.

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter as EStateFP
from rdkit import RDLogger
from sklearn.preprocessing import RobustScaler

RDLogger.DisableLog('rdApp.*')
_DESC_LIST = Descriptors._descList


def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    ecfp = np.zeros(1024, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, ecfp)

    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=1024)
    ecfp2 = np.zeros(1024, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp2, ecfp2)

    fp6 = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
    ecfp6 = np.zeros(1024, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp6, ecfp6)

    fcfp_fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, 2, nBits=1024, useFeatures=True)
    fcfp = np.zeros(1024, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fcfp_fp, fcfp)

    try:
        _, sum_e = EStateFP.FingerprintMol(mol)   # 79d dense continuous
        estate = np.array(sum_e, dtype=np.float64)
        estate = np.nan_to_num(estate, nan=0.0, posinf=0.0, neginf=0.0)
        estate = np.clip(estate, -1e6, 1e6).astype(np.float32)
    except Exception:
        estate = np.zeros(79, dtype=np.float32)

    mk = MACCSkeys.GenMACCSKeys(mol)
    maccs = np.zeros(167, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(mk, maccs)

    ap_fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    atom_pair = np.zeros(2048, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(ap_fp, atom_pair)

    tt_fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
    torsion = np.zeros(2048, dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(tt_fp, torsion)

    phys = []
    for _, func in _DESC_LIST:
        try:
            v = func(mol)
            v = float(v)
            phys.append(v if (np.isfinite(v) and abs(v) < 1e15) else 0.0)
        except Exception:
            phys.append(0.0)

    return {
        'ecfp':      ecfp,
        'ecfp2':     ecfp2,
        'ecfp6':     ecfp6,
        'fcfp':      fcfp,
        'estate':    estate,
        'maccs':     maccs,
        'atom_pair': atom_pair,
        'torsion':   torsion,
        'phys':      np.array(phys, dtype=np.float32),
    }


def extract_ligand_features(smiles_list, scaler=None, fit_scaler=False):
    ecfps, ecfp2s, ecfp6s, fcfps, estates, maccss, aps, tors, physs, valid_idx = \
        [], [], [], [], [], [], [], [], [], []

    for i, smi in enumerate(smiles_list):
        r = smiles_to_features(smi)
        if r is None:
            continue
        ecfps.append(r['ecfp'])
        ecfp2s.append(r['ecfp2'])
        ecfp6s.append(r['ecfp6'])
        fcfps.append(r['fcfp'])
        estates.append(r['estate'])
        maccss.append(r['maccs'])
        aps.append(r['atom_pair'])
        tors.append(r['torsion'])
        physs.append(r['phys'])
        valid_idx.append(i)

    n_fail = len(smiles_list) - len(valid_idx)
    if n_fail:
        print(f"  Ligand: {n_fail} SMILES failed → dropped")

    phys_arr = np.array(physs, dtype=np.float64)
    phys_arr = np.nan_to_num(phys_arr, nan=0.0, posinf=0.0, neginf=0.0)
    phys_arr = phys_arr.astype(np.float32)

    estate_arr = np.array(estates, dtype=np.float32)

    if fit_scaler:
        scaler = RobustScaler()
        scaler.fit(np.concatenate([phys_arr, estate_arr], axis=1))

    # Scale phys + estate together (both continuous)
    combined_scaled = scaler.transform(
        np.concatenate([phys_arr, estate_arr], axis=1))
    phys_scaled   = combined_scaled[:, :phys_arr.shape[1]]
    estate_scaled = combined_scaled[:, phys_arr.shape[1]:]

    feats = {
        'ecfp':      np.array(ecfps,  dtype=np.float32),
        'ecfp2':     np.array(ecfp2s, dtype=np.float32),
        'ecfp6':     np.array(ecfp6s, dtype=np.float32),
        'fcfp':      np.array(fcfps,  dtype=np.float32),
        'estate':    estate_scaled,
        'maccs':     np.array(maccss, dtype=np.float32),
        'atom_pair': np.array(aps,    dtype=np.float32),
        'torsion':   np.array(tors,   dtype=np.float32),
        'phys':      phys_scaled,
    }

    print(f"  Ligand: ECFP2/4/6 {feats['ecfp'].shape[1]}d×3 | "
          f"FCFP {feats['fcfp'].shape[1]}d | "
          f"E-state {feats['estate'].shape[1]}d | "
          f"MACCS {feats['maccs'].shape[1]}d | "
          f"AtomPair {feats['atom_pair'].shape[1]}d | "
          f"Torsion {feats['torsion'].shape[1]}d | "
          f"RDKit {feats['phys'].shape[1]}d")

    return feats, valid_idx, scaler

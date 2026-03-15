# src/features/ligand.py
#
# Ligand feature extraction — pure RDKit, zero ML models at inference.
# All operations: O(N_atoms) or O(N_atoms²) at worst → microseconds/mol.
#
# Feature blocks:
#
#   BINARY FINGERPRINTS (presence/absence of substructure)
#   ─────────────────────────────────────────────────────
#   ecfp2       1024d   Morgan r=1  — ultra-local atom neighbourhoods
#   ecfp        1024d   Morgan r=2  — standard local topology (ECFP4)
#   ecfp6       1024d   Morgan r=3  — extended neighbourhoods
#   fcfp        1024d   Functional class r=2 — pharmacophoric identity
#   maccs        167d   166 SMARTS pharmacophore keys
#   atom_pair   2048d   All-pairs graph distance (global topology)
#   torsion     2048d   4-atom rotatable bond paths (conformational)
#   avalon       512d   Avalon — completely different algorithm (Scitegic)
#   rdkit_pat   2048d   RDKit layered — ring + aromaticity + bond order
#
#   COUNT FINGERPRINTS (how many times each substructure appears)
#   ─────────────────────────────────────────────────────────────
#   ecfp_count  1024d   Morgan r=2 counts — 3 benzenes != 1 benzene
#   ecfp6_count 1024d   Morgan r=3 counts
#
#   DENSE CONTINUOUS
#   ────────────────
#   estate        79d   EState sum indices — electrotopological signal
#   phys         217d   RDKit full descriptor suite (RobustScaler normalised)
#
# Inference timing (HF Spaces free tier, 2 vCPU):
#   Per SMILES: ~3-5 ms total (all fingerprints + descriptors)
#   1M compounds: ~50-80 min on single CPU core
#   No GPU, no transformer, no external calls.

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors
from rdkit.Chem.EState import Fingerprinter as EStateFP
from rdkit import RDLogger
from sklearn.preprocessing import RobustScaler

RDLogger.DisableLog('rdApp.*')
_DESC_LIST = Descriptors._descList

try:
    from rdkit.Avalon.pyAvalonTools import GetAvalonFP as _GetAvalonFP
    _AVALON_OK = True
except ImportError:
    _AVALON_OK = False
    print("  WARNING: rdkit.Avalon not available — avalon features will be zeros. "
          "Reinstall RDKit with Avalon support if needed.")


def smiles_to_features(smiles: str):
    """
    Convert a SMILES string to the full ligand feature dict.
    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # ── Binary Morgan fingerprints ─────────────────────────────────────
    def _bin(radius, nbits=1024):
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    ecfp2 = _bin(1)
    ecfp  = _bin(2)    # ECFP4
    ecfp6 = _bin(3)

    fp_fcfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024, useFeatures=True)
    fcfp    = np.zeros(1024, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_fcfp, fcfp)

    # ── Morgan COUNT fingerprints ──────────────────────────────────────
    # Counts how many times each substructure hashes to each bit.
    # A drug with 3 chloro-phenyl groups looks different from one with 1.
    # Orthogonal to the binary versions above.
    def _cnt(radius, nbits=1024):
        fp  = AllChem.GetHashedMorganFingerprint(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    ecfp_count  = _cnt(2)
    ecfp6_count = _cnt(3)

    # ── Avalon fingerprint (512d) ──────────────────────────────────────
    # Completely different algorithm from Morgan family.
    # Graph-invariant path enumeration — catches heteroaromatic scaffold
    # patterns Morgan misses.
    if _AVALON_OK:
        try:
            fp_av  = _GetAvalonFP(mol, nBits=512)
            avalon = np.zeros(512, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp_av, avalon)
        except Exception:
            avalon = np.zeros(512, dtype=np.float32)
    else:
        avalon = np.zeros(512, dtype=np.float32)

    # ── RDKit Pattern (Layered) fingerprint (2048d) ────────────────────
    # Encodes atom connectivity WITH ring membership, aromaticity, bond
    # order layered in. Catches fused aromatic systems (indoles, purines,
    # quinolines) that ECFP treats as overlapping local neighbourhoods.
    try:
        fp_pat    = Chem.RDKFingerprint(mol, fpSize=2048)
        rdkit_pat = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp_pat, rdkit_pat)
    except Exception:
        rdkit_pat = np.zeros(2048, dtype=np.float32)

    # ── MACCS keys (167d) ─────────────────────────────────────────────
    mk   = MACCSkeys.GenMACCSKeys(mol)
    maccs = np.zeros(167, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(mk, maccs)

    # ── AtomPair binary (2048d) ────────────────────────────────────────
    fp_ap     = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=2048)
    atom_pair = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_ap, atom_pair)

    # ── Topological Torsion binary (2048d) ────────────────────────────
    fp_tt   = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=2048)
    torsion = np.zeros(2048, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp_tt, torsion)

    # ── EState sum indices (79d dense continuous) ──────────────────────
    try:
        _, sum_e = EStateFP.FingerprintMol(mol)
        estate   = np.array(sum_e, dtype=np.float64)
        estate   = np.nan_to_num(estate, nan=0.0, posinf=0.0, neginf=0.0)
        estate   = np.clip(estate, -1e6, 1e6).astype(np.float32)
    except Exception:
        estate = np.zeros(79, dtype=np.float32)

    # ── RDKit physicochemical descriptors (~217d) ──────────────────────
    phys = []
    for _, func in _DESC_LIST:
        try:
            v = float(func(mol))
            phys.append(v if (np.isfinite(v) and abs(v) < 1e15) else 0.0)
        except Exception:
            phys.append(0.0)

    return {
        'ecfp2':       ecfp2,
        'ecfp':        ecfp,
        'ecfp6':       ecfp6,
        'fcfp':        fcfp,
        'maccs':       maccs,
        'atom_pair':   atom_pair,
        'torsion':     torsion,
        'avalon':      avalon,
        'rdkit_pat':   rdkit_pat,
        'ecfp_count':  ecfp_count,
        'ecfp6_count': ecfp6_count,
        'estate':      estate,
        'phys':        np.array(phys, dtype=np.float32),
    }


def extract_ligand_features(smiles_list: list, scaler=None, fit_scaler: bool = False):
    """
    Extract ligand features for a list of SMILES strings.

    Args:
        smiles_list: list of SMILES strings
        scaler:      fitted RobustScaler (required if fit_scaler=False)
        fit_scaler:  if True, fit a new scaler on the continuous features

    Returns:
        feats:     dict of numpy arrays, one per feature type
        valid_idx: indices of successfully parsed SMILES
        scaler:    fitted RobustScaler

    Note: Binary + count fingerprints are NOT scaled.
          GBMs are invariant to monotone transforms on binary features.
          Count fingerprints are log1p-transformed for numerical stability.
    """
    ecfp2s, ecfps, ecfp6s, fcfps = [], [], [], []
    maccss, aps, tors             = [], [], []
    avalons, rdkit_pats           = [], []
    ecfp_counts, ecfp6_counts     = [], []
    estates, physs                = [], []
    valid_idx                     = []

    for i, smi in enumerate(smiles_list):
        r = smiles_to_features(smi)
        if r is None:
            continue
        ecfp2s.append(r['ecfp2'])
        ecfps.append(r['ecfp'])
        ecfp6s.append(r['ecfp6'])
        fcfps.append(r['fcfp'])
        maccss.append(r['maccs'])
        aps.append(r['atom_pair'])
        tors.append(r['torsion'])
        avalons.append(r['avalon'])
        rdkit_pats.append(r['rdkit_pat'])
        ecfp_counts.append(r['ecfp_count'])
        ecfp6_counts.append(r['ecfp6_count'])
        estates.append(r['estate'])
        physs.append(r['phys'])
        valid_idx.append(i)

    n_fail = len(smiles_list) - len(valid_idx)
    if n_fail:
        print(f"  Ligand: {n_fail} SMILES failed to parse — dropped")

    # Continuous: clean then scale together
    phys_arr   = np.nan_to_num(
        np.array(physs, dtype=np.float64),
        nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)
    estate_arr = np.array(estates, dtype=np.float32)

    continuous = np.concatenate([phys_arr, estate_arr], axis=1)
    if fit_scaler:
        scaler = RobustScaler()
        scaler.fit(continuous)
    continuous_scaled = scaler.transform(continuous)
    phys_scaled   = continuous_scaled[:, :phys_arr.shape[1]]
    estate_scaled = continuous_scaled[:, phys_arr.shape[1]:]

    # Count FPs: log1p stabilises large int values without losing magnitude info
    ecfp_cnt_arr  = np.log1p(np.array(ecfp_counts,  dtype=np.float32))
    ecfp6_cnt_arr = np.log1p(np.array(ecfp6_counts, dtype=np.float32))

    feats = {
        'ecfp2':       np.array(ecfp2s,     dtype=np.float32),
        'ecfp':        np.array(ecfps,      dtype=np.float32),
        'ecfp6':       np.array(ecfp6s,     dtype=np.float32),
        'fcfp':        np.array(fcfps,      dtype=np.float32),
        'maccs':       np.array(maccss,     dtype=np.float32),
        'atom_pair':   np.array(aps,        dtype=np.float32),
        'torsion':     np.array(tors,       dtype=np.float32),
        'avalon':      np.array(avalons,    dtype=np.float32),
        'rdkit_pat':   np.array(rdkit_pats, dtype=np.float32),
        'ecfp_count':  ecfp_cnt_arr,
        'ecfp6_count': ecfp6_cnt_arr,
        'estate':      estate_scaled,
        'phys':        phys_scaled,
    }

    total_dim = sum(v.shape[1] for v in feats.values())
    print(f"  Ligand: {len(valid_idx)} molecules | {total_dim}d total")
    print(f"    Binary:  ecfp2={feats['ecfp2'].shape[1]} ecfp={feats['ecfp'].shape[1]} "
          f"ecfp6={feats['ecfp6'].shape[1]} fcfp={feats['fcfp'].shape[1]} "
          f"maccs={feats['maccs'].shape[1]} ap={feats['atom_pair'].shape[1]} "
          f"tors={feats['torsion'].shape[1]} avalon={feats['avalon'].shape[1]} "
          f"rdkit_pat={feats['rdkit_pat'].shape[1]}")
    print(f"    Counts:  ecfp_cnt={feats['ecfp_count'].shape[1]} "
          f"ecfp6_cnt={feats['ecfp6_count'].shape[1]}")
    print(f"    Dense:   estate={feats['estate'].shape[1]} "
          f"phys={feats['phys'].shape[1]}")

    return feats, valid_idx, scaler

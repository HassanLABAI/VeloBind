# src/data/loader.py
#
# Loads LP-PDBBind and CASF-2016 into clean DataFrames.
# Output columns: pdb_id, seq, smiles, label

import pandas as pd
from pathlib import Path


def load_lppdb(csv_path: Path,
               exclude_ids: set = None) -> pd.DataFrame:
    """
    Load LP-PDBBind flat CSV.

    Relevant columns:
      pdb_id  — PDB identifier
      seq     — protein sequence
      smiles  — ligand SMILES
      value   — pAffinity (already normalized from Kd/Ki/IC50)

    Args:
        csv_path:    path to LP_PDBBind.csv
        exclude_ids: set of lowercase PDB IDs to remove before training
                     (pass your CASF IDs here to prevent leakage)

    Drops rows with missing seq, smiles, or label.
    Strips whitespace from sequences and SMILES.
    """
    df = pd.read_csv(csv_path)

    df = df[['pdb_id', 'seq', 'smiles', 'value']].copy()
    df.columns = ['pdb_id', 'seq', 'smiles', 'label']

    before = len(df)
    df = df.dropna(subset=['seq', 'smiles', 'label'])
    df['seq']    = df['seq'].str.strip().str.upper()
    df['smiles'] = df['smiles'].str.strip()
    df['pdb_id'] = df['pdb_id'].str.lower().str.strip()
    df = df[df['seq'].str.len() > 0]
    df = df[df['smiles'].str.len() > 0]

    after_clean = len(df)

    # Remove CASF complexes to prevent data leakage
    if exclude_ids:
        before_excl = len(df)
        df = df[~df['pdb_id'].isin(exclude_ids)]
        n_removed = before_excl - len(df)
        print(f"  Removed {n_removed} CASF complexes from training (leakage prevention)")

    df = df.reset_index(drop=True)
    print(f"LP-PDBBind: {before} → {after_clean} (after cleaning) "
          f"→ {len(df)} (after CASF removal)")
    return df


def load_casf(casf_dir: Path) -> pd.DataFrame:
    """
    Load CASF-2016 CoreSet.

    Reads CoreSet.dat for pdb_ids and labels.
    Reads protein sequences from <pdb_id>/<pdb_id>_protein.pdb SEQRES records.
    Reads ligand SMILES from <pdb_id>/<pdb_id>_ligand.mol2 via RDKit.

    Returns DataFrame with same columns as load_lppdb.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    coreset_dat = casf_dir / "power_scoring" / "CoreSet.dat"
    coreset_dir = casf_dir / "coreset"

    # Parse CoreSet.dat — tab/space separated, first col = pdb_id, last = -logKd
    records = []
    with open(coreset_dat) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            pdb_id = parts[0].lower()
            label  = float(parts[-3])
            records.append({'pdb_id': pdb_id, 'label': label})

    dat_df = pd.DataFrame(records)
    print(f"CASF CoreSet.dat: {len(dat_df)} entries")

    rows    = []
    dropped = []

    for _, row in dat_df.iterrows():
        pid    = row['pdb_id']
        label  = row['label']
        folder = coreset_dir / pid

        # Protein sequence from SEQRES
        seq = _parse_seqres(folder / f"{pid}_protein.pdb")

        # Ligand SMILES — try mol2 first, then sdf
        smiles = _parse_ligand_smiles(folder, pid)

        if seq is None or smiles is None:
            dropped.append((pid, "seq missing" if seq is None else "smiles missing"))
            continue

        rows.append({'pdb_id': pid, 'seq': seq, 'smiles': smiles, 'label': label})

    df = pd.DataFrame(rows)
    print(f"CASF parsed: {len(df)} complexes  |  dropped: {len(dropped)}")
    for pid, reason in dropped:
        print(f"  dropped {pid}: {reason}")

    return df, dropped


def load_casf2013(casf13_dir: Path) -> pd.DataFrame:
    """
    Load CASF-2013 CoreSet.  Identical structure to CASF-2016:
      power_scoring/CoreSet.dat  — labels
      coreset/<pid>/             — PDB + mol2/sdf files
    Returns same (df, dropped) as load_casf.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    coreset_dat = casf13_dir / "power_scoring" / "CoreSet.dat"
    coreset_dir = casf13_dir / "coreset"

    records = []
    with open(coreset_dat) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts  = line.split()
            pdb_id = parts[0].lower()
            label  = float(parts[-3])
            records.append({'pdb_id': pdb_id, 'label': label})

    dat_df = pd.DataFrame(records)
    print(f"CASF-2013 CoreSet.dat: {len(dat_df)} entries")

    rows, dropped = [], []
    for _, row in dat_df.iterrows():
        pid    = row['pdb_id']
        label  = row['label']
        folder = coreset_dir / pid

        seq    = _parse_seqres(folder / f"{pid}_protein.pdb")
        smiles = _parse_ligand_smiles(folder, pid)

        if seq is None or smiles is None:
            dropped.append((pid, "seq missing" if seq is None else "smiles missing"))
            continue

        rows.append({'pdb_id': pid, 'seq': seq, 'smiles': smiles, 'label': label})

    df = pd.DataFrame(rows)
    print(f"CASF-2013 parsed: {len(df)} complexes  |  dropped: {len(dropped)}")
    for pid, reason in dropped:
        print(f"  dropped {pid}: {reason}")

    return df, dropped


# ── Private helpers ───────────────────────────────────────────────────

_AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C',
    'GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P',
    'SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    # common non-standard → closest standard
    'MSE':'M','SEP':'S','TPO':'T','PTR':'Y','HYP':'P',
}


def _parse_seqres(pdb_path: Path) -> str | None:
    if not pdb_path.exists():
        return None

    # Try SEQRES records first (canonical, includes all residues)
    seq_by_chain = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('SEQRES'):
                chain = line[11]
                residues = line[19:].split()
                seq_by_chain.setdefault(chain, []).extend(residues)

    if seq_by_chain:
        chain    = max(seq_by_chain, key=lambda c: len(seq_by_chain[c]))
        residues = seq_by_chain[chain]
        seq      = ''.join(_AA3TO1.get(r, 'X') for r in residues)
        seq      = seq.replace('X', '')
        if seq:
            return seq

    # Fallback: parse ATOM records (some PDB files lack SEQRES)
    # Collects unique residues in order of appearance
    atom_by_chain = {}
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            chain   = line[21]
            res_name = line[17:20].strip()
            res_seq  = line[22:26].strip()   # residue sequence number
            atom_by_chain.setdefault(chain, {})[res_seq] = res_name

    if not atom_by_chain:
        return None

    chain    = max(atom_by_chain, key=lambda c: len(atom_by_chain[c]))
    residues = [atom_by_chain[chain][k]
                for k in sorted(atom_by_chain[chain],
                                key=lambda x: int(x) if x.lstrip('-').isdigit() else 0)]
    seq = ''.join(_AA3TO1.get(r, 'X') for r in residues)
    seq = seq.replace('X', '')
    return seq if seq else None


def _parse_ligand_smiles(folder: Path, pid: str) -> str | None:
    from rdkit import Chem

    # Try mol2
    mol2_path = folder / f"{pid}_ligand.mol2"
    if mol2_path.exists():
        mol = Chem.MolFromMol2File(str(mol2_path), removeHs=True)
        if mol:
            return Chem.MolToSmiles(mol)

    # Try sdf
    sdf_path = folder / f"{pid}_ligand.sdf"
    if sdf_path.exists():
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
        for mol in suppl:
            if mol:
                return Chem.MolToSmiles(mol)

    return None

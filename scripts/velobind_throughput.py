import sys
import time
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import config
from src.features.ligand import extract_ligand_features
from src.features.assembly import assemble_flagged, load_winner_kwargs

_spec = importlib.util.spec_from_file_location("lit09", ROOT / "scripts" / "09_litpcba.py")
lit = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lit)

MODEL_DIR = config.OUTPUT_DIR / "models"
N = 5000


def main():
    import joblib
    models, meta, iso, scaler = lit.load_ensemble(MODEL_DIR, config.SEEDS, config.N_FOLDS)
    winner = load_winner_kwargs(MODEL_DIR / "best_cfg.json")
    lig_scaler = joblib.load(config.OUTPUT_DIR / "preprocessors" / "ligand_scaler.pkl")

    d = np.load(config.DATA_DIR / "X_train.npz", allow_pickle=True)
    ser = pd.read_csv(config.DATA_DIR / "train_clean.csv")["smiles"].dropna()
    smis = ser.sample(min(N, len(ser)), random_state=0).tolist()   # never over-sample
    prot = {k: d[k][0:1] for k in ("prot_esm_mean", "prot_esm_attn", "prot_esm_var", "prot_seqfeat")}

    t0 = time.time()
    lig, valid, _ = extract_ligand_features(smis, scaler=lig_scaler, fit_scaler=False)
    nval = len(valid)
    data = {k: np.tile(v, (nval, 1)) for k, v in prot.items()}
    data.update({
        'lig_ecfp': lig['ecfp'], 'lig_ecfp2': lig['ecfp2'], 'lig_ecfp6': lig['ecfp6'],
        'lig_fcfp': lig['fcfp'], 'lig_maccs': lig['maccs'], 'lig_ap': lig['atom_pair'],
        'lig_torsion': lig['torsion'], 'lig_avalon': lig['avalon'], 'lig_rdkit_pat': lig['rdkit_pat'],
        'lig_ecfp_cnt': lig['ecfp_count'], 'lig_ecfp6_cnt': lig['ecfp6_count'],
        'lig_estate': lig['estate'], 'lig_phys': lig['phys'],
    })
    X = assemble_flagged(data, **winner)
    scores, _ = lit.predict(X, models, meta, iso, scaler, config.SEEDS, config.N_FOLDS)
    dt = time.time() - t0

    rate = nval / dt
    import os
    ncpu = os.cpu_count() or 1
    lines = [
        "VeloBind throughput (per-compound screening path; protein cached)",
        f"  compounds scored : {nval:,}",
        f"  wall time        : {dt:.2f} s",
        f"  rate             : {rate:,.0f} compounds/s  ({rate/ncpu:,.0f} per core, {ncpu} cores)",
        f"  => 1,000,000 compounds in ~{1_000_000/rate/60:.1f} min on this machine",
    ]
    txt = "\n".join(lines)
    print(txt)
    (config.OUTPUT_DIR / "throughput.txt").write_text(txt + "\n")


if __name__ == "__main__":
    main()

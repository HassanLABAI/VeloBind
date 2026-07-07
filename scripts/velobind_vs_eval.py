import sys
import argparse
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # so `import velobind_expand` resolves
from src.config import config
from src.features.protein import load_esm, sequence_features
from src.features.ligand import extract_ligand_features
from src.features.assembly import assemble_flagged, load_winner_kwargs
import torch

_spec = importlib.util.spec_from_file_location("lit09", ROOT / "scripts" / "09_litpcba.py")
lit = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lit)
from velobind_expand import ecfp, max_sim_to_set, expand_score

MODEL_DIR = config.OUTPUT_DIR / "models"
OUT = config.OUTPUT_DIR / "vs_eval"
SUPPORT_FRAC, SUPPORT_MIN, SUPPORT_MAX = 0.2, 3, 30


def pdb_to_seq(pdb_path):
    """1-letter sequence from a PDB receptor (CA atoms, first chain occurrence)."""
    seq, seen = [], set()
    for line in open(pdb_path):
        if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == "CA":
            key = line[21] + line[22:27].strip()      # chain + resseq + icode
            if key in seen:
                continue
            seen.add(key)
            aa = lit.THREE_TO_ONE.get(line[17:20].strip())
            if aa:
                seq.append(aa)
    return "".join(seq)


def load_target(tdir, benchmark):
    """Return (sequence, actives_smiles, inactives_smiles) for one target folder."""
    if benchmark == "litpcba":
        seq = lit.seq_from_mol2(lit.get_representative_mol2(tdir))
        act = pd.read_csv(tdir / "actives.smi", sep=r"\s+", header=None).iloc[:, 0].tolist()
        ina = pd.read_csv(tdir / "inactives.smi", sep=r"\s+", header=None).iloc[:, 0].tolist()
    else:  # DUD-E layout
        rec = tdir / "receptor.pdb"
        seq = pdb_to_seq(rec)
        af = next((tdir / f"actives_final.{e}" for e in ("ism", "smi") if (tdir / f"actives_final.{e}").exists()), None)
        df = next((tdir / f"decoys_final.{e}" for e in ("ism", "smi") if (tdir / f"decoys_final.{e}").exists()), None)
        if af is None or df is None:
            raise FileNotFoundError("no actives_final/decoys_final (.ism/.smi)")
        act = [ln.split()[0] for ln in open(af) if ln.strip()]
        ina = [ln.split()[0] for ln in open(df) if ln.strip()]
    return seq, act, ina


def score_affinity(seq, smiles, tok, esm_model, device, lig_scaler, models, meta, iso, scaler, winner):
    """Affinity score for all `smiles` against one target sequence (reuses 09 path)."""
    enc = tok(seq, return_tensors='pt', truncation=True,
              max_length=config.MAX_SEQ_LEN + 2).to(device)
    with torch.no_grad():
        out = esm_model(**enc, output_hidden_states=True, output_attentions=True)
    mask = enc['attention_mask'].unsqueeze(-1).float()
    denom = mask.sum(1).clamp(min=1e-9)
    means, vars = [], []
    for li in config.ESM_LAYERS:
        h = out.hidden_states[li + 1]
        mv = (h * mask).sum(1) / denom
        means.append(mv.squeeze(0).cpu().numpy())
        vars.append((((h - mv.unsqueeze(1)) ** 2) * mask).sum(1).div(denom).squeeze(0).cpu().numpy())
    esm_mean = np.concatenate(means); esm_var = np.concatenate(vars)
    a = out.attentions[-1][0].mean(0).mean(0); a = a / a.sum().clamp(min=1e-9)
    esm_attn = (out.hidden_states[-1][0] * a.unsqueeze(-1)).sum(0).cpu().numpy()
    seqfeat = sequence_features(seq)
    prot = dict(prot_esm_mean=esm_mean, prot_esm_attn=esm_attn,
                prot_esm_var=esm_var, prot_seqfeat=seqfeat)

    CHUNK = 2000
    scores_all, valid_all = [], []
    for start in range(0, len(smiles), CHUNK):
        sub = smiles[start:start + CHUNK]
        lig, valid, _ = extract_ligand_features(sub, scaler=lig_scaler, fit_scaler=False)
        if len(valid) == 0:
            continue
        n = len(valid)
        data = {k: np.tile(v, (n, 1)) for k, v in prot.items()}
        data.update({'lig_ecfp': lig['ecfp'], 'lig_ecfp2': lig['ecfp2'], 'lig_ecfp6': lig['ecfp6'],
                     'lig_fcfp': lig['fcfp'], 'lig_maccs': lig['maccs'], 'lig_ap': lig['atom_pair'],
                     'lig_torsion': lig['torsion'], 'lig_avalon': lig['avalon'],
                     'lig_rdkit_pat': lig['rdkit_pat'], 'lig_ecfp_cnt': lig['ecfp_count'],
                     'lig_ecfp6_cnt': lig['ecfp6_count'], 'lig_estate': lig['estate'],
                     'lig_phys': lig['phys']})
        X = assemble_flagged(data, **winner)
        sc, _ = lit.predict(X, models, meta, iso, scaler, config.SEEDS, config.N_FOLDS)
        scores_all.append(np.asarray(sc))
        valid_all.extend(start + i for i in valid)
        del X, data, lig
    return np.concatenate(scores_all), np.array(valid_all, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["litpcba", "dude"])
    ap.add_argument("--data", required=True, help="Root dir of target folders")
    ap.add_argument("--max_inactives", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    import joblib
    models, meta, iso, scaler = lit.load_ensemble(MODEL_DIR, config.SEEDS, config.N_FOLDS)
    winner = load_winner_kwargs(MODEL_DIR / "best_cfg.json")
    lig_scaler = joblib.load(config.OUTPUT_DIR / "preprocessors" / "ligand_scaler.pkl")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok, esm_model = load_esm(config.ESM_MODEL, device); esm_model.eval()

    rows = []
    for tdir in sorted(p for p in Path(args.data).iterdir() if p.is_dir()):
        try:
            seq, act, ina = load_target(tdir, args.benchmark)
            if len(seq) < 30:
                continue
            if args.max_inactives and len(ina) > args.max_inactives:
                ina = list(pd.Series(ina).sample(args.max_inactives, random_state=args.seed))
            all_smi = act + ina
            labels = np.array([1] * len(act) + [0] * len(ina))
            aff, valid = score_affinity(seq, all_smi, tok, esm_model, device,
                                        lig_scaler, models, meta, iso, scaler, winner)
            labels = labels[valid]
            smi_v = [all_smi[i] for i in valid]
            n_act = int(labels.sum())
            if n_act < 5:
                continue

            act_idx = np.where(labels == 1)[0]
            n_sup = int(np.clip(int(SUPPORT_FRAC * n_act), SUPPORT_MIN, SUPPORT_MAX))
            n_sup = min(n_sup, n_act - 1)
            sup = rng.choice(act_idx, n_sup, replace=False)
            test_mask = np.ones(len(labels), bool); test_mask[sup] = False
            yt = labels[test_mask]; aff_t = aff[test_mask]
            q_fps = ecfp([smi_v[i] for i in np.where(test_mask)[0]])
            sup_fps = ecfp([smi_v[i] for i in sup])
            sim = max_sim_to_set(q_fps, sup_fps)

            tani = sim
            blend = expand_score(aff_t, sim, None, args.alpha)
            randr = rng.random(len(yt))

            def metr(scores):
                return (lit.enrichment_factor(yt, scores, 0.01),
                        lit.enrichment_factor(yt, scores, 0.05),
                        lit.bedroc(yt, scores, 20.0))
            ef1_a, ef5_a, bd_a = metr(aff_t)
            ef1_t, ef5_t, bd_t = metr(tani)
            ef1_b, ef5_b, bd_b = metr(blend)
            ef1_r, _, _ = metr(randr)
            rows.append(dict(target=tdir.name, n_act=n_act, n_support=n_sup, n_test=len(yt),
                             EF1_affinity=round(ef1_a, 3), EF1_tanimoto=round(ef1_t, 3),
                             EF1_expand=round(ef1_b, 3), EF1_random=round(ef1_r, 3),
                             BEDROC_affinity=round(bd_a, 4), BEDROC_tanimoto=round(bd_t, 4),
                             BEDROC_expand=round(bd_b, 4)))
            print(f"  {tdir.name:12s} EF1%: aff={ef1_a:.2f} tani={ef1_t:.2f} "
                  f"expand={ef1_b:.2f} (rand {ef1_r:.2f})")
        except Exception as e:
            print(f"  ERROR {tdir.name}: {e}")

    if not rows:
        print("No targets evaluated."); return
    df = pd.DataFrame(rows)
    out_csv = OUT / f"warmcold_{args.benchmark}.csv"
    df.to_csv(out_csv, index=False)
    m = df.mean(numeric_only=True)
    print("\n=== MEAN over targets ===")
    print(f"  EF1%  affinity={m['EF1_affinity']:.2f}  tanimoto={m['EF1_tanimoto']:.2f}  "
          f"EXPAND={m['EF1_expand']:.2f}  random={m['EF1_random']:.2f}")
    print(f"  The bar: EXPAND should be >= tanimoto. Honest expectation: competitive, modest.")
    print(f"  Saved -> {out_csv}")


if __name__ == "__main__":
    main()

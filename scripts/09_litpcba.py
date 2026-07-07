# LIT-PCBA retrospective virtual-screening enrichment benchmark.

import sys
import re
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import RidgeCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.features.protein import load_esm, sequence_features
from src.features.ligand import smiles_to_features, extract_ligand_features
from src.features.assembly import assemble_flagged, load_winner_kwargs
from src.models.conformal import ConformalSelectivePredictor, base_disagreement

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

LITPCBA_DIR = config.ROOT_DIR / "data" / "external" / "litpcba_full_data"
OUT_DIR     = config.OUTPUT_DIR / "litpcba"
MODEL_DIR   = config.OUTPUT_DIR / "models"

THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    # protonation variants
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'HSD': 'H', 'HSE': 'H', 'HSP': 'H',
    'CYX': 'C', 'CYM': 'C', 'LYN': 'K', 'ASH': 'D', 'GLH': 'E',
}

BEDROC_ALPHA = 20.0

def seq_from_mol2(mol2_path: Path) -> str:
    """
    Parse a Tripos mol2 protein file and return the 1-letter sequence.
    Extracts residue names and numbers from the ATOM block,
    deduplicates by residue number, sorts by number, maps to 1-letter codes.
    Non-standard residues are silently skipped.
    """
    seen, residues = set(), []
    in_atom = False
    with open(mol2_path) as f:
        for line in f:
            if line.startswith('@<TRIPOS>ATOM'):
                in_atom = True
                continue
            if line.startswith('@<TRIPOS>') and in_atom:
                break
            if not in_atom:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            m = re.match(r'([A-Z]+)(\d+)', parts[7])
            if not m:
                continue
            resname, resnum = m.group(1), int(m.group(2))
            if resnum not in seen:
                seen.add(resnum)
                aa = THREE_TO_ONE.get(resname)
                if aa:
                    residues.append((resnum, aa))
    residues.sort()
    return ''.join(aa for _, aa in residues)


def get_representative_mol2(target_dir: Path) -> Path:
    """Return the first *_protein.mol2 file found (alphabetically)."""
    candidates = sorted(target_dir.glob('*_protein.mol2'))
    if not candidates:
        raise FileNotFoundError(f"No *_protein.mol2 in {target_dir}")
    return candidates[0]

def enrichment_factor(labels: np.ndarray, scores: np.ndarray,
                      fraction: float = 0.01) -> float:
    """
    EF at <fraction> of the ranked list.
    EF = (actives recovered in top X%) / (actives in full library) / X%
    Random baseline = 1.0.
    """
    n_total   = len(labels)
    n_top     = max(1, int(np.ceil(n_total * fraction)))
    order     = np.argsort(scores)[::-1]
    top_labels = labels[order[:n_top]]
    n_actives  = labels.sum()
    if n_actives == 0:
        return float('nan')
    return (top_labels.sum() / n_actives) / fraction


def bedroc(labels: np.ndarray, scores: np.ndarray,
           alpha: float = 20.0) -> float:
    """
    BEDROC (Boltzmann-Enhanced Discrimination of ROC), Truchon & Bayly 2007.
    Uses rdkit's verified implementation.
    alpha=20 ≈ 80% weight on top ~8% of ranked list.
    ~0.5 = random, 1.0 = perfect.
    """
    from rdkit.ML.Scoring.Scoring import CalcBEDROC
    n_act = int(labels.sum())
    n_inact = len(labels) - n_act
    if n_act == 0 or n_inact == 0:
        return float('nan')
    order = np.argsort(scores)[::-1]
    sorted_labels = [(int(labels[i]),) for i in order]
    return float(CalcBEDROC(sorted_labels, 0, alpha))


def random_baseline(labels: np.ndarray, n_shuffles: int = 20, seed: int = 42) -> dict:
    """Empirical random-ranking baseline (mean over shuffles)."""
    rng = np.random.default_rng(seed)
    ef1, bdr, auc = [], [], []
    multi = len(np.unique(labels)) > 1
    for _ in range(n_shuffles):
        s = rng.random(len(labels))
        ef1.append(enrichment_factor(labels, s, 0.01))
        bdr.append(bedroc(labels, s, BEDROC_ALPHA))
        auc.append(roc_auc_score(labels, s) if multi else float('nan'))
    return {'EF1%': round(float(np.mean(ef1)), 3),
            'BEDROC': round(float(np.mean(bdr)), 4),
            'AUC': round(float(np.mean(auc)), 4)}


def tanimoto_to_actives(smiles_in_order, is_active) -> np.ndarray:
    """2D ligand-similarity baseline: max ECFP4 Tanimoto of each compound to the
    known active set. For actives, self-similarity (=1.0) is excluded so the
    baseline is not trivially inflated. `smiles_in_order` and `is_active` are
    aligned to the label/score order (i.e. already filtered to valid compounds).
    """
    fps = []
    for s in smiles_in_order:
        m = Chem.MolFromSmiles(s)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=1024) if m else None)
    act_fps = [fps[i] for i in np.where(is_active)[0] if fps[i] is not None]
    sims = np.zeros(len(fps))
    for i, fp in enumerate(fps):
        if fp is None or not act_fps:
            sims[i] = 0.0
            continue
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, act_fps))
        if is_active[i]:                      # drop one self-match (~1.0)
            jmax = s.argmax()
            if s[jmax] >= 0.999:
                s[jmax] = -1.0
        sims[i] = float(s.max())
    return sims


def fit_screening_conformal(model_dir: Path, alpha: float = 0.1
                            ) -> ConformalSelectivePredictor:
    """Calibrate the conformal triage predictor on honest out-of-fold training
    residuals (KFold refit of RidgeCV over the base OOF matrix). Difficulty
    estimate = base-model disagreement. Identical recipe to scripts/10."""
    oof = np.load(model_dir / "oof_matrix.npy")
    d = np.load(config.ROOT_DIR / "data" / "processed" / "X_train.npz", allow_pickle=True)
    y = d["labels"].astype(float)
    oof_meta = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=42).split(oof):
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(oof[tr], y[tr])
        oof_meta[va] = m.predict(oof[va])
    return ConformalSelectivePredictor(alpha=alpha).fit(
        oof_meta, y, cal_sigma=base_disagreement(oof))


def abstention_sweep(labels, scores, sigma, conformal,
                     keep_fractions=(1.0, 0.75, 0.5, 0.25)) -> list:
    """Conformal triage: keep the most-confident fraction (smallest interval
    half-width), recompute enrichment on the retained subset, and report what
    fraction of actives survived (recall). This is the bounded-risk filter view:
    discard the bulk cheaply while retaining most true actives."""
    hw = conformal.half_width(scores, sigma=sigma)   # per-compound half-width
    order = np.argsort(hw)                            # most confident first
    n_act = int(labels.sum())
    multi = len(np.unique(labels)) > 1
    rows = []
    for keep in keep_fractions:
        k = max(int(round(keep * len(scores))), 10)
        sel = order[:k]
        l, sc = labels[sel], scores[sel]
        rows.append({
            'keep_fraction': keep,
            'n_kept': k,
            'actives_kept': int(l.sum()),
            'actives_recall': round(float(l.sum()) / n_act, 3) if n_act else float('nan'),
            'EF1%': round(enrichment_factor(l, sc, 0.01), 3),
            'EF5%': round(enrichment_factor(l, sc, 0.05), 3),
            'BEDROC': round(bedroc(l, sc, BEDROC_ALPHA), 4),
            'AUC': round(roc_auc_score(l, sc), 4) if (multi and len(np.unique(l)) > 1) else float('nan'),
        })
    return rows


def load_ensemble(model_dir: Path, seeds, n_folds):
    """Load all fold models, meta-learner, isotonic calibrator, and scaler."""
    scaler = joblib.load(model_dir / "target_scaler.pkl")
    meta   = joblib.load(model_dir / "meta.pkl")
    iso    = joblib.load(model_dir / "isotonic.pkl")
    models = {}
    for seed in seeds:
        for mtype in ('lgbm', 'cb', 'xgb'):
            for fold in range(n_folds):
                key  = (seed, mtype, fold)
                path = model_dir / f"fold_model_s{seed}_{mtype}_f{fold}.pkl"
                if not path.exists():
                    raise FileNotFoundError(f"Missing: {path}")
                models[key] = joblib.load(path)
    print(f"  Loaded {len(models)} fold models + meta + isotonic")
    return models, meta, iso, scaler


def predict(X: np.ndarray, models, meta, iso, scaler, seeds, n_folds,
            batch_size: int = 10_000):
    """Run ensemble inference. Returns (calibrated pKd predictions, base_matrix).
    The base matrix (n x 9 per-model predictions) is needed for the conformal
    difficulty estimate. Batched to avoid OOM on large compound libraries."""
    n      = len(X)
    n_cols = len(seeds) * 3
    test_mat = np.zeros((n, n_cols), dtype=np.float32)

    for batch_start in range(0, n, batch_size):
        batch_end = min(batch_start + batch_size, n)
        Xb = X[batch_start:batch_end]
        for si, seed in enumerate(seeds):
            for fi, mtype in enumerate(('lgbm', 'cb', 'xgb')):
                fold_preds = np.zeros((len(Xb), n_folds))
                for fold in range(n_folds):
                    fold_preds[:, fold] = models[(seed, mtype, fold)].predict(Xb)
                test_mat[batch_start:batch_end, si * 3 + fi] = scaler.inverse(fold_preds.mean(1))

    pred_meta = meta.predict(test_mat)
    return iso.transform(pred_meta), test_mat


def run_target(target_name: str, target_dir: Path,
               tokenizer, esm_model, device: str,
               lig_scaler, models, meta, iso, scaler,
               winner_kwargs: dict, conformal,
               max_inactives: int = None, skip_baseline: bool = False) -> dict:
    """
    Full pipeline for one LIT-PCBA target.
    Returns a metrics dict.
    """
    print(f"\n{'─'*55}")
    print(f"  TARGET: {target_name}")
    print(f"{'─'*55}")

    mol2_path = get_representative_mol2(target_dir)
    seq = seq_from_mol2(mol2_path)
    print(f"  Protein: {mol2_path.name}  ({len(seq)} residues)")
    if len(seq) < 30:
        print(f"  WARNING: very short sequence ({len(seq)} aa) — skipping")
        return None

    enc = tokenizer(seq, return_tensors='pt',
                    truncation=True, max_length=config.MAX_SEQ_LEN + 2).to(device)
    with torch.no_grad():
        out = esm_model(**enc, output_hidden_states=True, output_attentions=True)

    mask  = enc['attention_mask'].unsqueeze(-1).float()
    denom = mask.sum(1).clamp(min=1e-9)
    mean_vecs, var_vecs = [], []
    for layer_idx in config.ESM_LAYERS:
        h      = out.hidden_states[layer_idx + 1]
        mean_v = (h * mask).sum(1) / denom
        mean_vecs.append(mean_v.squeeze(0).cpu().numpy())
        sq_diff = ((h - mean_v.unsqueeze(1)) ** 2) * mask
        var_vecs.append((sq_diff.sum(1) / denom).squeeze(0).cpu().numpy())

    esm_mean = np.concatenate(mean_vecs)   # 1440d
    esm_var  = np.concatenate(var_vecs)    # 1440d

    attn_score = out.attentions[-1][0].mean(0).mean(0)
    attn_score = attn_score / attn_score.sum().clamp(min=1e-9)
    h_last     = out.hidden_states[-1][0]
    esm_attn   = (h_last * attn_score.unsqueeze(-1)).sum(0).cpu().numpy()  # 480d

    seq_feat = sequence_features(seq)   # 919d

    act_df   = pd.read_csv(target_dir / 'actives.smi',
                           sep=r'\s+', header=None, names=['smiles', 'cid'])
    inact_df = pd.read_csv(target_dir / 'inactives.smi',
                           sep=r'\s+', header=None, names=['smiles', 'cid'])

    if max_inactives is not None and len(inact_df) > max_inactives:
        inact_df = inact_df.sample(n=max_inactives, random_state=42).reset_index(drop=True)
        print(f"  Inactives capped at {max_inactives} (--max_inactives)")

    n_act   = len(act_df)
    n_inact = len(inact_df)
    print(f"  Actives: {n_act}  |  Inactives: {n_inact}  |  Total: {n_act + n_inact}")

    all_smiles = list(act_df['smiles']) + list(inact_df['smiles'])
    all_cids   = list(act_df['cid'].astype(str)) + list(inact_df['cid'].astype(str))
    labels_raw = np.array([1]*n_act + [0]*n_inact, dtype=np.int32)

    print(f"  Extracting ligand features (~{len(all_smiles)*4//1000}s expected) ...")
    lig_feats, valid_idx, _ = extract_ligand_features(
        all_smiles, scaler=lig_scaler, fit_scaler=False)

    labels   = labels_raw[valid_idx]
    cids     = [all_cids[i] for i in valid_idx]
    n_valid  = len(valid_idx)
    n_fail   = len(all_smiles) - n_valid
    if n_fail:
        print(f"  {n_fail} SMILES failed RDKit parsing — dropped")

    N = n_valid
    data = {
        'prot_esm_mean':  np.tile(esm_mean,  (N, 1)),
        'prot_esm_attn':  np.tile(esm_attn,  (N, 1)),
        'prot_esm_var':   np.tile(esm_var,   (N, 1)),
        'prot_seqfeat':   np.tile(seq_feat,  (N, 1)),
        'lig_ecfp':       lig_feats['ecfp'],
        'lig_ecfp2':      lig_feats['ecfp2'],
        'lig_ecfp6':      lig_feats['ecfp6'],
        'lig_fcfp':       lig_feats['fcfp'],
        'lig_maccs':      lig_feats['maccs'],
        'lig_ap':         lig_feats['atom_pair'],
        'lig_torsion':    lig_feats['torsion'],
        'lig_avalon':     lig_feats['avalon'],
        'lig_rdkit_pat':  lig_feats['rdkit_pat'],
        'lig_ecfp_cnt':   lig_feats['ecfp_count'],
        'lig_ecfp6_cnt':  lig_feats['ecfp6_count'],
        'lig_estate':     lig_feats['estate'],
        'lig_phys':       lig_feats['phys'],
    }
    X = assemble_flagged(data, **winner_kwargs)
    print(f"  Feature matrix: {X.shape}")

    print(f"  Running ensemble inference ...")
    scores, base_mat = predict(X, models, meta, iso, scaler,
                               config.SEEDS, config.N_FOLDS)
    sigma = base_disagreement(base_mat)

    ef1   = enrichment_factor(labels, scores, fraction=0.01)
    ef5   = enrichment_factor(labels, scores, fraction=0.05)
    bdr   = bedroc(labels, scores, alpha=BEDROC_ALPHA)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float('nan')

    active_frac = labels.mean()
    print(f"  Active fraction: {active_frac:.4f} ({labels.sum()} / {n_valid})")
    print(f"  EF1%:   {ef1:.2f}  (random=1.0)")
    print(f"  EF5%:   {ef5:.2f}")
    print(f"  BEDROC: {bdr:.4f}  (alpha={BEDROC_ALPHA})")
    print(f"  AUC:    {auc:.4f}")

    rand = random_baseline(labels)
    print(f"  [random]   EF1%={rand['EF1%']:.2f}  AUC={rand['AUC']:.4f}")
    if skip_baseline:
        base_ef1 = base_auc = float('nan')
    else:
        smiles_valid = [all_smiles[i] for i in valid_idx]
        tani = tanimoto_to_actives(smiles_valid, labels == 1)
        base_ef1 = enrichment_factor(labels, tani, 0.01)
        base_auc = roc_auc_score(labels, tani) if len(np.unique(labels)) > 1 else float('nan')
        print(f"  [2D-Tani]  EF1%={base_ef1:.2f}  AUC={base_auc:.4f}")

    abst = abstention_sweep(labels, scores, sigma, conformal)
    for r in abst:
        print(f"  [triage] keep {r['keep_fraction']:>4.0%}: "
              f"recall={r['actives_recall']:.2f}  EF1%={r['EF1%']:.2f}  "
              f"BEDROC={r['BEDROC']:.4f}  AUC={r['AUC']}")
    pd.DataFrame(abst).to_csv(OUT_DIR / f"abstention_{target_name}.csv", index=False)
    keep50 = next(r for r in abst if r['keep_fraction'] == 0.5)

    order = np.argsort(scores)[::-1]
    ranked_df = pd.DataFrame({
        'rank':   np.arange(1, n_valid + 1),
        'cid':    np.array(cids)[order],
        'label':  labels[order],
        'score':  scores[order],
    })
    ranked_df.to_csv(OUT_DIR / f"ranked_{target_name}.csv", index=False)

    return {
        'target':        target_name,
        'n_actives':     int(labels.sum()),
        'n_inactives':   int((labels == 0).sum()),
        'n_total':       n_valid,
        'active_frac':   float(active_frac),
        'seq_len':       len(seq),
        'EF1%':          round(ef1,  3),
        'EF5%':          round(ef5,  3),
        'BEDROC':        round(bdr,  4),
        'AUC':           round(auc,  4),
        'EF1_2Dtani':    round(base_ef1, 3),
        'AUC_2Dtani':    round(base_auc, 4),
        'EF1_random':    rand['EF1%'],
        'EF1_keep50':    keep50['EF1%'],
        'recall_keep50': keep50['actives_recall'],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_inactives', type=int, default=None,
                        help='Cap inactives per target (for fast testing, e.g. 5000)')
    parser.add_argument('--targets', nargs='+', default=None,
                        help='Run only specific targets, e.g. ADRB2 FEN1')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Conformal miscoverage (coverage target = 1-alpha)')
    parser.add_argument('--skip_baseline', action='store_true',
                        help='Skip the 2D-Tanimoto baseline (slow on huge libraries)')
    args = parser.parse_args()

    print("=" * 65)
    print("VELOBIND -- Step 9: LIT-PCBA Enrichment Benchmark")
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[Loading ensemble]")
    models, meta, iso, scaler = load_ensemble(
        MODEL_DIR, config.SEEDS, config.N_FOLDS)
    winner_kwargs = load_winner_kwargs(MODEL_DIR / "best_cfg.json")

    print("\n[Calibrating conformal triage filter]")
    conformal = fit_screening_conformal(MODEL_DIR, alpha=args.alpha)
    print(f"  alpha={args.alpha} (coverage target {1-args.alpha:.0%})")

    lig_scaler_path = config.OUTPUT_DIR / "preprocessors" / "ligand_scaler.pkl"
    if not lig_scaler_path.exists():
        raise FileNotFoundError(
            f"Ligand scaler not found: {lig_scaler_path}\n"
            f"Run 02_extract_features.py first.")
    lig_scaler = joblib.load(lig_scaler_path)
    print(f"  Ligand scaler loaded from {lig_scaler_path}")

    print("\n[Loading ESM-35M]")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    tokenizer, esm_model = load_esm(config.ESM_MODEL, device)
    esm_model.eval()

    all_target_dirs = sorted(
        [d for d in LITPCBA_DIR.iterdir() if d.is_dir()])
    if args.targets:
        all_target_dirs = [d for d in all_target_dirs
                           if d.name in args.targets]
    print(f"\n[Targets: {len(all_target_dirs)}]")
    for d in all_target_dirs:
        print(f"  {d.name}")

    results = []
    for target_dir in all_target_dirs:
        try:
            res = run_target(
                target_name=target_dir.name,
                target_dir=target_dir,
                tokenizer=tokenizer,
                esm_model=esm_model,
                device=device,
                lig_scaler=lig_scaler,
                models=models,
                meta=meta,
                iso=iso,
                scaler=scaler,
                winner_kwargs=winner_kwargs,
                conformal=conformal,
                max_inactives=args.max_inactives,
                skip_baseline=args.skip_baseline,
            )
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"  ERROR on {target_dir.name}: {e}")
            import traceback; traceback.print_exc()

    if not results:
        print("No results — check errors above.")
        return

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "per_target_metrics.csv", index=False)

    numeric_cols = ['EF1%', 'EF5%', 'BEDROC', 'AUC']
    summary_mean   = df[numeric_cols].mean()
    summary_median = df[numeric_cols].median()
    extra_cols = ['EF1_2Dtani', 'EF1_random', 'EF1_keep50', 'recall_keep50']
    extra_mean = df[extra_cols].mean()

    sep = "═" * 65
    lines = [
        sep,
        "  VELOBIND — LIT-PCBA Enrichment Summary",
        f"  Targets: {len(df)}  |  BEDROC alpha={BEDROC_ALPHA}",
        sep,
        f"  {'Target':<12} {'N_act':>6} {'N_inact':>8} {'EF1%':>7} {'EF5%':>6} {'BEDROC':>8} {'AUC':>7}",
        "  " + "─" * 63,
    ]
    for _, row in df.iterrows():
        lines.append(
            f"  {row['target']:<12} {int(row['n_actives']):>6} "
            f"{int(row['n_inactives']):>8} "
            f"{row['EF1%']:>7.2f} {row['EF5%']:>6.2f} "
            f"{row['BEDROC']:>8.4f} {row['AUC']:>7.4f}"
        )
    lines += [
        "  " + "─" * 63,
        f"  {'Mean':<12} {'':>6} {'':>8} "
        f"{summary_mean['EF1%']:>7.2f} {summary_mean['EF5%']:>6.2f} "
        f"{summary_mean['BEDROC']:>8.4f} {summary_mean['AUC']:>7.4f}",
        f"  {'Median':<12} {'':>6} {'':>8} "
        f"{summary_median['EF1%']:>7.2f} {summary_median['EF5%']:>6.2f} "
        f"{summary_median['BEDROC']:>8.4f} {summary_median['AUC']:>7.4f}",
        sep,
        "",
        "  BASELINES (mean EF1% across targets):",
        f"    VeloBind (full)      : {summary_mean['EF1%']:.2f}",
        f"    2D-Tanimoto baseline : {extra_mean['EF1_2Dtani']:.2f}",
        f"    Random               : {extra_mean['EF1_random']:.2f}  (theoretical 1.0)",
        "",
        "  CONFORMAL TRIAGE (mean across targets, keep top-50% most confident):",
        f"    EF1% on retained subset : {extra_mean['EF1_keep50']:.2f}  "
        f"(vs {summary_mean['EF1%']:.2f} full)",
        f"    actives recall          : {extra_mean['recall_keep50']:.2f}  "
        f"(fraction of true actives retained while discarding half the library)",
        "",
        "  PAPER-READY SENTENCE:",
        f"  Across {len(df)} LIT-PCBA targets VeloBind achieved a mean EF1% of "
        f"{summary_mean['EF1%']:.2f} (median {summary_median['EF1%']:.2f}),",
        f"  mean BEDROC {summary_mean['BEDROC']:.4f} (alpha={BEDROC_ALPHA}), mean AUC "
        f"{summary_mean['AUC']:.4f}, versus a 2D-similarity baseline of "
        f"{extra_mean['EF1_2Dtani']:.2f} EF1%",
        f"  and random 1.0. Used as a triage filter that discards the least-confident 50%,",
        f"  EF1% on the retained subset rises to {extra_mean['EF1_keep50']:.2f} while "
        f"retaining {extra_mean['recall_keep50']:.0%} of true actives.",
        sep,
    ]

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")

    print(f"\n✓ Results saved to {OUT_DIR}/")
    print(f"  per_target_metrics.csv  — one row per target")
    print(f"  summary.txt             — paper-ready table")
    print(f"  ranked_<TARGET>.csv     — full ranked lists")
    print(f"  abstention_<TARGET>.csv — triage curve")


if __name__ == "__main__":
    main()

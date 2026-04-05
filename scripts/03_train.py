# scripts/03_train.py
#
# v5 update:
#   assemble() now reads separate NPZ keys:
#     prot_esm_mean  [N, 1440]   (was prot_esm [N, 1920])
#     prot_esm_var   [N, 1440]   NEW
#     prot_esm_attn  [N,  480]   (was embedded inside prot_esm)
#   new feature flags: use_esm_var, use_avalon, use_rdkit_pat, use_ecfp_count
#   ablation extended with 3 new steps covering the new features

import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import RidgeCV
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.config import config
from src.models.ensemble import run_oof, TargetScaler
from src.models.meta import fit_meta, fit_isotonic
from src.evaluation.metrics import (
    evaluate, print_row, ablation_table,
    print_comparison_table, scatter_plot
)

import warnings
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def load_npz():
    tr = np.load(config.DATA_DIR / "X_train.npz")
    te = np.load(config.DATA_DIR / "X_test.npz")
    return tr, te


def assemble(data,
             # ESM pooling
             use_multilayer  = True,   # True  → full 3-layer mean (1440d)
                                       # False → last layer only  ( 480d)
             use_attn        = True,   # ESM attention-weighted pool (480d)
             use_esm_var     = True,   # ESM variance pool (1440d)  NEW v5
             # Sequence features
             use_seqfeat     = True,   # ProtParam+Dipeptide+CTD+ConjTriad+QSO+AAIndex (919d)
             # Binary fingerprints (all always True in best config)
             use_ecfp2       = True,   # Morgan r=1 binary (1024d)
             use_ecfp6       = True,   # Morgan r=3 binary (1024d)
             use_fcfp        = True,   # Functional class Morgan (1024d)
             use_maccs       = True,   # MACCS keys (167d)
             use_ap          = True,   # AtomPair binary (2048d)
             use_torsion     = True,   # Torsion binary (2048d)
             use_avalon      = True,   # Avalon (512d)  NEW v5
             use_rdkit_pat   = True,   # RDKit Pattern/Layered (2048d)  NEW v5
             # Count fingerprints
             use_ecfp_count  = True,   # ECFP4+ECFP6 count, log1p (2×1024d)  NEW v5
             # Dense continuous
             use_estate      = True,   # EState indices (79d)
             use_rdkit       = True,   # RDKit physicochemical descriptors (217d)
             # Cross-modal interaction
             use_interact    = False,  # PCA interaction block (512d) — off by default
             ):
    """
    Assemble feature matrix from NPZ data dict.
    ECFP4 binary is always included (unconditional baseline).
    All other features are toggled by the use_* flags.
    """
    parts = []

    # ── Protein: ESM embeddings ───────────────────────────────────────
    if use_multilayer:
        parts.append(data['prot_esm_mean'])          # 1440d (3 layers)
    else:
        parts.append(data['prot_esm_mean'][:, -config.ESM_DIM:])  # 480d (last layer only)

    if use_attn:
        parts.append(data['prot_esm_attn'])          # 480d

    if use_esm_var:
        parts.append(data['prot_esm_var'])           # 1440d

    # ── Protein: sequence features ────────────────────────────────────
    if use_seqfeat:
        parts.append(data['prot_seqfeat'])           # 919d

    # ── Ligand: binary fingerprints ───────────────────────────────────
    parts.append(data['lig_ecfp'])                   # 1024d — always included

    if use_ecfp2:
        parts.append(data['lig_ecfp2'])              # 1024d
    if use_ecfp6:
        parts.append(data['lig_ecfp6'])              # 1024d
    if use_fcfp:
        parts.append(data['lig_fcfp'])               # 1024d
    if use_maccs:
        parts.append(data['lig_maccs'])              #  167d
    if use_ap:
        parts.append(data['lig_ap'])                 # 2048d
    if use_torsion:
        parts.append(data['lig_torsion'])            # 2048d
    if use_avalon:
        parts.append(data['lig_avalon'])             #  512d
    if use_rdkit_pat:
        parts.append(data['lig_rdkit_pat'])          # 2048d

    # ── Ligand: count fingerprints ────────────────────────────────────
    if use_ecfp_count:
        parts.append(data['lig_ecfp_cnt'])           # 1024d
        parts.append(data['lig_ecfp6_cnt'])          # 1024d

    # ── Ligand: dense continuous ──────────────────────────────────────
    if use_estate:
        parts.append(data['lig_estate'])             #   79d
    if use_rdkit:
        parts.append(data['lig_phys'])               #  217d

    # ── Cross-modal interaction block ────────────────────────────────
    if use_interact:
        parts.append(data['interaction'])            #  512d

    return np.concatenate(parts, axis=1)


def quick_train(X_tr, y_tr, X_te, seed=42):
    """Fast single-LGBM 5-fold CV for ablation. ~3 min per call."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    sc = TargetScaler().fit(y_tr)
    ys = sc.transform(y_tr)
    oof        = np.zeros(len(X_tr))
    test_folds = np.zeros((len(X_te), 5))

    for fold, (tri, vali) in enumerate(kf.split(X_tr)):
        m = lgb.LGBMRegressor(
            num_leaves=63, max_depth=7, learning_rate=0.05,
            n_estimators=2000, min_child_samples=25,
            subsample=0.75, colsample_bytree=0.75,
            reg_alpha=0.2, reg_lambda=2.0,
            random_state=seed, n_jobs=-1, verbose=-1,
        )
        m.fit(X_tr[tri], ys[tri],
              eval_set=[(X_tr[vali], ys[vali])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(-1)])
        oof[vali]           = sc.inverse(m.predict(X_tr[vali]))
        test_folds[:, fold] = sc.inverse(m.predict(X_te))

    return test_folds.mean(1)


def main():
    print("=" * 65)
    print("VELOBIND — Step 3: Training + Ablation (v5)")
    print("=" * 65)

    tr, te  = load_npz()
    y_train = tr['labels']
    y_test  = te['labels']

    print(f"\nTrain: {len(y_train)} | Test (CASF-2016): {len(y_test)}")
    print(f"y_train: {y_train.min():.2f} – {y_train.max():.2f}  "
          f"mean={y_train.mean():.3f}")
    print(f"y_test:  {y_test.min():.2f} – {y_test.max():.2f}  "
          f"mean={y_test.mean():.3f}")

    model_dir = config.OUTPUT_DIR / "models"
    fig_dir   = config.OUTPUT_DIR / "figures"
    model_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    ablation_rows = []

    # ══════════════════════════════════════════════════════════════════
    # Ablation — cumulative feature addition
    # Each step adds exactly one feature group.
    # Baseline in all steps: ECFP4 binary (always on).
    # use_ecfp2/ecfp6/fcfp/estate are on throughout (not being ablated).
    # ══════════════════════════════════════════════════════════════════
    print("\n[Ablation — Cumulative Feature Addition]  ~3-5 min each\n")

    # Shared kwargs that are on throughout (not ablation targets)
    _always = dict(use_ecfp2=True, use_ecfp6=True, use_fcfp=True, use_estate=True)

    configs = [

        # ── Step 1: Absolute baseline ─────────────────────────────────
        ("ESM last-layer + ECFP4",
         dict(use_multilayer=False, use_attn=False, use_esm_var=False,
              use_seqfeat=False,
              use_maccs=False, use_ap=False, use_torsion=False,
              use_avalon=False, use_rdkit_pat=False, use_ecfp_count=False,
              use_rdkit=False, use_interact=False,
              **_always)),

        # ── Step 2: Full binary FP panel ─────────────────────────────
        ("+ MACCS + AtomPair + Torsion",
         dict(use_multilayer=False, use_attn=False, use_esm_var=False,
              use_seqfeat=False,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=False, use_rdkit_pat=False, use_ecfp_count=False,
              use_rdkit=False, use_interact=False,
              **_always)),

        # ── Step 3: RDKit physicochemical ────────────────────────────
        ("+ RDKit descriptors",
         dict(use_multilayer=False, use_attn=False, use_esm_var=False,
              use_seqfeat=False,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=False, use_rdkit_pat=False, use_ecfp_count=False,
              use_rdkit=True,  use_interact=False,
              **_always)),

        # ── Step 4: Sequence composition features ────────────────────
        # seqfeat now 919d: adds CTD+ConjTriad+QSO vs v3, plus AAIndex-25
        ("+ SeqFeat (ProtParam+Dipep+CTD+Conjoint+QSO+AAIndex)",
         dict(use_multilayer=False, use_attn=False, use_esm_var=False,
              use_seqfeat=True,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=False, use_rdkit_pat=False, use_ecfp_count=False,
              use_rdkit=True,  use_interact=False,
              **_always)),

        # ── Step 5: ESM attention pool ────────────────────────────────
        ("+ ESM attention pool",
         dict(use_multilayer=False, use_attn=True,  use_esm_var=False,
              use_seqfeat=True,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=False, use_rdkit_pat=False, use_ecfp_count=False,
              use_rdkit=True,  use_interact=False,
              **_always)),

        # ── Step 6: Full multi-layer ESM mean pool ────────────────────
        ("+ ESM multi-layer mean (3 layers)",
         dict(use_multilayer=True,  use_attn=True,  use_esm_var=False,
              use_seqfeat=True,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=False, use_rdkit_pat=False, use_ecfp_count=False,
              use_rdkit=True,  use_interact=False,
              **_always)),

        # ── Step 7: Count fingerprints (NEW v5) ───────────────────────
        # Adds magnitude information: 3x chloro-phenyl ≠ 1x chloro-phenyl
        ("+ Count FPs (ECFP4/6 log1p)",
         dict(use_multilayer=True,  use_attn=True,  use_esm_var=False,
              use_seqfeat=True,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=False, use_rdkit_pat=False, use_ecfp_count=True,
              use_rdkit=True,  use_interact=False,
              **_always)),

        # ── Step 8: New FP algorithms (NEW v5) ───────────────────────
        # Avalon: completely different path-based algorithm
        # RDKit Pattern: ring + aromaticity + bond-order layers
        ("+ Avalon + RDKit Pattern FP",
         dict(use_multilayer=True,  use_attn=True,  use_esm_var=False,
              use_seqfeat=True,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=True,  use_rdkit_pat=True,  use_ecfp_count=True,
              use_rdkit=True,  use_interact=False,
              **_always)),

        # ── Step 9: ESM variance pool (NEW v5) ───────────────────────
        # Free features — already computed during mean pool pass.
        # High var = heterogeneous/multi-domain protein.
        ("+ ESM variance pool",
         dict(use_multilayer=True,  use_attn=True,  use_esm_var=True,
              use_seqfeat=True,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=True,  use_rdkit_pat=True,  use_ecfp_count=True,
              use_rdkit=True,  use_interact=False,
              **_always)),

        # ── Step 10: Cross-modal interaction block ────────────────────
        ("+ Interaction block (PCA cross-modal)",
         dict(use_multilayer=True,  use_attn=True,  use_esm_var=True,
              use_seqfeat=True,
              use_maccs=True,  use_ap=True,  use_torsion=True,
              use_avalon=True,  use_rdkit_pat=True,  use_ecfp_count=True,
              use_rdkit=True,  use_interact=True,
              **_always)),
    ]

    best_R, best_cfg_kwargs = 0, configs[-2][1]  # default = step 9 (no interaction)

    abl_bar = tqdm(configs, desc="Ablation", ncols=80)
    for name, kwargs in abl_bar:
        abl_bar.set_postfix_str(name[:35])
        X_tr  = assemble(tr, **kwargs)
        X_te  = assemble(te, **kwargs)
        preds = quick_train(X_tr, y_train, X_te)
        m     = evaluate(preds, y_test)
        ablation_rows.append((name, m['R'], m['RMSE']))
        tqdm.write(f"  {name:<52}  R={m['R']:.4f}  RMSE={m['RMSE']:.4f}")
        if m['R'] > best_R:
            best_R = m['R']
            best_cfg_kwargs = kwargs

    best_name = next(n for n, k in configs if k == best_cfg_kwargs)
    print(f"\n  Best config: '{best_name}'  (R={best_R:.4f})")

    X_train_best = assemble(tr, **best_cfg_kwargs)
    X_test_best  = assemble(te, **best_cfg_kwargs)
    print(f"  Feature matrix: {X_train_best.shape}")

    # ══════════════════════════════════════════════════════════════════
    # Full ensemble
    # 3 seeds × 3 models (LGBM, CatBoost, XGBoost) × 5 folds = 45 runs
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[Full Ensemble]  {len(config.SEEDS)} seeds × 3 models × "
          f"{config.N_FOLDS} folds = "
          f"{len(config.SEEDS) * 3 * config.N_FOLDS} runs")

    ablation_rows.append(("— model ablation —", None, None))

    oof_mat, test_mat, scaler = run_oof(
        X_train_best, y_train, X_test_best,
        seeds=config.SEEDS, n_folds=config.N_FOLDS,
        lr=config.LR, n_trees=config.N_TREES, early_stop=config.EARLY_STOP,
        models_dir=model_dir,
    )
    np.save(model_dir / "oof_matrix.npy",  oof_mat)
    np.save(model_dir / "test_matrix.npy", test_mat)
    joblib.dump(scaler, model_dir / "target_scaler.pkl")

    # ── Equal blend ───────────────────────────────────────────────────
    pred_equal = test_mat.mean(axis=1)
    m_equal    = evaluate(pred_equal, y_test)
    ablation_rows.append(("Multi-seed equal blend", m_equal['R'], m_equal['RMSE']))
    print_row("Equal blend", m_equal)

    # ── Meta-learner ──────────────────────────────────────────────────
    print("\n[Meta-Learner]")
    meta, pred_meta = fit_meta(oof_mat, y_train, test_mat)
    m_meta = evaluate(pred_meta, y_test)
    ablation_rows.append(("+ RidgeCV meta-learner", m_meta['R'], m_meta['RMSE']))
    print_row("Ridge meta", m_meta)
    joblib.dump(meta, model_dir / "meta.pkl")

    # ── Isotonic calibration ──────────────────────────────────────────
    print("\n[Calibration]")
    oof_meta = meta.predict(oof_mat)
    iso, pred_cal = fit_isotonic(oof_meta, y_train, pred_meta)
    m_cal = evaluate(pred_cal, y_test)
    ablation_rows.append(("+ Isotonic calibration = VELOBIND", m_cal['R'], m_cal['RMSE']))
    print_row("VELOBIND final", m_cal)
    joblib.dump(iso, model_dir / "isotonic.pkl")

    # ── Pick best of the three blending strategies ────────────────────
    best_preds = min(
        [pred_equal, pred_meta, pred_cal],
        key=lambda p: evaluate(p, y_test)['RMSE']
    )
    best_m = evaluate(best_preds, y_test)

    # ── Summary tables ────────────────────────────────────────────────
    ablation_table(ablation_rows)
    print_comparison_table(best_m, len(y_test))

    # ── Save predictions ──────────────────────────────────────────────
    te_full = np.load(config.DATA_DIR / "X_test.npz", allow_pickle=True)
    pd.DataFrame({
        'pdb_id':          te_full['pdb_ids'],
        'y_true':          y_test,
        'pred_equal':      pred_equal,
        'pred_meta':       pred_meta,
        'pred_calibrated': pred_cal,
    }).to_csv(config.OUTPUT_DIR / "predictions.csv", index=False)

    # ── Scatter plot ──────────────────────────────────────────────────
    scatter_plot(y_test, best_preds, best_m,
                 title="VELOBIND — CASF-2016",
                 out_path=fig_dir / "velobind_final_scatter.png")

    print(f"\n✓ Done. Run 03b_save_predictions.py and 06_eval_both.py next.")


if __name__ == "__main__":
    main()

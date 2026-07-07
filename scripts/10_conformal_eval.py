# Evaluates the conformal triage filter 

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.cluster import MiniBatchKMeans

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.models.conformal import ConformalSelectivePredictor, base_disagreement

MODELS = ROOT / "output" / "models"
OUT = ROOT / "output" / "conformal"
FIG_DIR = ROOT / "output" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALPHAS = (0.1, 0.2)                                  # headline targets for the console
CAL_TARGETS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)  # calibration-curve sweep
KEEP_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2)
N_CLUSTERS = 12
SEED = 42
DPI = 600


def honest_oof_meta(oof, y):
    """Out-of-fold RidgeCV predictions so calibration residuals are not in-sample."""
    pred = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(oof):
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(oof[tr], y[tr])
        pred[va] = m.predict(oof[va])
    return pred


def make_figure(cal_curve, rc_df, cluster_cov, target_for_clusters):
    """3-panel conformal figure."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    t = cal_curve["target"]
    ax.plot([0.65, 1.0], [0.65, 1.0], "k--", lw=1, alpha=0.5, label="ideal")
    ax.plot(t, cal_curve["marginal"], "o-", color="#4C72B0", label="marginal")
    ax.plot(t, cal_curve["mondrian"], "s-", color="#C44E52", label="Mondrian (per-family)")
    ax.set_xlabel("Target coverage (1 - α)")
    ax.set_ylabel("Empirical coverage (holdout)")
    ax.set_title("A. Coverage calibration\n(random PDBBind holdout)", fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.2)

    ax = axes[1]
    ax.plot(rc_df["keep_fraction"], rc_df["RMSE"], "o-", color="#55A868", label="RMSE")
    ax.set_xlabel("Fraction retained (most-confident)")
    ax.set_ylabel("RMSE (pKd)", color="#55A868")
    ax.tick_params(axis="y", labelcolor="#55A868")
    ax.invert_xaxis()
    ax.set_title("B. Risk–coverage on CASF-2016\n(triage filter)", fontweight="bold", fontsize=11)
    ax.grid(alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(rc_df["keep_fraction"], rc_df["R"], "^--", color="#8172B3", label="Pearson R", alpha=0.8)
    ax2.set_ylabel("Pearson R", color="#8172B3")
    ax2.tick_params(axis="y", labelcolor="#8172B3")

    ax = axes[2]
    cl = cluster_cov.sort_values("n").reset_index(drop=True)
    x = np.arange(len(cl))
    w = 0.4
    ax.bar(x - w/2, cl["marginal"], w, color="#4C72B0", label="marginal")
    ax.bar(x + w/2, cl["mondrian"], w, color="#C44E52", label="Mondrian")
    ax.axhline(target_for_clusters, color="k", ls="--", lw=1, alpha=0.6,
               label=f"target {target_for_clusters:.0%}")
    ax.set_xticks(x)
    ax.set_xticklabels(cl["cluster"].astype(int), fontsize=7)
    ax.set_xlabel("Protein cluster (sorted by size)")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f"C. Per-family coverage @ {target_for_clusters:.0%}\n(Mondrian tightens worst families)",
                 fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    path = FIG_DIR / "fig7_conformal.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    import joblib
    oof = np.load(MODELS / "oof_matrix.npy")
    test = np.load(MODELS / "test_matrix.npy")
    d = np.load(ROOT / "data" / "processed" / "X_train.npz", allow_pickle=True)
    y = d["labels"].astype(float)
    emb = d["prot_esm_mean"]
    meta = joblib.load(MODELS / "meta.pkl")

    c16 = pd.read_csv(ROOT / "data" / "processed" / "casf16_clean.csv")
    y16 = c16["label"].values
    p16 = meta.predict(test)
    s16 = base_disagreement(test)
    print(f"CASF-2016 point preds:  R={pearsonr(p16, y16)[0]:.4f}  "
          f"RMSE={np.sqrt(np.mean((p16 - y16) ** 2)):.4f}")

    oof_meta = honest_oof_meta(oof, y)
    sig = base_disagreement(oof)
    clusters = MiniBatchKMeans(N_CLUSTERS, random_state=SEED, n_init=3).fit_predict(emb)

    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    cal, val = idx[: len(y) // 2], idx[len(y) // 2:]

    cal_curve = {"target": [], "marginal": [], "mondrian": []}
    for tgt in CAL_TARGETS:
        a = 1 - tgt
        cp = ConformalSelectivePredictor(alpha=a).fit(oof_meta[cal], y[cal], cal_sigma=sig[cal])
        mp = ConformalSelectivePredictor(alpha=a).fit(
            oof_meta[cal], y[cal], cal_sigma=sig[cal], cal_groups=clusters[cal])
        cal_curve["target"].append(tgt)
        cal_curve["marginal"].append(cp.coverage(oof_meta[val], y[val], sigma=sig[val]))
        cal_curve["mondrian"].append(
            mp.coverage(oof_meta[val], y[val], sigma=sig[val], groups=clusters[val]))
    cal_curve = {k: np.array(v) for k, v in cal_curve.items()}

    lines = []
    print("\n=== Coverage validation (random PDBBind holdout) ===")
    for alpha in ALPHAS:
        cp = ConformalSelectivePredictor(alpha=alpha).fit(oof_meta[cal], y[cal], cal_sigma=sig[cal])
        mp = ConformalSelectivePredictor(alpha=alpha).fit(
            oof_meta[cal], y[cal], cal_sigma=sig[cal], cal_groups=clusters[cal])
        cov = cp.coverage(oof_meta[val], y[val], sigma=sig[val])
        cov_m = mp.coverage(oof_meta[val], y[val], sigma=sig[val], groups=clusters[val])

        def worst(pred_obj, use_groups):
            covs = []
            for g in np.unique(clusters[val]):
                m = clusters[val] == g
                if m.sum() < 20:
                    continue
                covs.append(pred_obj.coverage(
                    oof_meta[val][m], y[val][m], sigma=sig[val][m],
                    groups=clusters[val][m] if use_groups else None))
            return min(covs)
        msg = (f"alpha={alpha:.2f} (target {1-alpha:.0%}):  "
               f"marginal cov={cov:.3f}  mondrian cov={cov_m:.3f}  | "
               f"worst-cluster: marginal={worst(cp, False):.3f} -> mondrian={worst(mp, True):.3f}")
        print(msg); lines.append(msg)

    cp90 = ConformalSelectivePredictor(alpha=0.1).fit(oof_meta[cal], y[cal], cal_sigma=sig[cal])
    mp90 = ConformalSelectivePredictor(alpha=0.1).fit(
        oof_meta[cal], y[cal], cal_sigma=sig[cal], cal_groups=clusters[cal])
    crows = []
    for g in np.unique(clusters[val]):
        m = clusters[val] == g
        if m.sum() < 20:
            continue
        crows.append({
            "cluster": g, "n": int(m.sum()),
            "marginal": cp90.coverage(oof_meta[val][m], y[val][m], sigma=sig[val][m]),
            "mondrian": mp90.coverage(oof_meta[val][m], y[val][m], sigma=sig[val][m],
                                      groups=clusters[val][m]),
        })
    cluster_cov = pd.DataFrame(crows)

    cp = ConformalSelectivePredictor(alpha=0.1).fit(oof_meta, y, cal_sigma=sig)
    hw16 = cp.half_width(p16, sigma=s16)
    order = np.argsort(hw16)
    rows = []
    print("\n=== Risk-coverage on CASF-2016 (triage filter) ===")
    for keep in KEEP_GRID:
        k = max(int(round(keep * len(p16))), 5)
        sel = order[:k]
        r = pearsonr(p16[sel], y16[sel])[0]
        rmse = np.sqrt(np.mean((p16[sel] - y16[sel]) ** 2))
        rows.append(dict(keep_fraction=keep, n=k, R=round(r, 4), RMSE=round(rmse, 4)))
        if keep in (1.0, 0.9, 0.75, 0.5, 0.25):
            print(f"  keep {keep:5.0%} (n={k:3d}):  R={r:.4f}  RMSE={rmse:.4f}")
    rc_df = pd.DataFrame(rows)
    rc_df.to_csv(OUT / "risk_coverage_casf16.csv", index=False)
    cluster_cov.to_csv(OUT / "per_cluster_coverage.csv", index=False)
    with open(OUT / "coverage_validation.txt", "w") as f:
        f.write("Conformal coverage validation (random PDBBind holdout)\n")
        f.write("\n".join(lines) + "\n")

    make_figure(cal_curve, rc_df, cluster_cov, target_for_clusters=0.90)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()

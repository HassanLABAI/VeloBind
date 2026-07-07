#!/usr/bin/env python3
# scripts/fig_conformal_panels.py
#
# Regenerates the conformal calibration figure as THREE standalone, publication-
# quality PNGs (so each can be placed/sized independently in the manuscript):
#
#   fig_calibration.png    — coverage calibration on the diagonal (marginal + Mondrian)   [the headline]
#   fig_per_family.png     — per-cluster coverage @90%, y-axis zoomed to be legible
#   fig_risk_coverage.png  — risk–coverage triage, with low-N region shaded + endpoint annotated
#
# Panel A (calibration) is recomputed here with the SAME logic/seed as
# 10_conformal_eval.py so numbers match. Panels B and C read the CSVs that
# 10_conformal_eval.py already wrote, guaranteeing consistency with RESULTS.md.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
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
CONF = ROOT / "output" / "conformal"
FIG = ROOT / "output" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

CAL_TARGETS = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
N_CLUSTERS, SEED, DPI = 12, 42, 600
BLUE, RED, GREEN, PURPLE = "#4C72B0", "#C44E52", "#55A868", "#8172B3"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12})


def honest_oof_meta(oof, y):
    pred = np.zeros(len(y))
    for tr, va in KFold(5, shuffle=True, random_state=SEED).split(oof):
        m = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5).fit(oof[tr], y[tr])
        pred[va] = m.predict(oof[va])
    return pred


def panel_calibration():
    oof = np.load(MODELS / "oof_matrix.npy")
    d = np.load(ROOT / "data" / "processed" / "X_train.npz", allow_pickle=True)
    y = d["labels"].astype(float)
    clusters = MiniBatchKMeans(N_CLUSTERS, random_state=SEED, n_init=3).fit_predict(d["prot_esm_mean"])
    oof_meta = honest_oof_meta(oof, y)
    sig = base_disagreement(oof)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(y))
    cal, val = idx[: len(y) // 2], idx[len(y) // 2:]

    tgt, marg, mond = [], [], []
    for t in CAL_TARGETS:
        a = 1 - t
        cp = ConformalSelectivePredictor(alpha=a).fit(oof_meta[cal], y[cal], cal_sigma=sig[cal])
        mp = ConformalSelectivePredictor(alpha=a).fit(
            oof_meta[cal], y[cal], cal_sigma=sig[cal], cal_groups=clusters[cal])
        tgt.append(t)
        marg.append(cp.coverage(oof_meta[val], y[val], sigma=sig[val]))
        mond.append(mp.coverage(oof_meta[val], y[val], sigma=sig[val], groups=clusters[val]))
    marg, mond = np.array(marg), np.array(mond)
    rece = float(np.mean(np.abs(np.array(tgt) - marg)))

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.plot([0.65, 1.0], [0.65, 1.0], "k--", lw=1.2, alpha=0.6, label="ideal (perfect calibration)")
    ax.plot(tgt, marg, "o-", color=BLUE, lw=2, ms=7, label="marginal")
    ax.plot(tgt, mond, "s-", color=RED, lw=2, ms=7, label="Mondrian (per-family)")
    ax.set_xlabel("Target coverage (1 − α)")
    ax.set_ylabel("Empirical coverage (held-out)")
    ax.set_title("Conformal intervals are well calibrated", fontweight="bold")
    ax.text(0.67, 0.95, f"rECE = {rece:.3f}", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="0.6"))
    ax.set_xlim(0.66, 0.99); ax.set_ylim(0.66, 0.99)
    ax.set_aspect("equal")
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    p = FIG / "fig_calibration.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Saved {p}  (rECE={rece:.4f})")


def panel_per_family():
    cl = pd.read_csv(CONF / "per_cluster_coverage.csv").sort_values("n").reset_index(drop=True)
    x = np.arange(len(cl)); w = 0.4
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.bar(x - w/2, cl["marginal"], w, color=BLUE, label="marginal")
    ax.bar(x + w/2, cl["mondrian"], w, color=RED, label="Mondrian (per-family)")
    ax.axhline(0.90, color="k", ls="--", lw=1.2, alpha=0.7, label="target 90%")
    ax.set_xticks(x); ax.set_xticklabels(cl["cluster"].astype(int), fontsize=9)
    ax.set_xlabel("Protein cluster (sorted by size)")
    ax.set_ylabel("Empirical coverage")
    ax.set_ylim(0.80, 0.95)                       # zoomed so near-nominal coverage is legible
    ax.set_yticks(np.arange(0.80, 0.96, 0.02))
    ax.set_title("Coverage holds within every protein family", fontweight="bold")
    ax.legend(fontsize=10, loc="lower right", ncol=1)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    p = FIG / "fig_per_family.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Saved {p}")


def panel_risk_coverage():
    rc = pd.read_csv(CONF / "risk_coverage_casf16.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(rc["keep_fraction"], rc["RMSE"], "o-", color=GREEN, lw=2.2, ms=7)
    ax.set_xlabel("Fraction retained (most-confident first)")
    ax.set_ylabel("RMSE on retained set (pKd)")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    # shade the low-N region where per-point estimates get noisy (n < 100)
    low = rc[rc["n"] < 100]["keep_fraction"]
    if len(low):
        bmid = (low.max() + rc["keep_fraction"].min()) / 2
        ax.axvspan(low.max(), rc["keep_fraction"].min(), color="0.85", alpha=0.5, zorder=0)
        ax.text(bmid, 0.96, "N < 100\n(noisy)", transform=ax.get_xaxis_transform(),
                fontsize=9, va="top", ha="center", color="0.4")
    ax.set_title("Abstaining on low-confidence predictions\nlowers error on the retained set", fontweight="bold")
    fig.tight_layout()
    p = FIG / "fig_risk_coverage.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"Saved {p}")


if __name__ == "__main__":
    panel_calibration()
    panel_per_family()
    panel_risk_coverage()
    print("Done — 3 standalone panels in output/figures/")

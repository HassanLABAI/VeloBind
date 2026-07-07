#!/usr/bin/env python3
# scripts/velobind_expand.py
#
# Hit-expansion: the VeloBind product feature. Given a library that VeloBind has
# already scored for affinity, sharpen the ranking using a handful of KNOWN ACTIVES
# (and optionally known inactives), then report a shortlist + confidence.
#
#   score(x) = (1-alpha) * z(affinity)  +  alpha * z( simActive(x) - simInactive(x) )
#
# - z() = standardize within the library (so affinity and similarity are comparable).
# - alpha in [0,1]: 0 = pure affinity (the cold / no-actives case), 1 = pure similarity.
# - simActive = max ECFP4 Tanimoto to the known actives; simInactive likewise to inactives.
#   Using known inactives to push down their look-alikes is the one edge over a plain
#   similarity search.
# - If no actives are supplied, falls back to affinity-only and flags low confidence.
#
# This module is pure + testable: it consumes a CSV that already has an `affinity`
# column (produced by the VeloBind affinity pipeline), so it does NOT re-run ESM/GBMs.
#
# CLI:
#   python scripts/velobind_expand.py --scored scored.csv --actives actives.smi \
#       [--inactives inactives.smi] [--alpha 0.5] [--keep_frac 0.01] --out ranked.csv

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

N_BITS = 2048


def ecfp(smiles_list):
    fps = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(str(s))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=N_BITS) if m else None)
    return fps


def max_sim_to_set(query_fps, ref_fps):
    """Max ECFP4 Tanimoto of each query compound to a reference set."""
    ref = [f for f in ref_fps if f is not None]
    out = np.zeros(len(query_fps))
    if not ref:
        return out
    for i, fp in enumerate(query_fps):
        if fp is not None:
            out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, ref))
    return out


def _z(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def expand_score(affinity, sim_active, sim_inactive=None, alpha=0.5):
    """Blend standardized affinity with standardized similarity signal."""
    sim = np.asarray(sim_active, float)
    if sim_inactive is not None:
        sim = sim - np.asarray(sim_inactive, float)
    if alpha <= 0 or sim.std() < 1e-9:
        return _z(affinity)
    return (1.0 - alpha) * _z(affinity) + alpha * _z(sim)


def _read_smi(path):
    return [ln.split()[0] for ln in open(path) if ln.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True,
                    help="CSV with columns: smiles, affinity [, half_width]")
    ap.add_argument("--actives", help=".smi of known actives (optional -> cold mode)")
    ap.add_argument("--inactives", help=".smi of known inactives (optional)")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--keep_frac", type=float, default=0.01)
    ap.add_argument("--out", default="ranked.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.scored)
    q = ecfp(df["smiles"].tolist())

    if args.actives:
        sa = max_sim_to_set(q, ecfp(_read_smi(args.actives)))
        si = max_sim_to_set(q, ecfp(_read_smi(args.inactives))) if args.inactives else None
        df["score"] = expand_score(df["affinity"].values, sa, si, args.alpha)
        df["sim_active"] = sa
        mode = f"hit-expansion (alpha={args.alpha:.2f}, {len(_read_smi(args.actives))} actives)"
    else:
        df["score"] = _z(df["affinity"].values)
        mode = "cold mode — affinity only, LOW CONFIDENCE (no known actives supplied)"

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    k = max(1, int(round(args.keep_frac * len(df))))
    df.to_csv(args.out, index=False)

    print(f"Mode      : {mode}")
    print(f"Scored    : {len(df):,} compounds")
    print(f"Shortlist : top {k:,} (top {args.keep_frac:.1%}) -> hand to docking")
    if "half_width" in df.columns:
        conf = float((df["half_width"] <= df["half_width"].median()).mean())
        print(f"Confidence: ~{conf:.0%} of compounds are above-median confidence "
              f"(narrower conformal interval)")
    print(f"Wrote     -> {args.out}")


if __name__ == "__main__":
    main()

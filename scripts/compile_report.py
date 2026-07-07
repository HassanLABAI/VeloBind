import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
CONF = OUT / "conformal"

# Tuple: (Model, Input, R, RMSE, MAE, SD, CI[concordance]).
LIT_BASELINES = {
    "CASF-2013": [
        ("DeepDTA", "1D seq", 0.662, 1.684, 1.309, 1.686, 0.736),
        ("GraphDTA", "1D seq", 0.670, 1.669, 1.320, 1.670, 0.737),
        ("S2DTA", "1D seq", 0.683, 1.644, 1.338, 1.642, 0.739),
        ("MREDTA", "1D seq", 0.659, 1.699, 1.306, 1.691, 0.739),
        ("IGN", "3D", 0.642, 1.732, 1.319, 1.724, 0.737),
        ("DeepDTAF", "3D", 0.734, 1.535, 1.207, 1.526, 0.767),
        ("MDF-DTA", "3D", 0.730, 1.586, 1.289, 1.536, 0.761),
        ("MMPD-DTA", "3D", 0.763, 1.474, 1.218, 1.453, 0.775),
        ("CAPLA", "3D", 0.765, 1.462, 1.184, 1.449, 0.781),
        ("PocketDTA", "3D", 0.739, 1.277, 0.942, 1.248, 0.777),
        ("HPDAF", "3D", 0.811, 1.248, 1.024, 1.269, 0.809),
    ],
    "CASF-2016": [
        ("DeepDTA", "1D seq", 0.709, 1.584, 1.211, 1.587, 0.759),
        ("GraphDTA", "1D seq", 0.687, 1.638, 1.287, 1.633, 0.747),
        ("S2DTA", "1D seq", 0.728, 1.553, 1.236, 1.542, 0.769),
        ("MREDTA", "1D seq", 0.749, 1.449, 1.108, 1.439, 0.776),
        ("IGN", "3D", 0.758, 1.447, 1.108, 1.417, 0.791),
        ("DeepDTAF", "3D", 0.744, 1.468, 1.123, 1.451, 0.778),
        ("MDF-DTA", "3D", 0.772, 1.386, 1.048, 1.379, 0.788),
        ("MMPD-DTA", "3D", 0.795, 1.342, 1.058, 1.316, 0.795),
        ("CAPLA", "3D", 0.786, 1.362, 1.054, 1.343, 0.797),
        ("PocketDTA", "3D", 0.806, 1.105, 0.861, 1.091, 0.805),
        ("HPDAF", "3D", 0.849, 0.991, 0.766, 0.977, 0.831),
    ],
}


def read_text(p):
    p = Path(p)
    return p.read_text(encoding="utf-8", errors="ignore") if p.exists() else None


def read_csv(p):
    p = Path(p)
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def read_json(p):
    p = Path(p)
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def _velobind_metrics(mfull, split):
    """Pull (R, RMSE, MAE, SD, CI) for one split from long-format metrics_full.csv."""
    if mfull is None:
        return None
    d = {str(r["metric"]): float(r["value"])
         for _, r in mfull.iterrows() if str(r["split"]) == split}
    need = ["R", "RMSE", "MAE", "SD", "CI"]
    return tuple(d[k] for k in need) if all(k in d for k in need) else None


def comparison_table(mfull, split):
    """Literature comparison table for one CASF split, VeloBind row filled from data."""
    base = LIT_BASELINES[split]
    head = "| Model | Input | R | RMSE | MAE | SD | CI |"
    sep = "| --- | --- | --- | --- | --- | --- | --- |"
    rows = [f"| {n} | {i} | {R:.3f} | {rm:.3f} | {ma:.3f} | {sd:.3f} | {ci:.3f} |"
            for n, i, R, rm, ma, sd, ci in base]
    vb = _velobind_metrics(mfull, split)
    if vb is None:
        rows.append("| **VeloBind (Ours)** | **1D seq** | _[metrics_full.csv not found]_ |  |  |  |  |")
        return "\n".join([head, sep] + rows)
    R, rm, ma, sd, ci = vb
    rows.append(f"| **VeloBind (Ours)** | **1D seq** | **{R:.4f}** | **{rm:.3f}** | "
                f"**{ma:.3f}** | **{sd:.3f}** | **{ci:.3f}** |")
    above = [b[0] for b in base if b[2] > R]
    note = (f"\n_VeloBind rank by Pearson R: **#{len(above) + 1}/{len(base) + 1}** "
            f"({'only ' + ', '.join(above) + ' above' if above else 'top method'}; "
            "all higher-ranked methods use 3D structure)._")
    return "\n".join([head, sep] + rows) + note


def md_table(df, max_rows=30, floatfmt=4):
    if df is None or len(df) == 0:
        return "_[not yet run]_"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.{floatfmt}g}" if pd.notna(v) else "")
    head = "| " + " | ".join(map(str, d.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in d.values]
    extra = f"\n_(showing {max_rows} of {len(df)} rows)_" if len(df) > max_rows else ""
    return "\n".join([head, sep] + rows) + extra


def section(title):
    return f"\n\n## {title}\n"


def main():
    L = []
    L.append("# VeloBind — RESULTS (everything in one place)")
    L.append(f"_Auto-compiled {datetime.now():%Y-%m-%d %H:%M} by compile_report.py "
             "from the project's own outputs. Numbers here are the source of truth "
             "for the paper._")

    L.append(section("1. Headline affinity metrics (CASF — regression test only)"))
    L.append("> Report these as a regression test confirming the base model is intact — "
             "NOT as a 'competitive with structure-based' claim.")
    mfull = read_csv(OUT / "metrics_full.csv")
    msum = read_text(OUT / "metrics_summary.txt")
    if mfull is not None:
        L.append(md_table(mfull))
    elif msum:
        L.append("```\n" + msum.strip() + "\n```")
    else:
        L.append("_[not yet run — run 06_casf_eval.py + 07b_full_metrics.py]_")

    L.append("\n**Literature comparison — CASF-2016 (N=285):**")
    L.append(comparison_table(mfull, "CASF-2016"))
    L.append("\n**Literature comparison — CASF-2013 (N=195):**")
    L.append(comparison_table(mfull, "CASF-2013"))

    em = read_json(OUT / "extra_metrics.json")
    if em:
        L.append(f"\n**CASF-2016 scoring power (Pearson R) = {em.get('CASF16_scoring_power_R')}; "
                 f"ranking power (Spearman) = {em.get('CASF16_ranking_power_Spearman')}.**")
    else:
        L.append("\n_Scoring/ranking power [not yet run — velobind_extra_metrics.py]_")

    L.append(section("2. Calibrated uncertainty (the paper's lead contribution)"))
    if em:
        L.append(f"**Regression calibration error (rECE) = {em.get('regression_calibration_error_rECE')}** "
                 "— mean |nominal − empirical coverage| across 95→60% confidence levels "
                 "(the regression analogue of ECE; lower is better).")
        cc = em.get("calibration_curve")
        if cc:
            L.append(md_table(pd.DataFrame(cc)))
    cov = read_text(CONF / "coverage_validation.txt")
    L.append("**Coverage validation (random holdout):**")
    L.append("```\n" + cov.strip() + "\n```" if cov else "_[not yet run — 10_conformal_eval.py]_")
    L.append("\n**Risk–coverage on CASF-2016 (abstain on least-confident):**")
    L.append(md_table(read_csv(CONF / "risk_coverage_casf16.csv")))
    L.append("\n**Per-protein-cluster coverage (Mondrian, nominal 90%):**")
    L.append(md_table(read_csv(CONF / "per_cluster_coverage.csv")))

    L.append(section("3. Protein-family generalization (answers PDBBind-skew critique)"))
    L.append("**By ESM cluster (anonymous):**")
    L.append(md_table(read_csv(CONF / "family_stratified.csv")))
    L.append("\n**By named functional class (from velobind_family_named.py):**")
    fam_named = read_csv(CONF / "family_named.csv")
    L.append(md_table(fam_named) if fam_named is not None
             else "_[not yet run — velobind_family_named.py (needs RCSB network)]_")

    L.append(section("4. Virtual screening & hit-expansion"))
    lit = read_text(OUT / "litpcba" / "summary.txt")
    L.append("**LIT-PCBA summary:**")
    L.append("```\n" + lit.strip() + "\n```" if lit else "_[not yet run — 09_litpcba.py]_")
    L.append("\n**LIT-PCBA per-target:**")
    L.append(md_table(read_csv(OUT / "litpcba" / "per_target_metrics.csv"), max_rows=20))
    L.append("\n**Warm/cold hit-expansion (with known actives — the product mode):**")
    L.append("> The bar is the `EF1_tanimoto` column; `EF1_expand` should match or beat it. "
             "`EF1_affinity` (no support) is the floor.")
    vs_dir = OUT / "vs_eval"
    found_vs = False
    for label, fn in [("LIT-PCBA", "warmcold_litpcba.csv"), ("LIT-PCBA (legacy)", "warmcold_metrics.csv"),
                      ("DUD-E", "warmcold_dude.csv")]:
        df_ = read_csv(vs_dir / fn)
        if df_ is not None:
            found_vs = True
            L.append(f"\n_{label}_ — mean EF1%: affinity={df_['EF1_affinity'].mean():.2f}, "
                     f"tanimoto={df_['EF1_tanimoto'].mean():.2f}, **expand={df_['EF1_expand'].mean():.2f}**, "
                     f"random={df_['EF1_random'].mean():.2f}")
            L.append(md_table(df_, max_rows=25))
    if not found_vs:
        L.append("_[not yet run — velobind_vs_eval.py]_")

    L.append(section("5. Throughput (deployment / scale)"))
    thr = read_text(OUT / "throughput.txt")
    L.append("```\n" + thr.strip() + "\n```" if thr else "_[not yet run — velobind_throughput.py]_")

    L.append(section("6. Temporal holdout (pre-2023 train → 2023–24 test)"))
    temp = read_text(OUT / "temporal" / "summary.txt")
    L.append("```\n" + temp.strip() + "\n```" if temp else "_[not yet run — temporal split + retrain]_")

    L.append(section("7. Reviewer-criticism → evidence map (for the rebuttal/cover letter)"))
    rebut = [
        ("No novel contribution / feature engineering",
         "§2 calibrated family-conditional conformal AD — the contribution beyond feature engineering"),
        ("'Competitive with structure-based' misleading",
         "§1 reported as regression test only; value claim is the §2 coverage–accuracy tool"),
        ("Only CASF; no screening / prospective / independent data",
         "§4 LIT-PCBA + DUD-E enrichment; §6 temporal 2023–24 holdout; retrospective case study"),
        ("kNN AD standard; poly-alanine trivial; no per-family AD",
         "§2 conformal coverage + §3 per-family Mondrian coverage with intervals"),
        ("PDBBind skewed; 'any protein' unsupported",
         "§3 per-protein-family holdout R + coverage"),
        ("Kd/Ki/IC50 aggregation; ESM chunking",
         "assay-type sensitivity + long-protein pooling (optional experiments)"),
        ("No throughput / UQ / workflow",
         "§5 throughput; §2 conformal UQ; hit-expansion workflow demo"),
    ]
    L.append("| Reviewer criticism | Where it's answered |\n| --- | --- |")
    for a, b in rebut:
        L.append(f"| {a} | {b} |")

    L.append(section("8. Status checklist"))
    checks = [
        ("CASF metrics", (OUT / "metrics_full.csv").exists() or (OUT / "metrics_summary.txt").exists()),
        ("Conformal coverage", (CONF / "coverage_validation.txt").exists()),
        ("Per-cluster coverage", (CONF / "per_cluster_coverage.csv").exists()),
        ("Family-stratified R", (CONF / "family_stratified.csv").exists()),
        ("LIT-PCBA raw enrichment", (OUT / "litpcba" / "summary.txt").exists()),
        ("Warm/cold LIT-PCBA", (OUT / "vs_eval" / "warmcold_litpcba.csv").exists()
                               or (OUT / "vs_eval" / "warmcold_metrics.csv").exists()),
        ("Warm/cold DUD-E", (OUT / "vs_eval" / "warmcold_dude.csv").exists()),
        ("Throughput", (OUT / "throughput.txt").exists()),
        ("Temporal holdout", (OUT / "temporal" / "summary.txt").exists()),
    ]
    for name, done in checks:
        L.append(f"- [{'x' if done else ' '}] {name}")

    text = "\n".join(L) + "\n"
    (OUT / "RESULTS.md").write_text(text, encoding="utf-8")
    done = sum(d for _, d in checks)
    print(f"Wrote {OUT/'RESULTS.md'}  ({done}/{len(checks)} experiments present)")


if __name__ == "__main__":
    main()
# VeloBind

**Structure-free protein–ligand binding affinity prediction with calibrated,
protein-family-conditional uncertainty.**

VeloBind predicts binding affinity (pKd) from a protein sequence and a ligand
SMILES string alone — no 3D structure, docking, or GPU required at inference.
Every prediction comes with a calibrated conformal confidence interval whose
coverage holds *within* individual protein families. On CASF-2016 it reaches
Pearson R = 0.847, on par with the best structure-based scoring function, in a
model compact enough to run on a laptop CPU.

## Highlights
- **Inputs:** protein sequence + ligand SMILES. No structure, pocket, or docking.
- **Model:** frozen ESM-2 (35M) embeddings + classical sequence-composition
  descriptors + RDKit fingerprints → gradient-boosted ensemble
  (LightGBM · CatBoost · XGBoost) + RidgeCV stacking + isotonic calibration.
- **Uncertainty:** family-conditional (Mondrian) conformal prediction intervals;
  regression calibration error rECE = 0.009.
- **Compact variant:** a single 12 MB LightGBM model for constrained/local
  deployment (included in `output/single/`).

## Installation
```bash
git clone https://github.com/HassanLABAI/VeloBind
cd VeloBind
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
ESM-2 weights download automatically from Hugging Face on first use.

## Usage
The compact model and its calibrator ship in `output/single/`. Feature assembly
and the prediction path are implemented in `src/features/` and
`scripts/velobind_single.py`; see that script for the exact sequence + SMILES →
pKd + interval workflow. To score a new set, extract features with
`scripts/02_extract_features.py`, then predict with the saved model.

## Repository structure
src/
config.py                  global paths & settings
features/                  protein (ESM-2 + SeqFeat) and ligand (RDKit) featurizers
models/                    ensemble, meta-learner, and conformal calibration
evaluation/                metrics
data/                      loaders + leakage checks
applicability_domain.py    kNN applicability-domain filter
scripts/
01–11                      end-to-end pipeline (data → features → train → eval)
velobind_single.py         compact 12 MB model + conformal
velobind_extra_metrics.py  scoring/ranking power + rECE
velobind_family_named.py   per-protein-class accuracy & coverage
velobind_vs_eval.py        LIT-PCBA / DUD-E screening evaluation
velobind_throughput.py     CPU throughput benchmark
fig_conformal_panels.py    calibration figures
compile_report.py          aggregates everything → output/RESULTS.md
output/single/               the deployable 12 MB model
run_all.sh                   full-pipeline orchestration

## Reproducing the paper
Run the numbered scripts in order (or `bash run_all.sh`), then
`python scripts/compile_report.py` to regenerate `output/RESULTS.md`, which holds
every number reported in the manuscript. The full 45-model ensemble and all
prediction outputs are archived on Zenodo (below); only the compact model is
kept in this repository.

## Data and trained models
- **Full ensemble, prediction outputs, and figures:** Zenodo,
  DOI [10.5281/zenodo.19039903](https://doi.org/10.5281/zenodo.19039903).
- **Benchmark datasets** (PDBBind v2020, CASF-2013/2016, LIT-PCBA, DUD-E) are
  available from their original sources; we provide the PDB-ID split lists and
  feature-extraction scripts to regenerate all derived inputs.

## Citation
> Mathur Y, Hassan MI. *VeloBind: A Compact, Structure-Free Model for Calibrated
> Protein-Ligand Binding Affinity Prediction.* (2026). [journal TBD]

## License
MIT
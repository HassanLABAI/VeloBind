# VeloBind

> **Structure-free protein–ligand binding affinity prediction**  
> Sequence + SMILES only · No 3D co-crystal structure required · Primary screening scale

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HF%20Spaces-VeloBind-blue)](https://huggingface.co/spaces/ym59/velobind)
[![Zenodo](https://img.shields.io/badge/Zenodo-Data%20%26%20Models-blue)](https://doi.org/10.5281/zenodo.19039903)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Virtually all high-performing binding affinity models require a solved 3D protein–ligand complex at inference, a bottleneck that precludes their use at the primary screening stage where millions of compounds must be evaluated before any structural data is available.

**VeloBind** removes this bottleneck entirely. Given only a protein sequence and a ligand SMILES string, it predicts binding affinity (pKd) at ~0.35 seconds per query on CPU, achieving **Pearson R = 0.8485** on CASF-2016, competitive with 3D structure-based methods and state-of-the-art among all sequence-only models.

### Why frozen ESM-2 + gradient boosting?

End-to-end transformer fine-tuning on ~18k complexes leads to catastrophic forgetting and overfitting. Instead:

- **ESM-2 (35M, frozen)** acts as a teacher, distilling evolutionary co-evolutionary knowledge from 250M protein sequences into fixed representations
- **A gradient-boosted ensemble** acts as a sample-efficient student, learning the affinity mapping with strong inductive bias from an extended ligand fingerprint suite

This follows Grinsztajn et al. (NeurIPS 2022) showing GBMs match or exceed fine-tuned transformers on tabular tasks below ~100k samples.

---

## Results

### CASF-2016

| Model | Input | CI | R | MAE | RMSE |
|---|---|---|---|---|---|
| DeepDTA | 1D seq | 0.759 | 0.709 | 1.211 | 1.584 |
| GraphDTA | 1D seq | 0.747 | 0.687 | 1.287 | 1.638 |
| S2DTA | 1D seq | 0.769 | 0.728 | 1.236 | 1.553 |
| MREDTA | 1D seq | 0.776 | 0.749 | 1.108 | 1.449 |
| IGN | 3D | 0.791 | 0.758 | 1.108 | 1.447 |
| DeepDTAF | 3D | 0.778 | 0.744 | 1.123 | 1.468 |
| CAPLA | 3D | 0.797 | 0.786 | 1.054 | 1.362 |
| PocketDTA | 3D | 0.805 | 0.806 | 0.861 | 1.105 |
| HPDAF | 3D | 0.831 | 0.849 | 0.766 | 0.991 |
| **VeloBind** | **1D seq** | **0.828** | **0.8485** | **0.9240** | **1.1977** |

### CASF-2013 (zero-shot)

| Model | Input | CI | R | MAE | RMSE |
|---|---|---|---|---|---|
| DeepDTA | 1D seq | 0.736 | 0.662 | 1.309 | 1.684 |
| MREDTA | 1D seq | 0.739 | 0.659 | 1.306 | 1.699 |
| IGN | 3D | 0.737 | 0.642 | 1.319 | 1.732 |
| DeepDTAF | 3D | 0.767 | 0.734 | 1.207 | 1.535 |
| CAPLA | 3D | 0.781 | 0.765 | 1.184 | 1.462 |
| HPDAF | 3D | 0.809 | 0.811 | 1.024 | 1.248 |
| **VeloBind** | **1D seq** | **0.792** | **0.779** | **1.241** | **1.504** |

> Every model ranked above VeloBind on either benchmark requires a co-crystal complex at inference. VeloBind uses sequence + SMILES only.

---

## Installation

```bash
conda create -n velobind python=3.10
conda activate velobind
pip install -r requirements.txt
```

### Data

Place the following before running any scripts:

```
data/raw/LP_PDBBind.csv        ← from github.com/THGLab/LP-PDBBind
data/external/CASF-2016/       ← from https://www.pdbbind-plus.org.cn/casf
data/external/CASF-2013/       ← from https://www.pdbbind-plus.org.cn/casf
```

---

## Reproducing Results

```bash
# 1. Verify data, check leakage (~2 min)
python scripts/01_check_data.py

# 2. Extract features — ESM-2 embeddings + ligand fingerprints (~2-3 hr)
python scripts/02_extract_features.py

# 3. Ablation + full ensemble training (~4 hr)
python scripts/03_train.py

# 4. Optimise blending strategy
python scripts/03b_save_predictions.py

# 5. SHAP interpretability
python scripts/04_explain.py

# 6. Generate paper figures
python scripts/05_figures.py

# 7. Joint evaluation — CASF-2016 + CASF-2013
python scripts/06_casf_eval.py

# 8. Full metrics + bootstrap CIs
python scripts/07b_full_metrics.py

# 9. Compute applicability domain centroid
python scripts/09_compute_ad.py
```

---

## Web App

A live demonstration app is available on Hugging Face Spaces:

👉 **https://ym59-velobind.hf.space/**

Three modes:
- **Single query** — paste one sequence + one SMILES, get pKd + confidence interval + applicability domain badge
- **Batch screening** — upload a CSV of compounds against one target, download ranked results
- **Selectivity profiling** — one compound against multiple targets

To run locally:
```bash
streamlit run app.py
```

---

## Repository Structure

```
├── scripts/                   pipeline scripts (01–09)
├── src/
│   ├── config.py              hyperparameters and paths
│   ├── data/
│   │   ├── loader.py          LP-PDBBind + CASF parsing
│   │   └── leakage.py         train/test overlap check
│   ├── features/
│   │   ├── protein.py         ESM-2 multi-layer pooling + sequence features
│   │   ├── ligand.py          ECFP + MACCS + AtomPair + Torsion + RDKit
│   │   └── interaction.py     cross-modal PCA interaction block
│   ├── models/
│   │   ├── ensemble.py        multi-seed OOF stacking (LGBM + CatBoost + XGBoost)
│   │   └── meta.py            RidgeCV meta-learner
│   └── evaluation/
│       └── metrics.py         Pearson R, RMSE, MAE, CI + scatter plots
├── app.py                     Streamlit inference app
├── logo.svg
└── requirements.txt
```

---

## Data & Models

Pre-trained models, feature matrices, and predictions are available on Zenodo:

📦 **https://doi.org/10.5281/zenodo.19039903**

| Archive | Contents |
|---|---|
| `velobind_models.zip` | 45 fold models + meta-learner + scalers + AD files |
| `velobind_features.zip` | NPZ feature matrices (train + CASF-2016 + CASF-2013) |
| `velobind_predictions.zip` | Prediction CSVs + full metrics |

---

## Feature Engineering

VeloBind uses a 10,054-dimensional feature vector per complex:

| Block | Features | Dim |
|---|---|---|
| ESM-2 last layer | Mean-pooled frozen embeddings | 480 |
| Sequence features | ProtParam + Dipeptide + CTD + ConjointTriad + QSO + AAIndex | 919 |
| ECFP2/4/6 | Morgan fingerprints (radii 1–3) | 3 × 1024 |
| FCFP | Functional-class Morgan | 1024 |
| MACCS | MACCS keys | 167 |
| AtomPair | Atom-pair fingerprint | 2048 |
| Torsion | Topological torsion | 2048 |
| EState | Electrotopological state indices | 79 |
| RDKit | Physicochemical descriptors | 217 |

---

## Citation

```bibtex
@article{velobind2026,
  title   = {VeloBind: A structure-free protein-ligand binding affinity predictor intended for primary drug screening},
  author  = {},
  journal = {},
  year    = {},
  doi     = {}
}
```

---

## License

MIT License. See [LICENSE](LICENSE).

The LP-PDBBind dataset and CASF benchmarks are subject to their own respective licenses.

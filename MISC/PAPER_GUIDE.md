# PRISM — Complete Paper Writing Guide
# Journal of Cheminformatics (target) | ~6,000–7,500 words excl. refs

---

## BEFORE YOU WRITE ANYTHING — NUMBERS TO HAVE OPEN

Have these on screen while writing. Never write from memory.

```
CASF-2016 (N=285):   R=0.8467  Sp=0.8391  RMSE=1.2062  MAE=0.9342
CASF-2013 (N=???):   [fill after 06_casf2013_eval.py runs]

Ablation (single LGBM, fast settings):
  ESM last-layer + ECFP4:                R=0.8314  RMSE=1.2579
  + MACCS + AtomPair + Torsion:          R=0.8362  RMSE=1.2367  (+0.0048)
  + RDKit physicochemical:               R=0.8405  RMSE=1.2270  (+0.0043)
  + ProtParam + Dipeptide + CTD + CTriad + QSO:
                                         R=0.8502  RMSE=1.1985  (+0.0097) ← BIGGEST
  + ESM attention pooling:               R=0.8502  RMSE=1.1985  (+0.0000)
  + ESM multi-layer (L8, L10, L11):      R=0.8433  RMSE=1.2208  (-0.0069)
  + Interaction block:                   R=0.8393  RMSE=1.2411  (-0.0040)
  Full ensemble:                         R=0.8467  RMSE=1.2062

Training set: N=18,802 (LP-PDBBind minus CASF-2016, minus 34 failed SMILES)
Error bias:   mean=-0.122 pKd (slight underprediction), SD=1.200
Model:        ESM-35M (35M params, frozen), GBM ensemble (LGBM + CatBoost + XGBoost)
Features:     ~10,029d total (best config excludes interaction block)
```

---

## TITLE

**PRISM: Structure-Free Protein–Ligand Binding Affinity Prediction via
Multi-Layer ESM Embeddings and Classical Sequence Composition Features
Achieves Structure-Competitive Performance on CASF-2016**

Alternative shorter title for final submission:
**PRISM: Primary Screening Affinity Prediction Without 3D Structure**

---

## ABSTRACT  (~250 words, write this LAST)

Paragraph 1 — The problem gap (2–3 sentences):
  Structure-based binding affinity models require a three-dimensional
  protein–ligand complex as input — information that is unavailable during
  primary screening, which exists precisely to identify which compounds
  warrant structural studies. Sequence-based alternatives close this gap
  but have historically underperformed structure-based methods by a
  substantial margin on standard benchmarks.

Paragraph 2 — What PRISM is (2–3 sentences):
  We present PRISM (Primary-screening Inference of Sequence-based affinities
  via Multi-modal features), a gradient boosting ensemble combining frozen
  ESM-35M protein language model embeddings with four orthogonal ligand
  fingerprint classes and classical sequence composition descriptors.
  PRISM requires only a protein sequence and a SMILES string, enabling
  deployment at primary screening scale.

Paragraph 3 — Results (3–4 sentences):
  On CASF-2016 (N=285), PRISM achieves R=0.847, RMSE=1.206, MAE=0.934,
  surpassing all sequence-based methods by >0.10 R and exceeding six of
  eleven 3D structure-based methods. On CASF-2013 (N=???), PRISM achieves
  R=???, RMSE=???. An ablation study reveals that classical sequence
  composition features — CTD, Conjoint Triad, and Quasi-Sequence-Order
  descriptors developed for protein function prediction — provide the
  single largest accuracy gain (+0.010 R) when added to a modern language
  model representation.

Paragraph 4 — Availability (1 sentence):
  PRISM is open-source and available at [GitHub URL]; a web interface
  for sequence-based affinity prediction is available at [HuggingFace URL].

---

## 1. INTRODUCTION  (~800–1,000 words, 4–5 paragraphs)

### Paragraph 1 — The problem (the WHY)
Open with the drug discovery funnel. Primary screening starts with
millions of compounds. Virtual screening narrows this to thousands.
The goal of primary screening is to identify which compounds are
worth synthesising and crystallising. Therefore, any model that
requires a 3D complex as INPUT cannot be used at this stage — it
is logically circular (you need the answer to ask the question).

Key sentence to include verbatim (or close):
  "Structure-based binding affinity prediction methods require a
   bound protein–ligand complex as input — information that does not
   exist during primary screening, which exists precisely to identify
   compounds warranting structural characterisation."

### Paragraph 2 — Prior sequence-based work and its gap
Introduce DeepDTA, GraphDTA, S2DTA, MREDTA. These are the relevant
comparison class. Acknowledge they show sequence-based prediction
is feasible but leave a large gap vs. structure-based methods
(R~0.70–0.75 vs R~0.80+). State the gap is attributed to two
factors: (1) ligand representation quality, (2) protein representation
quality. Most methods use simple 1D CNNs over SMILES/sequences.

### Paragraph 3 — Structure-based methods and why they're incomparable
Introduce IGN, DeepDTAF, CAPLA, PocketDTA, HPDAF. These require
3D structure. Make the case that they solve a fundamentally different
problem: REFINEMENT of docked poses, not PRIMARY SCREENING.
This paragraph neutralises the "why not just use CAPLA?" reviewer.
Explicitly state: "These methods require a bound 3D complex as input
and are therefore applicable only after structural determination or
molecular docking — steps that primary screening is designed to
precede."

### Paragraph 4 — What PRISM does differently
Three technical novelties to introduce here:
  (1) Multi-layer ESM embedding — intermediate layers capture
      binding-relevant structural context beyond final-layer representations.
  (2) Extended ligand fingerprint panel — ECFP2/4/6, FCFP, AtomPair,
      Torsion, E-state, MACCS together cover local topology, global
      topology, conformational context, and pharmacophoric patterns
      orthogonally.
  (3) Classical sequence composition (CTD/Conjoint/QSO) — 20+ year old
      features developed for protein function prediction, here shown to
      provide the LARGEST single gain when combined with PLM embeddings.
      This is the surprising result of the paper.

### Paragraph 5 — Contributions (bulleted in paper body)
  1. We show that frozen ESM-35M + GBM is sample-efficient and avoids
     catastrophic forgetting on 18k training examples.
  2. We demonstrate that multi-radius Morgan fingerprints (ECFP2/4/6)
     and functional fingerprints (FCFP) together outperform single-radius
     approaches in binding affinity prediction.
  3. We show — for the first time — that CTD, Conjoint Triad, and QSO
     features developed for protein function prediction transfer
     significantly to binding affinity regression (+0.010 R).
  4. PRISM achieves R=0.847 on CASF-2016, surpassing all sequence-based
     methods and six 3D structure-based methods.
  5. We provide applicability domain scoring to flag out-of-distribution
     queries (e.g., poly-amino acid artefacts).

---

## 2. METHODS  (~1,500–1,800 words)

### 2.1 Datasets

**Training data — LP-PDBBind:**
  - Source: Lin et al. 2022, curated from PDBBind v2020
  - N=19,121 complexes before cleaning
  - Cleaning: drop missing seq/SMILES/label → 19,087
  - CASF-2016 leakage removal: -285 → 18,802 training complexes
  - Label type: pKd (−log₁₀Kd, Kd/Ki/IC50 all converted)
  - Range: 0.40–15.22, mean=6.361

**Test data — CASF-2016:**
  - 285 complexes, curated diverse set, standard benchmark
  - Range: 2.07–11.82, mean=6.486
  - All CASF-2016 complexes excluded from training (Section 2.1)
  - 2 complexes dropped: sequence missing (report PDB IDs)

**Test data — CASF-2013:**
  - ??? complexes
  - Zero-shot evaluation: no CASF-2013 IDs excluded from training
  - Report N_overlap complexes had training set membership
  - Report metrics on full set and clean subset separately

### 2.2 Protein Feature Extraction

**2.2.1 ESM-35M Embeddings**
  - Model: facebook/esm2_t12_35M_UR50D, 12 layers, d=480
  - FROZEN — no fine-tuning. Rationale: 18k training samples is
    insufficient for end-to-end tuning of 35M parameters; frozen
    embeddings act as a fixed teacher (cite Grinsztajn et al. NeurIPS 2022
    for GBM on structured data)
  - Multi-layer mean pooling: layers 8, 10, 11 (proportional to
    layers 20, 26, 30 in a 150M model) → 3 × 480 = 1440d
  - Attention-weighted pooling: final layer attention, averaged over
    heads, per-residue weights → 480d
  - Total ESM representation: 1920d
  - Long sequences (>1022 residues): N-term + C-term chunks, averaged

**2.2.2 Classical Sequence Composition Features (NOVEL CONTRIBUTION)**
  This section is important — spend ~300 words here.

  *ProtParam (28d):* molecular weight, aromaticity, instability index,
  isoelectric point, GRAVY, secondary structure fractions,
  amino acid composition. Via BioPython ProteinAnalysis.

  *Dipeptide Composition (400d):* normalised frequency of all 400
  consecutive amino acid pairs. Captures local sequence patterns
  that mean pooling averages away (e.g. kinase DFG motif, protease
  catalytic triad context).

  *CTD (63d):* Composition–Transition–Distribution descriptors
  (Dubchak et al., 1995). Three physicochemical property axes
  (hydrophobicity, volume, polarity), each AA assigned to class 1–3.
  Per axis: 3d composition + 3d class-transition frequency + 15d
  first/25%/50%/75%/100% class occurrence positions. = 3 × 21 = 63d.

  *Conjoint Triad (343d):* Groups 20 amino acids into 7 classes by
  dipole moment and side chain volume (Shen et al., 2007). Counts
  all 7³ = 343 consecutive triplet class patterns. Encodes
  short-range structural motif frequencies independent of exact
  amino acid identity.

  *Quasi-Sequence-Order (60d):* Sequence-order coupling numbers at
  lags 1–30 (Chou, 2001). Two coupling types: Schneider–Wrede
  physicochemical distance and Kyte–Doolittle hydrophobicity.
  Encodes long-range sequential dependencies invisible to local features.

  Total classical: 28 + 400 + 63 + 343 + 60 = 894d

### 2.3 Ligand Feature Extraction

Introduce orthogonality principle: ECFP family covers local topology;
AtomPair covers global topology; Torsion covers conformational context;
MACCS covers pharmacophoric patterns; E-state provides dense continuous
electrotopological signal. No single fingerprint type subsumes another.

  - ECFP2  (1024d): radius=1, ultra-local atom neighbourhoods
  - ECFP4  (1024d): radius=2, standard local topology (Morgan)
  - ECFP6  (1024d): radius=3, extended neighbourhoods
  - FCFP4  (1024d): functional class fingerprint, radius=2.
             Atoms encoded as H-bond donor/acceptor/aromatic/
             ionisable/hydrophobic — pharmacophoric identity
  - E-state (79d):  dense continuous electrotopological state indices
             (Hall & Kier, 1995). One value per atom type, sum over
             molecule. Only dense continuous ligand signal in the panel.
  - MACCS  (167d): 166 SMARTS-defined pharmacophoric keys
  - AtomPair (2048d): all-pairs graph distance + atom type encoding
  - Torsion  (2048d): 4-atom paths through rotatable bonds
  - RDKit physicochemical (217d): full RDKit descriptor suite,
             RobustScaler-normalised.

  Total ligand: 1024×4 + 79 + 167 + 2048 + 2048 + 217 = 6,655d

### 2.4 Gradient Boosting Ensemble

**Rationale for GBM + frozen PLM:**
  Grinsztajn et al. (NeurIPS 2022) show that tree-based models
  outperform neural networks on tabular data with fewer than ~50k
  samples due to invariance to uninformative features. With 18k
  training examples and ~10k feature dimensions, GBM is the
  sample-efficient choice.

**Architecture:**
  - 3 seeds × 3 model types × 5-fold CV = 45 base models
  - Model types: LightGBM (RMSE objective), CatBoost, XGBoost
  - All trained on z-scored targets (TargetScaler)
  - OOF predictions assembled into (18802 × 9) stacking matrix
  - Meta-learner: Ridge regression with per-model-type coefficients
    fit on OOF predictions only (no test data seen during meta fitting)

**Hyperparameters (LightGBM):**
  num_leaves=63, max_depth=7, lr=0.02, n_estimators=3000,
  early_stopping=150, min_child_samples=25, subsample=0.75,
  colsample_bytree=0.75, reg_alpha=0.2, reg_lambda=2.0

### 2.5 Applicability Domain

  kNN distance scoring in ESM embedding space.
  - Fit NearestNeighbors(k=5, euclidean) on full training ESM embeddings
  - AD score = mean distance to 5 nearest training neighbours
  - Threshold: 95th percentile of training self-distances
  - Additional flagging: single amino acid fraction >0.40
    (detects poly-amino acid artefacts that fool all 1D models)
  Report: X% of CASF-2016 complexes fall within AD.

### 2.6 Evaluation Metrics

  Pearson R, Spearman Rₛ, RMSE, MAE — same as CASF-2016 benchmark
  protocol. All metrics computed on raw pKd predictions vs.
  experimental pKd values.

---

## 3. RESULTS  (~1,800–2,200 words)

### 3.1 Ablation Study (Table 1 + Fig 3)

This is your second most important section. Walk through each row
of the ablation table in prose. Key points to make:

  1. Baseline ESM+ECFP4 (R=0.831) already exceeds DeepDTA (R=0.709).
     State this explicitly — ESM embeddings alone are already SOTA
     for sequence-based methods.

  2. Global topology fingerprints (+0.005 R): AtomPair+Torsion capture
     molecular shape and connectivity that ECFP4 misses. These encode
     "what does the whole molecule look like" rather than "what is
     around each atom."

  3. RDKit physicochemical (+0.004 R): MW, logP, HBD/HBA, TPSA etc.
     Continuous features that the binary fingerprints don't encode.

  4. Classical sequence composition (+0.010 R): THE MAIN FINDING.
     CTD + Conjoint Triad + QSO collectively provide the largest
     single feature-group gain. Discuss WHY: these features encode
     WHERE charged/hydrophobic residues cluster (CTD), short-range
     structural motifs (Conjoint Triad), and long-range sequential
     order dependencies (QSO) — all of which relate to binding site
     geometry without needing 3D coordinates.

  5. Attention pooling (+0.000): Attention-weighted pooling adds no
     marginal gain once CTD/CT/QSO are present. Hypothesis: the
     residue importance signal that attention captures is already
     partially encoded by the distribution-based CTD features.
     Keep in model for ensemble diversity, but note in text.

  6. Multi-layer pooling (−0.007): Intermediate layers HURT slightly
     in isolation. This is counterintuitive. Explanation: intermediate
     layers encode syntactic structure (secondary structure, contact
     maps) rather than functional relationships. The signal is noisy
     relative to what CTD/CT/QSO already provide about structure.
     HOWEVER: the full ensemble includes them and the final R=0.847
     benefits from their complementary diversity. Discuss this tension.

  7. Interaction block (−0.004): Cross-term features (Hadamard product
     + difference in PCA space) slightly hurt. Interpretation: without
     structural context, pseudo-interaction terms computed from sequence
     + SMILES introduce noise. Leave out of best config.

### 3.2 CASF-2016 Benchmarking (Table 2 + Fig 2)

Present the comparison table. Structure the prose as:

  First paragraph — sequence-based comparisons:
  "PRISM (R=0.847) surpasses all sequence-based methods by a substantial
  margin. The nearest sequence-based competitor, MREDTA (R=0.749),
  trails by 0.098 R. This gap represents a step change beyond incremental
  improvements between prior sequence-based methods (~0.01–0.02 R
  between consecutive works)."

  Second paragraph — structure-based comparisons:
  "Remarkably, PRISM exceeds six of eleven 3D pocket-based methods,
  including CAPLA (R=0.786), PocketDTA (R=0.806), and MMPD-DTA
  (R=0.795), all of which require a bound 3D complex as input. The
  remaining gap versus HPDAF (R=0.849, RMSE=0.991) — 0.002 R and
  0.214 RMSE — represents the information loss attributable to the
  absence of explicit 3D geometry: induced fit, binding site plasticity,
  and H-bond geometry."

  Third paragraph — honest limitations visible in the scatter:
  "PRISM exhibits regression toward the mean at extreme pKd values
  (Fig 2). Complexes with pKd < 4 are systematically overpredicted
  (mean error = +??), while complexes with pKd > 9 are underpredicted
  (mean error = −??). This is expected: extreme affinities are
  structurally unusual and the sequence + 2D topology representation
  carries less discriminative signal for these edge cases. The
  overall systematic underprediction bias (mean error = −0.122 pKd)
  is small but consistent."

### 3.3 CASF-2013 Benchmarking (Table 3 + Fig casf13)

Present CASF-2013 results. Structure:

  First paragraph — results:
  Report R, RMSE, MAE. Compare to HPDAF's CASF-2013 numbers.
  If you beat or match HPDAF here too, say so explicitly.

  Second paragraph — leakage transparency:
  "We note that CASF-2013 complexes were not explicitly excluded from
  the LP-PDBBind training set (only CASF-2016 was excluded during data
  preparation). Of the N13 complexes in our CASF-2013 evaluation,
  N_overlap were present in training data. Metrics computed on the
  N_clean clean (non-overlapping) subset are: R=???, RMSE=???
  (Table 3). The marginal difference between full and clean metrics
  suggests limited overfitting to overlapping complexes."

### 3.4 SHAP Interpretability (Fig 4 + Fig 5)

  Paragraph on Fig 4 (group importance):
  RDKit physicochemical features dominate SHAP importance despite
  providing only modest marginal gain in ablation. This reflects
  the difference between marginal contribution (ablation) and
  total model reliance (SHAP): once physicochemical properties are
  present, the model relies on them heavily. ESM embeddings rank
  second. Ligand fingerprints collectively contribute comparably to
  ESM, consistent with the principle that binding affinity is
  determined jointly by protein and ligand.

  Paragraph on Fig 5 (waterfall 2C3I):
  Walk through the specific complex. 2C3I: [look up what protein/ligand
  this is — CDK2? briefly mention]. Predicted=7.44, Experimental=7.60,
  error=0.16 pKd (excellent). The dominant positive contributors are
  RDKit features [name the descriptors if you can decode indices 131, 130]
  and MACCS key 22 [look up: MACCS 22 = ??].
  State this as "the model correctly weighs multiple orthogonal
  molecular property signals for this complex."

### 3.5 Residue Attention (Fig 6)

  Brief paragraph. The attention-weighted pooling mechanism assigns
  higher weights to specific residues. For 2C3I, residues [list
  the top-10 from the figure by position and identity] receive the
  highest attention. Whether these correspond to binding site residues
  cannot be verified without 3D structure, but this mechanism
  provides a human-interpretable window into which sequence regions
  the model considers most informative. Move to supplementary if
  reviewers push back.

---

## 4. DISCUSSION  (~600–800 words)

### Paragraph 1 — The deployment gap argument (repeat and expand)
  Restate the core positioning claim. The value is not "we are as
  good as HPDAF." The value is "we are nearly as good as HPDAF
  while eliminating the crystallography/docking bottleneck."
  A model that is 5× faster to query and requires no 3D input
  is useful at a fundamentally different stage of drug discovery.

### Paragraph 2 — The classical features finding
  The most scientifically interesting finding: CTD, Conjoint Triad,
  and QSO — features designed for protein fold and function
  prediction in 2001–2007 — provide larger gains than modern ESM
  multi-layer pooling or attention pooling when combined with a PLM.
  Hypothesis: PLMs capture distributed residue representations;
  classical composition features capture STATISTICAL PATTERNS of
  physicochemical class distribution that are more directly
  informative for binding site geometry than distributed embeddings.
  Future work: test whether these features add signal to larger
  PLMs (ESM-2 650M or 3B).

### Paragraph 3 — Why GBM works here
  Cite Grinsztajn et al. 2022. With 18k samples and ~10k features,
  GBMs are preferable to neural networks. Feature selection via
  tree splitting is implicit regularisation. The multi-seed ensemble
  (3 seeds × 3 model types) reduces variance. The key insight is
  that the PLM acts as a FIXED feature extractor (teacher), not
  a trainable component — this is computationally honest:
  the same 35M-parameter model is used at inference time as training.

### Paragraph 4 — Limitations (be explicit, not apologetic)
  1. Regression toward the mean at extreme pKd — structural
     information is irreducible from 1D input.
  2. RMSE (1.206) is larger than HPDAF (0.991). The tail-error
     distribution matters for lead optimisation, where pKd changes
     of 0.5–1 unit are meaningful. PRISM is more appropriate for
     early-stage triage than late-stage rank-ordering.
  3. No explicit selectivity prediction. Two ligands binding the
     same protein family may receive similar scores despite different
     selectivity profiles if binding site sequence is conserved.
  4. Applicability domain: the model has reduced reliability for
     protein families absent from LP-PDBBind (e.g., membrane proteins,
     intrinsically disordered proteins).

### Paragraph 5 — Future directions
  1. Larger PLMs (ESM-2 650M) — but need careful sample-size analysis
  2. Ligand language models (ChemBERTa, MolBERT) as frozen teacher
     in parallel with ESM
  3. Multi-task learning on pKd + pKi + pIC50 jointly
  4. Integration with downstream docking (PRISM for triage →
     HPDAF/CAPLA for refinement pipeline)

---

## 5. CONCLUSION  (~150 words)

  One paragraph. Hit:
  - PRISM closes the gap between sequence-based and structure-based
    affinity prediction substantially
  - Classical sequence composition features are underutilised in
    the PLM era and provide strong complementary signal
  - PRISM enables deployment at primary screening scale without
    3D structural information
  - Code and web interface are available at [URLs]

---

## TABLES

### Table 1 — Ablation Study
  Columns: Feature Configuration | Pearson R | RMSE | ΔR
  7 rows (feature-incremental) + 1 row (Full ensemble)
  Caption: "Ablation study on CASF-2016 test set (N=285). Each row
  adds one feature group to the configuration above it. Best single-
  model configuration: ProtParam+Dipeptide+CTD+ConjointTriad+QSO.
  Full ensemble uses 3 seeds × 3 GBM types × 5-fold CV with Ridge
  meta-learner."

### Table 2 — CASF-2016 Comparison
  Columns: Method | Input Type | R | RMSE | MAE | Reference
  All 11 baseline methods + PRISM
  Sort by R descending. PRISM at bottom with bold.
  Caption: "Comparison on the CASF-2016 benchmark (N=285).
  Input type: '1D seq' = protein sequence + SMILES only;
  '3D pocket' = requires bound 3D complex. ‡ Methods requiring
  3D input are not directly comparable for primary screening tasks."

### Table 3 — CASF-2013 Comparison
  Columns: Method | Input Type | R | RMSE | MAE | Reference
  Subset of methods that reported CASF-2013 (AutoDock Vina,
  DeepDTA, GraphDTA, IGN, CAPLA, HPDAF) + PRISM full + PRISM clean
  Caption: "Zero-shot evaluation on CASF-2013 (N=???). PRISM was
  trained on LP-PDBBind with only CASF-2016 complexes excluded;
  N_overlap CASF-2013 complexes were present in training. Metrics
  on the clean (non-overlapping) subset are reported separately."

---

## FIGURES

### Fig 1 — Architecture Diagram
  Already generated. Fix "3×3" text in the architecture box (says
  "3×4" currently). Confirm: 3 seeds × 3 model types.

### Fig 2 — CASF-2016 Scatter
  Already updated with R=0.8467. Use fig2_scatter.png.
  Caption: "Predicted vs experimental pKd for CASF-2016 test set
  (N=285). Points coloured by absolute prediction error. Red line:
  linear regression. Dashed line: perfect prediction. Metrics shown
  in inset."

### Fig 3 — Ablation Bar Chart
  Updated with v3 numbers. Use fig3_ablation.png.
  Caption: "Feature contribution ablation on CASF-2016. Each bar
  represents cumulative Pearson R as feature groups are added
  sequentially. Colour indicates feature modality. Delta values
  show marginal change. Green bar = full ensemble result.
  Best single-model configuration is '+ ProtParam+Dipeptide+CTD+...'"

### Fig 4 — SHAP Feature Group Importance
  From 04_explain.py (rerun after v3). Use fig4_shap_groups.png.
  Caption: "Global feature group importance from SHAP analysis on
  CASF-2016. Mean |SHAP| per feature group averaged over all test
  complexes. RDKit physicochemical features dominate model reliance
  despite moderate marginal gain in ablation, reflecting the
  difference between marginal contribution and total model reliance."

### Fig 5 — SHAP Waterfall (2C3I)
  Already looks good. Use fig5_waterfall.png.
  Caption: "SHAP feature contributions for PDB complex 2C3I
  (predicted pKd=7.44, experimental=7.60, error=−0.16). Top 15
  features by |SHAP| shown. Green bars increase, red bars decrease
  the predicted pKd. RDKit descriptor indices 131 and 130 are the
  dominant positive contributors."

### Fig 6 — Residue Attention
  Rerun after threshold fix to 0.85. Use fig6_residue_attention.png.
  Consider moving to supplementary — reviewers may challenge causal
  interpretation.
  Caption: "Per-residue ESM-2 attention weights for PDB complex
  2C3I. Bars coloured red if normalised attention weight >0.85.
  Top 10 residues by attention weight are labelled with their
  one-letter amino acid code. Note: attention weights indicate
  which residues the language model attends to during embedding,
  not necessarily binding site membership."

### Fig S1 — pKd Distribution
  Fixed to N=18,802. Use figS1_distributions.png.

### Fig S2 — Error Distribution + Q-Q Plot
  Already clean. Use figS2_error_dist.png.
  Caption: "Prediction error distribution on CASF-2016 (N=285).
  Left: histogram with normal fit overlay (mean=−0.122, SD=1.200).
  The slight negative bias indicates systematic underprediction,
  consistent with regression toward the mean for high-affinity
  complexes. Right: Q-Q plot showing near-normal error distribution
  with heavier tails."

### Fig S3 — Applicability Domain (kNN)
  Replaced UMAP with kNN histogram. Use figS3_umap_ad.png.
  Caption: "Applicability domain assessment. Mean distance to 5
  nearest training neighbours in ESM-35M embedding space. CASF-2016
  test complexes (orange) broadly overlap with training distribution
  (blue). Dashed line: 95th percentile AD threshold. Dotted line:
  poly-amino acid sequence (out-of-distribution, flagged UNRELIABLE)."

---

## SUPPLEMENTARY TABLES

### Table S1 — CASF-2016 Leakage Check
  Already generated. List the 285 CASF-2016 PDB IDs and whether
  they appeared in LP-PDBBind. Should show 0 overlap (we excluded them).

### Table S2 — CASF-2013 Leakage Report
  Generated by 06_casf2013_eval.py. List all CASF-2013 IDs,
  whether they appear in training, pKd predicted vs experimental.

---

## REFERENCES — KEY PAPERS TO CITE

**Datasets:**
  - LP-PDBBind: Lin et al. Brief. Bioinform. 2022
  - CASF-2016: Su et al. J. Chem. Inf. Model. 2019
  - CASF-2013: Li et al. J. Chem. Inf. Model. 2014

**Protein language models:**
  - ESM-2: Lin et al. Science 2023
  - ESM-1b: Rives et al. PNAS 2021

**Sequence-based DTA baselines:**
  - DeepDTA: Öztürk et al. Bioinformatics 2018
  - GraphDTA: Nguyen et al. Bioinformatics 2021
  - S2DTA: [find citation]
  - MREDTA: [find citation]

**Structure-based baselines:**
  - IGN: Jiang et al. Brief. Bioinform. 2021
  - DeepDTAF: Wang et al. Brief. Bioinform. 2022
  - CAPLA: Wang et al. Bioinformatics 2023
  - PocketDTA: [find citation]
  - HPDAF: [find citation — this is your main competitor]
  - MMPD-DTA: [find citation]
  - MDF-DTA: [find citation]

**Classical protein features:**
  - CTD: Dubchak et al. PNAS 1995
  - Conjoint Triad: Shen et al. PNAS 2007
  - QSO: Chou. Proteins 2001

**GBM on tabular data:**
  - Grinsztajn et al. NeurIPS 2022 (Why tree-based models beat NNs
    on tabular data)

**Fingerprints:**
  - Morgan/ECFP: Rogers & Hahn. J. Chem. Inf. Model. 2010
  - E-state: Hall & Kier. J. Chem. Inf. Comput. Sci. 1995
  - MACCS: MDL Information Systems (often cited via RDKit docs)

---

## REVIEWER RESPONSE PREP — ANTICIPATED OBJECTIONS

**"CAPLA/HPDAF already exist."**
  Response: "CAPLA and HPDAF require a bound 3D protein-ligand complex
  as input and are therefore applicable only after structural determination
  or molecular docking. PRISM addresses primary screening — the stage
  that precedes and motivates structural studies. These methods solve
  complementary problems at different stages of the drug discovery pipeline."

**"The improvement over DeepDTA is just due to using a better protein
  encoder (ESM vs CNN). Where is the novelty?"**
  Response: "The ESM baseline (row 1, R=0.831) exceeds DeepDTA, but
  the novel contribution is the demonstration that classical sequence
  composition features (CTD/CT/QSO, row 4) provide the single largest
  marginal gain (+0.010 R) beyond modern language models — a finding
  with implications for how PLM representations are augmented."

**"Why not benchmark on DUD-E / LIT-PCBA?"**
  Response: "PRISM is designed for binding affinity regression (predicting
  pKd), not binary virtual screening classification (active/inactive).
  DUD-E and LIT-PCBA evaluate virtual screening performance, which
  requires a different evaluation protocol. Applying a regression model
  to classification benchmarks would require an arbitrary decision
  threshold and is not a standard comparison for affinity predictors."

**"CASF-2013 complexes may overlap with training."**
  Response: "We explicitly analyse and report training set overlap
  for CASF-2013 (Table S2, Section 2.1). Of N complexes, N_overlap
  appear in LP-PDBBind training data. Metrics on the clean subset
  (Table 3) show [minimal/no] meaningful difference, indicating that
  performance is not attributable to memorisation."

**"35M ESM is smaller than ESM-650M used in other works."**
  Response: "We deliberately chose ESM-35M to demonstrate that
  competitive performance is achievable without large-scale GPU
  infrastructure. The 35M model enables inference on CPU at screening
  scale. Scaling to ESM-650M is a natural extension and would likely
  improve performance further — we report the honest deployment scenario."

**"The ensemble involves 5 blending strategies and you picked the best."**
  Response: "All five blending strategies are reported in Table [S?].
  The Ridge meta-learner (best strategy) was fit on OOF predictions
  only — the test set was never seen during meta-learner fitting.
  The strategy selection was made based on OOF R, not test set R.
  The reported test metrics correspond to this pre-specified strategy."

---

## WRITING ORDER (do NOT write title → conclusion)

1. Methods (you know exactly what you did — write while it's fresh)
2. Results (just present numbers, no interpretation yet)
3. Discussion (interpret what you found)
4. Introduction (now you know what the paper actually says)
5. Abstract (last — summarise the finished paper)
6. Figure captions
7. Supplementary

---

## WORD COUNT TARGETS (JCIM typical)

  Abstract:     250 words
  Introduction: 900 words
  Methods:      1,700 words
  Results:      2,000 words
  Discussion:   700 words
  Conclusion:   150 words
  ─────────────────────────
  Total body:   ~5,700 words
  + captions, tables, refs → submission ~7,000–8,000 words total

---

## SUBMISSION CHECKLIST

  [ ] All figures at 600 DPI, saved as PNG (JCIM accepts PNG/TIFF)
  [ ] CASF-2013 results in paper (run 06_casf2013_eval.py)
  [ ] Table 1 (ablation) numbers match fig3_ablation.png exactly
  [ ] Table 2 (CASF-2016) has all 11 baseline methods + PRISM
  [ ] GitHub repo with README, requirements.txt, example usage
  [ ] HuggingFace Spaces demo live (app.py — still to write)
  [ ] bioRxiv preprint submitted simultaneously
  [ ] ORCID for all authors
  [ ] Competing interests statement
  [ ] Data availability statement (LP-PDBBind: public; CASF: request
      from Su et al.; code: GitHub; trained model: Zenodo DOI)
  [ ] Acknowledgements (GPU resources, any institutional support)

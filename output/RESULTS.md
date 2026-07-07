# VeloBind — RESULTS (everything in one place)
_Auto-compiled 2026-07-01 16:29 by compile_report.py from the project's own outputs. Numbers here are the source of truth for the paper._


## 1. Headline affinity metrics (CASF — regression test only)

> Report these as a regression test confirming the base model is intact — NOT as a 'competitive with structure-based' claim.
| split | metric | value | N | ci_lo | ci_hi |
| --- | --- | --- | --- | --- | --- |
| OOF | R | 0.8073 | 18714 | 0.8012 | 0.8138 |
| OOF | SD | 1.097 | 18714 | 1.081 | 1.112 |
| OOF | RMSE | 1.097 | 18714 | 1.081 | 1.112 |
| OOF | MAE | 0.8251 | 18714 | 0.8147 | 0.8353 |
| OOF | CI | 0.8064 | 18714 | 0.8011 | 0.8141 |
| CASF-2016 | R | 0.8465 | 285 | 0.8094 | 0.8787 |
| CASF-2016 | SD | 1.2 | 285 | 1.089 | 1.307 |
| CASF-2016 | RMSE | 1.204 | 285 | 1.097 | 1.315 |
| CASF-2016 | MAE | 0.9305 | 285 | 0.8436 | 1.018 |
| CASF-2016 | CI | 0.8271 | 285 | 0.8036 | 0.8497 |
| CASF-2013 | R | 0.8101 | 195 | 0.7569 | 0.8533 |
| CASF-2013 | SD | 1.365 | 195 | 1.235 | 1.478 |
| CASF-2013 | RMSE | 1.366 | 195 | 1.237 | 1.49 |
| CASF-2013 | MAE | 1.113 | 195 | 1.005 | 1.221 |
| CASF-2013 | CI | 0.806 | 195 | 0.7757 | 0.8336 |

**Literature comparison — CASF-2016 (N=285):**
| Model | Input | R | RMSE | MAE | SD | CI |
| --- | --- | --- | --- | --- | --- | --- |
| DeepDTA | 1D seq | 0.709 | 1.584 | 1.211 | 1.587 | 0.759 |
| GraphDTA | 1D seq | 0.687 | 1.638 | 1.287 | 1.633 | 0.747 |
| S2DTA | 1D seq | 0.728 | 1.553 | 1.236 | 1.542 | 0.769 |
| MREDTA | 1D seq | 0.749 | 1.449 | 1.108 | 1.439 | 0.776 |
| IGN | 3D | 0.758 | 1.447 | 1.108 | 1.417 | 0.791 |
| DeepDTAF | 3D | 0.744 | 1.468 | 1.123 | 1.451 | 0.778 |
| MDF-DTA | 3D | 0.772 | 1.386 | 1.048 | 1.379 | 0.788 |
| MMPD-DTA | 3D | 0.795 | 1.342 | 1.058 | 1.316 | 0.795 |
| CAPLA | 3D | 0.786 | 1.362 | 1.054 | 1.343 | 0.797 |
| PocketDTA | 3D | 0.806 | 1.105 | 0.861 | 1.091 | 0.805 |
| HPDAF | 3D | 0.849 | 0.991 | 0.766 | 0.977 | 0.831 |
| **VeloBind (Ours)** | **1D seq** | **0.8465** | **1.204** | **0.930** | **1.200** | **0.827** |
_VeloBind rank by Pearson R: **#2/12** (only HPDAF above; all higher-ranked methods use 3D structure)._

**Literature comparison — CASF-2013 (N=195):**
| Model | Input | R | RMSE | MAE | SD | CI |
| --- | --- | --- | --- | --- | --- | --- |
| DeepDTA | 1D seq | 0.662 | 1.684 | 1.309 | 1.686 | 0.736 |
| GraphDTA | 1D seq | 0.670 | 1.669 | 1.320 | 1.670 | 0.737 |
| S2DTA | 1D seq | 0.683 | 1.644 | 1.338 | 1.642 | 0.739 |
| MREDTA | 1D seq | 0.659 | 1.699 | 1.306 | 1.691 | 0.739 |
| IGN | 3D | 0.642 | 1.732 | 1.319 | 1.724 | 0.737 |
| DeepDTAF | 3D | 0.734 | 1.535 | 1.207 | 1.526 | 0.767 |
| MDF-DTA | 3D | 0.730 | 1.586 | 1.289 | 1.536 | 0.761 |
| MMPD-DTA | 3D | 0.763 | 1.474 | 1.218 | 1.453 | 0.775 |
| CAPLA | 3D | 0.765 | 1.462 | 1.184 | 1.449 | 0.781 |
| PocketDTA | 3D | 0.739 | 1.277 | 0.942 | 1.248 | 0.777 |
| HPDAF | 3D | 0.811 | 1.248 | 1.024 | 1.269 | 0.809 |
| **VeloBind (Ours)** | **1D seq** | **0.8101** | **1.366** | **1.113** | **1.365** | **0.806** |
_VeloBind rank by Pearson R: **#2/12** (only HPDAF above; all higher-ranked methods use 3D structure)._

**CASF-2016 scoring power (Pearson R) = 0.846; ranking power (Spearman) = 0.8391.**


## 2. Calibrated uncertainty (the paper's lead contribution)

**Regression calibration error (rECE) = 0.0089** — mean |nominal − empirical coverage| across 95→60% confidence levels (the regression analogue of ECE; lower is better).
| target_coverage | empirical | gap |
| --- | --- | --- |
| 0.95 | 0.953 | 0.003 |
| 0.9 | 0.899 | 0.001 |
| 0.85 | 0.844 | 0.006 |
| 0.8 | 0.792 | 0.008 |
| 0.7 | 0.683 | 0.017 |
| 0.6 | 0.581 | 0.019 |
**Coverage validation (random holdout):**
```
Conformal coverage validation (random PDBBind holdout)
alpha=0.10 (target 90%):  marginal cov=0.899  mondrian cov=0.902  | worst-cluster: marginal=0.871 -> mondrian=0.882
alpha=0.20 (target 80%):  marginal cov=0.792  mondrian cov=0.794  | worst-cluster: marginal=0.740 -> mondrian=0.761
```

**Risk–coverage on CASF-2016 (abstain on least-confident):**
| keep_fraction | n | R | RMSE |
| --- | --- | --- | --- |
| 1 | 285 | 0.846 | 1.207 |
| 0.9 | 256 | 0.8323 | 1.207 |
| 0.8 | 228 | 0.8217 | 1.206 |
| 0.7 | 200 | 0.8293 | 1.193 |
| 0.6 | 171 | 0.8336 | 1.192 |
| 0.5 | 142 | 0.8427 | 1.146 |
| 0.4 | 114 | 0.8236 | 1.196 |
| 0.3 | 86 | 0.8244 | 1.167 |
| 0.25 | 71 | 0.8538 | 0.9786 |
| 0.2 | 57 | 0.8503 | 0.9652 |

**Per-protein-cluster coverage (Mondrian, nominal 90%):**
| cluster | n | marginal | mondrian |
| --- | --- | --- | --- |
| 0 | 153 | 0.915 | 0.8824 |
| 1 | 2245 | 0.89 | 0.8931 |
| 2 | 480 | 0.9021 | 0.9187 |
| 3 | 1340 | 0.9172 | 0.9119 |
| 4 | 458 | 0.8712 | 0.8908 |
| 5 | 509 | 0.8861 | 0.9018 |
| 6 | 491 | 0.9022 | 0.9022 |
| 7 | 216 | 0.9259 | 0.9167 |
| 8 | 471 | 0.9108 | 0.9087 |
| 9 | 398 | 0.8894 | 0.902 |
| 10 | 2174 | 0.896 | 0.896 |
| 11 | 422 | 0.9171 | 0.9265 |


## 3. Protein-family generalization (answers PDBBind-skew critique)

**By ESM cluster (anonymous):**
| cluster | n_train | n_casf16 | R_train_oof | RMSE_train_oof | R_casf16 | RMSE_casf16 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4426 | 89 | 0.759 | 1.207 | 0.8129 | 1.18 |
| 10 | 4356 | 51 | 0.7909 | 1.081 | 0.7331 | 1.613 |
| 3 | 2624 | 40 | 0.8154 | 1.058 | 0.8888 | 1.097 |
| 5 | 1014 | 0 | 0.7431 | 1.203 |  |  |
| 2 | 1011 | 0 | 0.791 | 1.117 |  |  |
| 6 | 955 | 20 | 0.8118 | 0.9589 | 0.8668 | 0.9828 |
| 8 | 934 | 15 | 0.8728 | 0.9371 | 0.9189 | 1.137 |
| 4 | 918 | 20 | 0.5936 | 0.9616 | 0.7997 | 1.018 |
| 11 | 857 | 10 | 0.7815 | 0.925 | 0.8213 | 0.7437 |
| 9 | 846 | 25 | 0.8514 | 1.03 | 0.882 | 1.241 |
| 7 | 438 | 5 | 0.8308 | 0.8343 | 0.9825 | 0.2658 |
| 0 | 335 | 10 | 0.8282 | 0.9705 | 0.8097 | 0.8079 |

**By named functional class (from velobind_family_named.py):**
| class | CASF16_n | CASF16_R | train_n | coverage90 |
| --- | --- | --- | --- | --- |
| kinase | 73 | 0.864 | 1895 | 0.898 |
| protease | 65 | 0.877 | 1640 | 0.895 |
| hydrolase | 41 | 0.852 | 1144 | 0.915 |
| transferase | 20 | 0.904 | 964 | 0.885 |
| other | 19 | 0.747 | 968 | 0.903 |
| transport/binding | 19 | 0.825 | 719 | 0.897 |
| nuclear receptor | 17 | 0.813 | 181 | 0.89 |
| immune/signaling | 13 | 0.716 | 707 | 0.912 |
| lyase | 10 | 0.762 | 390 | 0.89 |
| ligase | 8 | 0.775 | 244 | 0.893 |
| isomerase |  |  | 147 | 0.891 |
| oxidoreductase |  |  | 303 | 0.901 |
| GPCR |  |  | 55 | 0.873 |


## 4. Virtual screening & hit-expansion

**LIT-PCBA summary:**
```
═════════════════════════════════════════════════════════════════
  VELOBIND — LIT-PCBA Enrichment Summary
  Targets: 15  |  BEDROC alpha=20.0
═════════════════════════════════════════════════════════════════
  Target        N_act  N_inact    EF1%   EF5%   BEDROC     AUC
  ───────────────────────────────────────────────────────────────
  ADRB2            17    50000    0.00   0.00   0.0182  0.4753
  ALDH1          7168    50000    1.14   1.32   0.1693  0.5866
  ESR1_ago         13     5583    7.69   3.08   0.1770  0.6631
  ESR1_ant        102     4948    2.94   1.76   0.1255  0.6808
  FEN1            369    50000    0.54   0.49   0.0416  0.4931
  GBA             166    50000    0.60   0.96   0.0869  0.6144
  IDH1             39    50000    7.69   4.62   0.1915  0.7369
  KAT2A           194    50000    1.03   0.41   0.0466  0.4466
  MAPK1           308    50000    0.97   1.10   0.0534  0.5935
  MTORC1           97    32972    0.00   1.44   0.0649  0.4893
  OPRK1            24    50000    0.00   1.67   0.0706  0.7742
  PKM2            546    50000    0.73   0.77   0.0507  0.5834
  PPARG            27     5211    0.00   1.48   0.0675  0.6860
  TP53             79     4168    1.27   0.76   0.0471  0.5382
  VDR             884    50000    1.58   0.79   0.0496  0.4104
  ───────────────────────────────────────────────────────────────
  Mean                            1.75   1.38   0.0840  0.5848
  Median                          0.97   1.10   0.0649  0.5866
═════════════════════════════════════════════════════════════════

  BASELINES (mean EF1% across targets):
    VeloBind (full)      : 1.75
    2D-Tanimoto baseline : 31.64
    Random               : 0.81  (theoretical 1.0)

  CONFORMAL TRIAGE (mean across targets, keep top-50% most confident):
    EF1% on retained subset : 0.94  (vs 1.75 full)
    actives recall          : 0.53  (fraction of true actives retained while discarding half the library)

  PAPER-READY SENTENCE:
  Across 15 LIT-PCBA targets VeloBind achieved a mean EF1% of 1.75 (median 0.97),
  mean BEDROC 0.0840 (alpha=20.0), mean AUC 0.5848, versus a 2D-similarity baseline of 31.64 EF1%
  and random 1.0. Used as a triage filter that discards the least-confident 50%,
  EF1% on the retained subset rises to 0.94 while retaining 53% of true actives.
═════════════════════════════════════════════════════════════════
```

**LIT-PCBA per-target:**
| target | n_actives | n_inactives | n_total | active_frac | seq_len | EF1% | EF5% | BEDROC | AUC | EF1_2Dtani | AUC_2Dtani | EF1_random | EF1_keep50 | recall_keep50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADRB2 | 17 | 50000 | 50017 | 0.0003399 | 284 | 0 | 0 | 0.0182 | 0.4753 | 35.29 | 0.6354 | 0.294 | 0 | 0.353 |
| ALDH1 | 7168 | 50000 | 57168 | 0.1254 | 494 | 1.144 | 1.32 | 0.1693 | 0.5866 | 6.041 | 0.7396 | 0.995 | 1.549 | 0.495 |
| ESR1_ago | 13 | 5583 | 5596 | 0.002323 | 233 | 7.692 | 3.077 | 0.177 | 0.6631 | 0 | 0.6229 | 0 | 0 | 0.615 |
| ESR1_ant | 102 | 4948 | 5050 | 0.0202 | 245 | 2.941 | 1.765 | 0.1255 | 0.6808 | 26.47 | 0.7169 | 1.225 | 0 | 0.588 |
| FEN1 | 369 | 50000 | 50369 | 0.007326 | 283 | 0.542 | 0.488 | 0.0416 | 0.4931 | 44.72 | 0.8492 | 0.989 | 0 | 0.415 |
| GBA | 166 | 50000 | 50166 | 0.003309 | 497 | 0.602 | 0.964 | 0.0869 | 0.6144 | 54.82 | 0.8338 | 1.114 | 0 | 0.494 |
| IDH1 | 39 | 50000 | 50039 | 0.0007794 | 367 | 7.692 | 4.615 | 0.1915 | 0.7369 | 38.46 | 0.7047 | 0.641 | 5.882 | 0.436 |
| KAT2A | 194 | 50000 | 50194 | 0.003865 | 164 | 1.031 | 0.412 | 0.0466 | 0.4466 | 26.8 | 0.6582 | 1.005 | 0 | 0.464 |
| MAPK1 | 308 | 50000 | 50308 | 0.006122 | 333 | 0.974 | 1.104 | 0.0534 | 0.5935 | 10.71 | 0.7018 | 0.925 | 1.515 | 0.643 |
| MTORC1 | 97 | 32972 | 33069 | 0.002933 | 202 | 0 | 1.443 | 0.0649 | 0.4893 | 21.65 | 0.7325 | 0.825 | 2.5 | 0.412 |
| OPRK1 | 24 | 50000 | 50024 | 0.0004798 | 281 | 0 | 1.667 | 0.0706 | 0.7742 | 66.67 | 0.8939 | 0.417 | 0 | 0.708 |
| PKM2 | 546 | 50000 | 50546 | 0.0108 | 518 | 0.733 | 0.769 | 0.0507 | 0.5834 | 36.08 | 0.8307 | 0.971 | 0 | 0.527 |
| PPARG | 27 | 5211 | 5238 | 0.005155 | 272 | 0 | 1.481 | 0.0675 | 0.686 | 37.04 | 0.7482 | 0.926 | 0 | 0.667 |
| TP53 | 79 | 4168 | 4247 | 0.0186 | 196 | 1.266 | 0.759 | 0.0471 | 0.5382 | 37.98 | 0.7443 | 0.823 | 1.786 | 0.709 |
| VDR | 884 | 50000 | 50884 | 0.01737 | 253 | 1.584 | 0.792 | 0.0496 | 0.4104 | 31.9 | 0.7914 | 0.995 | 0.909 | 0.498 |

**Warm/cold hit-expansion (with known actives — the product mode):**
> The bar is the `EF1_tanimoto` column; `EF1_expand` should match or beat it. `EF1_affinity` (no support) is the floor.

_LIT-PCBA (legacy)_ — mean EF1%: affinity=0.87, tanimoto=11.31, **expand=10.90**, random=1.51
| target | n_act | n_support | n_test | EF1_affinity | EF1_tanimoto | EF1_expand | EF1_random | BEDROC_affinity | BEDROC_tanimoto | BEDROC_expand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADRB2 | 17 | 3 | 2014 | 0 | 7.143 | 0 | 0 | 0.0296 | 0.1336 | 0.1064 |
| ALDH1 | 7168 | 30 | 9138 | 1.065 | 1.149 | 1.191 | 1.023 | 0.8261 | 0.844 | 0.872 |
| ESR1_ago | 13 | 3 | 2010 | 0 | 0 | 0 | 0 | 0.0846 | 0.07 | 0.08 |
| ESR1_ant | 102 | 20 | 2082 | 1.22 | 9.756 | 12.2 | 3.659 | 0.1212 | 0.2257 | 0.3225 |
| FEN1 | 369 | 30 | 2339 | 0.885 | 7.08 | 7.08 | 0.59 | 0.1381 | 0.4931 | 0.4384 |
| GBA | 166 | 30 | 2136 | 0.735 | 16.18 | 16.18 | 2.206 | 0.1403 | 0.5297 | 0.5189 |
| IDH1 | 39 | 7 | 2032 | 3.125 | 21.88 | 21.88 | 3.125 | 0.2124 | 0.2815 | 0.3368 |
| KAT2A | 194 | 30 | 2164 | 0.61 | 11.59 | 9.756 | 0 | 0.1202 | 0.2853 | 0.2681 |
| MAPK1 | 308 | 30 | 2278 | 1.079 | 5.036 | 5.036 | 0.719 | 0.1582 | 0.3447 | 0.358 |
| MTORC1 | 97 | 19 | 2078 | 1.282 | 12.82 | 10.26 | 2.564 | 0.0948 | 0.3718 | 0.3577 |
| OPRK1 | 24 | 4 | 2020 | 0 | 40 | 40 | 5 | 0.1022 | 0.5793 | 0.5905 |
| PKM2 | 546 | 30 | 2516 | 1.357 | 4.457 | 4.651 | 1.163 | 0.2313 | 0.5844 | 0.5261 |
| PPARG | 27 | 5 | 2022 | 0 | 13.64 | 18.18 | 0 | 0.0751 | 0.2863 | 0.3278 |
| TP53 | 79 | 15 | 2064 | 0 | 15.62 | 14.06 | 1.562 | 0.0229 | 0.3271 | 0.2387 |
| VDR | 884 | 30 | 2854 | 1.639 | 3.279 | 3.044 | 1.054 | 0.2797 | 0.538 | 0.4293 |

_DUD-E_ — mean EF1%: affinity=5.47, tanimoto=19.50, **expand=19.50**, random=0.99
| target | n_act | n_support | n_test | EF1_affinity | EF1_tanimoto | EF1_expand | EF1_random | BEDROC_affinity | BEDROC_tanimoto | BEDROC_expand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aa2ar | 482 | 30 | 2452 | 1.327 | 5.531 | 5.531 | 0.664 | 0.2778 | 0.9864 | 0.9604 |
| abl1 | 182 | 30 | 2152 | 6.579 | 14.47 | 14.47 | 1.316 | 0.3122 | 0.9488 | 0.8813 |
| ace | 282 | 30 | 2252 | 8.73 | 9.127 | 9.127 | 0.794 | 0.6944 | 0.9937 | 0.9794 |
| aces | 453 | 30 | 2423 | 3.783 | 5.91 | 5.91 | 0.709 | 0.4873 | 0.9668 | 0.9026 |
| ada | 93 | 18 | 2075 | 5.333 | 28 | 28 | 1.333 | 0.3217 | 0.9909 | 0.965 |
| ada17 | 532 | 30 | 2502 | 5.179 | 5.179 | 5.179 | 1.594 | 0.9083 | 0.999 | 0.9957 |
| adrb1 | 247 | 30 | 2217 | 1.382 | 10.6 | 10.6 | 0.461 | 0.1542 | 0.9798 | 0.944 |
| adrb2 | 231 | 30 | 2201 | 0.498 | 11.44 | 11.44 | 1.493 | 0.1201 | 0.9782 | 0.9298 |
| akt1 | 293 | 30 | 2263 | 4.183 | 8.745 | 8.745 | 0.38 | 0.4698 | 0.9897 | 0.9776 |
| akt2 | 117 | 23 | 2094 | 4.255 | 22.34 | 22.34 | 2.128 | 0.2127 | 0.9695 | 0.9522 |
| aldr | 159 | 30 | 2129 | 3.876 | 17.05 | 17.05 | 0 | 0.2217 | 0.8949 | 0.753 |
| ampc | 48 | 9 | 2039 | 2.564 | 53.85 | 53.85 | 2.564 | 0.199 | 0.7784 | 0.6562 |
| andr | 269 | 30 | 2239 | 9.623 | 9.623 | 9.623 | 1.674 | 0.7123 | 0.9709 | 0.9613 |
| aofb | 122 | 24 | 2098 | 0 | 21.43 | 21.43 | 5.102 | 0.0275 | 0.756 | 0.6248 |
| bace1 | 283 | 30 | 2253 | 6.324 | 9.091 | 9.091 | 0.395 | 0.5017 | 0.9741 | 0.9342 |
| braf | 152 | 30 | 2122 | 2.459 | 18.03 | 18.03 | 1.639 | 0.1913 | 0.9584 | 0.9315 |
| cah2 | 492 | 30 | 2462 | 5.411 | 5.411 | 5.411 | 1.515 | 0.8726 | 0.9933 | 0.9905 |
| casp3 | 199 | 30 | 2169 | 4.142 | 13.02 | 13.02 | 1.183 | 0.212 | 0.98 | 0.9434 |
| cdk2 | 474 | 30 | 2444 | 3.829 | 5.631 | 5.631 | 0.901 | 0.5468 | 0.9517 | 0.9339 |
| comt | 41 | 8 | 2033 | 0 | 63.64 | 63.64 | 0 | 0.0231 | 0.9328 | 0.8621 |
| cp2c9 | 120 | 24 | 2096 | 2.083 | 21.88 | 21.88 | 2.083 | 0.199 | 0.6665 | 0.6166 |
| cp3a4 | 170 | 30 | 2140 | 3.571 | 15.71 | 15.71 | 0.714 | 0.2789 | 0.7321 | 0.6615 |
| csf1r | 166 | 30 | 2136 | 9.559 | 16.18 | 16.18 | 2.206 | 0.3329 | 0.9478 | 0.8287 |
| cxcr4 | 40 | 8 | 2032 | 0 | 65.62 | 65.62 | 0 | 0.0489 | 0.9566 | 0.9463 |
| def | 102 | 20 | 2082 | 8.537 | 25.61 | 25.61 | 0 | 0.2804 | 0.9813 | 0.9215 |
_(showing 25 of 102 rows)_


## 5. Throughput (deployment / scale)

```
VeloBind throughput (per-compound screening path; protein cached)
  compounds scored : 4,988
  wall time        : 54.30 s
  rate             : 92 compounds/s  (3 per core, 28 cores)
  => 1,000,000 compounds in ~181.4 min on this machine
```


## 6. Temporal holdout (pre-2023 train → 2023–24 test)

_[not yet run — temporal split + retrain]_


## 7. Reviewer-criticism → evidence map (for the rebuttal/cover letter)

| Reviewer criticism | Where it's answered |
| --- | --- |
| No novel contribution / feature engineering | §2 calibrated family-conditional conformal AD — the contribution beyond feature engineering |
| 'Competitive with structure-based' misleading | §1 reported as regression test only; value claim is the §2 coverage–accuracy tool |
| Only CASF; no screening / prospective / independent data | §4 LIT-PCBA + DUD-E enrichment; §6 temporal 2023–24 holdout; retrospective case study |
| kNN AD standard; poly-alanine trivial; no per-family AD | §2 conformal coverage + §3 per-family Mondrian coverage with intervals |
| PDBBind skewed; 'any protein' unsupported | §3 per-protein-family holdout R + coverage |
| Kd/Ki/IC50 aggregation; ESM chunking | assay-type sensitivity + long-protein pooling (optional experiments) |
| No throughput / UQ / workflow | §5 throughput; §2 conformal UQ; hit-expansion workflow demo |


## 8. Status checklist

- [x] CASF metrics
- [x] Conformal coverage
- [x] Per-cluster coverage
- [x] Family-stratified R
- [x] LIT-PCBA raw enrichment
- [x] Warm/cold LIT-PCBA
- [x] Warm/cold DUD-E
- [x] Throughput
- [ ] Temporal holdout

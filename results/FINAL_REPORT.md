# MXene gravimetric capacitance: SHAP descriptor analysis and
# Bayesian-network synthetic data augmentation

Source corpus: `C:\Users\ASUS\Desktop\mxene-geometry\terra\h2so4_corpus_final.csv`  |  target: `target_cap_F_g` (F/g)  |  random seed 42  |  all evaluation grouped by `paper_id`.

---

## 1. Dataset

- Rows in the raw corpus: **192** across **83** columns.
- Unique source papers in the raw corpus: **49** (median 3 rows per paper, max 15).
- Rows with a usable single-valued gravimetric capacitance: **118** from **40** papers.

Target dispositions (nothing is hidden - the full per-row log is in `01_target_disposition_log.csv`):

| disposition | n |
| --- | ---: |
| `kept:exact` | 116 |
| `dropped:no_value_reported` | 70 |
| `dropped:inequality` | 4 |
| `kept:approximate` | 2 |

Capacitance range in the modelling set: 7.26 - 1609.5 F/g (median 266.5, mean 284.5, SD 158.0). 2 value(s) fall outside a physically typical 20-1500 F/g band and are retained but flagged.

### Descriptor availability among the modelled rows

| descriptor | non-missing | missingness | distinct values |
| --- | ---: | ---: | ---: |
| `interlayer_A` | 88 | 25% | 54 |
| `scan_rate_mV_s` | 88 | 25% | 8 |
| `current_density_A_g` | 35 | 70% | 4 |
| `mass_loading_mg_cm2` | 55 | 53% | 34 |
| `electrode_thickness_um` | 54 | 54% | 43 |
| `ssa_m2_g` | 35 | 70% | 35 |
| `flake_size_um` | 11 | 91% | 7 |
| `pore_diameter_nm` | 4 | 97% | 4 |
| `h2so4_M` | 111 | 6% | 4 |
| `layer_class` | 95 | 19% | 3 |
| `composition_family` | 118 | 0% | 6 |
| `synthesis_family` | 118 | 0% | 1 |
| `electrolyte_is_gel` | 118 | 0% | 2 |

Key missingness problems: gas-sorption surface area, gravimetric current density, flake size and pore diameter are reported by only a minority of papers, and lateral flake size is usually given qualitatively ("micrometer-sized", "several") or as a range, which this pipeline refuses to convert into a number. `synthesis_family` collapses to a single level (LiF/HCl MILD) among the rows that survive target cleaning, i.e. it carries no information in this cohort.

### Target-leakage audit

14 of 83 raw columns are admissible as predictors (`00_leakage_audit.csv`). Explicitly excluded as leakage or post-hoc information:

| column | reason |
| --- | --- |
| `volumetric_capacitance` | LEAKAGE: C_vol = C_grav x electrode density - a direct algebraic transform of the target. |
| `areal_capacitance` | LEAKAGE: C_areal = C_grav x mass loading - a direct algebraic transform of the target. |
| `role` | Annotation of the electrode's role in its own paper (primary/control/variant); assigned partly BECAUSE of the measured performance - post-hoc, not knowable before the experiment. |
| `is_primary` | Same as `role`: the 'primary' electrode is usually the best-performing one, i.e. selected on the outcome. |
| `year` | Publication metadata; a proxy for which paper a row came from (paper-level leakage under grouped CV). |
| `overall_reasoning` | Free-text extraction rationale; may quote the capacitance value verbatim. |
| `gravimetric_capacitance__conf` | Directly attached to the target column (`gravimetric_capacitance`). |
| `gravimetric_capacitance__src` | Directly attached to the target column (`gravimetric_capacitance`). |

No energy-density or power-density columns exist in this corpus, so the classic `E = 0.5*C*V^2` leakage route is absent; the analogous routes that *are* present are volumetric and areal capacitance, which are algebraic transforms of the target and are removed.

## 2. Real-only model

XGBoost regressor, 13 descriptors, 5 folds used (40 unique papers available). Hyper-parameters were tuned by `RandomizedSearchCV` inside an **inner** `GroupKFold` on the training papers of each outer fold only; the outer test papers never influenced model selection, encoding, or imputation.

| fold | train rows | test rows | test papers | R2 | MAE (F/g) | RMSE (F/g) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 94 | 24 | 8 | 0.151 | 61.3 | 90.4 |
| 2 | 94 | 24 | 8 | 0.099 | 134.7 | 278.7 |
| 3 | 94 | 24 | 8 | -0.092 | 90.3 | 115.7 |
| 4 | 95 | 23 | 8 | -0.102 | 52.4 | 73.9 |
| 5 | 95 | 23 | 8 | 0.095 | 69.0 | 87.5 |
| **mean** | | | | **0.030** | **81.5** | **129.2** |
| **SD** | | | | 0.119 | 32.830 | 84.924 |
| **median** | | | | 0.095 | 69.0 | 90.4 |

Pooled over all held-out predictions: R2 = 0.081, MAE = 81.9 F/g, RMSE = 150.8 F/g.

Fold-to-fold variation is large (R2 spans -0.102 to 0.151). That spread is the honest signal here: with 40 papers, whole-paper hold-out means each fold tests a different corner of a heterogeneous literature, and a single unusual paper dominates its fold.

Sensitivity to the two physically extreme capacitances (7.26 and 1609.5 F/g): recomputing the fold metrics with those held-out points excluded from the *scoring* only (they remain in training, nothing is deleted) gives R2 = 0.012 +/- 0.110, MAE = 70.1 F/g, RMSE = 93.2 F/g (vs 0.030, 81.5, 129.2 with them). Full table in `02_real_only_outlier_sensitivity.csv`.

Figures: `figures/real_only_predicted_vs_actual.png`, `figures/real_only_residuals.png`.

## 3. SHAP on real data

SHAP values are computed with `shap.TreeExplainer` on the **held-out** rows of each outer fold, so the attribution describes generalisation behaviour rather than training fit. One-hot columns are reported individually and aggregated back to the source variable (per-row sum of |SHAP| over a variable's dummies).

| rank | descriptor | mean abs SHAP (F/g) | SD across folds | mean rank | rank SD | folds in top-5 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `h2so4_M` | 25.66 | 9.29 | 1.4 | 0.89 | 5/5 |
| 2 | `composition_family` | 22.80 | 9.38 | 2.4 | 0.89 | 5/5 |
| 3 | `scan_rate_mV_s` | 16.91 | 5.92 | 2.8 | 1.10 | 5/5 |
| 4 | `interlayer_A` | 13.04 | 8.48 | 4.0 | 1.58 | 4/5 |
| 5 | `layer_class` | 9.13 | 2.61 | 4.8 | 0.45 | 5/5 |
| 6 | `ssa_m2_g` | 7.59 | 4.87 | 6.4 | 1.52 | 1/5 |
| 7 | `electrode_thickness_um` | 6.08 | 3.37 | 7.0 | 1.00 | 0/5 |
| 8 | `current_density_A_g` | 3.28 | 1.17 | 7.8 | 1.48 | 0/5 |
| 9 | `mass_loading_mg_cm2` | 2.10 | 0.93 | 9.6 | 0.89 | 0/5 |
| 10 | `electrolyte_is_gel` | 1.81 | 1.41 | 10.0 | 2.00 | 0/5 |

### Direction of the continuous descriptors

Read off the pooled held-out SHAP values as the Spearman correlation between a descriptor's value and its own SHAP contribution. A positive value means the model pushes the prediction up as the descriptor increases.

| descriptor | rho(value, SHAP) | model behaviour | physically expected? |
| --- | ---: | --- | --- |
| `h2so4_M` | +0.70 | raises the prediction as it grows | yes - more protons means more Ti-O pseudocapacitance |
| `scan_rate_mV_s` | -0.84 | lowers the prediction as it grows | yes - faster sweeps starve the interlayer of ions |
| `interlayer_A` | -0.79 | lowers the prediction as it grows | **contradicts the physical expectation** - a wider gallery should ease ion access |
| `ssa_m2_g` | +0.82 | raises the prediction as it grows | yes - more accessible area, though BET is a weak proxy in restacked films |
| `electrode_thickness_um` | -0.08 | no monotone effect | flat - no claim |
| `current_density_A_g` | +0.04 | no monotone effect | flat - no claim |
| `mass_loading_mg_cm2` | +0.22 | raises the prediction as it grows | **contradicts the physical expectation** |
| `flake_size_um` | -0.61 | lowers the prediction as it grows | plausible - longer in-plane diffusion path |

Rank stability across folds is in the `rank SD` and `folds in top-5` columns above: a descriptor with a low mean rank but a high rank SD is being driven by one or two papers, not by a consistent trend. **Caution:** `interlayer_A`, `mass_loading_mg_cm2` run against the physical expectation. In a literature corpus that usually means confounding rather than physics - which paper a measurement came from is entangled with the descriptor value. For `interlayer_A` specifically: the largest reported d-spacings belong mostly to pillared and composite architectures whose *gravimetric* capacitance is diluted by the intercalant mass, and d-spacing is typically measured on a dry film rather than in the hydrated, polarised state that actually stores charge. For `mass_loading_mg_cm2`: groups that push high loadings tend to be the ones with an architecture good enough to survive it, so loading partly encodes electrode quality rather than transport penalty. Both are concrete reasons not to read a SHAP direction as a causal law, and concrete things for a PINN to settle with a mechanistic term instead of a fitted one.

Figures: `figures/real_shap_bar.png`, `figures/real_shap_beeswarm.png`, `figures/shap_dependence/`.

## 4. Bayesian-network synthesis

Generator: **DataSynthesizer 0.1.13, correlated-attribute mode** (greedy Bayes network, epsilon = 0 - the inputs are already-published literature values, so no differential-privacy noise is warranted). No substitute generator (CTGAN, copula, SMOTE, KDE) was used anywhere.

Selected Bayesian-network descriptors (5 + target):

| descriptor | mean abs SHAP | SHAP rank | missingness | BN treatment | why |
| --- | ---: | ---: | ---: | --- | --- |
| `h2so4_M` | 25.66 | 1 | 6% | categorical | Electrolyte proton activity; governs pseudocapacitive Ti-O redox and double-layer screening. |
| `composition_family` | 22.80 | 2 | 0% | categorical | Pristine vs carbon / polymer / inorganic composite vs doped MXene. |
| `scan_rate_mV_s` | 16.91 | 3 | 25% | categorical | CV sweep rate; controls the diffusion length probed and therefore rate-limited capacitance loss. |
| `interlayer_A` | 13.04 | 4 | 25% | continuous | XRD (002) d-spacing: sets the ion-accessible gallery height; the central geometric descriptor for a PINN. |
| `layer_class` | 9.13 | 5 | 19% | categorical | Delamination state (multilayer / delaminated / single-or-few-layer). |
| `target_cap_F_g` | - | target | 0% | continuous | Gravimetric capacitance (F/g). |

Discrete-vs-continuous choice: `h2so4_M` and `scan_rate_mV_s` are experimenter *settings* that occur at a handful of values, so they are declared categorical to the Bayesian network. Treating them as continuous would let the generator invent molarities and sweep rates that no experiment in the corpus used.

BN cohort (complete cases on those descriptors + target): **47 rows / 18 papers**, down from 118 rows / 40 papers. This is the price of requiring every selected descriptor to be reported in the same publication. **All four training conditions below use exactly this cohort and exactly this feature set**, so the comparison is like-for-like.

### Anti-leakage protocol for generation

For every outer grouped fold: the Bayesian network is fitted on the *training papers only*; synthetic rows are drawn from that network; the physical-validity filter uses training-fold minima/maxima only; the one-hot encoder and the hyper-parameter search also see training papers only. The held-out papers are never described, never generated from, never augmented, and never appear in any training set.

Physical-validity filtering across all folds/k/ratios: **6474 rows drawn, 0 rejected (0.0%)**; the generator is asked for a small surplus over the required count and the surplus is discarded after filtering. Rejection reasons:

**No synthetic row was rejected.** That is not a sign the filter is inactive - it reflects how DataSynthesizer samples: continuous attributes are drawn uniformly inside histogram bins whose outer edges are the training-fold min and max, and categorical attributes are drawn from the observed level set, so the generator cannot leave the training domain by construction. The filter (non-positive capacitance, negative descriptors, unseen categories, out-of-range values) remains in the pipeline as a guard that would catch a change of generator or configuration.

Example learned network (fold 1, k=2): `[["target_cap_F_g", ["h2so4_M"]], ["interlayer_A", ["target_cap_F_g", "h2so4_M"]], ["scan_rate_mV_s", ["interlayer_A", "target_cap_F_g"]], ["composition_family", ["scan_rate_mV_s", "target_cap_F_g"]], ["layer_class", ["scan_rate_mV_s", "target_cap_F_g"]]]`. Full set in `05_bayesian_network_structures.csv`.

## 5. Synthetic-data fidelity

Every number below compares synthetic rows against the **real training rows of the same fold** (never the test papers). Averages are over folds and augmentation ratios at k = 2.

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| mean KS statistic (continuous) | 0.1545 | 0.09459 | 0.1982 |
| max KS statistic (continuous) | 0.1967 | 0.0991 | 0.2368 |
| # continuous vars with KS p < 0.05 | 0 | 0 | 0 |
| mean Jensen-Shannon distance (categorical) | 0.07896 | 0.02759 | 0.17 |
| Pearson corr-matrix mean abs difference | 0.04656 | 0.002495 | 0.3458 |
| Spearman corr-matrix mean abs difference | 0.09504 | 0.0009416 | 0.2917 |
| Mutual-information matrix mean abs difference | 0.05545 | 0.02458 | 0.09306 |
| descriptor-target association mean abs difference | 0.0648 | 0.01027 | 0.2629 |
| exact duplicates of real rows | 0 | 0 | 0 |
| duplicates within synthetic | 0 | 0 | 0 |
| synthetic->real NN distance (median) | 0.006458 | 0.004596 | 0.008607 |
| real->real NN distance (median) | 0.01947 | 0.01297 | 0.03506 |
| synthetic rows closer than the real-NN 5th pct | 3.467 | 0 | 19 |

**Verdict.** Marginals: mean KS statistic 0.154 with on average 0.0 of 2 continuous variables rejected at p<0.05 - marginal distributions are reproduced well. Categorical frequencies: mean Jensen-Shannon distance 0.079 (close agreement). Descriptor-target relationships: mean absolute change of 0.065 in the descriptor-target association (Spearman for continuous descriptors, eta-squared for categorical ones) - the relationships a regressor has to learn survive generation. Dependency structure: mean absolute difference of 0.047 in the Pearson correlation matrix and 0.055 in the normalised mutual-information matrix. Pairwise dependencies, including the descriptor-target relationships, are broadly preserved. Memorisation: 0 synthetic rows across all folds were exact copies of a real training row and 0 were duplicated within a synthetic set. The median synthetic-to-real nearest-neighbour distance is 0.38x the median real-to-real nearest-neighbour distance; a ratio below 1 is expected here rather than alarming, because 37-200 synthetic points are drawn over the same manifold as ~38 real ones and nearest-neighbour distances shrink with density alone. The sharper diagnostic is how many synthetic rows fall closer to a real record than the 5th percentile of real-to-real spacing: on average 3.5 of 116 rows (3.0%), a small tail rather than wholesale copying.

### Descriptor-target relationships

A generator can reproduce every marginal and still destroy the relationship the regressor has to learn, so the descriptor-target association is checked explicitly: Spearman correlation for continuous descriptors, eta-squared (share of capacitance variance explained by the level) for categorical ones. Averaged over folds at k = 2, ratio 3x:

| descriptor | association measure | real | synthetic | abs diff |
| --- | --- | ---: | ---: | ---: |
| `interlayer_A` | Spearman with target | -0.275 | -0.244 | 0.071 |
| `h2so4_M` | eta-squared on target | 0.149 | 0.129 | 0.054 |
| `composition_family` | eta-squared on target | 0.694 | 0.651 | 0.063 |
| `scan_rate_mV_s` | eta-squared on target | 0.120 | 0.131 | 0.031 |
| `layer_class` | eta-squared on target | 0.017 | 0.042 | 0.029 |

Note on the correlation matrices: only one BN descriptor (`interlayer_A`) is modelled as continuous, so the Pearson/Spearman matrix reduces to a single off-diagonal entry and a 'correlation of correlations' would be undefined. The normalised mutual-information matrix, which handles the categorical descriptors as well, is the informative dependency comparison here.

Per-variable detail: `05_synthetic_distribution_stats.csv`; dependency structure: `05_mutual_information_comparison.csv` and `05_target_relationship_comparison.csv`; memorisation: `05_nearest_neighbor_analysis.csv`. Figures in `figures/synthetic_vs_real/`.

## 6. Real-only vs augmented performance

Outer evaluation: 5 folds used (18 unique papers available). The test set of each fold is the identical set of **real** held-out papers for all four conditions, with the same feature space and the same XGBoost hyper-parameters (tuned once per fold on the real training papers, then held fixed across conditions so that tuning cannot confound the comparison).

Each augmented condition is generated **3 times with independent generator seeds** inside every fold; the fold score is the mean over those draws, so re-drawing the synthetic set cannot masquerade as an effect. The spread caused by re-drawing alone is reported in the last column.

| Training condition | Test data | n synthetic (mean/fold) | R2 mean +/- SD | MAE mean +/- SD | RMSE mean +/- SD | MAE SD from re-drawing |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Real only | unseen real papers | 0 | -1.906 +/- 2.667 | 129.6 +/- 68.0 | 189.1 +/- 149.3 | n/a (no generation) |
| Real + 1x synthetic | unseen real papers | 38 | -1.876 +/- 2.275 | 130.9 +/- 65.4 | 190.2 +/- 148.6 | 11.1 |
| Real + 3x synthetic | unseen real papers | 113 | -2.530 +/- 3.427 | 138.7 +/- 69.8 | 200.8 +/- 150.3 | 18.9 |
| Real + 5.25x synthetic | unseen real papers | 198 | -1.380 +/- 1.479 | 123.6 +/- 65.5 | 186.1 +/- 153.5 | 6.9 |

**Read this column first.** Simply re-drawing the synthetic set with a different seed moves fold MAE by 12.3 F/g on average, while the entire spread between the four training conditions is 15.1 F/g. Any apparent ranking of the augmentation ratios has to be read against that noise floor.

**Why the R2 values are negative.** R2 is measured against the variance of each *fold's own* held-out capacitances. Several folds contain only 3-4 papers whose capacitances sit within a narrow band (fold 1 spans 100-300 F/g, SD ~ 60 F/g), so even a model with a 60-90 F/g MAE scores below the fold mean. The absolute-error metrics are the trustworthy comparators on a cohort this small; R2 is reported because it was requested, and because its *ordering* across conditions is still informative.

Medians and pooled metrics (pooled = every held-out real prediction from every fold scored together, which avoids dividing by a near-zero within-fold variance while keeping the grouped hold-out completely intact):

| condition | R2 median | MAE median | RMSE median | pooled R2 | pooled MAE | pooled RMSE | pooled R2 (excl. extremes) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Real only | -0.336 | 94.5 | 111.8 | -0.160 | 129.5 | 228.6 | -1.040 |
| Real + 1x synthetic | -0.328 | 116.6 | 132.2 | -0.167 | 130.8 | 229.2 | -1.054 |
| Real + 3x synthetic | -0.407 | 102.3 | 121.3 | -0.279 | 138.6 | 239.9 | -1.526 |
| Real + 5.25x synthetic | -0.357 | 99.4 | 122.5 | -0.153 | 122.8 | 227.8 | -0.813 |

The pooled ranking agrees with the fold-mean ranking: best condition by pooled R2 is **Real + 5.25x synthetic**, by fold-mean R2 **Real + 5.25x synthetic**.

### Paired statistical comparison

| comparison | metric | real-only | augmented | abs diff | rel % | folds improved | Wilcoxon p | paired-t p |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B_real_plus_1.0x | r2 | -1.906 | -1.876 | +0.029 | +1.5% | 2/5 | 0.812 | 0.960 |
| B_real_plus_1.0x | mae | 129.564 | 130.949 | +1.385 | +1.1% | 2/5 | 0.812 | 0.856 |
| B_real_plus_1.0x | rmse | 189.067 | 190.208 | +1.141 | +0.6% | 2/5 | 0.812 | 0.890 |
| C_real_plus_3.0x | r2 | -1.906 | -2.530 | -0.625 | -32.8% | 0/5 | 0.062 | 0.147 |
| C_real_plus_3.0x | mae | 129.564 | 138.698 | +9.134 | +7.1% | 0/5 | 0.062 | 0.018 |
| C_real_plus_3.0x | rmse | 189.067 | 200.824 | +11.757 | +6.2% | 0/5 | 0.062 | 0.013 |
| D_real_plus_5.25x | r2 | -1.906 | -1.380 | +0.526 | +27.6% | 1/5 | 0.625 | 0.487 |
| D_real_plus_5.25x | mae | 129.564 | 123.573 | -5.991 | -4.6% | 3/5 | 0.812 | 0.551 |
| D_real_plus_5.25x | rmse | 189.067 | 186.102 | -2.965 | -1.6% | 1/5 | 0.625 | 0.809 |

**Bottom line.** Augmentation did not improve prediction on unseen real papers. The nominally best condition (5.25x) lowers fold-mean MAE by at most 6.0 F/g, which is inside the 12.3 F/g that re-drawing the synthetic set moves the same score, and it improves only 3/5 folds. Condition(s) 1x, 3x are worse than real-only training. Read together with the k-sensitivity table below - where changing the network degree alone moves MAE as much as changing the ratio does - the conclusion is that Bayesian-network augmentation neither helps nor systematically harms this cohort; it adds variance without adding information. Reported as found: no configuration search was run to make the synthetic data look useful.

With 5 paired folds the smallest attainable two-sided Wilcoxon p-value is 0.0625, so **no result here can reach p < 0.05 by that test**. The p-values are reported for completeness only; the fold-level direction of change and the effect size are the meaningful quantities.

### Bayesian-network complexity sensitivity

| k | condition | R2 mean +/- SD | MAE mean | RMSE mean |
| ---: | --- | ---: | ---: | ---: |
| 1 | Real + 1x synthetic | -2.109 +/- 2.158 | 141.6 | 203.7 |
| 2 | Real + 1x synthetic | -1.876 +/- 2.275 | 130.9 | 190.2 |
| 3 | Real + 1x synthetic | -1.669 +/- 2.104 | 127.6 | 182.6 |
| 1 | Real + 3x synthetic | -1.076 +/- 1.603 | 120.9 | 176.4 |
| 2 | Real + 3x synthetic | -2.530 +/- 3.427 | 138.7 | 200.8 |
| 3 | Real + 3x synthetic | -1.210 +/- 1.947 | 120.9 | 173.6 |
| 1 | Real + 5.25x synthetic | -1.101 +/- 1.910 | 121.3 | 179.3 |
| 2 | Real + 5.25x synthetic | -1.380 +/- 1.479 | 123.6 | 186.1 |
| 3 | Real + 5.25x synthetic | -0.766 +/- 1.323 | 109.5 | 162.2 |

Changing only the network degree at a fixed ratio moves MAE by up to 17.8 F/g, while changing the ratio at a fixed degree moves it by 15.1 F/g. The two spreads are the same order, which is the clearest single indication that the whole augmentation effect on this cohort sits inside configuration noise. No `k` is reliably better than another; k = 2 is reported as the pre-registered default, not as a winner.

Figures: `figures/augmentation_r2.png`, `figures/augmentation_mae.png`, `figures/augmentation_rmse.png` (+ per-fold line versions).

## 7. SHAP stability after augmentation

Comparison is cohort-real-only vs **Real + 5.25x synthetic** (the best-performing augmented condition on unseen real papers), on the identical cohort and feature set.

- Spearman rank correlation of source-level SHAP rankings: **0.900** (p = 0.037)
- Top-3 overlap: **3/3**; top-5 overlap: **5/5**
- Mean |rank change|: 0.40; max |rank change|: 1
- Mean |change| in normalised mean-|SHAP| share: 0.018

| descriptor | real rank | augmented rank | rank change | real SHAP share | augmented SHAP share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h2so4_M` | 1 | 2 | +1 | 0.245 | 0.248 |
| `composition_family` | 2 | 1 | -1 | 0.242 | 0.269 |
| `interlayer_A` | 3 | 3 | +0 | 0.215 | 0.229 |
| `scan_rate_mV_s` | 4 | 4 | +0 | 0.214 | 0.193 |
| `layer_class` | 5 | 5 | +0 | 0.084 | 0.061 |

The augmented model is driven by essentially the same descriptors as the real-only model: the top-3 and top-5 sets are preserved, the average rank displacement is 0.4 positions and each descriptor's share of total attribution moves by only 0.018 on average. Synthetic augmentation did not rewrite the model's physics story.

Figure: `figures/shap_rank_comparison.png`.

## 8. PINN descriptor implications

SHAP measures how a *particular fitted model* distributes credit; it is not evidence of physical causation. A descriptor is recommended for the PINN only when strong real-data SHAP importance, cross-fold stability, post-augmentation stability, adequate coverage and materials-science plausibility all hold.

| descriptor | real SHAP rank | augmented rank | stability | missingness | interpretability | recommendation |
| --- | ---: | ---: | --- | ---: | --- | --- |
| `composition_family` | 2 | 1 | stable | 0% | low/medium - a categorical bucket, not a physical parameter | **conditional - needs a continuous physical parameterisation** |
| `layer_class` | 5 | 5 | stable | 20% | medium - qualitative stand-in for the delamination state | **conditional - needs a continuous physical parameterisation** |
| `ssa_m2_g` | 6 | - | not evaluated (not in BN cohort) | 70% | medium - proxy for wetted area, poorly correlated with it in restacked films | **not recommended yet** |
| `electrode_thickness_um` | 7 | - | not evaluated (not in BN cohort) | 54% | high - explicit geometry | **not recommended yet** |
| `current_density_A_g` | 8 | - | not evaluated (not in BN cohort) | 70% | high - galvanostatic boundary condition | **not recommended yet** |
| `mass_loading_mg_cm2` | 9 | - | not evaluated (not in BN cohort) | 53% | high - sets the transport domain length scale | **not recommended yet** |
| `electrolyte_is_gel` | 10 | - | not evaluated (not in BN cohort) | 0% | medium - changes ionic conductivity and confinement | **not recommended yet** |
| `flake_size_um` | 11 | - | not evaluated (not in BN cohort) | 91% | medium - in-plane diffusion length | **not recommended yet** |
| `pore_diameter_nm` | 12 | - | not evaluated (not in BN cohort) | 97% | medium - only meaningful for engineered architectures | **not recommended yet** |
| `synthesis_family` | 13 | - | not evaluated (not in BN cohort) | 0% | low - processing label | **not recommended yet** |
| `h2so4_M` | 1 | 2 | stable | 6% | high - enters Nernst/Poisson-Nernst-Planck terms directly | **recommended** |
| `scan_rate_mV_s` | 3 | 4 | stable | 25% | high - sets the timescale in the transport PDE | **recommended** |
| `interlayer_A` | 4 | 3 | stable | 25% | high - direct geometric input to an ion-transport PINN | **recommended** |

**Sufficiently supported now:** `h2so4_M`, `scan_rate_mV_s`, `interlayer_A`.

**Conditional - well evidenced but categorical, so they must be recast as a continuous physical quantity before entering a PDE residual (e.g. `composition_family` -> intercalant mass fraction, `layer_class` -> measured stacking number):** `composition_family`, `layer_class`.

**Too sparse or unreliable at present:** `ssa_m2_g`, `electrode_thickness_um`, `current_density_A_g`, `mass_loading_mg_cm2`, `electrolyte_is_gel`, `flake_size_um`, `pore_diameter_nm`, `synthesis_family`.

## 9. Limitations

- **Literature-derived, heterogeneous data.** Every row is an LLM-assisted extraction from a different publication with its own cell geometry, reference electrode, mass-normalisation convention and reporting rate. Two rows with identical descriptors can legitimately differ by a hundred F/g.
- **Multiple observations per paper.** Rows are strongly clustered by publication (median ~3, up to 15 rows per paper). All evaluation is grouped by `paper_id` for exactly this reason; any random-split number would be optimistically biased and none is reported here.
- **Missing descriptor values dominate the design.** Requiring complete cases on 5 descriptors shrinks the corpus from 118 to 47 rows. The Bayesian-network experiment therefore describes a *sub-population* of the literature, not the whole corpus.
- **Synthetic data are not new experimental evidence.** Every synthetic row is a resample of dependencies already present in the training papers. Augmentation can regularise a model; it cannot add information the literature does not contain.
- **Bayesian networks reproduce learned dependencies, not physics.** The generator has no notion of charge conservation, ion transport or electrode kinetics; the physical-validity filter is a coarse guard (sign and training-domain bounds), not a physics check.
- **SHAP is attribution, not causation.** A high SHAP value means the fitted XGBoost model leans on that column, which can equally reflect a confounded reporting habit (e.g. groups who measure d-spacing also make better electrodes) as a causal mechanism.
- **Limited extrapolation validity.** Synthetic values are explicitly clipped to the training-fold support, and the model is only credible inside the descriptor ranges the corpus actually covers (e.g. H2SO4 0.5-3 M).
- **Small number of folds.** With a handful of grouped folds, paired significance testing is underpowered by construction; conclusions rest on effect direction and consistency, not on p-values.
- **Unit and definition ambiguity survives cleaning.** Some papers report an interlayer *gap* rather than a (002) d-spacing, and some report flake size only qualitatively. Ambiguous entries are dropped rather than guessed, which biases the cohort toward better-documented papers.

---

## 10. Summary answers

**A.** **Usable observations:** 118 of 192 corpus rows carried a defensible single-valued gravimetric capacitance. 70 rows reported no value at all and 4 reported only a range or an inequality, which were dropped rather than converted to midpoints.

**B.** **Papers represented:** 40 unique publications among the modelled rows (49 in the raw corpus); the Bayesian-network cohort covers 18 of them.

**C.** **Real-only XGBoost:** grouped 5-fold CV gives R2 = 0.030 +/- 0.119 (median 0.095), MAE = 81.5 +/- 32.8 F/g, RMSE = 129.2 +/- 84.9 F/g. Fold-to-fold spread is wide (R2 -0.10 to 0.15).

**D.** **Top real-data SHAP descriptors:** `h2so4_M`, `composition_family`, `scan_rate_mV_s`, `interlayer_A`, `layer_class`.

**E.** **Bayesian-network variables:** `h2so4_M`, `composition_family`, `scan_rate_mV_s`, `interlayer_A`, `layer_class` plus the target `target_cap_F_g`, on a complete-case cohort of 47 rows / 18 papers.

**F.** **Fidelity:** Marginals: mean KS statistic 0.154 with on average 0.0 of 2 continuous variables rejected at p<0.05 - marginal distributions are reproduced well. Categorical frequencies: mean Jensen-Shannon distance 0.079 (close agreement). Descriptor-target relationships: mean absolute change of 0.065 in the descriptor-target association (Spearman for continuous descriptors, eta-squared for categorical ones) - the relationships a regressor has to learn survive generation. Dependency structure: mean absolute difference of 0.047 in the Pearson correlation matrix and 0.055 in the normalised mutual-information matrix. Pairwise dependencies, including the descriptor-target relationships, are broadly preserved. Memorisation: 0 synthetic rows across all folds were exact copies of a real training row and 0 were duplicated within a synthetic set. The median synthetic-to-real nearest-neighbour distance is 0.38x the median real-to-real nearest-neighbour distance; a ratio below 1 is expected here rather than alarming, because 37-200 synthetic points are drawn over the same manifold as ~38 real ones and nearest-neighbour distances shrink with density alone. The sharper diagnostic is how many synthetic rows fall closer to a real record than the 5th percentile of real-to-real spacing: on average 3.5 of 116 rows (3.0%), a small tail rather than wholesale copying.

**G.** **Best augmentation ratio:** D_real_plus_5.25x (fold-mean R2 -1.380 and MAE 123.6 F/g, against -1.906 / 129.6 F/g for real-only). The ranking is not monotone in the amount of synthetic data (3x is the *worst* condition here), and the gap between the best and worst ratio (15.1 F/g MAE) is comparable to the 12.3 F/g that re-drawing the same configuration moves the score. More synthetic data is not automatically better, and on this cohort no ratio is reliably better than any other.

**H.** **No material improvement - the honest answer is that augmentation did not help.** The best augmented condition (D_real_plus_5.25x) shifted fold-mean MAE from 129.6 to 123.6 F/g (-4.6%) and pooled MAE from 129.5 to 122.8 F/g, improving MAE in 3/5 folds. That shift is inside the 12.3 F/g that simply re-drawing the synthetic set with a different generator seed moves the same score, and far inside the fold-to-fold spread (MAE SD 68.0 F/g). The ranking is not even monotone in the amount of synthetic data - the 3x condition is the worst of the four (MAE 138.7 F/g). No search was run to make synthetic data look beneficial, and none of the differences approach significance.

**I.** **SHAP stability:** Spearman rank correlation 0.900, top-3 overlap 3/3, top-5 overlap 5/5, mean |rank change| 0.40. The augmented model is driven by essentially the same descriptors as the real-only model: the top-3 and top-5 sets are preserved, the average rank displacement is 0.4 positions and each descriptor's share of total attribution moves by only 0.018 on average. Synthetic augmentation did not rewrite the model's physics story.

**J.** **PINN descriptors:** ready to use as physical inputs now - `h2so4_M`, `scan_rate_mV_s`, `interlayer_A`; well supported by the data but categorical, so they need recasting as continuous physical quantities before they can enter a PDE residual (`composition_family`, `layer_class`). See `08_pinn_descriptor_recommendation.csv`.

**K.** **Remaining limitations:** heterogeneous literature-derived rows, heavy clustering within papers, severe descriptor missingness that forces a small complete-case cohort, a near-constant synthesis route in the surviving rows, synthetic rows that carry no new experimental information, and a fold count too small for meaningful significance testing. Section 9 lists these in full.

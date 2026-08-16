# 00 - Inventory of existing pipeline outputs

Generated 2026-08-16 14:15 by `run_final_stage.py`.

## Provenance and version selection

`results/` contains exactly one generation of outputs: the whole pipeline is re-run from scratch by `run_pipeline.py` (it deletes and rebuilds `results/`), so there are no competing versions of any file to adjudicate between. All files carry timestamps from the same run, and `results/environment.txt` records the package versions and the three DataSynthesizer compatibility patches that run used.

One methodological point does need stating, because it is easy to conflate two different real-only models that both exist in `results/`:

| Model | Rows | Papers | Descriptors | Grouped-CV $R^2$ | Purpose |
| --- | ---: | ---: | ---: | ---: | --- |
| Full real-only baseline (`02_*`) | 118 | 40 | 13 | 0.030 +/- 0.119 | headline generalisation estimate |
| BN-cohort real-only (`06_*`, condition A) | 47 | 18 | 5 | -1.906 +/- 2.667 | control arm for the augmentation experiment only |

The strongly negative figure quoted in earlier summaries (-1.91) belongs to the **47-row complete-case BN cohort**, not to the 118-row corpus. The full real-only baseline reaches $R^2 = 0.030$. Both are weak; they are not the same number and this report keeps them apart throughout.

## Outputs reused unchanged

| File | Role in this stage |
| --- | --- |
| `results/01_cleaned_model_data.csv` | Cleaned modelling table (118 usable rows). Reused as the single source of truth for the real corpus. |
| `results/01_target_disposition_log.csv` | Per-row target disposition. Reused for the dataset-accounting figure. |
| `results/02_real_only_fold_metrics.csv` | Paper-grouped CV of the full-descriptor real-only model. Reused verbatim as the generalisation estimate. |
| `results/02_real_only_summary.csv` | Aggregated grouped-CV metrics including pooled scores. Reused. |
| `results/02_real_only_predictions.csv` | Held-out predictions per fold. Reused for the prediction figure. |
| `results/03_real_shap_importance.csv` | Fold-wise held-out SHAP. Reused for the cross-fold rank-stability columns. |
| `results/04_selected_bn_features.json` | Bayesian-network feature set and cohort definition. Reused. |
| `results/04_bn_real_cohort.csv` | 47-row complete-case BN cohort. Reused. |
| `results/05_synthetic_fidelity_summary.csv` | Per-fold synthetic fidelity metrics. Reused. |
| `results/05_synthetic_distribution_stats.csv` | Per-variable marginal comparisons. Reused for the fidelity figure. |
| `results/05_mutual_information_comparison.csv` | Real vs synthetic MI matrices. Reused for the fidelity figure. |
| `results/06_augmentation_summary.csv` | Four-condition augmentation comparison. Reused verbatim. |
| `results/06_augmentation_per_seed_metrics.csv` | Per-generator-seed fold metrics. Reused for the noise-floor figure. |
| `results/06_bn_k_sensitivity.csv` | BN degree sensitivity. Reused for the noise-floor figure. |
| `results/06_statistical_comparison.csv` | Paired fold comparisons. Reused. |
| `results/07_shap_stability.csv` | Real vs augmented SHAP ranking. Reused. |
| `results/07_shap_stability_summary.csv` | Rank-stability statistics. Reused. |
| `results/environment.txt` | Package versions and compatibility patches. Reused. |

## Recomputed or newly created here

| Analysis | Why it is new |
| --- | --- |
| Final real-only interpretation model | run_pipeline.py only ever fitted per-fold models. A single model on all 118 usable rows is required for descriptor interpretation and did not previously exist. |
| Full-corpus SHAP | results/03 holds fold-wise held-out SHAP. The final descriptor ranking needs SHAP on the complete real corpus; both are reported and compared. |
| Cross-paper heterogeneity / domain shift | Not previously computed. |
| Physics-variable availability and closure identities | Not previously computed. |
| Descriptor classification for the physics-guided model | results/08 gave a preliminary recommendation; it is superseded here by an A/B/C classification that also accounts for closure-identity support. |
| Publication figures and LaTeX tables | Previous figures were PNG working plots. |

## Verification of the previously reported numbers

Every value below was read back from the stored CSVs, not from prose.

| Claim | Stored value | Agrees |
| --- | --- | :---: |
| A_real_only: R2 -1.91, MAE 129.6 | R2 -1.91, MAE 129.6 | yes |
| B_real_plus_1.0x: R2 -1.88, MAE 130.9 | R2 -1.88, MAE 130.9 | yes |
| C_real_plus_3.0x: R2 -2.53, MAE 138.7 | R2 -2.53, MAE 138.7 | yes |
| D_real_plus_5.25x: R2 -1.38, MAE 123.6 | R2 -1.38, MAE 123.6 | yes |
| Generator seed shifts fold MAE by ~12.3 F/g | 12.3 F/g | yes |
| BN k changes MAE by as much as ~17.8 F/g | 17.8 F/g | yes |
| Zero KS rejections | 0 rejections | yes |
| Categorical JS ~ 0.08 | 0.079 | yes |
| Correlation / MI difference ~ 0.05 | Pearson 0.047, MI 0.055 | yes |
| Zero exact memorisation | 0 exact copies | yes |
| SHAP Spearman ~ 0.90, top-3 3/3, top-5 5/5 | rho 0.90, top-3 3/3, top-5 5/5 | yes |
| Augmentation improved MAE in 3 of 5 folds (best condition) | 3/5 | yes |
| Usable rows ~ 118 | 118 rows | yes |

## Inconsistencies found

- **Two distinct real-only baselines.** The R2 of -1.91 quoted in earlier summaries is condition A of the augmentation experiment (47 rows, 18 papers, 5 descriptors), not the headline real-only baseline (118 rows, 40 papers, 13 descriptors, R2 = 0.030). Both are reported separately here.
- **`n_folds_improved` counts differ by metric.** The best augmented condition improves MAE in 3/5 folds but R2 and RMSE in only 1/5. The '3 of 5' figure is the MAE count and is reported as such.
- **No other discrepancies.** Every numeric claim checked above reproduces the stored CSV values.

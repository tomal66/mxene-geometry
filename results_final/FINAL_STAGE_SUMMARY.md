# Final stage summary - numerical findings

Every number below is read from a stored CSV under `results/` or `results_final/`. Nothing is transcribed from prose.

## 1. Corpus

- Raw rows: 192; source publications: 49
- Usable single-valued capacitance rows: **118**
- Publications represented: **40**
- Capacitance: 7.26-1609.5 F/g (median 266.5)
- Standardised descriptors retained: 12 (dropped as constant: ['synthesis_family'])

## 2. Cross-paper generalisation (grouped CV, the unbiased estimate)

| Model | Rows | Papers | R2 | MAE (F/g) | RMSE (F/g) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full real-only, 13 descriptors | 118 | 40 | 0.030 +/- 0.119 | 81.5 +/- 32.8 | 129.2 +/- 84.9 |
| BN cohort real-only, 5 descriptors | 47 | 18 | -1.906 +/- 2.67 | 129.6 +/- 68.0 | 189.1 +/- 149.3 |

These are two different models and must not be conflated. Pooled over all held-out predictions the full model gives R2 = 0.081, MAE = 81.9 F/g.

## 3. Evidence for cross-paper domain shift

| Evidence | Value |
| --- | ---: |
| Between-paper share of $C_g$ variance (eta^2) | 0.645 |
| Intraclass correlation ICC(1) for $C_g$ | 0.474 |
| Mean between-paper variance share across numeric descriptors | 0.870 |
| Median within-paper descriptor range, as a fraction of the global range | 0.00 |

Per fold, the fraction of held-out capacitances lying inside the training range is 0.92-1.00, and the standardised mean difference between train and test targets reaches 0.12. Details in `domain_shift/fold_domain_overlap.csv`.

## 4. Bayesian augmentation (all conditions, k=2, 3 generator seeds)

| Condition | R2 | MAE (F/g) | RMSE (F/g) | Pooled MAE | Seed SD (MAE) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_real_only | -1.906 +/- 2.67 | 129.6 +/- 68.0 | 189.1 +/- 149.3 | 129.5 | -- |
| B_real_plus_1.0x | -1.876 +/- 2.27 | 130.9 +/- 65.4 | 190.2 +/- 148.6 | 130.8 | 11.1 |
| C_real_plus_3.0x | -2.530 +/- 3.43 | 138.7 +/- 69.8 | 200.8 +/- 150.3 | 138.6 | 18.9 |
| D_real_plus_5.25x | -1.380 +/- 1.48 | 123.6 +/- 65.5 | 186.1 +/- 153.5 | 122.8 | 6.9 |

**Conclusion: no robust improvement.** The best-vs-worst spread across ratios is 15.1 F/g MAE, against 12.3 F/g from re-drawing the synthetic set alone and a comparable amount from changing the network degree. The response is non-monotonic in the amount of synthetic data.

## 5. Synthetic fidelity (why the null result is interpretable)

| Metric | Value |
| --- | ---: |
| Mean KS statistic | 0.154 |
| Continuous variables with KS p<0.05 | 0 |
| Mean Jensen-Shannon distance | 0.079 |
| Pearson matrix mean |diff| | 0.047 |
| Mutual-information mean |diff| | 0.055 |
| Descriptor-target association mean |diff| | 0.065 |
| Exact duplicates of a real row | 0 |

## 6. Final real-only SHAP (interpretation model, all usable rows)

| Rank | Descriptor | mean abs SHAP (F/g) | Share | Missingness | Fold rank SD | Direction |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | Composition family | 25.05 | 22.3% | 0% | 0.89 | n/a |
| 2 | H2SO4 concentration (M) | 22.37 | 19.9% | 6% | 0.89 | increasing |
| 3 | Scan rate (mV/s) | 17.87 | 15.9% | 25% | 1.10 | decreasing |
| 4 | Interlayer spacing (A) | 11.78 | 10.5% | 25% | 1.58 | decreasing |
| 5 | Layer morphology | 10.31 | 9.2% | 19% | 0.45 | n/a |
| 6 | Specific surface area (m2/g) | 7.85 | 7.0% | 70% | 1.52 | increasing |
| 7 | Electrode thickness (um) | 5.14 | 4.6% | 54% | 1.00 | decreasing |
| 8 | Gel electrolyte | 4.43 | 3.9% | 0% | 2.00 | n/a |
| 9 | Current density (A/g) | 3.23 | 2.9% | 70% | 1.48 | non-monotone/flat |
| 10 | Mass loading (mg/cm2) | 2.33 | 2.1% | 53% | 0.89 | non-monotone/flat |
| 11 | Pore diameter (nm) | 1.43 | 1.3% | 97% | 1.22 | n/a |
| 12 | Flake size (um) | 0.49 | 0.4% | 91% | 1.10 | decreasing |

Rank stability under augmentation: Spearman 0.90 (p = 0.037), top-3 3/3, top-5 5/5, mean |rank change| 0.40.

## 7. Physics feasibility

| Identity | Rows | Median rel. error | Within 5% |
| --- | ---: | ---: | ---: |
| `C_A = C_g * m_A` | 39 | 0.38% | 85% |
| `C_V = C_g * m_A / t` | 34 | 0.87% | 76% |

| Model class | Verdict |
| --- | --- |
| A. Classical PDE/ODE PINN | **NOT JUSTIFIED** |
| B. Physics-guided neural network | **FEASIBLE** |
| C. Hybrid physics-constrained multi-task regression | **RECOMMENDED** |

## 8. Descriptor classification

- **Category A (physics-grade inputs):** `Interlayer spacing (A)`, `Scan rate (mV/s)`, `Mass loading (mg/cm2)`, `Electrode thickness (um)`, `H2SO4 concentration (M)`
- **Category B (ML covariates only):** `Layer morphology`, `Composition family`, `Gel electrolyte`
- **Category C (excluded):** `Current density (A/g)`, `Specific surface area (m2/g)`, `Flake size (um)`, `Pore diameter (nm)`, `Synthesis route`

Full reasoning per descriptor in `pinn/descriptor_classification.csv`.

## 9. Deliverables

- `results_final/00_existing_outputs_inventory.md`
- `results_final/ADVISOR_SUMMARY.md`
- `results_final/FINAL_STAGE_SUMMARY.md`
- `results_final/domain_shift/`
- `results_final/shap/`
- `results_final/pinn/PROPOSED_PINN.md`
- `results_final/pinn/`
- `results_final/figures/`
- `results_final/models/`
- `latex_report/main.pdf`
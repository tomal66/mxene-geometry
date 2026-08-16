# Advisor summary

_MXene / H2SO4 supercapacitor corpus - 118 usable observations from 40 publications. Target: gravimetric capacitance (F/g)._

## Main result

The bottleneck in this project is **cross-study heterogeneity, not sample count**. Roughly 64% of the total variance in gravimetric capacitance lies *between* publications rather than within them (ICC(1) = 0.47), and the descriptors are almost as paper-bound as the target, so holding out a whole paper forces the model to extrapolate. Consequently paper-grouped prediction is weak (R2 = 0.030 +/- 0.119), and Bayesian-network augmentation - which faithfully resamples the joint distribution it was shown - cannot repair it, because the missing information is about experimental domains the corpus never sampled. What *is* robust is the descriptor hierarchy: SHAP rankings barely move under Bayesian resampling, so the real-only SHAP analysis is a sound basis for choosing variables for the next-stage physics-guided model.

## Key numbers

| Quantity | Value |
| --- | ---: |
| Usable observations / publications | 118 / 40 |
| Grouped-CV R2 (full real-only model, 13 descriptors) | 0.030 +/- 0.119 |
| Grouped-CV MAE / RMSE | 81.5 / 129.2 F/g |
| Between-paper variance share of $C_g$ (eta^2 / ICC) | 0.64 / 0.47 |
| Best augmented condition (MAE) | D_real_plus_5.25x (123.6 F/g vs 129.6) |
| MAE moved by re-drawing the synthetic set alone | 12.3 F/g |
| Synthetic fidelity: KS rejections / exact copies | 0 / 0 |
| SHAP rank stability (Spearman, top-3, top-5) | 0.90, 3/3, 5/5 |
| Areal closure C_A = C_g m_A verified | 85% of 39 rows within 5% |

## What Bayesian augmentation showed

Nothing improved beyond noise. The nominally best ratio (+5.25x) lowered fold MAE by 6.0 F/g, but simply re-drawing the same synthetic set with a different seed moves fold MAE by 12.3 F/g, and changing the network degree k moves it by a comparable amount. The response was also non-monotonic (+3x was the worst condition). Crucially, the generator itself was fine: zero KS rejections on marginals, Jensen-Shannon distance ~0.08 on categorical frequencies, correlation and mutual-information structure reproduced to ~0.05, descriptor-target associations preserved, and zero exact copies of real rows. **Distributional fidelity did not translate into out-of-domain generalisation** - which is the informative part of the null result.

## What SHAP showed

Top descriptors from the final real-only model fitted to all usable observations:

| Rank | Descriptor | Share of attribution | Missingness |
| ---: | --- | ---: | ---: |
| 1 | Composition family | 22.3% | 0% |
| 2 | H2SO4 concentration (M) | 19.9% | 6% |
| 3 | Scan rate (mV/s) | 15.9% | 25% |
| 4 | Interlayer spacing (A) | 10.5% | 25% |
| 5 | Layer morphology | 9.2% | 19% |

The ranking is stable: Spearman 0.90 between the real-only and Bayesian-augmented models, top-3 overlap 3/3, top-5 5/5, mean rank displacement 0.4 positions.

One caution worth raising: within this corpus interlayer spacing is associated with *lower* predicted capacitance, opposite to the naive expectation. That is most likely confounding (widest galleries belong to pillared/composite films whose gravimetric capacitance is diluted by intercalant mass; d-spacing is measured dry, not hydrated) and it is a concrete argument against imposing monotonicity priors from SHAP.

## What this means for the next model

A classical PDE-based PINN is **not** justified: the corpus has no time axis, no potential window, no discharge time and no current, so even the algebraic galvanostatic definition cannot be evaluated for a single row. What *is* available is a pair of exact closure identities, verified against the published values:

```
  C_A = C_g * m_A          median error 0.4%, 85% of rows within 5%
  C_V = C_g * m_A / t      median error 0.6%, 78% of rows within 5%
```
So the recommended direction is a **hybrid physics-constrained multi-task regression**: predict C_g with a positivity-constrained head, add auxiliary heads for areal and volumetric capacitance, and penalise violation of the two identities. This turns two columns that are unusable as inputs (they are algebraic transforms of the target) into legitimate extra supervision. Inputs: `Interlayer spacing (A)`, `Scan rate (mV/s)`, `Mass loading (mg/cm2)`, `Electrode thickness (um)`, `H2SO4 concentration (M)`.

## Immediate next experiment

**Add potential window and discharge time to the extraction schema and re-extract.** These two fields are currently at 0% coverage and are the only missing pieces of `C_g = I dt / (m dV)`. With them, the physics term stops being a definitional unit-closure and becomes the actual measurement equation, applicable to every galvanostatic row rather than the minority that report areal capacitance. It would also let rows be labelled GCD vs CV, removing a known and currently unmodelled source of between-paper scatter. That is a higher-value use of effort than any further tuning of the current feature set, and it addresses the domain-shift diagnosis directly rather than working around it.

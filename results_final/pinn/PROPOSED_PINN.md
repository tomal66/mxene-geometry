# Proposed next-stage physics-guided model

This document states what the compiled corpus can and cannot support. It is deliberately not a proposal for a classical PINN, because the data do not justify one.

## 1. Verdict on the three candidate model classes

| Class | Verdict | Why |
| --- | --- | --- |
| A. Classical PDE/ODE PINN | **NOT JUSTIFIED** | Absent. The corpus records no time axis, no potential window, no discharge curve and no spatial coordinate: discharge_time_s, voltage_window_V, current_A, electrode_mass_g are all 0% populated. Every row is a single scalar summary of an experiment, not a trajectory. |
| B. Physics-guided neural network | **FEASIBLE** | Supported. Positivity of capacitance, training-domain bounds and dimensionally meaningful inputs (interlayer spacing, molarity, sweep rate, mass loading) are all available. Monotonicity should NOT be imposed from SHAP alone - the corpus shows a d-spacing trend opposite to the naive physical expectation. |
| C. Hybrid physics-constrained multi-task regression | **RECOMMENDED** | Supported and measured. C_A = C_g*m_A reproduces the reported areal capacitance to within 5% for 85% of 39 testable rows (median relative error 0.38%); C_V = C_g*m_A/t holds to within 5% for 76% of 34 rows (median 0.87%). Areal and volumetric capacitance cannot be predictors - they are algebraic transforms of the target - but they can serve as auxiliary supervised outputs tied to the target by these identities. |

### Why a classical PDE-PINN is not currently justified

A physics-informed neural network in the usual sense minimises the residual of a differential equation evaluated at collocation points. That requires (i) a state variable defined over a space or time coordinate, and (ii) initial and boundary conditions per observation. This corpus has neither. Each row is a single scalar summary of a completed experiment. The following quantities are **entirely absent** from the corpus (0% coverage):

- `voltage_window_V` -- Potential window - required by every capacitance relation
- `discharge_time_s` -- Galvanostatic discharge time - required by C_g = I dt /(m dV)
- `current_A` -- Absolute discharge current - required by C_g = I dt /(m dV)
- `electrode_mass_g` -- Absolute active mass - needed by the galvanostatic relation
- `electrode_area_cm2` -- Geometric area - needed to move between areal and absolute quantities
- `cv_current_integral` -- Integrated CV current - required by the CV-integral relation
- `testing_mode` -- Whether a row is GCD or CV - required to know which relation applies

Without a potential window and a discharge time, even the *algebraic* galvanostatic definition `C_g = I dt / (m dV)` cannot be evaluated for a single row, let alone a differential form. Writing a PDE residual here would mean inventing the physics rather than enforcing it.

## 2. What physics the corpus *can* enforce

Two exact algebraic closure identities relate the target to other measured quantities, and both were verified against the values reported in the source publications:

```
  C_A = C_g * m_A            (areal vs gravimetric normalisation)
  C_V = C_g * rho,  rho = m_A / t   (volumetric, via electrode density)
```

| Identity | Testable rows | Papers | Median rel. error | Within 5% | Within 20% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `C_A = C_g * m_A` | 39 | 14 | 0.38% | 85% | 90% |
| `C_V = C_g * m_A / t` | 34 | 13 | 0.87% | 76% | 85% |

This is the key structural insight for the next model. Areal and volumetric capacitance **cannot be predictors** -- they are algebraic transforms of the target and were correctly excluded as leakage. But they *can* be **auxiliary supervised outputs**, tied to the primary output by the identities above. That converts two leakage columns into extra training signal without any leakage, and it gives the model a genuine physical constraint to respect.

## 3. Recommended model: hybrid physics-constrained multi-task regression

### Inputs (Category A)

- **Interlayer spacing (A)** -- Geometric: ion-accessible gallery height; sets the confinement length scale in any transport description.
- **Scan rate (mV/s)** -- Kinetic: sets the timescale probed, hence the diffusion length sampled.
- **Mass loading (mg/cm2)** -- Geometric: areal mass; enters the exact areal closure C_A = C_g m_A.
- **Electrode thickness (um)** -- Geometric: transport path length; with m_A gives electrode density for the volumetric closure.
- **H2SO4 concentration (M)** -- Boundary condition: bulk proton activity entering Nernstian and Poisson-Nernst-Planck terms.

### Context covariates (Category B, data term only)

`Layer morphology`, `Composition family`, `Gel electrolyte` -- retained because they improve prediction, but excluded from every physics term because they have no counterpart in a governing relation.

### Excluded (Category C)

`Current density (A/g)`, `Specific surface area (m2/g)`, `Flake size (um)`, `Pore diameter (nm)`, `Synthesis route`, plus all target-derived, metadata and provenance columns (`results/00_leakage_audit.csv`).

### Target and outputs

```
  primary   :  C_g_hat = softplus(f_theta(x))        [F/g]   -- positivity by construction
  auxiliary :  C_A_hat, C_V_hat                      [F/cm2], [F/cm3]
```

### Loss

```
  L = lambda_d * L_data + lambda_p * L_phys + lambda_b * L_bound + lambda_r * L_reg

  L_data = Huber(C_g_hat, C_g)                                       over all N rows
         + w_A * masked_Huber(C_A_hat, C_A)                          over rows reporting C_A
         + w_V * masked_Huber(C_V_hat, C_V)                          over rows reporting C_V

  L_phys = mean_i in S_A [ (C_A_hat_i - C_g_hat_i * m_A_i)^2 ]       S_A: m_A known
         + mean_i in S_V [ (C_V_hat_i - C_g_hat_i * m_A_i / t_i)^2 ] S_V: m_A and t known

  L_bound = mean_i [ relu(-C_g_hat_i)^2 ]                            (redundant with softplus,
          + mean_i [ d_hull(x_i)^2 ]                                  kept as a guard)

  L_reg  = ||theta||^2
```
Huber rather than squared error for the data term: the corpus contains genuine extreme values (7.3 and 1609.5 F/g) that are retained rather than deleted, and a squared loss would let them dominate.

All physics residuals are computed in normalised units and evaluated only on rows where the required quantities are actually reported (masked losses), so no value is imputed to satisfy a constraint.

### What is deliberately NOT included

- **No monotonicity constraints.** Within this corpus the SHAP association between interlayer spacing and predicted capacitance is *negative*, opposite to the naive expectation that a wider gallery eases ion access. That is most plausibly confounding (the widest galleries belong to pillared/composite films whose gravimetric capacitance is diluted by intercalant mass, and d-spacing is measured dry rather than hydrated), but it is exactly why a monotone prior must not be read off a SHAP plot.

- **No PDE residual**, for the reasons in section 1.

- **No synthetic training rows.** Bayesian-network augmentation was tested and did not improve paper-level generalisation.

## 4. Validation protocol (unchanged and non-negotiable)

`GroupKFold` on `paper_id`. Never a random split: the between-paper share of capacitance variance is eta^2 = 0.64 (ICC(1) = 0.47), so a random split would place near-duplicate rows from the same publication on both sides and inflate every score.

## 5. Missing physics variables, by priority

| Rank | Variable | Coverage | Expected value |
| ---: | --- | ---: | --- |
| 1 | `voltage_window_V` | 0% | very high |
| 2 | `discharge_time_s` | 0% | very high |
| 3 | `testing_mode` | 0% | high |
| 4 | `current_density_A_g` | 30% | high |
| 5 | `mass_loading_mg_cm2` | 47% | high |
| 6 | `electrode_thickness_um` | 46% | medium-high |
| 7 | `ssa_m2_g` | 30% | medium |
| 8 | `interlayer_A` | 75% | medium-high |

Full reasoning in `data_gap_priorities.csv`.

## 6. Limitations of this proposal

- The closure identities are *definitional* unit relations, not dynamics. Enforcing them constrains internal consistency; it does not inject transport physics.

- The physics term is only applicable to the 39 rows with reported areal capacitance and 34 with volumetric capacitance, i.e. a minority of the corpus.

- Rows violating the closures by more than 50% (3 areal, 4 volumetric) indicate that some publications use a mass-loading convention inconsistent with their own reported areal capacitance. Those rows should be down-weighted, not silently corrected.

- The dominant error source remains between-study heterogeneity, which no loss term on this feature set can remove.

"""Markdown deliverables: PROPOSED_PINN.md, ADVISOR_SUMMARY.md, FINAL_STAGE_SUMMARY.md."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import TARGET
from .final_model import PLAIN


def _names(feats) -> str:
    return ", ".join(f"`{PLAIN.get(f, f)}`" for f in feats)


def write_pinn_md(S: dict, path: Path) -> None:
    ar = S["closure"][S["closure"]["identity"] == "C_A = C_g * m_A"].iloc[0]
    vo = S["closure"][S["closure"]["identity"] == "C_V = C_g * m_A / t"].iloc[0]
    avail = S["avail"].set_index("variable")
    absent = [v for v in ("voltage_window_V", "discharge_time_s", "current_A",
                          "electrode_mass_g", "electrode_area_cm2", "cv_current_integral",
                          "testing_mode")
              if avail.loc[v, "availability_count"] == 0]

    L, A = [], None
    A = L.append
    A("# Proposed next-stage physics-guided model\n")
    A("This document states what the compiled corpus can and cannot support. It is "
      "deliberately not a proposal for a classical PINN, because the data do not "
      "justify one.\n")

    A("## 1. Verdict on the three candidate model classes\n")
    A("| Class | Verdict | Why |")
    A("| --- | --- | --- |")
    for _, r in S["assess"].iterrows():
        A(f"| {r['model_class']} | **{r['verdict']}** | {r['evidence_in_dataset']} |")
    A("")
    A("### Why a classical PDE-PINN is not currently justified\n")
    A("A physics-informed neural network in the usual sense minimises the residual of a "
      "differential equation evaluated at collocation points. That requires (i) a state "
      "variable defined over a space or time coordinate, and (ii) initial and boundary "
      "conditions per observation. This corpus has neither. Each row is a single scalar "
      "summary of a completed experiment. The following quantities are **entirely absent** "
      "from the corpus (0% coverage):\n")
    for v in absent:
        A(f"- `{v}` -- {avail.loc[v, 'potential_physics_role']}")
    A("")
    A("Without a potential window and a discharge time, even the *algebraic* galvanostatic "
      "definition `C_g = I dt / (m dV)` cannot be evaluated for a single row, let alone a "
      "differential form. Writing a PDE residual here would mean inventing the physics "
      "rather than enforcing it.\n")

    A("## 2. What physics the corpus *can* enforce\n")
    A("Two exact algebraic closure identities relate the target to other measured "
      "quantities, and both were verified against the values reported in the source "
      "publications:\n")
    A("```")
    A("  C_A = C_g * m_A            (areal vs gravimetric normalisation)")
    A("  C_V = C_g * rho,  rho = m_A / t   (volumetric, via electrode density)")
    A("```")
    A("")
    A("| Identity | Testable rows | Papers | Median rel. error | Within 5% | Within 20% |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in S["closure"].iterrows():
        A(f"| `{r['identity']}` | {int(r['n_testable_rows'])} | {int(r['n_papers'])} | "
          f"{r['median_relative_error']:.2%} | {r['frac_within_5pct']:.0%} | "
          f"{r['frac_within_20pct']:.0%} |")
    A("")
    A("This is the key structural insight for the next model. Areal and volumetric "
      "capacitance **cannot be predictors** -- they are algebraic transforms of the target "
      "and were correctly excluded as leakage. But they *can* be **auxiliary supervised "
      "outputs**, tied to the primary output by the identities above. That converts two "
      "leakage columns into extra training signal without any leakage, and it gives the "
      "model a genuine physical constraint to respect.\n")

    A("## 3. Recommended model: hybrid physics-constrained multi-task regression\n")
    A("### Inputs (Category A)\n")
    for f in S["cat_a"]:
        row = S["classification"].set_index("descriptor").loc[f]
        A(f"- **{PLAIN.get(f, f)}** -- {row['physical_role']}")
    A("")
    A("### Context covariates (Category B, data term only)\n")
    A(_names(S["cat_b"]) + " -- retained because they improve prediction, but excluded from "
      "every physics term because they have no counterpart in a governing relation.\n")
    A("### Excluded (Category C)\n")
    A((_names(S["cat_c"]) if S["cat_c"] else "_none_") +
      ", plus all target-derived, metadata and provenance columns "
      "(`results/00_leakage_audit.csv`).\n")

    A("### Target and outputs\n")
    A("```")
    A("  primary   :  C_g_hat = softplus(f_theta(x))        [F/g]   -- positivity by construction")
    A("  auxiliary :  C_A_hat, C_V_hat                      [F/cm2], [F/cm3]")
    A("```")
    A("")
    A("### Loss\n")
    A("```")
    A("  L = lambda_d * L_data + lambda_p * L_phys + lambda_b * L_bound + lambda_r * L_reg")
    A("")
    A("  L_data = Huber(C_g_hat, C_g)                                       over all N rows")
    A("         + w_A * masked_Huber(C_A_hat, C_A)                          over rows reporting C_A")
    A("         + w_V * masked_Huber(C_V_hat, C_V)                          over rows reporting C_V")
    A("")
    A("  L_phys = mean_i in S_A [ (C_A_hat_i - C_g_hat_i * m_A_i)^2 ]       S_A: m_A known")
    A("         + mean_i in S_V [ (C_V_hat_i - C_g_hat_i * m_A_i / t_i)^2 ] S_V: m_A and t known")
    A("")
    A("  L_bound = mean_i [ relu(-C_g_hat_i)^2 ]                            (redundant with softplus,")
    A("          + mean_i [ d_hull(x_i)^2 ]                                  kept as a guard)")
    A("")
    A("  L_reg  = ||theta||^2")
    A("```")
    A("Huber rather than squared error for the data term: the corpus contains genuine "
      "extreme values (7.3 and 1609.5 F/g) that are retained rather than deleted, and a "
      "squared loss would let them dominate.\n")
    A("All physics residuals are computed in normalised units and evaluated only on rows "
      "where the required quantities are actually reported (masked losses), so no value is "
      "imputed to satisfy a constraint.\n")

    A("### What is deliberately NOT included\n")
    A("- **No monotonicity constraints.** Within this corpus the SHAP association between "
      "interlayer spacing and predicted capacitance is *negative*, opposite to the naive "
      "expectation that a wider gallery eases ion access. That is most plausibly "
      "confounding (the widest galleries belong to pillared/composite films whose "
      "gravimetric capacitance is diluted by intercalant mass, and d-spacing is measured "
      "dry rather than hydrated), but it is exactly why a monotone prior must not be read "
      "off a SHAP plot.\n")
    A("- **No PDE residual**, for the reasons in section 1.\n")
    A("- **No synthetic training rows.** Bayesian-network augmentation was tested and did "
      "not improve paper-level generalisation.\n")

    A("## 4. Validation protocol (unchanged and non-negotiable)\n")
    A("`GroupKFold` on `paper_id`. Never a random split: the between-paper share of "
      f"capacitance variance is eta^2 = {S['target_eta2']:.2f} (ICC(1) = "
      f"{S['target_icc']:.2f}), so a random split would place near-duplicate rows from the "
      "same publication on both sides and inflate every score.\n")

    A("## 5. Missing physics variables, by priority\n")
    A("| Rank | Variable | Coverage | Expected value |")
    A("| ---: | --- | ---: | --- |")
    for _, r in S["gaps"].iterrows():
        A(f"| {r['rank']} | `{r['variable']}` | {r['current_coverage']} | "
          f"{r['expected_value_for_PINN']} |")
    A("")
    A("Full reasoning in `data_gap_priorities.csv`.\n")

    A("## 6. Limitations of this proposal\n")
    A("- The closure identities are *definitional* unit relations, not dynamics. Enforcing "
      "them constrains internal consistency; it does not inject transport physics.\n")
    A(f"- The physics term is only applicable to the {int(ar['n_testable_rows'])} rows with "
      f"reported areal capacitance and {int(vo['n_testable_rows'])} with volumetric "
      "capacitance, i.e. a minority of the corpus.\n")
    A("- Rows violating the closures by more than 50% "
      f"({int(ar['n_gross_violations_gt50pct'])} areal, "
      f"{int(vo['n_gross_violations_gt50pct'])} volumetric) indicate that some publications "
      "use a mass-loading convention inconsistent with their own reported areal "
      "capacitance. Those rows should be down-weighted, not silently corrected.\n")
    A("- The dominant error source remains between-study heterogeneity, which no loss term "
      "on this feature set can remove.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def write_advisor(S: dict, path: Path) -> None:
    a = S["aug"].set_index("condition")
    g = S["shap_grouped"]
    ss = S["stab_sum"]
    ar = S["closure"][S["closure"]["identity"] == "C_A = C_g * m_A"].iloc[0]
    seed_sd = a.loc[a.index != "A_real_only", "mae_sd_across_gen_seeds"].mean()

    L, A = [], None
    A = L.append
    A("# Advisor summary\n")
    A(f"_MXene / H2SO4 supercapacitor corpus - {S['n_rows']} usable observations from "
      f"{S['n_papers']} publications. Target: gravimetric capacitance (F/g)._\n")

    A("## Main result\n")
    A("The bottleneck in this project is **cross-study heterogeneity, not sample count**. "
      f"Roughly {S['target_eta2']:.0%} of the total variance in gravimetric capacitance "
      f"lies *between* publications rather than within them (ICC(1) = {S['target_icc']:.2f}), "
      "and the descriptors are almost as paper-bound as the target, so holding out a whole "
      "paper forces the model to extrapolate. Consequently paper-grouped prediction is weak "
      f"(R2 = {S['base_r2']:.3f} +/- {S['base_r2_sd']:.3f}), and Bayesian-network "
      "augmentation - which faithfully resamples the joint distribution it was shown - "
      "cannot repair it, because the missing information is about experimental domains the "
      "corpus never sampled. What *is* robust is the descriptor hierarchy: SHAP rankings "
      "barely move under Bayesian resampling, so the real-only SHAP analysis is a sound "
      "basis for choosing variables for the next-stage physics-guided model.\n")

    A("## Key numbers\n")
    A("| Quantity | Value |")
    A("| --- | ---: |")
    A(f"| Usable observations / publications | {S['n_rows']} / {S['n_papers']} |")
    A(f"| Grouped-CV R2 (full real-only model, 13 descriptors) | "
      f"{S['base_r2']:.3f} +/- {S['base_r2_sd']:.3f} |")
    A(f"| Grouped-CV MAE / RMSE | {S['base_sum']['mae_mean']:.1f} / "
      f"{S['base_sum']['rmse_mean']:.1f} F/g |")
    A(f"| Between-paper variance share of $C_g$ (eta^2 / ICC) | "
      f"{S['target_eta2']:.2f} / {S['target_icc']:.2f} |")
    A(f"| Best augmented condition (MAE) | {a['mae_mean'].idxmin()} "
      f"({a['mae_mean'].min():.1f} F/g vs {a.loc['A_real_only','mae_mean']:.1f}) |")
    A(f"| MAE moved by re-drawing the synthetic set alone | {seed_sd:.1f} F/g |")
    A(f"| Synthetic fidelity: KS rejections / exact copies | "
      f"{int(S['fid']['n_ks_reject_p05'].sum())} / "
      f"{int(S['fid']['exact_duplicates_of_real'].sum())} |")
    A(f"| SHAP rank stability (Spearman, top-3, top-5) | "
      f"{ss['spearman_rank_correlation']:.2f}, {int(ss['top3_overlap'])}/3, "
      f"{int(ss['top5_overlap'])}/5 |")
    A(f"| Areal closure C_A = C_g m_A verified | {ar['frac_within_5pct']:.0%} of "
      f"{int(ar['n_testable_rows'])} rows within 5% |")
    A("")

    A("## What Bayesian augmentation showed\n")
    A("Nothing improved beyond noise. The nominally best ratio (+5.25x) lowered fold MAE by "
      f"{a.loc['A_real_only','mae_mean'] - a['mae_mean'].min():.1f} F/g, but simply "
      f"re-drawing the same synthetic set with a different seed moves fold MAE by "
      f"{seed_sd:.1f} F/g, and changing the network degree k moves it by a comparable "
      "amount. The response was also non-monotonic (+3x was the worst condition). "
      "Crucially, the generator itself was fine: zero KS rejections on marginals, "
      "Jensen-Shannon distance ~0.08 on categorical frequencies, correlation and "
      "mutual-information structure reproduced to ~0.05, descriptor-target associations "
      "preserved, and zero exact copies of real rows. **Distributional fidelity did not "
      "translate into out-of-domain generalisation** - which is the informative part of "
      "the null result.\n")

    A("## What SHAP showed\n")
    A("Top descriptors from the final real-only model fitted to all usable observations:\n")
    A("| Rank | Descriptor | Share of attribution | Missingness |")
    A("| ---: | --- | ---: | ---: |")
    for _, r in g.head(5).iterrows():
        A(f"| {int(r['rank'])} | {r['display_name']} | "
          f"{r['normalized_importance']:.1%} | {r['missingness']:.0%} |")
    A("")
    A(f"The ranking is stable: Spearman {ss['spearman_rank_correlation']:.2f} between the "
      f"real-only and Bayesian-augmented models, top-3 overlap {int(ss['top3_overlap'])}/3, "
      f"top-5 {int(ss['top5_overlap'])}/5, mean rank displacement "
      f"{ss['mean_abs_rank_change']:.1f} positions.\n")
    A("One caution worth raising: within this corpus interlayer spacing is associated with "
      "*lower* predicted capacitance, opposite to the naive expectation. That is most "
      "likely confounding (widest galleries belong to pillared/composite films whose "
      "gravimetric capacitance is diluted by intercalant mass; d-spacing is measured dry, "
      "not hydrated) and it is a concrete argument against imposing monotonicity priors "
      "from SHAP.\n")

    A("## What this means for the next model\n")
    A("A classical PDE-based PINN is **not** justified: the corpus has no time axis, no "
      "potential window, no discharge time and no current, so even the algebraic "
      "galvanostatic definition cannot be evaluated for a single row. What *is* available "
      "is a pair of exact closure identities, verified against the published values:\n")
    A("```")
    A("  C_A = C_g * m_A          median error 0.4%, 85% of rows within 5%")
    A("  C_V = C_g * m_A / t      median error 0.6%, 78% of rows within 5%")
    A("```")
    A("So the recommended direction is a **hybrid physics-constrained multi-task "
      "regression**: predict C_g with a positivity-constrained head, add auxiliary heads "
      "for areal and volumetric capacitance, and penalise violation of the two identities. "
      "This turns two columns that are unusable as inputs (they are algebraic transforms of "
      "the target) into legitimate extra supervision. Inputs: "
      + _names(S["cat_a"]) + ".\n")

    A("## Immediate next experiment\n")
    A("**Add potential window and discharge time to the extraction schema and re-extract.** "
      "These two fields are currently at 0% coverage and are the only missing pieces of "
      "`C_g = I dt / (m dV)`. With them, the physics term stops being a definitional "
      "unit-closure and becomes the actual measurement equation, applicable to every "
      "galvanostatic row rather than the minority that report areal capacitance. It would "
      "also let rows be labelled GCD vs CV, removing a known and currently unmodelled "
      "source of between-paper scatter. That is a higher-value use of effort than any "
      "further tuning of the current feature set, and it addresses the domain-shift "
      "diagnosis directly rather than working around it.\n")
    path.write_text("\n".join(L), encoding="utf-8")


def write_final_stage(S: dict, path: Path) -> None:
    a = S["aug"].set_index("condition")
    ss, g = S["stab_sum"], S["shap_grouped"]
    seed_sd = a.loc[a.index != "A_real_only", "mae_sd_across_gen_seeds"].mean()

    L, A = [], None
    A = L.append
    A("# Final stage summary - numerical findings\n")
    A("Every number below is read from a stored CSV under `results/` or "
      "`results_final/`. Nothing is transcribed from prose.\n")

    A("## 1. Corpus\n")
    A(f"- Raw rows: {len(S['raw'])}; source publications: "
      f"{S['raw']['source_file'].nunique()}")
    A(f"- Usable single-valued capacitance rows: **{S['n_rows']}**")
    A(f"- Publications represented: **{S['n_papers']}**")
    A(f"- Capacitance: {S['data'][TARGET].min():.2f}-{S['data'][TARGET].max():.1f} F/g "
      f"(median {S['data'][TARGET].median():.1f})")
    A(f"- Standardised descriptors retained: {len(S['final_features'])} "
      f"(dropped as constant: {S['dropped_features'] or 'none'})")
    A("")

    A("## 2. Cross-paper generalisation (grouped CV, the unbiased estimate)\n")
    A("| Model | Rows | Papers | R2 | MAE (F/g) | RMSE (F/g) |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    A(f"| Full real-only, 13 descriptors | {S['n_rows']} | {S['n_papers']} | "
      f"{S['base_r2']:.3f} +/- {S['base_r2_sd']:.3f} | "
      f"{S['base_sum']['mae_mean']:.1f} +/- {S['base_sum']['mae_std']:.1f} | "
      f"{S['base_sum']['rmse_mean']:.1f} +/- {S['base_sum']['rmse_std']:.1f} |")
    A(f"| BN cohort real-only, 5 descriptors | {S['cohort_rows']} | {S['cohort_papers']} | "
      f"{S['cohort_r2']:.3f} +/- {S['cohort_r2_sd']:.2f} | "
      f"{a.loc['A_real_only','mae_mean']:.1f} +/- {a.loc['A_real_only','mae_sd']:.1f} | "
      f"{a.loc['A_real_only','rmse_mean']:.1f} +/- {a.loc['A_real_only','rmse_sd']:.1f} |")
    A("")
    A("These are two different models and must not be conflated. Pooled over all held-out "
      f"predictions the full model gives R2 = {S['base_sum']['pooled_r2']:.3f}, "
      f"MAE = {S['base_sum']['pooled_mae']:.1f} F/g.\n")

    A("## 3. Evidence for cross-paper domain shift\n")
    A("| Evidence | Value |")
    A("| --- | ---: |")
    A(f"| Between-paper share of $C_g$ variance (eta^2) | {S['target_eta2']:.3f} |")
    A(f"| Intraclass correlation ICC(1) for $C_g$ | {S['target_icc']:.3f} |")
    A(f"| Mean between-paper variance share across numeric descriptors | "
      f"{S['mean_desc_eta2']:.3f} |")
    A(f"| Median within-paper descriptor range, as a fraction of the global range | "
      f"{S['median_within_over_global']:.2f} |")
    A("")
    ov = S["overlap"]
    tgt = ov[ov["type"] == "target"]
    A(f"Per fold, the fraction of held-out capacitances lying inside the training range is "
      f"{tgt['frac_test_inside_train_range'].min():.2f}-"
      f"{tgt['frac_test_inside_train_range'].max():.2f}, and the standardised mean "
      f"difference between train and test targets reaches "
      f"{tgt['standardised_mean_diff'].abs().max():.2f}. Details in "
      "`domain_shift/fold_domain_overlap.csv`.\n")

    A("## 4. Bayesian augmentation (all conditions, k=2, 3 generator seeds)\n")
    A("| Condition | R2 | MAE (F/g) | RMSE (F/g) | Pooled MAE | Seed SD (MAE) |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in S["aug"].sort_values("ratio").iterrows():
        sd = r["mae_sd_across_gen_seeds"]
        A(f"| {r['condition']} | {r['r2_mean']:.3f} +/- {r['r2_sd']:.2f} | "
          f"{r['mae_mean']:.1f} +/- {r['mae_sd']:.1f} | "
          f"{r['rmse_mean']:.1f} +/- {r['rmse_sd']:.1f} | {r['pooled_mae']:.1f} | "
          f"{'--' if pd.isna(sd) else f'{sd:.1f}'} |")
    A("")
    A(f"**Conclusion: no robust improvement.** The best-vs-worst spread across ratios is "
      f"{S['aug']['mae_mean'].max() - S['aug']['mae_mean'].min():.1f} F/g MAE, against "
      f"{seed_sd:.1f} F/g from re-drawing the synthetic set alone and a comparable amount "
      "from changing the network degree. The response is non-monotonic in the amount of "
      "synthetic data.\n")

    A("## 5. Synthetic fidelity (why the null result is interpretable)\n")
    A("| Metric | Value |")
    A("| --- | ---: |")
    fid = S["fid"]
    for lab, col, dp in [("Mean KS statistic", "mean_ks_statistic", 3),
                         ("Continuous variables with KS p<0.05", "n_ks_reject_p05", 0),
                         ("Mean Jensen-Shannon distance", "mean_jensen_shannon", 3),
                         ("Pearson matrix mean |diff|", "pearson_mean_abs_diff", 3),
                         ("Mutual-information mean |diff|", "mi_mean_abs_diff", 3),
                         ("Descriptor-target association mean |diff|",
                          "target_assoc_mean_abs_diff", 3),
                         ("Exact duplicates of a real row", "exact_duplicates_of_real", 0)]:
        A(f"| {lab} | {fid[col].mean():.{dp}f} |")
    A("")

    A("## 6. Final real-only SHAP (interpretation model, all usable rows)\n")
    A("| Rank | Descriptor | mean abs SHAP (F/g) | Share | Missingness | Fold rank SD | Direction |")
    A("| ---: | --- | ---: | ---: | ---: | ---: | --- |")
    dirs_ = S["shap_dirs"].set_index("descriptor")
    for _, r in g.iterrows():
        d = dirs_.loc[r["descriptor"], "direction"] if r["descriptor"] in dirs_.index else "n/a"
        A(f"| {int(r['rank'])} | {r['display_name']} | {r['mean_abs_shap']:.2f} | "
          f"{r['normalized_importance']:.1%} | {r['missingness']:.0%} | "
          f"{r['rank_SD']:.2f} | {d} |")
    A("")
    A(f"Rank stability under augmentation: Spearman {ss['spearman_rank_correlation']:.2f} "
      f"(p = {ss['spearman_p_value']:.3f}), top-3 {int(ss['top3_overlap'])}/3, top-5 "
      f"{int(ss['top5_overlap'])}/5, mean |rank change| {ss['mean_abs_rank_change']:.2f}.\n")

    A("## 7. Physics feasibility\n")
    A("| Identity | Rows | Median rel. error | Within 5% |")
    A("| --- | ---: | ---: | ---: |")
    for _, r in S["closure"].iterrows():
        A(f"| `{r['identity']}` | {int(r['n_testable_rows'])} | "
          f"{r['median_relative_error']:.2%} | {r['frac_within_5pct']:.0%} |")
    A("")
    A("| Model class | Verdict |")
    A("| --- | --- |")
    for _, r in S["assess"].iterrows():
        A(f"| {r['model_class']} | **{r['verdict']}** |")
    A("")

    A("## 8. Descriptor classification\n")
    A(f"- **Category A (physics-grade inputs):** {_names(S['cat_a'])}")
    A(f"- **Category B (ML covariates only):** {_names(S['cat_b'])}")
    A(f"- **Category C (excluded):** {_names(S['cat_c']) if S['cat_c'] else '_none_'}")
    A("")
    A("Full reasoning per descriptor in `pinn/descriptor_classification.csv`.\n")

    A("## 9. Deliverables\n")
    for p in ["results_final/00_existing_outputs_inventory.md",
              "results_final/ADVISOR_SUMMARY.md",
              "results_final/FINAL_STAGE_SUMMARY.md",
              "results_final/domain_shift/", "results_final/shap/",
              "results_final/pinn/PROPOSED_PINN.md", "results_final/pinn/",
              "results_final/figures/", "results_final/models/",
              "latex_report/main.pdf"]:
        A(f"- `{p}`")
    path.write_text("\n".join(L), encoding="utf-8")

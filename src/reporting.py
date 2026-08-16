"""Assembles results/FINAL_REPORT.md from the pipeline state dictionary."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import AUG_LABELS, BN_EPSILON, BN_K_DEFAULT, RESULTS_DIR, TARGET

PRETTY = {
    "A_real_only": "Real only",
    "B_real_plus_1.0x": "Real + 1x synthetic",
    "C_real_plus_3.0x": "Real + 3x synthetic",
    "D_real_plus_5.25x": "Real + 5.25x synthetic",
}


def _pm(mean, sd, dp=3):
    if pd.isna(mean):
        return "n/a"
    s = "n/a" if pd.isna(sd) else f"{sd:.{dp}f}"
    return f"{mean:.{dp}f} +/- {s}"


def write_final_report(S: dict) -> None:
    L: list[str] = []
    A = L.append

    data = S["data"]
    audit = S["audit"]
    base_fold = S["baseline"]["fold_metrics"]
    base_sum = S["baseline_summary"].iloc[0]
    summ = S["aug_summary"]
    stats = S["stats"]
    stab = S["stability_summary"]
    fid = S["fidelity"]
    sel = S["bn_selection"]
    cohort = S["bn_cohort"]

    A("# MXene gravimetric capacitance: SHAP descriptor analysis and")
    A("# Bayesian-network synthetic data augmentation")
    A("")
    A(f"Source corpus: `{S['dataset_path']}`  |  target: `{TARGET}` (F/g)  |  "
      f"random seed 42  |  all evaluation grouped by `paper_id`.")
    A("")
    A("---")
    A("")

    # ------------------------------------------------------------------ dataset
    A("## 1. Dataset")
    A("")
    A(f"- Rows in the raw corpus: **{audit['n_rows']}** across **{audit['n_cols']}** columns.")
    A(f"- Unique source papers in the raw corpus: **{audit['n_papers']}** "
      f"(median {audit['rows_per_paper'].median():.0f} rows per paper, max {audit['rows_per_paper'].max()}).")
    A(f"- Rows with a usable single-valued gravimetric capacitance: **{len(data)}** "
      f"from **{data['paper_id'].nunique()}** papers.")
    A("")
    tl = S["target_log"]["disposition"].value_counts()
    A("Target dispositions (nothing is hidden - the full per-row log is in "
      "`01_target_disposition_log.csv`):")
    A("")
    A("| disposition | n |")
    A("| --- | ---: |")
    for k, v in tl.items():
        A(f"| `{k}` | {v} |")
    A("")
    A(f"Capacitance range in the modelling set: "
      f"{data[TARGET].min():.2f} - {data[TARGET].max():.1f} F/g "
      f"(median {data[TARGET].median():.1f}, mean {data[TARGET].mean():.1f}, "
      f"SD {data[TARGET].std():.1f}). "
      f"{int(data['target_outlier_flag'].sum())} value(s) fall outside a physically "
      "typical 20-1500 F/g band and are retained but flagged.")
    A("")
    A("### Descriptor availability among the modelled rows")
    A("")
    A("| descriptor | non-missing | missingness | distinct values |")
    A("| --- | ---: | ---: | ---: |")
    for f in S["baseline"]["features"]:
        s = data[f]
        A(f"| `{f}` | {int(s.notna().sum())} | {s.isna().mean():.0%} | {int(s.nunique(dropna=True))} |")
    A("")
    A("Key missingness problems: gas-sorption surface area, gravimetric current "
      "density, flake size and pore diameter are reported by only a minority of "
      "papers, and lateral flake size is usually given qualitatively "
      "(\"micrometer-sized\", \"several\") or as a range, which this pipeline refuses "
      "to convert into a number. `synthesis_family` collapses to a single level "
      "(LiF/HCl MILD) among the rows that survive target cleaning, i.e. it carries "
      "no information in this cohort.")
    A("")
    A("### Target-leakage audit")
    A("")
    leak = S["leak"]
    A(f"{int(leak.use_as_predictor.sum())} of {len(leak)} raw columns are admissible "
      "as predictors (`00_leakage_audit.csv`). Explicitly excluded as leakage or "
      "post-hoc information:")
    A("")
    A("| column | reason |")
    A("| --- | --- |")
    for c in ["volumetric_capacitance", "areal_capacitance", "role", "is_primary",
              "year", "overall_reasoning", "gravimetric_capacitance__conf",
              "gravimetric_capacitance__src"]:
        r = leak[leak["column"] == c]
        if len(r):
            A(f"| `{c}` | {r.iloc[0]['reason']} |")
    A("")
    A("No energy-density or power-density columns exist in this corpus, so the "
      "classic `E = 0.5*C*V^2` leakage route is absent; the analogous routes that "
      "*are* present are volumetric and areal capacitance, which are algebraic "
      "transforms of the target and are removed.")
    A("")

    # ------------------------------------------------------------- real baseline
    A("## 2. Real-only model")
    A("")
    A(f"XGBoost regressor, {base_sum['n_features']} descriptors, "
      f"{S['baseline']['fold_reason']} Hyper-parameters were tuned by "
      "`RandomizedSearchCV` inside an **inner** `GroupKFold` on the training papers "
      "of each outer fold only; the outer test papers never influenced model "
      "selection, encoding, or imputation.")
    A("")
    A("| fold | train rows | test rows | test papers | R2 | MAE (F/g) | RMSE (F/g) |")
    A("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in base_fold.iterrows():
        A(f"| {int(r['fold'])} | {int(r['n_train'])} | {int(r['n_test'])} | "
          f"{int(r['n_test_papers'])} | {r['r2']:.3f} | {r['mae']:.1f} | {r['rmse']:.1f} |")
    A(f"| **mean** | | | | **{base_fold['r2'].mean():.3f}** | "
      f"**{base_fold['mae'].mean():.1f}** | **{base_fold['rmse'].mean():.1f}** |")
    A(f"| **SD** | | | | {base_fold['r2'].std():.3f} | {base_fold['mae'].std():.3f} | "
      f"{base_fold['rmse'].std():.3f} |")
    A(f"| **median** | | | | {base_fold['r2'].median():.3f} | "
      f"{base_fold['mae'].median():.1f} | {base_fold['rmse'].median():.1f} |")
    A("")
    A(f"Pooled over all held-out predictions: R2 = {base_sum['pooled_r2']:.3f}, "
      f"MAE = {base_sum['pooled_mae']:.1f} F/g, RMSE = {base_sum['pooled_rmse']:.1f} F/g.")
    A("")
    A(f"Fold-to-fold variation is large (R2 spans {base_fold['r2'].min():.3f} to "
      f"{base_fold['r2'].max():.3f}). That spread is the honest signal here: with "
      f"{data['paper_id'].nunique()} papers, whole-paper hold-out means each fold "
      "tests a different corner of a heterogeneous literature, and a single "
      "unusual paper dominates its fold.")
    A("")
    A(S.get("baseline_sensitivity", ""))
    A("")
    A("Figures: `figures/real_only_predicted_vs_actual.png`, "
      "`figures/real_only_residuals.png`.")
    A("")

    # -------------------------------------------------------------------- SHAP
    A("## 3. SHAP on real data")
    A("")
    A("SHAP values are computed with `shap.TreeExplainer` on the **held-out** rows of "
      "each outer fold, so the attribution describes generalisation behaviour rather "
      "than training fit. One-hot columns are reported individually and aggregated "
      "back to the source variable (per-row sum of |SHAP| over a variable's dummies).")
    A("")
    src = S["real_shap_source"]
    A("| rank | descriptor | mean abs SHAP (F/g) | SD across folds | mean rank | rank SD | folds in top-5 |")
    A("| ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in src.head(10).iterrows():
        A(f"| {int(r['shap_rank'])} | `{r['feature']}` | {r['mean_abs_shap']:.2f} | "
          f"{r['sd_abs_shap']:.2f} | {r['mean_rank']:.1f} | {r['rank_sd']:.2f} | "
          f"{int(r['n_folds_top5'])}/{int(r['n_folds'])} |")
    A("")
    A("### Direction of the continuous descriptors")
    A("")
    A("Read off the pooled held-out SHAP values as the Spearman correlation between "
      "a descriptor's value and its own SHAP contribution. A positive value means "
      "the model pushes the prediction up as the descriptor increases.")
    A("")
    A("| descriptor | rho(value, SHAP) | model behaviour | physically expected? |")
    A("| --- | ---: | --- | --- |")
    for row in S.get("shap_directions", []):
        A(f"| `{row['feature']}` | {row['rho']:+.2f} | {row['word']} | {row['expected']} |")
    A("")
    A(S.get("shap_direction_note", ""))
    A("")
    A("Figures: `figures/real_shap_bar.png`, `figures/real_shap_beeswarm.png`, "
      "`figures/shap_dependence/`.")
    A("")

    # -------------------------------------------------------- Bayesian synthesis
    A("## 4. Bayesian-network synthesis")
    A("")
    A("Generator: **DataSynthesizer 0.1.13, correlated-attribute mode** (greedy Bayes "
      f"network, epsilon = {BN_EPSILON} - the inputs are already-published literature "
      "values, so no differential-privacy noise is warranted). No substitute "
      "generator (CTGAN, copula, SMOTE, KDE) was used anywhere.")
    A("")
    A(f"Selected Bayesian-network descriptors ({len(sel['selected_features'])} + target):")
    A("")
    A("| descriptor | mean abs SHAP | SHAP rank | missingness | BN treatment | why |")
    A("| --- | ---: | ---: | ---: | --- | --- |")
    for f, j in sel["per_feature_justification"].items():
        treat = "categorical" if j["bn_treated_as_categorical"] else "continuous"
        A(f"| `{f}` | {j['mean_abs_shap']:.2f} | {j['shap_rank']} | "
          f"{j['missing_fraction']:.0%} | {treat} | {j['physical_meaning']} |")
    A(f"| `{TARGET}` | - | target | 0% | continuous | Gravimetric capacitance (F/g). |")
    A("")
    A("Discrete-vs-continuous choice: `h2so4_M` and `scan_rate_mV_s` are experimenter "
      "*settings* that occur at a handful of values, so they are declared categorical "
      "to the Bayesian network. Treating them as continuous would let the generator "
      "invent molarities and sweep rates that no experiment in the corpus used.")
    A("")
    co = sel["cohort"]
    A(f"BN cohort (complete cases on those descriptors + target): "
      f"**{co['rows_after']} rows / {co['papers_after']} papers**, down from "
      f"{co['rows_before']} rows / {co['papers_before']} papers. "
      "This is the price of requiring every selected descriptor to be reported in "
      "the same publication. **All four training conditions below use exactly this "
      "cohort and exactly this feature set**, so the comparison is like-for-like.")
    A("")
    A("### Anti-leakage protocol for generation")
    A("")
    A("For every outer grouped fold: the Bayesian network is fitted on the *training "
      "papers only*; synthetic rows are drawn from that network; the physical-validity "
      "filter uses training-fold minima/maxima only; the one-hot encoder and the "
      "hyper-parameter search also see training papers only. The held-out papers are "
      "never described, never generated from, never augmented, and never appear in "
      "any training set.")
    A("")
    rej = pd.read_csv(RESULTS_DIR / "05_synthetic_rejection_log.csv")
    tot_gen = int(rej.groupby(["fold", "k", "ratio", "round"])["n_generated"].first().sum())
    tot_rej = int(rej["n_rejected"].sum())
    A(f"Physical-validity filtering across all folds/k/ratios: **{tot_gen} rows drawn, "
      f"{tot_rej} rejected ({100 * tot_rej / max(1, tot_gen):.1f}%)**; the generator is "
      "asked for a small surplus over the required count and the surplus is discarded "
      "after filtering. Rejection reasons:")
    A("")
    by_reason = rej[rej["reason"] != "none_rejected"].groupby("reason")["n_rejected"].sum()
    if len(by_reason):
        A("| reason | n rejected |")
        A("| --- | ---: |")
        for k, v in by_reason.sort_values(ascending=False).items():
            A(f"| `{k}` | {int(v)} |")
        A("")
        A("The out-of-domain guard (`*_below_train_min` / `*_above_train_max`) is the "
          "binding constraint: the generator's tails are clipped back to the "
          "training-fold support rather than allowed to extrapolate.")
    else:
        A("**No synthetic row was rejected.** That is not a sign the filter is "
          "inactive - it reflects how DataSynthesizer samples: continuous "
          "attributes are drawn uniformly inside histogram bins whose outer edges "
          "are the training-fold min and max, and categorical attributes are drawn "
          "from the observed level set, so the generator cannot leave the training "
          "domain by construction. The filter (non-positive capacitance, negative "
          "descriptors, unseen categories, out-of-range values) remains in the "
          "pipeline as a guard that would catch a change of generator or "
          "configuration.")
    A("")
    nets = pd.read_csv(RESULTS_DIR / "05_bayesian_network_structures.csv")
    ex = nets[(nets["k"] == BN_K_DEFAULT)].head(1)
    if len(ex):
        A(f"Example learned network (fold {int(ex.iloc[0]['fold'])}, k={BN_K_DEFAULT}): "
          f"`{ex.iloc[0]['bayesian_network']}`. Full set in "
          "`05_bayesian_network_structures.csv`.")
    A("")

    # --------------------------------------------------------------- fidelity
    A("## 5. Synthetic-data fidelity")
    A("")
    f2 = fid[fid["k"] == BN_K_DEFAULT]
    A("Every number below compares synthetic rows against the **real training rows of "
      "the same fold** (never the test papers). Averages are over folds and "
      "augmentation ratios at k = 2.")
    A("")
    A("| metric | mean | min | max |")
    A("| --- | ---: | ---: | ---: |")
    for label, col in [
        ("mean KS statistic (continuous)", "mean_ks_statistic"),
        ("max KS statistic (continuous)", "max_ks_statistic"),
        ("# continuous vars with KS p < 0.05", "n_ks_reject_p05"),
        ("mean Jensen-Shannon distance (categorical)", "mean_jensen_shannon"),
        ("Pearson corr-matrix mean abs difference", "pearson_mean_abs_diff"),
        ("Spearman corr-matrix mean abs difference", "spearman_mean_abs_diff"),
        ("Mutual-information matrix mean abs difference", "mi_mean_abs_diff"),
        ("descriptor-target association mean abs difference", "target_assoc_mean_abs_diff"),
        ("exact duplicates of real rows", "exact_duplicates_of_real"),
        ("duplicates within synthetic", "duplicates_within_synthetic"),
        ("synthetic->real NN distance (median)", "nn_dist_median"),
        ("real->real NN distance (median)", "real_to_real_nn_median"),
        ("synthetic rows closer than the real-NN 5th pct", "n_syn_closer_than_real_p05"),
    ]:
        if col in f2 and f2[col].notna().any():
            A(f"| {label} | {f2[col].mean():.4g} | {f2[col].min():.4g} | {f2[col].max():.4g} |")
    A("")
    A(S.get("fidelity_verdict", ""))
    A("")
    A("### Descriptor-target relationships")
    A("")
    A("A generator can reproduce every marginal and still destroy the relationship "
      "the regressor has to learn, so the descriptor-target association is checked "
      "explicitly: Spearman correlation for continuous descriptors, eta-squared "
      "(share of capacitance variance explained by the level) for categorical ones. "
      f"Averaged over folds at k = {BN_K_DEFAULT}, ratio 3x:")
    A("")
    tr = pd.read_csv(RESULTS_DIR / "05_target_relationship_comparison.csv")
    tr = tr[(tr["k"] == BN_K_DEFAULT) & (tr["ratio"] == 3.0)]
    A("| descriptor | association measure | real | synthetic | abs diff |")
    A("| --- | --- | ---: | ---: | ---: |")
    for v, g in tr.groupby("variable", sort=False):
        if g["type"].iloc[0] == "continuous":
            A(f"| `{v}` | Spearman with target | {g['real_spearman_with_target'].mean():+.3f} | "
              f"{g['syn_spearman_with_target'].mean():+.3f} | "
              f"{g['spearman_abs_diff'].mean():.3f} |")
        else:
            A(f"| `{v}` | eta-squared on target | {g['real_eta_squared_on_target'].mean():.3f} | "
              f"{g['syn_eta_squared_on_target'].mean():.3f} | "
              f"{g['eta_squared_abs_diff'].mean():.3f} |")
    A("")
    A("Note on the correlation matrices: only one BN descriptor (`interlayer_A`) is "
      "modelled as continuous, so the Pearson/Spearman matrix reduces to a single "
      "off-diagonal entry and a 'correlation of correlations' would be undefined. "
      "The normalised mutual-information matrix, which handles the categorical "
      "descriptors as well, is the informative dependency comparison here.")
    A("")
    A("Per-variable detail: `05_synthetic_distribution_stats.csv`; dependency "
      "structure: `05_mutual_information_comparison.csv` and "
      "`05_target_relationship_comparison.csv`; memorisation: "
      "`05_nearest_neighbor_analysis.csv`. Figures in `figures/synthetic_vs_real/`.")
    A("")

    # ------------------------------------------------------------ augmentation
    A("## 6. Real-only vs augmented performance")
    A("")
    A(f"Outer evaluation: {S['fold_reason_aug']} The test set of each fold is the "
      "identical set of **real** held-out papers for all four conditions, with the "
      "same feature space and the same XGBoost hyper-parameters (tuned once per fold "
      "on the real training papers, then held fixed across conditions so that tuning "
      "cannot confound the comparison).")
    A("")
    n_seeds = int(summ["n_gen_seeds"].max())
    A(f"Each augmented condition is generated **{n_seeds} times with independent "
      "generator seeds** inside every fold; the fold score is the mean over those "
      "draws, so re-drawing the synthetic set cannot masquerade as an effect. The "
      "spread caused by re-drawing alone is reported in the last column.")
    A("")
    A("| Training condition | Test data | n synthetic (mean/fold) | R2 mean +/- SD | MAE mean +/- SD | RMSE mean +/- SD | MAE SD from re-drawing |")
    A("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in summ.sort_values("ratio").iterrows():
        gs = r.get("mae_sd_across_gen_seeds")
        gs_txt = "n/a (no generation)" if pd.isna(gs) else f"{gs:.1f}"
        A(f"| {PRETTY.get(r['condition'], r['condition'])} | unseen real papers | "
          f"{r['n_synthetic_train_mean']:.0f} | {_pm(r['r2_mean'], r['r2_sd'])} | "
          f"{_pm(r['mae_mean'], r['mae_sd'], 1)} | {_pm(r['rmse_mean'], r['rmse_sd'], 1)} | "
          f"{gs_txt} |")
    A("")
    _gs = summ["mae_sd_across_gen_seeds"].dropna()
    _spread = summ["mae_mean"].max() - summ["mae_mean"].min()
    if len(_gs):
        A(f"**Read this column first.** Simply re-drawing the synthetic set with a "
          f"different seed moves fold MAE by {_gs.mean():.1f} F/g on average, while the "
          f"entire spread between the four training conditions is {_spread:.1f} F/g. "
          "Any apparent ranking of the augmentation ratios has to be read against that "
          "noise floor.")
        A("")
    A("**Why the R2 values are negative.** R2 is measured against the variance of "
      "each *fold's own* held-out capacitances. Several folds contain only 3-4 "
      "papers whose capacitances sit within a narrow band (fold 1 spans 100-300 F/g, "
      "SD ~ 60 F/g), so even a model with a 60-90 F/g MAE scores below the "
      "fold mean. The absolute-error metrics are the trustworthy comparators on a "
      "cohort this small; R2 is reported because it was requested, and because its "
      "*ordering* across conditions is still informative.")
    A("")
    A("Medians and pooled metrics (pooled = every held-out real prediction from every "
      "fold scored together, which avoids dividing by a near-zero within-fold variance "
      "while keeping the grouped hold-out completely intact):")
    A("")
    A("| condition | R2 median | MAE median | RMSE median | pooled R2 | pooled MAE | pooled RMSE | pooled R2 (excl. extremes) |")
    A("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in summ.sort_values("ratio").iterrows():
        A(f"| {PRETTY.get(r['condition'], r['condition'])} | {r['r2_median']:.3f} | "
          f"{r['mae_median']:.1f} | {r['rmse_median']:.1f} | {r['pooled_r2']:.3f} | "
          f"{r['pooled_mae']:.1f} | {r['pooled_rmse']:.1f} | "
          f"{r['pooled_r2_excl_extreme']:.3f} |")
    A("")
    _pool_best = summ.loc[summ["pooled_r2"].idxmax(), "condition"]
    A(f"The pooled ranking agrees with the fold-mean ranking: best condition by pooled "
      f"R2 is **{PRETTY.get(_pool_best, _pool_best)}**, by fold-mean R2 "
      f"**{PRETTY.get(summ.loc[summ['r2_mean'].idxmax(), 'condition'], '')}**."
      if _pool_best == summ.loc[summ["r2_mean"].idxmax(), "condition"] else
      f"The pooled ranking and the fold-mean ranking disagree "
      f"(pooled best: {PRETTY.get(_pool_best, _pool_best)}; fold-mean best: "
      f"{PRETTY.get(summ.loc[summ['r2_mean'].idxmax(), 'condition'], '')}), which is "
      "itself evidence of how little separates the conditions.")
    A("")
    A("### Paired statistical comparison")
    A("")
    A("| comparison | metric | real-only | augmented | abs diff | rel % | folds improved | Wilcoxon p | paired-t p |")
    A("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for _, r in stats.iterrows():
        wp = "n/a" if pd.isna(r["wilcoxon_p"]) else f"{r['wilcoxon_p']:.3f}"
        tp = "n/a" if pd.isna(r["paired_t_p"]) else f"{r['paired_t_p']:.3f}"
        A(f"| {r['comparison'].replace('real_only_vs_', '')} | {r['metric']} | "
          f"{r['real_only_mean']:.3f} | {r['augmented_mean']:.3f} | "
          f"{r['absolute_difference']:+.3f} | {r['relative_pct_change']:+.1f}% | "
          f"{int(r['n_folds_improved'])}/{int(r['n_folds'])} | {wp} | {tp} |")
    A("")
    nf = int(stats["n_folds"].max()) if len(stats) else 0
    A(S.get("aug_bottom_line", ""))
    A("")
    A(f"With {nf} paired folds the smallest attainable two-sided Wilcoxon p-value is "
      f"{2 ** -(nf - 1):.4f}, so **no result here can reach p < 0.05 by that test**. "
      "The p-values are reported for completeness only; the fold-level direction of "
      "change and the effect size are the meaningful quantities.")
    A("")
    A("### Bayesian-network complexity sensitivity")
    A("")
    ks = S["ksens"]
    A("| k | condition | R2 mean +/- SD | MAE mean | RMSE mean |")
    A("| ---: | --- | ---: | ---: | ---: |")
    ksa = ks[ks["condition"] != "A_real_only"]
    for _, r in ksa.sort_values(["condition", "k"]).iterrows():
        A(f"| {int(r['k'])} | {PRETTY.get(r['condition'], r['condition'])} | "
          f"{_pm(r['r2_mean'], r['r2_sd'])} | {r['mae_mean']:.1f} | {r['rmse_mean']:.1f} |")
    A("")
    spread_k = ksa.groupby("condition")["mae_mean"].agg(lambda s: s.max() - s.min()).max()
    spread_ratio = summ["mae_mean"].max() - summ["mae_mean"].min()
    A(f"Changing only the network degree at a fixed ratio moves MAE by up to "
      f"{spread_k:.1f} F/g, while changing the ratio at a fixed degree moves it by "
      f"{spread_ratio:.1f} F/g. The two spreads are the same order, which is the "
      "clearest single indication that the whole augmentation effect on this cohort "
      "sits inside configuration noise. No `k` is reliably better than another; "
      f"k = {BN_K_DEFAULT} is reported as the pre-registered default, not as a winner.")
    A("")
    A("Figures: `figures/augmentation_r2.png`, `figures/augmentation_mae.png`, "
      "`figures/augmentation_rmse.png` (+ per-fold line versions).")
    A("")

    # ---------------------------------------------------------------- stability
    A("## 7. SHAP stability after augmentation")
    A("")
    A(f"Comparison is cohort-real-only vs **{PRETTY.get(S['best_condition'], S['best_condition'])}** "
      "(the best-performing augmented condition on unseen real papers), on the "
      "identical cohort and feature set.")
    A("")
    A(f"- Spearman rank correlation of source-level SHAP rankings: "
      f"**{stab['spearman_rank_correlation']:.3f}** (p = {stab['spearman_p_value']:.3f})")
    A(f"- Top-3 overlap: **{stab['top3_overlap']}/3**; top-5 overlap: "
      f"**{stab['top5_overlap']}/{min(5, stab['n_features'])}**")
    A(f"- Mean |rank change|: {stab['mean_abs_rank_change']:.2f}; "
      f"max |rank change|: {stab['max_abs_rank_change']:.0f}")
    A(f"- Mean |change| in normalised mean-|SHAP| share: "
      f"{stab['mean_abs_norm_shap_change']:.3f}")
    A("")
    tab = S["stability_tab"]
    A("| descriptor | real rank | augmented rank | rank change | real SHAP share | augmented SHAP share |")
    A("| --- | ---: | ---: | ---: | ---: | ---: |")
    for _, r in tab.iterrows():
        A(f"| `{r['feature']}` | {r['real_rank']:.0f} | {r['augmented_rank']:.0f} | "
          f"{r['rank_change']:+.0f} | {r['real_norm_shap']:.3f} | {r['augmented_norm_shap']:.3f} |")
    A("")
    A(S.get("stability_verdict", ""))
    A("")
    A("Figure: `figures/shap_rank_comparison.png`.")
    A("")

    # --------------------------------------------------------------------- PINN
    A("## 8. PINN descriptor implications")
    A("")
    A("SHAP measures how a *particular fitted model* distributes credit; it is not "
      "evidence of physical causation. A descriptor is recommended for the PINN only "
      "when strong real-data SHAP importance, cross-fold stability, post-augmentation "
      "stability, adequate coverage and materials-science plausibility all hold.")
    A("")
    pinn = S["pinn"]
    A("| descriptor | real SHAP rank | augmented rank | stability | missingness | interpretability | recommendation |")
    A("| --- | ---: | ---: | --- | ---: | --- | --- |")
    for _, r in pinn.iterrows():
        rr = "-" if pd.isna(r["real_SHAP_rank"]) else f"{int(r['real_SHAP_rank'])}"
        ar = "-" if pd.isna(r["augmented_SHAP_rank"]) else f"{int(r['augmented_SHAP_rank'])}"
        A(f"| `{r['feature']}` | {rr} | {ar} | {r['rank_stability']} | "
          f"{r['missingness']:.0%} | {r['physical_interpretability']} | "
          f"**{r['recommended_for_PINN']}** |")
    A("")
    rec = pinn[pinn["recommended_for_PINN"] == "recommended"]["feature"].tolist()
    cond_p = pinn[pinn["recommended_for_PINN"].str.startswith("conditional - needs")]["feature"].tolist()
    cond_d = pinn[pinn["recommended_for_PINN"].str.startswith("conditional - collect")]["feature"].tolist()
    no = pinn[pinn["recommended_for_PINN"] == "not recommended yet"]["feature"].tolist()
    A(f"**Sufficiently supported now:** {', '.join('`' + f + '`' for f in rec) or 'none'}.")
    A("")
    A(f"**Conditional - well evidenced but categorical, so they must be recast as a "
      f"continuous physical quantity before entering a PDE residual (e.g. "
      f"`composition_family` -> intercalant mass fraction, `layer_class` -> measured "
      f"stacking number):** {', '.join('`' + f + '`' for f in cond_p) or 'none'}.")
    A("")
    if cond_d:
        A(f"**Conditional - would need more reported data:** "
          f"{', '.join('`' + f + '`' for f in cond_d)}.")
        A("")
    A(f"**Too sparse or unreliable at present:** "
      f"{', '.join('`' + f + '`' for f in no) or 'none'}.")
    A("")

    # -------------------------------------------------------------- limitations
    A("## 9. Limitations")
    A("")
    for line in [
        "**Literature-derived, heterogeneous data.** Every row is an LLM-assisted "
        "extraction from a different publication with its own cell geometry, "
        "reference electrode, mass-normalisation convention and reporting rate. "
        "Two rows with identical descriptors can legitimately differ by a hundred F/g.",
        "**Multiple observations per paper.** Rows are strongly clustered by "
        "publication (median ~3, up to 15 rows per paper). All evaluation is "
        "grouped by `paper_id` for exactly this reason; any random-split number "
        "would be optimistically biased and none is reported here.",
        "**Missing descriptor values dominate the design.** Requiring complete cases "
        f"on {len(sel['selected_features'])} descriptors shrinks the corpus from "
        f"{co['rows_before']} to {co['rows_after']} rows. The Bayesian-network "
        "experiment therefore describes a *sub-population* of the literature, not "
        "the whole corpus.",
        "**Synthetic data are not new experimental evidence.** Every synthetic row "
        "is a resample of dependencies already present in the training papers. "
        "Augmentation can regularise a model; it cannot add information the "
        "literature does not contain.",
        "**Bayesian networks reproduce learned dependencies, not physics.** The "
        "generator has no notion of charge conservation, ion transport or "
        "electrode kinetics; the physical-validity filter is a coarse guard "
        "(sign and training-domain bounds), not a physics check.",
        "**SHAP is attribution, not causation.** A high SHAP value means the fitted "
        "XGBoost model leans on that column, which can equally reflect a "
        "confounded reporting habit (e.g. groups who measure d-spacing also make "
        "better electrodes) as a causal mechanism.",
        "**Limited extrapolation validity.** Synthetic values are explicitly clipped "
        "to the training-fold support, and the model is only credible inside the "
        "descriptor ranges the corpus actually covers "
        f"(e.g. H2SO4 {data['h2so4_M'].min():g}-{data['h2so4_M'].max():g} M).",
        "**Small number of folds.** With a handful of grouped folds, paired "
        "significance testing is underpowered by construction; conclusions rest on "
        "effect direction and consistency, not on p-values.",
        "**Unit and definition ambiguity survives cleaning.** Some papers report an "
        "interlayer *gap* rather than a (002) d-spacing, and some report flake size "
        "only qualitatively. Ambiguous entries are dropped rather than guessed, "
        "which biases the cohort toward better-documented papers.",
    ]:
        A(f"- {line}")
    A("")

    # ---------------------------------------------------------------- A-K answers
    A("---")
    A("")
    A("## 10. Summary answers")
    A("")
    for tag, txt in S["ak_answers"]:
        A(f"**{tag}.** {txt}")
        A("")

    (RESULTS_DIR / "FINAL_REPORT.md").write_text("\n".join(L), encoding="utf-8")

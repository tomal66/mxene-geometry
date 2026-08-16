"""CSV -> LaTeX table fragments (booktabs / siunitx). No metric is hard-coded."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _esc(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return "--"
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#"), ("$", r"\$"), ("{", r"\{"),
                 ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")]:
        s = s.replace(a, b)
    return s


def _num(v, dp=2, na="--") -> str:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return na
    return f"{v:.{dp}f}"


def write_table(path: Path, colspec: str, header: list[str], rows: list[list[str]],
                caption: str, label: str, note: str = "",
                small: bool = True, width_cmd: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    L = [r"\begin{table}[htbp]", r"\centering",
         rf"\caption{{{caption}}}", rf"\label{{{label}}}"]
    if small:
        L.append(r"\footnotesize")
    if width_cmd:
        L.append(width_cmd)
    L += [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
          " & ".join(header) + r" \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(rf"\par\vspace{{2pt}}\begin{{minipage}}{{\linewidth}}\scriptsize {note}\end{{minipage}}")
    L.append(r"\end{table}")
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------

def dataset_summary_table(stats: dict, out: Path) -> None:
    # keys and values are authored with intentional LaTeX, so they are NOT escaped
    rows = [[str(k), str(v)] for k, v in stats.items()]
    write_table(
        out, "lr", [r"\textbf{Quantity}", r"\textbf{Value}"], rows,
        caption="Composition of the literature-derived H\\textsubscript{2}SO\\textsubscript{4} "
                "MXene corpus after target cleaning and descriptor standardisation.",
        label="tab:dataset",
        note="Range and inequality literals (e.g.\\ \\texttt{300--350}, \\texttt{over 200}) were "
             "discarded rather than converted to midpoints; every discarded row is logged in "
             "\\texttt{01\\_target\\_disposition\\_log.csv}.")


def grouped_metrics_table(fold: pd.DataFrame, summ: pd.Series, out: Path) -> None:
    rows = []
    for _, r in fold.sort_values("fold").iterrows():
        rows.append([str(int(r["fold"])), str(int(r["n_train"])), str(int(r["n_test"])),
                     str(int(r["n_test_papers"])), _num(r["r2"], 3),
                     _num(r["mae"], 1), _num(r["rmse"], 1)])
    rows.append([r"\midrule \textbf{Mean}", "", "", "",
                 r"\textbf{" + _num(fold["r2"].mean(), 3) + "}",
                 r"\textbf{" + _num(fold["mae"].mean(), 1) + "}",
                 r"\textbf{" + _num(fold["rmse"].mean(), 1) + "}"])
    rows.append(["SD", "", "", "", _num(fold["r2"].std(), 3),
                 _num(fold["mae"].std(), 1), _num(fold["rmse"].std(), 1)])
    rows.append(["Median", "", "", "", _num(fold["r2"].median(), 3),
                 _num(fold["mae"].median(), 1), _num(fold["rmse"].median(), 1)])
    rows.append(["Pooled", "", "", "", _num(summ["pooled_r2"], 3),
                 _num(summ["pooled_mae"], 1), _num(summ["pooled_rmse"], 1)])
    write_table(
        out, "lrrrrrr",
        [r"\textbf{Fold}", r"\textbf{$n_{\mathrm{train}}$}", r"\textbf{$n_{\mathrm{test}}$}",
         r"\textbf{Papers}", r"\textbf{$R^2$}", r"\textbf{MAE}", r"\textbf{RMSE}"],
        rows,
        caption="Paper-grouped cross-validation of the real-only XGBoost model on the full "
                "descriptor set. MAE and RMSE in F\\,g\\textsuperscript{-1}. Rows from one "
                "publication never appear in both the training and test partition.",
        label="tab:grouped",
        note="``Pooled'' scores every held-out prediction from every fold together, which "
             "avoids normalising by a near-zero within-fold target variance while keeping the "
             "grouped hold-out intact.")


def augmentation_table(summ: pd.DataFrame, out: Path) -> None:
    pretty = {"A_real_only": "Real only", "B_real_plus_1.0x": r"Real $+\,1\times$",
              "C_real_plus_3.0x": r"Real $+\,3\times$",
              "D_real_plus_5.25x": r"Real $+\,5.25\times$"}
    rows = []
    for _, r in summ.sort_values("ratio").iterrows():
        gs = r.get("mae_sd_across_gen_seeds")
        rows.append([
            pretty.get(r["condition"], _esc(r["condition"])),
            f"{r['n_synthetic_train_mean']:.0f}",
            f"{_num(r['r2_mean'], 2)} $\\pm$ {_num(r['r2_sd'], 2)}",
            f"{_num(r['mae_mean'], 1)} $\\pm$ {_num(r['mae_sd'], 1)}",
            f"{_num(r['rmse_mean'], 1)} $\\pm$ {_num(r['rmse_sd'], 1)}",
            _num(r["pooled_mae"], 1),
            _num(gs, 1),
        ])
    write_table(
        out, "lrrrrrr",
        [r"\textbf{Training set}", r"\textbf{$n_{\mathrm{syn}}$}", r"\textbf{$R^2$}",
         r"\textbf{MAE}", r"\textbf{RMSE}", r"\textbf{Pooled MAE}",
         r"\textbf{Seed SD}"],
        rows,
        caption="Real-only versus Bayesian-augmented training, evaluated on the identical set "
                "of held-out real papers in every condition ($k=2$, five grouped folds, three "
                "independent generator seeds averaged per fold). Errors in "
                "F\\,g\\textsuperscript{-1}; $\\pm$ denotes the fold-to-fold SD.",
        label="tab:augmentation",
        note="``Seed SD'' is the within-fold MAE standard deviation across the three generator "
             "seeds, i.e.\\ the score movement produced by re-drawing the synthetic set alone. "
             "It is of the same magnitude as the entire spread between augmentation ratios, "
             "and for the $+3\\times$ condition it exceeds it.")


def fidelity_table(fid: pd.DataFrame, out: Path) -> None:
    spec = [
        ("Mean KS statistic (continuous)", "mean_ks_statistic", 3),
        ("Continuous variables with KS $p<0.05$", "n_ks_reject_p05", 1),
        ("Mean Jensen--Shannon distance (categorical)", "mean_jensen_shannon", 3),
        ("Pearson correlation matrix, mean $|\\Delta|$", "pearson_mean_abs_diff", 3),
        ("Spearman correlation matrix, mean $|\\Delta|$", "spearman_mean_abs_diff", 3),
        ("Mutual-information matrix, mean $|\\Delta|$", "mi_mean_abs_diff", 3),
        ("Descriptor--target association, mean $|\\Delta|$", "target_assoc_mean_abs_diff", 3),
        ("Exact duplicates of a real training row", "exact_duplicates_of_real", 1),
        ("Duplicates within a synthetic set", "duplicates_within_synthetic", 1),
        ("Synthetic$\\rightarrow$real NN distance (median)", "nn_dist_median", 4),
        ("Real$\\rightarrow$real NN distance (median)", "real_to_real_nn_median", 4),
    ]
    rows = []
    for lab, col, dp in spec:
        if col not in fid:
            continue
        rows.append([lab, _num(fid[col].mean(), dp),
                     _num(fid[col].min(), dp), _num(fid[col].max(), dp)])
    write_table(
        out, "lrrr",
        [r"\textbf{Fidelity metric}", r"\textbf{Mean}", r"\textbf{Min}", r"\textbf{Max}"],
        rows,
        caption="Fidelity of the Bayesian-network synthetic observations, computed against the "
                "real training rows of the same fold ($k=2$; mean, min and max over folds, "
                "augmentation ratios and generator seeds).",
        label="tab:fidelity",
        note="Nearest-neighbour distances use a range-normalised mixed-type (Gower) metric. "
             "No synthetic row reproduced a real training row exactly.")


def shap_table(grouped: pd.DataFrame, directions: pd.DataFrame, out: Path,
               top_n: int = 10) -> None:
    dd = directions.set_index("descriptor")
    rows = []
    for _, r in grouped.head(top_n).iterrows():
        f = r["descriptor"]
        rho = dd.loc[f, "spearman_value_vs_shap"] if f in dd.index else np.nan
        rows.append([
            str(int(r["rank"])), _TEXNAME.get(r["descriptor"], _esc(r["display_name"])),
            _num(r["mean_abs_shap"], 2), _num(100 * r["normalized_importance"], 1),
            _num(r["missingness"] * 100, 0),
            _num(r["mean_rank_across_folds"], 1),
            _num(r["rank_SD"], 2),
            _num(100 * r["top5_frequency"], 0) if not pd.isna(r["top5_frequency"]) else "--",
            _num(rho, 2),
        ])
    write_table(
        out, "rlrrrrrrr",
        [r"\textbf{Rank}", r"\textbf{Descriptor}", r"\textbf{$\overline{|\mathrm{SHAP}|}$}",
         r"\textbf{Share}", r"\textbf{Miss.}", r"\textbf{Fold rank}", r"\textbf{Rank SD}",
         r"\textbf{Top-5}", r"\textbf{$\rho$}"],
        rows,
        caption="Descriptor-level SHAP attribution of the final real-only interpretation model "
                "fitted to all usable real observations. $\\overline{|\\mathrm{SHAP}|}$ in "
                "F\\,g\\textsuperscript{-1}; share, missingness and top-5 frequency in per cent.",
        label="tab:shap",
        note="``Fold rank'' and ``Rank SD'' come from the independent held-out SHAP computation "
             "of the paper-grouped cross-validation; ``Top-5'' is the fraction of folds in which "
             "the descriptor entered the five most important. $\\rho$ is the Spearman correlation "
             "between a descriptor's value and its own SHAP value (direction of the association, "
             "not a causal claim); it is defined for continuous descriptors only.")


_TEXNAME = {
    "interlayer_A": r"Interlayer spacing (\AA)",
    "h2so4_M": r"H\textsubscript{2}SO\textsubscript{4} conc.\ (M)",
    "scan_rate_mV_s": r"Scan rate (mV\,s$^{-1}$)",
    "current_density_A_g": r"Current density (A\,g$^{-1}$)",
    "mass_loading_mg_cm2": r"Mass loading (mg\,cm$^{-2}$)",
    "electrode_thickness_um": r"Electrode thickness ($\upmu$m)",
    "ssa_m2_g": r"Specific surface area (m$^2$\,g$^{-1}$)",
    "flake_size_um": r"Flake size ($\upmu$m)",
    "pore_diameter_nm": "Pore diameter (nm)",
    "layer_class": "Layer morphology",
    "composition_family": "Composition family",
    "synthesis_family": "Synthesis route",
    "electrolyte_is_gel": "Gel electrolyte",
}


def pinn_table(cls: pd.DataFrame, out: Path) -> None:
    cat_order = {"A": 0, "B": 1, "C": 2}
    d = cls.copy()
    d["_o"] = d["category"].str[0].map(cat_order).fillna(9)
    d = d.sort_values(["_o", "SHAP_rank"], na_position="last")
    rows = []
    for _, r in d.iterrows():
        rows.append([
            _TEXNAME.get(r["descriptor"], _esc(r["descriptor"])),
            "--" if pd.isna(r["SHAP_rank"]) else str(int(r["SHAP_rank"])),
            _num(r["missing_fraction"] * 100, 0),
            _esc(r["SHAP_stability"]),
            _esc(r["category"].split(" -- ")[0]),
            _esc(str(r["recommended_as_PINN_input"]).replace(
                "yes (as a closure variable)", "yes (closure)")),
        ])
    write_table(
        out, "lrrlll",
        [r"\textbf{Descriptor}", r"\textbf{SHAP rank}", r"\textbf{Miss.\ (\%)}",
         r"\textbf{Stability}", r"\textbf{Category}", r"\textbf{PINN input?}"],
        rows,
        caption="Classification of every standardised descriptor for the next-stage "
                "physics-guided model. Category A: physically meaningful, well-covered and "
                "stable inputs. Category B: informative machine-learning covariates that are "
                "categorical proxies rather than physical parameters. Category C: excluded.",
        label="tab:pinn",
        note="Stability combines the cross-fold SHAP rank spread with the rank change observed "
             "between the real-only and Bayesian-augmented models. Full reasoning is in "
             "\\texttt{results\\_final/pinn/descriptor\\_classification.csv}.")


def physics_availability_table(avail: pd.DataFrame, out: Path) -> None:
    rows = []
    for _, r in avail.iterrows():
        rows.append([
            _esc(r["variable"].replace("_", " ")),
            f"${_esc(r['symbol'])}$" if r["symbol"] != "-" else "--",
            str(int(r["availability_count"])),
            _num(r["availability_fraction"] * 100, 0),
            _esc(r["usable_now"]),
        ])
    write_table(
        out, "llrrl",
        [r"\textbf{Variable}", r"\textbf{Symbol}", r"\textbf{$n$}",
         r"\textbf{Coverage (\%)}", r"\textbf{Usable now}"],
        rows,
        caption="Availability of the quantities required by the standard electrochemical "
                "capacitance relations, evaluated over the usable real observations.",
        label="tab:physavail",
        note="Coverage is the fraction of usable rows for which the quantity can be parsed with "
             "an unambiguous unit. The four quantities required by "
             "$C_g = I\\,\\Delta t/(m\\,\\Delta V)$ that are entirely absent from the corpus are "
             "the reason a classical PDE-based PINN cannot presently be posed.")


def closure_table(summary: pd.DataFrame, out: Path) -> None:
    rows = []
    for _, r in summary.iterrows():
        rows.append([
            "$" + r["identity"].replace("C_A", "C_A").replace("C_g", "C_g")
                 .replace("C_V", "C_V").replace("m_A", "m_A").replace("*", r"\cdot")
                 .replace(" = ", " = ").replace("/ t", "/t") + "$",
            str(int(r["n_testable_rows"])), str(int(r["n_papers"])),
            _num(100 * r["median_relative_error"], 2),
            _num(100 * r["frac_within_5pct"], 0),
            _num(100 * r["frac_within_20pct"], 0),
            str(int(r["n_gross_violations_gt50pct"])),
        ])
    write_table(
        out, "lrrrrrr",
        [r"\textbf{Closure identity}", r"\textbf{$n$ rows}", r"\textbf{Papers}",
         r"\textbf{Median err.\ (\%)}", r"\textbf{$<5\%$}", r"\textbf{$<20\%$}",
         r"\textbf{$>50\%$}"],
        rows,
        caption="Empirical verification of the algebraic capacitance closure identities against "
                "the values reported in the source publications. These identities are the "
                "physics that the corpus can actually enforce.",
        label="tab:closure",
        note="$m_A$ is areal mass loading and $t$ electrode thickness, so $m_A/t$ is the "
             "electrode density. Rows failing by more than 50\\% are dominated by publications "
             "whose mass-loading convention differs from the one implied by their reported "
             "areal capacitance, which is itself evidence of cross-study protocol heterogeneity.")


def heterogeneity_table(het: pd.DataFrame, out: Path) -> None:
    d = het[het["eta_squared"].notna()].sort_values("eta_squared", ascending=False)
    rows = []
    for _, r in d.iterrows():
        rows.append([_esc(r["variable"].replace("_", " ")), _esc(r["role"]),
                     str(int(r["n"])), str(int(r["n_groups"])),
                     _num(r["eta_squared"], 3), _num(r["icc1"], 3)])
    write_table(
        out, "llrrrr",
        [r"\textbf{Variable}", r"\textbf{Role}", r"\textbf{$n$}", r"\textbf{Papers}",
         r"\textbf{$\eta^2$}", r"\textbf{ICC(1)}"],
        rows,
        caption="One-way variance decomposition by source publication. $\\eta^2$ is the share of "
                "total sum-of-squares lying between papers; ICC(1) is the corresponding "
                "intraclass correlation corrected for unequal group sizes.",
        label="tab:heterogeneity",
        note="Values approaching unity mean a variable is essentially constant within a "
             "publication and varies only between publications, so holding out a whole paper "
             "requires the model to extrapolate rather than interpolate.")

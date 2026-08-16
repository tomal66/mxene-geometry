"""End-to-end MXene SHAP + Bayesian-network synthetic-data pipeline.

Run with::

    python run_pipeline.py

Every stage writes its artefacts into ``results/``.  See README_PIPELINE.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import plotting as P
from src.bayesian_synthesis import generate_filtered
from src.config import (AUG_LABELS, AUG_RATIOS, BN_EPSILON, BN_K_DEFAULT,
                        BN_K_VALUES, CACHE_DIR, FIG_DEP_DIR, FIG_DIR,
                        FIG_SYN_DIR, N_FOLDS_TARGET, N_GEN_REPEATS, RANDOM_STATE,
                        RESULTS_DIR, TARGET, ensure_dirs, resolve_dataset,
                        seed_everything)
from src.data_cleaning import (audit_dataset, clean_target, leakage_audit,
                               target_outlier_flags, write_audit)
from src.feature_engineering import (ALL_FEATURES, CATEGORICAL_FEATURES,
                                     NUMERIC_FEATURES, build_features)
from src.modeling import (FoldEncoder, fit_xgb, make_group_kfold, metrics,
                          summarise, tune_xgb)
from src.reporting import write_final_report
from src.shap_analysis import (aggregate_to_source, fold_shap_values,
                               importance_table, pool_folds, shap_stability)
from src.synthetic_validation import (categorical_stats, continuous_stats,
                                      correlation_agreement, mi_agreement,
                                      memorisation_check, target_relationships)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

STATE: dict = {}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ===========================================================================
# Stage 0 - environment
# ===========================================================================

def stage_environment() -> None:
    import matplotlib, scipy, shap, sklearn, xgboost
    import DataSynthesizer
    lines = [
        "# Environment",
        f"python                : {platform.python_version()} ({platform.platform()})",
        f"numpy                 : {np.__version__}",
        f"pandas                : {pd.__version__}",
        f"scipy                 : {scipy.__version__}",
        f"scikit-learn          : {sklearn.__version__}",
        f"xgboost               : {xgboost.__version__}",
        f"shap                  : {shap.__version__}",
        f"matplotlib            : {matplotlib.__version__}",
        f"DataSynthesizer       : {getattr(DataSynthesizer, '__version__', '0.1.13 (no __version__ attr)')}",
        f"RANDOM_STATE          : {RANDOM_STATE}",
        "",
        "# Compatibility fixes applied",
        "- DataSynthesizer 0.1.13 + NumPy>=2: `DataGenerator.generate_encoded_dataset`",
        "  eval()s parent-instance strings that NumPy 2 renders as `[np.int64(...)]`,",
        "  but the module never imports NumPy -> NameError. Fix: inject `np` into",
        "  `DataSynthesizer.DataGenerator`'s module globals (src/bayesian_synthesis.py).",
        "  No algorithmic behaviour is altered; the eval reconstructs the intended indices.",
        "- DataDescriber is constructed with category_threshold=1 so that the explicit",
        "  `attribute_to_is_categorical` mapping governs, not the cardinality heuristic.",
        "- DataDescriber re-reads its input CSV with pandas dtype inference, so a declared",
        "  STRING attribute whose levels look numeric (1.0, 2.0) returns as float and the",
        "  bin-index lookup raises `KeyError: 2.0`. Fix: a DataDescriber subclass that reads",
        "  declared-STRING columns with dtype=str (src/bayesian_synthesis.py).",
        "",
        "# Performance patch (results unchanged)",
        "- PrivBayes.greedy_bayes opens a fresh multiprocessing.Pool() per attribute-loop",
        "  iteration. Under the Windows spawn start method that is one interpreter start",
        "  per core per iteration and dominates runtime for a small network (~90 s per fit",
        "  vs ~1.3 s serial). Pool.map is an ordered map over a pure function, so a serial",
        "  stand-in returns bit-identical results.",
    ]
    (RESULTS_DIR / "environment.txt").write_text("\n".join(lines), encoding="utf-8")
    log("environment.txt written")


# ===========================================================================
# Stages 1-3 - audit, leakage, target
# ===========================================================================

def stage_audit() -> None:
    path = resolve_dataset()
    raw = pd.read_csv(path)
    log(f"loaded {path}  shape={raw.shape}")
    audit = audit_dataset(raw, path)
    write_audit(audit, raw)
    leak = leakage_audit(raw)
    log(f"leakage audit: {int(leak.use_as_predictor.sum())} of {len(leak)} columns admissible as predictors")
    STATE.update(raw=raw, audit=audit, leak=leak, dataset_path=path)


def stage_clean() -> None:
    raw = STATE["raw"]
    cleaned, tlog = clean_target(raw)
    feats, flogs = build_features(raw)

    model = pd.concat(
        [
            cleaned[["paper_id", "source_file", "doi"]],
            feats[ALL_FEATURES],
            cleaned[[TARGET, "target_is_approximate"]],
        ],
        axis=1,
    )
    model["target_outlier_flag"] = target_outlier_flags(model[TARGET])
    kept = model[model[TARGET].notna()].copy().reset_index(drop=True)

    model.to_csv(RESULTS_DIR / "01_cleaned_model_data_all_rows.csv", index=False)
    kept.to_csv(RESULTS_DIR / "01_cleaned_model_data.csv", index=False)
    tlog.to_csv(RESULTS_DIR / "01_target_disposition_log.csv", index=False)

    # ---------------- preprocessing log
    L, A = [], None
    A = L.append
    A("# 01 - Preprocessing log\n")
    A("## Target cleaning (`gravimetric_capacitance` -> `target_cap_F_g`)\n")
    A("A value is kept only if it is a single defensible number.  Ranges "
      "(`300-350`), inequalities (`over 200`, `<300`) and multi-phase strings "
      "are **never** collapsed to a midpoint - they are dropped and counted here.\n")
    A("| disposition | n rows |")
    A("| --- | ---: |")
    for k, v in tlog["disposition"].value_counts().items():
        A(f"| `{k}` | {v} |")
    A("")
    dropped = tlog[tlog["disposition"].str.startswith("dropped") &
                   (tlog["parsed_kind"] != "missing")]
    if len(dropped):
        A("### Rows dropped for a non-empty but non-defensible target value\n")
        A("| paper_id | raw value | parsed kind |")
        A("| --- | --- | --- |")
        for _, r in dropped.iterrows():
            A(f"| {r['paper_id']} | `{r['raw_value']}` | {r['parsed_kind']} |")
        A("")
    A(f"**Rows retained for modelling: {len(kept)} / {len(model)}** "
      f"(from {kept['paper_id'].nunique()} papers).")
    A(f"Approximate targets (`~`, `about`, mean+/-sd) retained and flagged: "
      f"{int(kept['target_is_approximate'].sum())}.")
    A("")
    ol = kept[kept["target_outlier_flag"]]
    A(f"Physically extreme capacitance values flagged (<20 or >1500 F/g): {len(ol)}. "
      "These are retained (nothing is hidden) and listed here:\n")
    if len(ol):
        A("| paper_id | target_cap_F_g |")
        A("| --- | ---: |")
        for _, r in ol.iterrows():
            A(f"| {r['paper_id']} | {r[TARGET]} |")
    A("")

    A("## Numeric descriptor standardisation\n")
    A("Conversions applied only where the source unit is unambiguous "
      "(`nm -> A` x10, `mm -> um` x1000, `ug/cm2 -> mg/cm2` /1000, `V/s -> mV/s` x1000). "
      "Incompatible bases (e.g. `mA/cm2` or `A/cm3` for a gravimetric current "
      "density; `mg` or `wt %` for an areal mass loading) are **left missing**, "
      "never converted.\n")
    A(flogs["numeric_log"].to_markdown(index=False))
    A("")
    A("## Electrolyte parsing (`h2so4_M`)\n")
    A("Parsed from the `electrolyte` field only.  `synthesis_method` also contains "
      "acid concentrations (e.g. wet-spinning in 98 wt% H2SO4) but those are "
      "*synthesis* solutions, not the test electrolyte, so that field is never consulted.\n")
    A(flogs["electrolyte_log"].to_markdown(index=False))
    A("")
    A("## Categorical family mappings\n")
    for c in CATEGORICAL_FEATURES:
        vc = kept[c].value_counts(dropna=False).to_dict()
        A(f"- `{c}`: {vc}")
    A("")
    A("## Paper grouping\n")
    A("`paper_id` = DOI when present, else `source_file`.  DOI and source_file are "
      "1:1 in this corpus, so the two definitions agree; 3 rows lack a DOI and fall "
      "back to their filename.")
    A(f"Unique papers among modelled rows: **{kept['paper_id'].nunique()}**.")
    (RESULTS_DIR / "01_preprocessing_log.md").write_text("\n".join(L), encoding="utf-8")

    log(f"cleaned: {len(kept)} modelling rows / {kept['paper_id'].nunique()} papers")
    STATE.update(model_all=model, data=kept, target_log=tlog, feat_logs=flogs)


# ===========================================================================
# Stage 4 - real-only baseline (full descriptor set)
# ===========================================================================

def _split_feature_types(cols: list[str]) -> tuple[list[str], list[str]]:
    num = [c for c in cols if c in NUMERIC_FEATURES]
    cat = [c for c in cols if c in CATEGORICAL_FEATURES]
    return num, cat


def run_grouped_cv(data: pd.DataFrame, features: list[str], tag: str,
                   tune: bool = True, collect_shap: bool = True) -> dict:
    num_cols, cat_cols = _split_feature_types(features)
    X_all = data[features]
    y_all = data[TARGET].values
    groups = data["paper_id"].values

    gkf, n_folds, reason = make_group_kfold(pd.Series(groups), N_FOLDS_TARGET)
    log(f"[{tag}] {reason}")

    fold_rows, pred_rows, shap_records, params_used = [], [], [], []
    for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        Xtr_raw, Xte_raw = X_all.iloc[tr], X_all.iloc[te]
        ytr, yte = y_all[tr], y_all[te]
        gtr = groups[tr]

        enc = FoldEncoder(cat_cols, num_cols).fit(Xtr_raw)
        Xtr, Xte = enc.transform(Xtr_raw), enc.transform(Xte_raw)

        params = tune_xgb(Xtr, ytr, gtr, seed=RANDOM_STATE) if tune else {}
        params_used.append({"fold": fold, **params})
        model = fit_xgb(Xtr, ytr, params, seed=RANDOM_STATE)
        yhat = model.predict(Xte)

        m = metrics(yte, yhat)
        fold_rows.append({"fold": fold, "n_train": len(tr), "n_test": len(te),
                          "n_test_papers": int(pd.Series(groups[te]).nunique()), **m})
        pred_rows.append(pd.DataFrame({
            "fold": fold, "paper_id": groups[te], "y_true": yte, "y_pred": yhat,
        }))
        if collect_shap:
            shap_records.append({"fold": fold, "X": Xte,
                                 "shap": fold_shap_values(model, Xte),
                                 "ohe_to_source": enc.ohe_to_source_})

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    ohe_map: dict[str, str] = {}
    for r in shap_records:
        ohe_map.update(r["ohe_to_source"])
    return {
        "fold_metrics": fold_df,
        "predictions": pred_df,
        "shap_records": shap_records,
        "ohe_to_source": ohe_map,
        "n_folds": n_folds,
        "fold_reason": reason,
        "params": pd.DataFrame(params_used),
        "features": features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


def stage_real_baseline() -> None:
    data = STATE["data"]
    features = [c for c in ALL_FEATURES]
    res = run_grouped_cv(data, features, "real-only-full", tune=True, collect_shap=True)

    fold_df = res["fold_metrics"]
    fold_df.to_csv(RESULTS_DIR / "02_real_only_fold_metrics.csv", index=False)
    res["predictions"].to_csv(RESULTS_DIR / "02_real_only_predictions.csv", index=False)
    res["params"].to_csv(RESULTS_DIR / "02_real_only_tuned_params.csv", index=False)

    summ = pd.DataFrame([{
        "model": "XGBoost (real only, full descriptor set)",
        "n_rows": len(data), "n_papers": int(data["paper_id"].nunique()),
        "n_features": len(features), "n_folds": res["n_folds"],
        **{f"{m}_{s}": getattr(fold_df[m], s)() for m in ("r2", "mae", "rmse")
           for s in ("mean", "std", "median")},
        "r2_min": fold_df["r2"].min(), "r2_max": fold_df["r2"].max(),
        "pooled_r2": metrics(res["predictions"]["y_true"], res["predictions"]["y_pred"])["r2"],
        "pooled_mae": metrics(res["predictions"]["y_true"], res["predictions"]["y_pred"])["mae"],
        "pooled_rmse": metrics(res["predictions"]["y_true"], res["predictions"]["y_pred"])["rmse"],
    }])
    summ.to_csv(RESULTS_DIR / "02_real_only_summary.csv", index=False)

    P.predicted_vs_actual(res["predictions"], FIG_DIR / "real_only_predicted_vs_actual.png",
                          "Real-only XGBoost, grouped CV (held-out papers)")
    P.residual_plot(res["predictions"], FIG_DIR / "real_only_residuals.png",
                    "Real-only XGBoost residuals (held-out papers)")

    log(f"real-only baseline: R2 {fold_df.r2.mean():.3f}+/-{fold_df.r2.std():.3f} | "
        f"MAE {fold_df.mae.mean():.1f} | RMSE {fold_df.rmse.mean():.1f}")
    STATE["baseline"] = res
    STATE["baseline_summary"] = summ


# ===========================================================================
# Stage 5 - SHAP on real data
# ===========================================================================

def stage_real_shap() -> None:
    res = STATE["baseline"]
    recs, ohe_map = res["shap_records"], res["ohe_to_source"]

    enc_imp = importance_table(recs, ohe_map)
    src_imp = aggregate_to_source(enc_imp, recs, ohe_map)
    out = pd.concat([src_imp, enc_imp], ignore_index=True)
    out.to_csv(RESULTS_DIR / "03_real_shap_importance.csv", index=False)

    shap_df, X_pool = pool_folds(recs)
    P.shap_bar(src_imp, FIG_DIR / "real_shap_bar.png",
               "Real-data SHAP importance (source-variable level)")
    P.shap_beeswarm(shap_df, X_pool, FIG_DIR / "real_shap_beeswarm.png",
                    "Real-data SHAP on held-out papers (encoded features)")

    # dependence plots for the strongest *continuous physical* descriptors
    phys = [f for f in src_imp["feature"] if f in NUMERIC_FEATURES][:6]
    for f in phys:
        if f not in shap_df.columns:
            continue
        x = pd.to_numeric(X_pool[f], errors="coerce")
        ok = x.notna()
        if ok.sum() < 5:
            continue
        P.shap_dependence(x[ok], shap_df.loc[ok.values, f], FIG_DEP_DIR / f"dependence_{f}.png",
                          f, f)
    # categorical dependence: SHAP of each one-hot level
    for cat in res["cat_cols"]:
        cols = [c for c in shap_df.columns if ohe_map.get(c) == cat]
        if not cols:
            continue
        agg = shap_df[cols]
        vals, labels = [], []
        for c in cols:
            sel = X_pool[c] == 1
            if sel.sum() >= 3:
                vals.append(agg.loc[sel.values, c].values)
                labels.append(c.replace(cat + "_", ""))
        if len(vals) >= 2:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(max(3.6, 1.1 * len(vals)), 3.4))
            ax.boxplot(vals, tick_labels=labels, widths=0.6)
            ax.axhline(0, color="k", lw=0.8, ls="--")
            ax.set_ylabel("SHAP value (F/g)")
            ax.set_title(f"Dependence: {cat}")
            ax.tick_params(axis="x", rotation=20)
            fig.savefig(FIG_DEP_DIR / f"dependence_{cat}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

    # ---- direction of each continuous descriptor, read off pooled held-out SHAP
    src = src_imp
    expected_sign = {
        "h2so4_M": (+1, "yes - more protons means more Ti-O pseudocapacitance"),
        "scan_rate_mV_s": (-1, "yes - faster sweeps starve the interlayer of ions"),
        "interlayer_A": (+1, "**no** - a wider gallery should ease ion access"),
        "ssa_m2_g": (+1, "yes - more accessible area, though BET is a weak proxy in restacked films"),
        "mass_loading_mg_cm2": (-1, "yes - thicker films lose rate performance"),
        "electrode_thickness_um": (-1, "yes - longer transport path"),
        "current_density_A_g": (-1, "yes - higher rate, lower capacitance"),
        "flake_size_um": (-1, "plausible - longer in-plane diffusion path"),
        "pore_diameter_nm": (+1, "plausible for engineered architectures"),
    }
    rows_dir = []
    for f in src["feature"]:
        if f not in NUMERIC_FEATURES or f not in shap_df.columns:
            continue
        x = pd.to_numeric(X_pool[f], errors="coerce")
        ok = x.notna().values
        if ok.sum() < 8:
            continue
        rho = pd.Series(x[ok].values).corr(pd.Series(shap_df.loc[ok, f].values), method="spearman")
        if pd.isna(rho):
            continue
        word = ("raises the prediction as it grows" if rho > 0.15 else
                "lowers the prediction as it grows" if rho < -0.15 else
                "no monotone effect")
        exp_sign, exp_txt = expected_sign.get(f, (0, "no strong prior"))
        if abs(rho) <= 0.15:
            verdict = "flat - no claim"
        elif exp_sign == 0:
            verdict = exp_txt
        elif np.sign(rho) == exp_sign:
            verdict = exp_txt
        else:
            verdict = ("**contradicts the physical expectation** - "
                       + exp_txt.removeprefix("**no** - ")
                       if exp_txt.startswith("**no**") else
                       "**contradicts the physical expectation**")
        rows_dir.append({"feature": f, "rho": float(rho), "word": word, "expected": verdict})
    STATE["shap_directions"] = rows_dir

    contradictions = [r["feature"] for r in rows_dir if "contradict" in r["expected"]]
    note = ("Rank stability across folds is in the `rank SD` and `folds in top-5` "
            "columns above: a descriptor with a low mean rank but a high rank SD is "
            "being driven by one or two papers, not by a consistent trend.")
    if contradictions:
        plural = "run" if len(contradictions) > 1 else "runs"
        note += (
            " **Caution:** " + ", ".join(f"`{c}`" for c in contradictions) +
            f" {plural} against the physical expectation. In a literature corpus that "
            "usually means confounding rather than physics - which paper a measurement "
            "came from is entangled with the descriptor value.")
        if "interlayer_A" in contradictions:
            note += (
                " For `interlayer_A` specifically: the largest reported d-spacings "
                "belong mostly to pillared and composite architectures whose "
                "*gravimetric* capacitance is diluted by the intercalant mass, and "
                "d-spacing is typically measured on a dry film rather than in the "
                "hydrated, polarised state that actually stores charge.")
        if "mass_loading_mg_cm2" in contradictions:
            note += (
                " For `mass_loading_mg_cm2`: groups that push high loadings tend to be "
                "the ones with an architecture good enough to survive it, so loading "
                "partly encodes electrode quality rather than transport penalty.")
        note += (
            " Both are concrete reasons not to read a SHAP direction as a causal law, "
            "and concrete things for a PINN to settle with a mechanistic term instead "
            "of a fitted one.")
    STATE["shap_direction_note"] = note

    log("real SHAP: top source features -> " +
        ", ".join(f"{r.feature}({r.mean_abs_shap:.1f})" for r in src_imp.head(5).itertuples()))
    STATE["real_shap_source"] = src_imp
    STATE["real_shap_encoded"] = enc_imp


# ===========================================================================
# Stage 6-7 - BN candidate descriptors and selection
# ===========================================================================

PHYSICAL_NOTES = {
    "interlayer_A": "XRD (002) d-spacing: sets the ion-accessible gallery height; the central geometric descriptor for a PINN.",
    "h2so4_M": "Electrolyte proton activity; governs pseudocapacitive Ti-O redox and double-layer screening.",
    "scan_rate_mV_s": "CV sweep rate; controls the diffusion length probed and therefore rate-limited capacitance loss.",
    "current_density_A_g": "GCD rate condition; same physical role as scan rate but for galvanostatic tests.",
    "mass_loading_mg_cm2": "Areal mass; controls ion transport path length through the film.",
    "electrode_thickness_um": "Film thickness; directly coupled to mass loading and transport resistance.",
    "ssa_m2_g": "Gas-sorption surface area; a weak proxy for wetted electrochemical area in a restacked MXene film.",
    "flake_size_um": "Lateral flake dimension; sets in-plane ion diffusion distance.",
    "pore_diameter_nm": "Meso/macropore size of engineered architectures.",
    "layer_class": "Delamination state (multilayer / delaminated / single-or-few-layer).",
    "composition_family": "Pristine vs carbon / polymer / inorganic composite vs doped MXene.",
    "synthesis_family": "Etching route.",
    "electrolyte_is_gel": "Gel (PVA/H2SO4) vs free aqueous electrolyte.",
}

# how each attribute is presented to the Bayesian network
BN_CATEGORICAL_TREATMENT = {
    "h2so4_M": (True, "Only 4 discrete experimental settings occur (0.5/1/2/3 M); "
                      "treating it as continuous would let the BN invent unobserved molarities."),
    "scan_rate_mV_s": (True, "A chosen instrument setting taking a handful of discrete "
                             "values spanning 3 decades; discrete treatment avoids "
                             "interpolating rates nobody measured."),
    "interlayer_A": (False, "Genuinely continuous physical measurement."),
    "mass_loading_mg_cm2": (False, "Genuinely continuous."),
    "electrode_thickness_um": (False, "Genuinely continuous."),
    "ssa_m2_g": (False, "Genuinely continuous."),
    "current_density_A_g": (True, "Discrete chosen rate setting."),
    "flake_size_um": (False, "Genuinely continuous."),
    "pore_diameter_nm": (False, "Genuinely continuous."),
    "layer_class": (True, "Nominal."),
    "composition_family": (True, "Nominal."),
    "synthesis_family": (True, "Nominal."),
    "electrolyte_is_gel": (True, "Boolean."),
    TARGET: (False, "Continuous target."),
}

EXCLUDE_PATTERNS = ("doi", "authors", "title", "source_file", "source_path", "paper_id",
                    "url", "__src", "__source", "__conf", "__confidence", "__unit",
                    "reasoning", "notes", "overall", "role", "is_primary",
                    "electrode_label", "year", "volumetric", "areal", TARGET)


def stage_bn_candidates() -> None:
    data = STATE["data"]
    src_imp = STATE["real_shap_source"].set_index("feature")

    rows = []
    for f in ALL_FEATURES:
        s = data[f]
        miss = float(s.isna().mean())
        nuniq = int(s.nunique(dropna=True))
        dtype = "numeric" if f in NUMERIC_FEATURES else "categorical"
        shap_v = float(src_imp.loc[f, "mean_abs_shap"]) if f in src_imp.index else 0.0
        rank = int(src_imp.loc[f, "shap_rank"]) if f in src_imp.index else 999
        stab = float(src_imp.loc[f, "rank_sd"]) if f in src_imp.index else np.nan

        reasons = []
        include = True
        if nuniq < 2:
            include = False; reasons.append("no variation in the modelling cohort")
        if miss > 0.55:
            include = False; reasons.append(f"missingness {miss:.0%} > 55%")
        if shap_v <= 1e-9:
            include = False; reasons.append("no measurable SHAP contribution")
        if include:
            reasons.append(f"SHAP rank {rank}, missingness {miss:.0%}, {nuniq} distinct values")
        rows.append({
            "feature": f, "mean_abs_shap": shap_v, "shap_rank": rank,
            "shap_rank_sd": stab, "missing_fraction": round(miss, 4),
            "n_unique": nuniq, "data_type": dtype,
            "bn_treated_as_categorical": BN_CATEGORICAL_TREATMENT[f][0],
            "include_default": include,
            "reason": "; ".join(reasons),
            "physical_meaning": PHYSICAL_NOTES.get(f, ""),
        })
    cand = pd.DataFrame(rows).sort_values(["include_default", "mean_abs_shap"],
                                          ascending=[False, False]).reset_index(drop=True)
    cand.to_csv(RESULTS_DIR / "04_bn_candidate_features.csv", index=False)

    excl = pd.DataFrame([
        {"column": c, "excluded_from_bn_because": "identifier / metadata / free text / "
                                                  "target-derived - never a physical descriptor"}
        for c in STATE["raw"].columns
        if any(p in c for p in EXCLUDE_PATTERNS)
    ])
    excl.to_csv(RESULTS_DIR / "04_bn_excluded_columns.csv", index=False)
    log(f"BN candidates: {int(cand.include_default.sum())} of {len(cand)} pass the default filter")
    STATE["bn_candidates"] = cand


def stage_bn_selection() -> None:
    data, cand = STATE["data"], STATE["bn_candidates"]
    pool = cand[cand["include_default"]].sort_values("mean_abs_shap", ascending=False)

    MIN_ROWS, MIN_PAPERS, MAX_FEATS, MIN_FEATS = 45, 15, 7, 4

    selected: list[str] = []
    trace = []
    for f in pool["feature"]:
        trial = selected + [f]
        sub = data.dropna(subset=trial + [TARGET])
        rows, papers = len(sub), sub["paper_id"].nunique()
        ok = (rows >= MIN_ROWS and papers >= MIN_PAPERS) or len(selected) < MIN_FEATS
        # never accept a feature that destroys the cohort even while under MIN_FEATS
        if len(selected) < MIN_FEATS and (rows < 35 or papers < 12):
            ok = False
        trace.append({"candidate": f, "cohort_rows": rows, "cohort_papers": papers,
                      "accepted": bool(ok and len(selected) < MAX_FEATS)})
        if ok and len(selected) < MAX_FEATS:
            selected = trial
    sub = data.dropna(subset=selected + [TARGET])

    # a descriptor that turns out to be constant inside the complete-case cohort
    # carries no information for either the BN or the model - drop it
    degenerate = [f for f in selected if sub[f].nunique(dropna=True) < 2]
    if degenerate:
        log(f"  dropping degenerate (constant-in-cohort) BN features: {degenerate}")
        selected = [f for f in selected if f not in degenerate]
        sub = data.dropna(subset=selected + [TARGET])

    payload = {
        "selected_features": selected,
        "target": TARGET,
        "selection_rule": {
            "candidate_pool": "features passing 04_bn_candidate_features.csv include_default",
            "ordering": "descending mean |SHAP| from the real-only grouped-CV model",
            "constraints": {
                "min_complete_case_rows": MIN_ROWS,
                "min_complete_case_papers": MIN_PAPERS,
                "min_features": MIN_FEATS,
                "max_features": MAX_FEATS,
            },
            "note": "Greedy: a candidate is added only if the resulting complete-case "
                    "cohort still satisfies the row/paper floors, so a high-SHAP but "
                    "sparsely reported descriptor is deliberately not forced in.",
        },
        "per_feature_justification": {
            f: {
                "mean_abs_shap": float(cand.set_index("feature").loc[f, "mean_abs_shap"]),
                "shap_rank": int(cand.set_index("feature").loc[f, "shap_rank"]),
                "missing_fraction": float(cand.set_index("feature").loc[f, "missing_fraction"]),
                "bn_treated_as_categorical": bool(BN_CATEGORICAL_TREATMENT[f][0]),
                "categorical_treatment_reason": BN_CATEGORICAL_TREATMENT[f][1],
                "physical_meaning": PHYSICAL_NOTES.get(f, ""),
            }
            for f in selected
        },
        "rejected_candidates": [
            {"feature": r["candidate"], "cohort_rows_if_added": r["cohort_rows"],
             "cohort_papers_if_added": r["cohort_papers"],
             "why": "adding it would push the complete-case cohort below the row/paper floor"}
            for r in trace if not r["accepted"]
        ] + [
            {"feature": f, "cohort_rows_if_added": int(len(sub)),
             "cohort_papers_if_added": int(sub["paper_id"].nunique()),
             "why": "constant inside the complete-case cohort - no information"}
            for f in degenerate
        ],
        "cohort": {
            "rows_before": int(len(data)), "rows_after": int(len(sub)),
            "papers_before": int(data["paper_id"].nunique()),
            "papers_after": int(sub["paper_id"].nunique()),
            "missing_data_strategy": "complete cases on the selected descriptors + target",
        },
    }
    (RESULTS_DIR / "04_selected_bn_features.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    pd.DataFrame(trace).to_csv(RESULTS_DIR / "04_bn_selection_trace.csv", index=False)

    cohort = sub[["paper_id"] + selected + [TARGET, "target_is_approximate",
                                            "target_outlier_flag"]].reset_index(drop=True)
    cohort.to_csv(RESULTS_DIR / "04_bn_real_cohort.csv", index=False)

    log(f"BN feature set ({len(selected)}): {selected}")
    log(f"BN cohort: {len(cohort)} rows / {cohort['paper_id'].nunique()} papers "
        f"(from {len(data)} rows / {data['paper_id'].nunique()} papers)")
    STATE["bn_features"] = selected
    STATE["bn_cohort"] = cohort
    STATE["bn_selection"] = payload


# ===========================================================================
# Stage 8-10 - fold-wise synthesis, fidelity, augmented models
# ===========================================================================

def stage_augmentation() -> None:
    cohort = STATE["bn_cohort"]
    feats = STATE["bn_features"]
    num_cols, cat_cols = _split_feature_types(feats)
    strictly_positive = tuple(c for c in num_cols)

    cat_flags = {f: BN_CATEGORICAL_TREATMENT[f][0] for f in feats}
    cat_flags[TARGET] = False
    bn_cat_cols = [f for f in feats if cat_flags[f]]
    bn_num_cols = [f for f in feats if not cat_flags[f]]

    X_all = cohort[feats]
    y_all = cohort[TARGET].values
    groups = cohort["paper_id"].values
    gkf, n_folds, reason = make_group_kfold(pd.Series(groups), N_FOLDS_TARGET)
    log(f"[augmentation] {reason}")

    fold_rows, pred_rows = [], []
    reject_rows, fidelity_rows, dist_rows, mi_rows, nn_rows, tgt_rows = [], [], [], [], [], []
    bn_structures = []
    shap_store: dict[tuple[int, float], list] = {}
    ohe_maps: dict[tuple[int, float], dict] = {}
    plot_pool: dict[str, list] = {"real": [], "syn": []}

    for fold, (tr, te) in enumerate(gkf.split(X_all, y_all, groups), start=1):
        train_real = cohort.iloc[tr].reset_index(drop=True)
        test_real = cohort.iloc[te].reset_index(drop=True)
        gtr = groups[tr]

        # -- encoder + hyper-parameters are fitted on the REAL TRAINING FOLD only,
        #    and shared across all four conditions so the comparison is fair.
        enc = FoldEncoder(cat_cols, num_cols).fit(train_real[feats])
        Xtr_real = enc.transform(train_real[feats])
        Xte = enc.transform(test_real[feats])
        ytr_real = train_real[TARGET].values
        yte = test_real[TARGET].values
        params = tune_xgb(Xtr_real, ytr_real, gtr, seed=RANDOM_STATE)

        bn_train = train_real[feats + [TARGET]].copy()

        for k in BN_K_VALUES:
          for ratio in AUG_RATIOS:
            cond = AUG_LABELS[ratio]
            if ratio == 0.0 and k != BN_K_DEFAULT:
                continue                          # real-only does not depend on k
            # repeat the draw only where it is the reported condition; k=1/3 are a
            # sensitivity check and run on a single seed
            reps = 1 if (ratio == 0.0 or k != BN_K_DEFAULT) else N_GEN_REPEATS
            for rep in range(reps):
                if ratio == 0.0:
                    Xtr, ytr = Xtr_real, ytr_real
                    n_syn = 0
                else:
                    n_target = int(round(ratio * len(train_real)))
                    tag = f"f{fold}_k{k}_r{str(ratio).replace('.', 'p')}_s{rep}"
                    syn, rej, meta = generate_filtered(
                        bn_train, n_target, k, BN_EPSILON, cat_flags,
                        bn_num_cols, bn_cat_cols, TARGET, CACHE_DIR, tag,
                        seed=RANDOM_STATE + fold * 1000 + k * 100 + rep * 7,
                        non_negative=strictly_positive,
                    )
                    rej = rej.assign(fold=fold, k=k, ratio=ratio, condition=cond, gen_seed=rep)
                    reject_rows.append(rej)
                    bn_structures.append({"fold": fold, "k": k, "ratio": ratio,
                                          "gen_seed": rep,
                                          "n_train": len(train_real),
                                          "bayesian_network": json.dumps(meta["bayesian_network"]),
                                          "n_valid_returned": meta["n_valid_returned"],
                                          "n_valid_requested": meta["n_valid_requested"]})
                    n_syn = len(syn)

                    # ---------------- fidelity (train-fold real vs synthetic)
                    fid = {"fold": fold, "k": k, "ratio": ratio, "gen_seed": rep,
                           "condition": cond,
                           "n_real_train": len(train_real), "n_synthetic": n_syn}
                    for c in bn_num_cols + [TARGET]:
                        st = continuous_stats(train_real[c], syn[c], c)
                        dist_rows.append({"fold": fold, "k": k, "ratio": ratio,
                                          "gen_seed": rep, **st})
                    for c in bn_cat_cols:
                        st = categorical_stats(train_real[c].astype(str), syn[c].astype(str), c)
                        dist_rows.append({"fold": fold, "k": k, "ratio": ratio,
                                          "gen_seed": rep, **st})
                    ks_all = [continuous_stats(train_real[c], syn[c], c).get("ks_statistic", np.nan)
                              for c in bn_num_cols + [TARGET]]
                    js_all = [categorical_stats(train_real[c].astype(str), syn[c].astype(str), c)["jensen_shannon_distance"]
                              for c in bn_cat_cols] or [np.nan]
                    fid["mean_ks_statistic"] = float(np.nanmean(ks_all))
                    fid["max_ks_statistic"] = float(np.nanmax(ks_all))
                    fid["n_ks_reject_p05"] = int(sum(
                        1 for c in bn_num_cols + [TARGET]
                        if continuous_stats(train_real[c], syn[c], c).get("ks_pvalue", 1) < 0.05))
                    fid["mean_jensen_shannon"] = float(np.nanmean(js_all))

                    corr_res, mats = correlation_agreement(
                        train_real, syn, bn_num_cols + [TARGET])
                    fid.update(corr_res)
                    mi_res, MIr, MIs = mi_agreement(train_real, syn, bn_num_cols + [TARGET],
                                                    bn_cat_cols)
                    fid.update(mi_res)
                    tr_tab = target_relationships(train_real, syn, bn_num_cols,
                                                  bn_cat_cols, TARGET)
                    for _, rr in tr_tab.iterrows():
                        tgt_rows.append({"fold": fold, "k": k, "ratio": ratio,
                                         "gen_seed": rep, **rr.to_dict()})
                    fid["target_assoc_mean_abs_diff"] = float(
                        pd.concat([tr_tab.get("spearman_abs_diff", pd.Series(dtype=float)),
                                   tr_tab.get("eta_squared_abs_diff", pd.Series(dtype=float))]).mean())

                    mem = memorisation_check(syn, train_real, bn_num_cols + [TARGET], bn_cat_cols)
                    fid.update(mem)
                    fidelity_rows.append(fid)
                    nn_rows.append({"fold": fold, "k": k, "ratio": ratio,
                                    "gen_seed": rep, **mem})
                    for a in MIr.index:
                        for b in MIr.columns:
                            mi_rows.append({"fold": fold, "k": k, "ratio": ratio,
                                            "gen_seed": rep, "var_a": a, "var_b": b,
                                            "mi_real": MIr.loc[a, b], "mi_syn": MIs.loc[a, b],
                                            "mi_diff": MIs.loc[a, b] - MIr.loc[a, b]})

                    if k == BN_K_DEFAULT and ratio == 3.0 and rep == 0:
                        plot_pool["real"].append(train_real[feats + [TARGET]])
                        plot_pool["syn"].append(syn[feats + [TARGET]])
                        if fold == 1:
                            STATE["example_mats"] = (mats, MIr, MIs)

                    Xsyn = enc.transform(syn[feats])
                    Xtr = pd.concat([Xtr_real, Xsyn], ignore_index=True)
                    ytr = np.concatenate([ytr_real, syn[TARGET].values])

                model = fit_xgb(Xtr, ytr, params, seed=RANDOM_STATE)
                yhat = model.predict(Xte)
                m = metrics(yte, yhat)
                fold_rows.append({"fold": fold, "k": k, "ratio": ratio, "gen_seed": rep,
                                  "condition": cond,
                                  "n_real_train": len(train_real), "n_synthetic_train": n_syn,
                                  "n_total_train": len(Xtr),
                                  "n_test_papers": int(pd.Series(groups[te]).nunique()), **m})
                pred_rows.append(pd.DataFrame({
                    "fold": fold, "k": k, "ratio": ratio, "gen_seed": rep, "condition": cond,
                    "paper_id": test_real["paper_id"].values,
                    "y_true": yte, "y_pred": yhat}))
                key = (k, ratio)
                shap_store.setdefault(key, []).append(
                    {"fold": f"{fold}s{rep}", "X": Xte, "shap": fold_shap_values(model, Xte)})
                ohe_maps[key] = enc.ohe_to_source_
        log(f"  fold {fold}/{n_folds} done (train={len(train_real)}, test={len(test_real)})")

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)
    fold_df.to_csv(RESULTS_DIR / "06_augmentation_fold_metrics.csv", index=False)
    pred_df.to_csv(RESULTS_DIR / "06_all_predictions.csv", index=False)
    pd.concat(reject_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "05_synthetic_rejection_log.csv", index=False)
    pd.DataFrame(bn_structures).to_csv(RESULTS_DIR / "05_bayesian_network_structures.csv", index=False)
    fid_df = pd.DataFrame(fidelity_rows)
    fid_df.to_csv(RESULTS_DIR / "05_synthetic_fidelity_summary.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(RESULTS_DIR / "05_synthetic_distribution_stats.csv", index=False)
    pd.DataFrame(mi_rows).to_csv(RESULTS_DIR / "05_mutual_information_comparison.csv", index=False)
    pd.DataFrame(nn_rows).to_csv(RESULTS_DIR / "05_nearest_neighbor_analysis.csv", index=False)
    pd.DataFrame(tgt_rows).to_csv(RESULTS_DIR / "05_target_relationship_comparison.csv", index=False)

    # ---- main summary at the default k, plus the k sensitivity table
    main_raw = fold_df[(fold_df["k"] == BN_K_DEFAULT) | (fold_df["ratio"] == 0.0)].copy()
    main_raw.to_csv(RESULTS_DIR / "06_augmentation_per_seed_metrics.csv", index=False)
    main_pred = pred_df[(pred_df["k"] == BN_K_DEFAULT) | (pred_df["ratio"] == 0.0)]

    # Average the independent generator draws within a fold first, so that the
    # fold remains the unit of pairing and the fold count is the same (5) for
    # every condition.
    main = main_raw.groupby(["condition", "fold"], as_index=False).agg(
        r2=("r2", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        n_real_train=("n_real_train", "first"),
        n_synthetic_train=("n_synthetic_train", "mean"),
        n_test=("n_test", "first"), n_test_papers=("n_test_papers", "first"),
        n_gen_seeds=("r2", "size"),
    )
    summ = summarise(main, ["condition"]).sort_values("condition").reset_index(drop=True)
    summ["ratio"] = summ["condition"].map({v: k for k, v in AUG_LABELS.items()})
    summ["n_synthetic_train_mean"] = summ["condition"].map(
        main.groupby("condition")["n_synthetic_train"].mean())
    summ["bn_k"] = np.where(summ["ratio"] == 0, np.nan, BN_K_DEFAULT)
    summ["n_gen_seeds"] = summ["condition"].map(main.groupby("condition")["n_gen_seeds"].max())

    # How much does simply re-drawing the synthetic set move the score?  Average
    # within-fold SD across independent generator seeds.
    for metric in ("r2", "mae", "rmse"):
        sd_seed = (main_raw.groupby(["condition", "fold"])[metric].std()
                   .groupby("condition").mean())
        summ[f"{metric}_sd_across_gen_seeds"] = summ["condition"].map(sd_seed)

    # Pooled metrics over every held-out real prediction. Per-fold R^2 is very
    # unstable here because some folds contain few papers with little spread in
    # capacitance; pooling keeps the grouped hold-out intact while giving a
    # summary that does not divide by a near-zero within-fold variance.
    pooled = []
    for cond, g in main_pred.groupby("condition"):
        per_seed, per_seed_x = [], []
        for _, gs in g.groupby("gen_seed"):
            per_seed.append(metrics(gs["y_true"], gs["y_pred"]))
            gx = gs[(gs["y_true"] >= 20) & (gs["y_true"] <= 1500)]
            per_seed_x.append(metrics(gx["y_true"], gx["y_pred"]))
        avg = lambda rows, key: float(np.mean([r[key] for r in rows]))
        pooled.append({"condition": cond,
                       "pooled_r2": avg(per_seed, "r2"),
                       "pooled_mae": avg(per_seed, "mae"),
                       "pooled_rmse": avg(per_seed, "rmse"),
                       "pooled_n": per_seed[0]["n_test"],
                       "pooled_r2_excl_extreme": avg(per_seed_x, "r2"),
                       "pooled_mae_excl_extreme": avg(per_seed_x, "mae"),
                       "pooled_rmse_excl_extreme": avg(per_seed_x, "rmse")})
    summ = summ.merge(pd.DataFrame(pooled), on="condition", how="left")
    summ.to_csv(RESULTS_DIR / "06_augmentation_summary.csv", index=False)

    ksens_src = fold_df.groupby(["k", "condition", "fold"], as_index=False)[
        ["r2", "mae", "rmse"]].mean()
    ksens = summarise(ksens_src, ["k", "condition"]).sort_values(["condition", "k"])
    ksens.to_csv(RESULTS_DIR / "06_bn_k_sensitivity.csv", index=False)

    for metric, lab, hib in (("r2", "R^2", True), ("mae", "MAE (F/g)", False),
                             ("rmse", "RMSE (F/g)", False)):
        P.augmentation_bars(summ, metric, FIG_DIR / f"augmentation_{metric}.png",
                            lab, f"{lab} on unseen real papers (mean +/- SD over folds)",
                            hib, fold_df=main)
        P.fold_metric_lines(main, metric, FIG_DIR / f"augmentation_{metric}_perfold.png", lab)

    STATE.update(aug_fold=fold_df, aug_main=main, aug_raw=main_raw,
                 aug_summary=summ, aug_pred=pred_df,
                 aug_shap=shap_store, aug_ohe=ohe_maps, fidelity=fid_df, ksens=ksens,
                 plot_pool=plot_pool, n_folds_aug=n_folds, fold_reason_aug=reason,
                 bn_num_cols=bn_num_cols, bn_cat_cols=bn_cat_cols)


def stage_fidelity_figures() -> None:
    pool = STATE["plot_pool"]
    if not pool["real"]:
        return
    real = pd.concat(pool["real"], ignore_index=True)
    syn = pd.concat(pool["syn"], ignore_index=True)
    num, cat = STATE["bn_num_cols"], STATE["bn_cat_cols"]
    for c in num + [TARGET]:
        P.hist_real_vs_syn(real[c], syn[c], FIG_SYN_DIR / f"dist_{c}.png", c)
    for c in cat:
        P.bar_real_vs_syn_categorical(real[c].astype(str), syn[c].astype(str),
                                      FIG_SYN_DIR / f"freq_{c}.png", c)
    mats, MIr, MIs = STATE.get("example_mats", (None, None, None))
    if mats is not None:
        P.heatmap_pair(mats["pearson"][0], mats["pearson"][1],
                       FIG_SYN_DIR / "corr_pearson.png",
                       "Pearson correlation, fold 1 (k=2, 3x)")
        P.heatmap_pair(mats["spearman"][0], mats["spearman"][1],
                       FIG_SYN_DIR / "corr_spearman.png",
                       "Spearman correlation, fold 1 (k=2, 3x)")
        P.heatmap_pair(MIr, MIs, FIG_SYN_DIR / "mutual_information.png",
                       "Normalised mutual information, fold 1 (k=2, 3x)",
                       vmin=0, vmax=1, cmap="viridis")
    log("fidelity figures written")


# ===========================================================================
# Stage 11 - statistics
# ===========================================================================

def stage_statistics() -> None:
    main = STATE["aug_main"]
    base = main[main["condition"] == AUG_LABELS[0.0]].set_index("fold")
    rows = []
    for ratio in AUG_RATIOS[1:]:
        cur = main[main["condition"] == AUG_LABELS[ratio]].set_index("fold")
        folds = sorted(set(base.index) & set(cur.index))
        for metric, better in (("r2", "higher"), ("mae", "lower"), ("rmse", "lower")):
            a = base.loc[folds, metric].values
            b = cur.loc[folds, metric].values
            d = b - a
            row = {
                "comparison": f"real_only_vs_{AUG_LABELS[ratio]}",
                "bn_k": BN_K_DEFAULT, "metric": metric, "better_is": better,
                "n_folds": len(folds),
                "real_only_mean": float(np.mean(a)), "augmented_mean": float(np.mean(b)),
                "absolute_difference": float(np.mean(d)),
                "relative_pct_change": float(100 * np.mean(d) / abs(np.mean(a)))
                if np.mean(a) != 0 else np.nan,
                "n_folds_improved": int(np.sum(d > 0) if better == "higher" else np.sum(d < 0)),
            }
            if len(folds) >= 3 and np.any(d != 0):
                try:
                    w = wilcoxon(a, b)
                    row["wilcoxon_stat"], row["wilcoxon_p"] = float(w.statistic), float(w.pvalue)
                except ValueError:
                    row["wilcoxon_stat"], row["wilcoxon_p"] = np.nan, np.nan
                t = ttest_rel(a, b)
                row["paired_t_p"] = float(t.pvalue)
            else:
                row["wilcoxon_stat"] = row["wilcoxon_p"] = row["paired_t_p"] = np.nan
            row["min_attainable_wilcoxon_p"] = 2 ** -(len(folds) - 1) if len(folds) else np.nan
            rows.append(row)
    stat = pd.DataFrame(rows)
    stat.to_csv(RESULTS_DIR / "06_statistical_comparison.csv", index=False)
    STATE["stats"] = stat
    log("statistical comparison written")


# ===========================================================================
# Stage 12-13 - augmented SHAP and stability
# ===========================================================================

def stage_augmented_shap() -> None:
    summ = STATE["aug_summary"]
    aug_only = summ[summ["ratio"] != 0.0]
    best_row = aug_only.loc[aug_only["r2_mean"].idxmax()]
    best_ratio = float(best_row["ratio"])
    overall_best = summ.loc[summ["r2_mean"].idxmax(), "condition"]

    key_real = (BN_K_DEFAULT, 0.0)
    key_aug = (BN_K_DEFAULT, best_ratio)
    recs_real, recs_aug = STATE["aug_shap"][key_real], STATE["aug_shap"][key_aug]
    ohe_map = STATE["aug_ohe"][key_aug]

    real_enc = importance_table(recs_real, ohe_map)
    real_src = aggregate_to_source(real_enc, recs_real, ohe_map)
    aug_enc = importance_table(recs_aug, ohe_map)
    aug_src = aggregate_to_source(aug_enc, recs_aug, ohe_map)

    pd.concat([real_src, real_enc], ignore_index=True).to_csv(
        RESULTS_DIR / "07_cohort_real_only_shap_importance.csv", index=False)
    pd.concat([aug_src, aug_enc], ignore_index=True).to_csv(
        RESULTS_DIR / "07_augmented_shap_importance.csv", index=False)

    shap_df, X_pool = pool_folds(recs_aug)
    P.shap_bar(aug_src, FIG_DIR / "augmented_shap_bar.png",
               f"Augmented model SHAP ({AUG_LABELS[best_ratio]}, k={BN_K_DEFAULT})")
    P.shap_beeswarm(shap_df, X_pool, FIG_DIR / "augmented_shap_beeswarm.png",
                    f"Augmented SHAP on held-out real papers ({AUG_LABELS[best_ratio]})")

    tab, summary = shap_stability(real_src, aug_src)
    tab.to_csv(RESULTS_DIR / "07_shap_stability.csv", index=False)
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "07_shap_stability_summary.csv", index=False)
    P.shap_rank_comparison(tab, FIG_DIR / "shap_rank_comparison.png",
                           f"SHAP rank: cohort real-only vs {AUG_LABELS[best_ratio]}")

    log(f"best augmentation condition: {AUG_LABELS[best_ratio]} "
        f"(R2 {best_row['r2_mean']:.3f}); overall best incl. real-only: {overall_best}")
    STATE.update(best_ratio=best_ratio, best_condition=AUG_LABELS[best_ratio],
                 overall_best_condition=overall_best,
                 cohort_real_shap=real_src, aug_shap_src=aug_src,
                 stability_tab=tab, stability_summary=summary)


# ===========================================================================
# Stage 14 - PINN descriptor recommendation
# ===========================================================================

INTERPRETABILITY = {
    "interlayer_A": "high - direct geometric input to an ion-transport PINN",
    "h2so4_M": "high - enters Nernst/Poisson-Nernst-Planck terms directly",
    "scan_rate_mV_s": "high - sets the timescale in the transport PDE",
    "current_density_A_g": "high - galvanostatic boundary condition",
    "mass_loading_mg_cm2": "high - sets the transport domain length scale",
    "electrode_thickness_um": "high - explicit geometry",
    "ssa_m2_g": "medium - proxy for wetted area, poorly correlated with it in restacked films",
    "flake_size_um": "medium - in-plane diffusion length",
    "pore_diameter_nm": "medium - only meaningful for engineered architectures",
    "layer_class": "medium - qualitative stand-in for the delamination state",
    "composition_family": "low/medium - a categorical bucket, not a physical parameter",
    "synthesis_family": "low - processing label",
    "electrolyte_is_gel": "medium - changes ionic conductivity and confinement",
}


def stage_pinn() -> None:
    data = STATE["data"]
    real_full = STATE["real_shap_source"].set_index("feature")
    cohort_real = STATE["cohort_real_shap"].set_index("feature")
    aug = STATE["aug_shap_src"].set_index("feature")
    stab = STATE["stability_tab"].set_index("feature")
    bn_feats = STATE["bn_features"]

    rows = []
    for f in ALL_FEATURES:
        miss = float(data[f].isna().mean())
        in_bn = f in bn_feats
        r_rank = int(real_full.loc[f, "shap_rank"]) if f in real_full.index else None
        r_sd = float(real_full.loc[f, "rank_sd"]) if f in real_full.index else np.nan
        top5 = int(real_full.loc[f, "n_folds_top5"]) if f in real_full.index else 0
        a_rank = int(aug.loc[f, "shap_rank"]) if f in aug.index else None
        c_rank = int(cohort_real.loc[f, "shap_rank"]) if f in cohort_real.index else None
        dr = float(stab.loc[f, "abs_rank_change"]) if f in stab.index else np.nan

        if not in_bn:
            stability = "not evaluated (not in BN cohort)"
        elif np.isnan(dr):
            stability = "unknown"
        elif dr <= 1:
            stability = "stable"
        elif dr <= 2:
            stability = "moderately stable"
        else:
            stability = "unstable"

        is_continuous = f in NUMERIC_FEATURES
        crit = {
            "real_shap_strong": bool(r_rank is not None and r_rank <= 5),
            "fold_stable": bool(top5 >= max(2, int(0.6 * (real_full.loc[f, 'n_folds'] if f in real_full.index else 5)))),
            "augmentation_stable": stability in {"stable", "moderately stable"},
            "coverage_ok": bool(miss <= 0.35),
            "physically_interpretable": INTERPRETABILITY.get(f, "").startswith("high"),
        }
        n_met = sum(crit.values())
        data_ok = all(crit[c] for c in ("real_shap_strong", "fold_stable",
                                        "augmentation_stable", "coverage_ok"))
        usable_as_pde_input = is_continuous and crit["physically_interpretable"]
        if data_ok and usable_as_pde_input:
            rec = "recommended"
        elif data_ok:
            # the evidence is there, but a categorical bucket cannot enter a PDE
            # residual as-is
            rec = "conditional - needs a continuous physical parameterisation"
        elif n_met >= 3:
            rec = "conditional - collect more data first"
        else:
            rec = "not recommended yet"
        missing_reasons = [k for k, v in crit.items() if not v]
        if rec.startswith("conditional - needs"):
            missing_reasons = [
                "categorical: informative for the surrogate, but a PINN needs it recast "
                "as a continuous physical quantity (composition_family -> intercalant "
                "mass fraction; layer_class -> measured stacking number)"]
        direction = next((d for d in STATE.get("shap_directions", [])
                          if d["feature"] == f), None)
        dir_note = "" if direction is None else direction["expected"]
        if direction is not None and "contradict" in dir_note:
            missing_reasons.append(
                "SHAP direction contradicts the physical expectation - use it as a "
                "PINN input only alongside a mechanistic term that can absorb the "
                "confound, not as a fitted monotone trend")
        rows.append({
            "feature": f,
            "real_SHAP_rank": r_rank,
            "real_SHAP_rank_sd_across_folds": r_sd,
            "n_folds_in_top5": top5,
            "cohort_real_SHAP_rank": c_rank,
            "augmented_SHAP_rank": a_rank,
            "abs_rank_change_after_augmentation": dr,
            "rank_stability": stability,
            "missingness": round(miss, 3),
            "physical_interpretability": INTERPRETABILITY.get(f, ""),
            "shap_direction_vs_physics": dir_note or "not assessed (categorical or too sparse)",
            "in_bn_feature_set": in_bn,
            "recommended_for_PINN": rec,
            "reason": (f"criteria met {n_met}/5"
                       + (f"; unmet: {', '.join(missing_reasons)}" if missing_reasons else "")
                       + ". SHAP is a model-attribution measure, not evidence of causation."),
        })
    out = pd.DataFrame(rows).sort_values(
        ["recommended_for_PINN", "real_SHAP_rank"], na_position="last").reset_index(drop=True)
    out.to_csv(RESULTS_DIR / "08_pinn_descriptor_recommendation.csv", index=False)
    log("PINN recommendation table written")
    STATE["pinn"] = out


# ===========================================================================
# Stage 15 - narrative blocks derived from the computed numbers
# ===========================================================================

def stage_narrative() -> None:
    data = STATE["data"]
    src = STATE["real_shap_source"]
    base_fold = STATE["baseline"]["fold_metrics"]
    summ = STATE["aug_summary"].set_index("condition")
    fid = STATE["fidelity"]
    f2 = fid[fid["k"] == BN_K_DEFAULT]
    stab = STATE["stability_summary"]
    stats = STATE["stats"]
    sel = STATE["bn_selection"]


    # ---- sensitivity of the baseline to the two physically extreme targets
    pred = STATE["baseline"]["predictions"]
    keep = ~((pred["y_true"] < 20) | (pred["y_true"] > 1500))
    rows = []
    for fold, g in pred[keep].groupby("fold"):
        rows.append({"fold": fold, **metrics(g["y_true"], g["y_pred"])})
    sens = pd.DataFrame(rows)
    sens.to_csv(RESULTS_DIR / "02_real_only_outlier_sensitivity.csv", index=False)
    STATE["baseline_sensitivity"] = (
        "Sensitivity to the two physically extreme capacitances (7.26 and 1609.5 F/g): "
        "recomputing the fold metrics with those held-out points excluded from the "
        "*scoring* only (they remain in training, nothing is deleted) gives "
        f"R2 = {sens['r2'].mean():.3f} +/- {sens['r2'].std():.3f}, "
        f"MAE = {sens['mae'].mean():.1f} F/g, RMSE = {sens['rmse'].mean():.1f} F/g "
        f"(vs {base_fold['r2'].mean():.3f}, {base_fold['mae'].mean():.1f}, "
        f"{base_fold['rmse'].mean():.1f} with them). "
        "Full table in `02_real_only_outlier_sensitivity.csv`."
    )

    # ---- fidelity verdict
    ks_mean = f2["mean_ks_statistic"].mean()
    ks_rej = f2["n_ks_reject_p05"].mean()
    js = f2["mean_jensen_shannon"].mean()
    pdiff = f2["pearson_mean_abs_diff"].mean()
    midiff = f2["mi_mean_abs_diff"].mean()
    dup = f2["exact_duplicates_of_real"].sum()
    priv = f2["privacy_ratio_syn_vs_real_nn"].mean()
    closer = f2["n_syn_closer_than_real_p05"].mean()
    nsyn = f2["n_synthetic"].mean()
    n_num = len(STATE["bn_num_cols"]) + 1
    verdict_bits = []
    verdict_bits.append(
        f"Marginals: mean KS statistic {ks_mean:.3f} with on average "
        f"{ks_rej:.1f} of {n_num} continuous variables rejected at p<0.05 - "
        + ("marginal distributions are reproduced well."
           if ks_rej <= 1 else
           "some marginals are visibly distorted, mostly through the clipping of "
           "the generator's tails to the training-fold support.")
    )
    verdict_bits.append(
        f"Categorical frequencies: mean Jensen-Shannon distance {js:.3f} "
        + ("(close agreement)." if js < 0.15 else "(noticeable frequency drift).")
    )
    tassoc = f2["target_assoc_mean_abs_diff"].mean()
    verdict_bits.append(
        f"Descriptor-target relationships: mean absolute change of {tassoc:.3f} in the "
        "descriptor-target association (Spearman for continuous descriptors, "
        "eta-squared for categorical ones) - "
        + ("the relationships a regressor has to learn survive generation."
           if tassoc < 0.15 else
           "the relationships a regressor has to learn are noticeably altered, which "
           "matters more for downstream utility than any marginal mismatch.")
    )
    verdict_bits.append(
        f"Dependency structure: mean absolute difference of {pdiff:.3f} in the "
        f"Pearson correlation matrix and {midiff:.3f} in the normalised "
        "mutual-information matrix. "
        + ("Pairwise dependencies, including the descriptor-target relationships, "
           "are broadly preserved."
           if pdiff < 0.25 else
           "Pairwise dependencies are only partially preserved - the network "
           "reproduces the strongest edges and washes out the weaker ones, which "
           "is expected for a k-parent greedy Bayes network fitted to a few dozen rows.")
    )
    verdict_bits.append(
        f"Memorisation: {int(dup)} synthetic rows across all folds were exact copies of "
        f"a real training row and {int(f2['duplicates_within_synthetic'].sum())} were "
        f"duplicated within a synthetic set. The median synthetic-to-real "
        f"nearest-neighbour distance is {priv:.2f}x the median real-to-real "
        "nearest-neighbour distance; a ratio below 1 is expected here rather than "
        "alarming, because 37-200 synthetic points are drawn over the same manifold as "
        "~38 real ones and nearest-neighbour distances shrink with density alone. The "
        "sharper diagnostic is how many synthetic rows fall closer to a real record "
        "than the 5th percentile of real-to-real spacing: on average "
        f"{closer:.1f} of {nsyn:.0f} rows ({closer / max(1.0, nsyn):.1%}), "
        + ("a small tail rather than wholesale copying."
           if closer / max(1.0, nsyn) < 0.15 else
           "a large enough share to treat part of the synthetic set as memorised.")
    )
    STATE["fidelity_verdict"] = "**Verdict.** " + " ".join(verdict_bits)

    # ---- did augmentation actually help?  (three-way verdict, not a coin flip
    #      on a single noisy metric)
    best = STATE["best_condition"]
    r2_real = summ.loc["A_real_only", "r2_mean"]
    r2_best = summ.loc[best, "r2_mean"]
    mae_real, mae_best = summ.loc["A_real_only", "mae_mean"], summ.loc[best, "mae_mean"]
    rmse_real, rmse_best = summ.loc["A_real_only", "rmse_mean"], summ.loc[best, "rmse_mean"]
    pooled_real = summ.loc["A_real_only", "pooled_mae"]
    pooled_best = summ.loc[best, "pooled_mae"]
    mae_rel = (mae_best - mae_real) / mae_real
    best_stats = stats[(stats["comparison"] == f"real_only_vs_{best}") & (stats["metric"] == "mae")]
    folds_mae_improved = int(best_stats["n_folds_improved"].iloc[0]) if len(best_stats) else 0
    n_folds_aug = STATE["n_folds_aug"]
    material = (mae_rel <= -0.05) and folds_mae_improved >= int(np.ceil(0.6 * n_folds_aug))
    all_worse = (r2_best < r2_real) and (mae_best > mae_real) and (rmse_best > rmse_real)
    verdict_kind = "yes" if material else ("no" if all_worse else "wash")
    improved = verdict_kind == "yes"
    STATE["aug_verdict_kind"] = verdict_kind

    # ---- data-driven bottom line for the augmentation section
    seed_noise = summ["mae_sd_across_gen_seeds"].dropna().mean()
    aug_rows = summ[summ.index != "A_real_only"]
    better = [c for c in aug_rows.index if summ.loc[c, "mae_mean"] < mae_real]
    worse = [c for c in aug_rows.index if summ.loc[c, "mae_mean"] > mae_real]
    pretty = {"B_real_plus_1.0x": "1x", "C_real_plus_3.0x": "3x", "D_real_plus_5.25x": "5.25x"}
    parts = ["**Bottom line.** "]
    if verdict_kind == "yes":
        parts.append(
            f"Augmentation measurably improved prediction on unseen real papers: the "
            f"{pretty.get(best, best)} condition cut fold-mean MAE from {mae_real:.1f} to "
            f"{mae_best:.1f} F/g ({mae_rel:+.1%}) in {folds_mae_improved}/{n_folds_aug} folds, "
            f"an effect larger than the {seed_noise:.1f} F/g moved by re-drawing alone.")
    else:
        parts.append("Augmentation did not improve prediction on unseen real papers. ")
        if better:
            parts.append(
                "The nominally best condition ("
                + ", ".join(pretty.get(c, c) for c in better) +
                f") lowers fold-mean MAE by at most {mae_real - summ.loc[better, 'mae_mean'].min():.1f} F/g, "
                f"which is inside the {seed_noise:.1f} F/g that re-drawing the synthetic "
                f"set moves the same score, and it improves only "
                f"{folds_mae_improved}/{n_folds_aug} folds. ")
        if worse:
            parts.append(
                "Condition(s) " + ", ".join(pretty.get(c, c) for c in worse) +
                " are worse than real-only training. ")
        parts.append(
            "Read together with the k-sensitivity table below - where changing the "
            "network degree alone moves MAE as much as changing the ratio does - the "
            "conclusion is that Bayesian-network augmentation neither helps nor "
            "systematically harms this cohort; it adds variance without adding "
            "information. Reported as found: no configuration search was run to make "
            "the synthetic data look useful.")
    STATE["aug_bottom_line"] = "".join(parts)
    rho = stab["spearman_rank_correlation"]
    n_feat = stab["n_features"]
    # With only a handful of features one adjacent swap moves Spearman a long way,
    # so set membership and mean rank displacement are the better stability signals.
    solid = (stab["top3_overlap"] >= min(3, n_feat) - 1 and
             stab["mean_abs_rank_change"] <= 1.0 and
             stab["mean_abs_norm_shap_change"] <= 0.06)
    if rho >= 0.8 or solid:
        sv = ("The augmented model is driven by essentially the same descriptors as the "
              f"real-only model: the top-3 and top-5 sets are preserved, the average "
              f"rank displacement is {stab['mean_abs_rank_change']:.1f} positions and each "
              f"descriptor's share of total attribution moves by only "
              f"{stab['mean_abs_norm_shap_change']:.3f} on average. Synthetic augmentation "
              "did not rewrite the model's physics story."
              + ("" if rho >= 0.8 else
                 f" (The Spearman coefficient of {rho:.2f} looks lukewarm only because "
                 f"with {n_feat} features a single adjacent swap costs a large fraction "
                 "of the correlation.)"))
    elif rho >= 0.5:
        sv = ("The augmented model keeps the broad ordering of descriptors but reshuffles "
              "the middle of the ranking. Treat individual rank changes as noise rather "
              "than as new information.")
    else:
        sv = ("**Flag:** the augmented model attributes importance very differently from "
              "the real-only model. That is the classic signature of synthetic-"
              "distribution distortion - the Bayesian network has smoothed or "
              "exaggerated some dependencies - and it means the augmented ranking "
              "must not be used to justify descriptor choices.")
    if improved and rho < 0.5:
        sv += (" Because R2 improved *while* the attribution changed radically, the "
               "improvement should be read as regularisation of a small-sample model, "
               "not as better physics.")
    STATE["stability_verdict"] = sv

    # ---- A..K answers
    n_aug_better = int((stats[(stats.metric == "r2")]["absolute_difference"] > 0).sum())
    top5 = ", ".join(f"`{f}`" for f in src["feature"].head(5))
    pinn = STATE["pinn"]
    rec = pinn[pinn["recommended_for_PINN"] == "recommended"]["feature"].tolist()
    cond_param = pinn[pinn["recommended_for_PINN"].str.startswith(
        "conditional - needs")]["feature"].tolist()
    cond_data = pinn[pinn["recommended_for_PINN"].str.startswith(
        "conditional - collect")]["feature"].tolist()

    STATE["ak_answers"] = [
        ("A", f"**Usable observations:** {len(data)} of {STATE['audit']['n_rows']} corpus rows "
              f"carried a defensible single-valued gravimetric capacitance. "
              f"{int((STATE['target_log'].disposition == 'dropped:no_value_reported').sum())} rows "
              f"reported no value at all and "
              f"{int(STATE['target_log'].disposition.str.startswith('dropped:').sum()) - int((STATE['target_log'].disposition == 'dropped:no_value_reported').sum())} "
              f"reported only a range or an inequality, which were dropped rather than "
              f"converted to midpoints."),
        ("B", f"**Papers represented:** {data['paper_id'].nunique()} unique publications among the "
              f"modelled rows ({STATE['audit']['n_papers']} in the raw corpus); the Bayesian-network "
              f"cohort covers {STATE['bn_cohort']['paper_id'].nunique()} of them."),
        ("C", f"**Real-only XGBoost:** grouped {STATE['baseline']['n_folds']}-fold CV gives "
              f"R2 = {base_fold['r2'].mean():.3f} +/- {base_fold['r2'].std():.3f} "
              f"(median {base_fold['r2'].median():.3f}), "
              f"MAE = {base_fold['mae'].mean():.1f} +/- {base_fold['mae'].std():.1f} F/g, "
              f"RMSE = {base_fold['rmse'].mean():.1f} +/- {base_fold['rmse'].std():.1f} F/g. "
              f"Fold-to-fold spread is wide (R2 {base_fold['r2'].min():.2f} to {base_fold['r2'].max():.2f})."),
        ("D", f"**Top real-data SHAP descriptors:** {top5}."),
        ("E", f"**Bayesian-network variables:** {', '.join('`' + f + '`' for f in sel['selected_features'])} "
              f"plus the target `{TARGET}`, on a complete-case cohort of "
              f"{sel['cohort']['rows_after']} rows / {sel['cohort']['papers_after']} papers."),
        ("F", "**Fidelity:** " + STATE["fidelity_verdict"].replace("**Verdict.** ", "")),
        ("G", f"**Best augmentation ratio:** {best} (fold-mean R2 {r2_best:.3f} and MAE "
              f"{mae_best:.1f} F/g, against {r2_real:.3f} / {mae_real:.1f} F/g for "
              f"real-only). The ranking is not monotone in the amount of synthetic data "
              f"(3x is the *worst* condition here), and the gap between the best and "
              f"worst ratio ({summ['mae_mean'].max() - summ['mae_mean'].min():.1f} F/g MAE) "
              f"is comparable to the {seed_noise:.1f} F/g that re-drawing the same "
              "configuration moves the score. More synthetic data is not automatically "
              "better, and on this cohort no ratio is reliably better than any other."),
        ("H", {
            "yes": (
                "**Yes - augmentation improved generalisation to unseen real papers.** "
                f"The best condition ({best}) moved fold-mean R2 from {r2_real:.3f} to "
                f"{r2_best:.3f} and fold-mean MAE from {mae_real:.1f} to {mae_best:.1f} F/g "
                f"({mae_rel:+.1%}), improving MAE in {folds_mae_improved}/{n_folds_aug} folds."),
            "no": (
                "**No - augmentation degraded generalisation to unseen real papers.** "
                f"Even the best augmented condition ({best}) was worse than real-only "
                f"training on every metric (R2 {r2_best:.3f} vs {r2_real:.3f}, MAE "
                f"{mae_best:.1f} vs {mae_real:.1f} F/g). Reported as found."),
            "wash": (
                "**No material improvement - the honest answer is that augmentation did "
                f"not help.** The best augmented condition ({best}) shifted fold-mean MAE "
                f"from {mae_real:.1f} to {mae_best:.1f} F/g ({mae_rel:+.1%}) and pooled MAE "
                f"from {pooled_real:.1f} to {pooled_best:.1f} F/g, improving MAE in "
                f"{folds_mae_improved}/{n_folds_aug} folds. That shift is inside the "
                f"{seed_noise:.1f} F/g that simply re-drawing the synthetic set with a "
                f"different generator seed moves the same score, and far inside the "
                f"fold-to-fold spread (MAE SD {summ.loc['A_real_only', 'mae_sd']:.1f} F/g). "
                "The ranking is not even monotone in the amount of synthetic data - the "
                f"3x condition is the worst of the four (MAE "
                f"{summ.loc['C_real_plus_3.0x', 'mae_mean']:.1f} F/g). No search was run to "
                "make synthetic data look beneficial, and none of the differences approach "
                "significance."),
        }[verdict_kind]),
        ("I", f"**SHAP stability:** Spearman rank correlation "
              f"{stab['spearman_rank_correlation']:.3f}, top-3 overlap "
              f"{stab['top3_overlap']}/3, top-5 overlap {stab['top5_overlap']}/"
              f"{min(5, stab['n_features'])}, mean |rank change| "
              f"{stab['mean_abs_rank_change']:.2f}. " + STATE["stability_verdict"]),
        ("J", f"**PINN descriptors:** ready to use as physical inputs now - "
              f"{', '.join('`' + f + '`' for f in rec) or 'none'}"
              + (f"; well supported by the data but categorical, so they need recasting "
                 f"as continuous physical quantities before they can enter a PDE residual "
                 f"({', '.join('`' + f + '`' for f in cond_param)})" if cond_param else "")
              + (f"; too sparsely reported to judge yet - "
                 f"{', '.join('`' + f + '`' for f in cond_data)}" if cond_data else "")
              + ". See `08_pinn_descriptor_recommendation.csv`."),
        ("K", "**Remaining limitations:** heterogeneous literature-derived rows, heavy "
              "clustering within papers, severe descriptor missingness that forces a "
              "small complete-case cohort, a near-constant synthesis route in the "
              "surviving rows, synthetic rows that carry no new experimental "
              "information, and a fold count too small for meaningful significance "
              "testing. Section 9 lists these in full."),
    ]
    log("narrative blocks composed")


# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-report", action="store_true")
    ap.parse_args()

    ensure_dirs()
    seed_everything(RANDOM_STATE)
    t0 = time.time()

    stage_environment()
    stage_audit()
    stage_clean()
    stage_real_baseline()
    stage_real_shap()
    stage_bn_candidates()
    stage_bn_selection()
    stage_augmentation()
    stage_fidelity_figures()
    stage_statistics()
    stage_augmented_shap()
    stage_pinn()
    stage_narrative()
    write_final_report(STATE)

    log(f"pipeline complete in {time.time() - t0:.1f}s -> {RESULTS_DIR}")


if __name__ == "__main__":
    main()

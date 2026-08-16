"""Stage F-3/F-5: the final real-only interpretation model and its SHAP analysis.

IMPORTANT SCOPE NOTE, enforced by naming throughout this module.

    Grouped cross-paper CV (results/02_*) measures *generalisation*.
    The model built here is fitted on every usable real observation and exists
    only to characterise multivariate descriptor associations within the
    compiled corpus.  Its in-sample fit is not, and must never be reported as,
    a predictive performance estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from .config import RANDOM_STATE, TARGET
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .modeling import FoldEncoder, fit_xgb, make_group_kfold, metrics, tune_xgb

# human-readable descriptor names and units for publication figures
DISPLAY = {
    "interlayer_A":           ("Interlayer spacing", r"$\AA$"),
    "h2so4_M":                ("H$_2$SO$_4$ concentration", "M"),
    "scan_rate_mV_s":         ("Scan rate", "mV s$^{-1}$"),
    "current_density_A_g":    ("Current density", "A g$^{-1}$"),
    "mass_loading_mg_cm2":    ("Mass loading", "mg cm$^{-2}$"),
    "electrode_thickness_um": ("Electrode thickness", r"$\mu$m"),
    "ssa_m2_g":               ("Specific surface area", "m$^2$ g$^{-1}$"),
    "flake_size_um":          ("Flake size", r"$\mu$m"),
    "pore_diameter_nm":       ("Pore diameter", "nm"),
    "layer_class":            ("Layer morphology", ""),
    "composition_family":     ("Composition family", ""),
    "synthesis_family":       ("Synthesis route", ""),
    "electrolyte_is_gel":     ("Gel electrolyte", ""),
    TARGET:                   ("Gravimetric capacitance", "F g$^{-1}$"),
}

PLAIN = {
    "interlayer_A": "Interlayer spacing (A)",
    "h2so4_M": "H2SO4 concentration (M)",
    "scan_rate_mV_s": "Scan rate (mV/s)",
    "current_density_A_g": "Current density (A/g)",
    "mass_loading_mg_cm2": "Mass loading (mg/cm2)",
    "electrode_thickness_um": "Electrode thickness (um)",
    "ssa_m2_g": "Specific surface area (m2/g)",
    "flake_size_um": "Flake size (um)",
    "pore_diameter_nm": "Pore diameter (nm)",
    "layer_class": "Layer morphology",
    "composition_family": "Composition family",
    "synthesis_family": "Synthesis route",
    "electrolyte_is_gel": "Gel electrolyte",
}


def label(feat: str, latex: bool = True) -> str:
    if not latex:
        return PLAIN.get(feat, feat)
    name, unit = DISPLAY.get(feat, (feat, ""))
    return f"{name} ({unit})" if unit else name


def drop_degenerate(data: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    keep, dropped = [], []
    for f in features:
        if data[f].nunique(dropna=True) < 2:
            dropped.append(f)
        else:
            keep.append(f)
    return keep, dropped


def fit_final_real_model(data: pd.DataFrame, features: list[str]) -> dict:
    """Fit the interpretation model on every usable real observation."""
    num = [f for f in features if f in NUMERIC_FEATURES]
    cat = [f for f in features if f in CATEGORICAL_FEATURES]

    enc = FoldEncoder(cat, num).fit(data[features])
    X = enc.transform(data[features])
    y = data[TARGET].values
    groups = data["paper_id"].values

    # Hyper-parameters are selected with a paper-grouped inner search so the
    # interpretation model is not tuned to memorise single papers.  This is a
    # model-selection step, not a performance estimate.
    params = tune_xgb(X, y, groups, seed=RANDOM_STATE)
    model = fit_xgb(X, y, params, seed=RANDOM_STATE)

    in_sample = metrics(y, model.predict(X))
    return {"model": model, "encoder": enc, "X": X, "y": y, "groups": groups,
            "params": params, "features": features, "num_cols": num, "cat_cols": cat,
            "in_sample": in_sample}


def final_shap(fit: dict) -> dict:
    """TreeExplainer SHAP on the full real corpus (in-sample attribution)."""
    explainer = shap.TreeExplainer(fit["model"])
    sv = np.asarray(explainer.shap_values(fit["X"], check_additivity=False))
    shap_df = pd.DataFrame(sv, columns=fit["X"].columns, index=fit["X"].index)

    ohe_map = fit["encoder"].ohe_to_source_
    enc = pd.DataFrame({
        "encoded_feature": shap_df.columns,
        "source_descriptor": [ohe_map.get(c, c) for c in shap_df.columns],
        "mean_abs_shap": shap_df.abs().mean(axis=0).values,
        "mean_shap": shap_df.mean(axis=0).values,
        "max_abs_shap": shap_df.abs().max(axis=0).values,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    enc.insert(0, "rank", np.arange(1, len(enc) + 1))
    enc["normalized_importance"] = enc["mean_abs_shap"] / enc["mean_abs_shap"].sum()

    # group one-hot dummies back to the parent descriptor: sum |SHAP| per row,
    # then average over rows (the correct group-level attribution).
    src = [ohe_map.get(c, c) for c in shap_df.columns]
    grouped_rows = shap_df.abs().T.groupby(pd.Index(src)).sum().T
    grp = pd.DataFrame({
        "descriptor": grouped_rows.columns,
        "mean_abs_shap": grouped_rows.mean(axis=0).values,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    grp.insert(0, "rank", np.arange(1, len(grp) + 1))
    grp["normalized_importance"] = grp["mean_abs_shap"] / grp["mean_abs_shap"].sum()
    return {"shap_df": shap_df, "encoded": enc, "grouped": grp,
            "grouped_rows": grouped_rows, "explainer": explainer}


def enrich_with_fold_stability(grouped: pd.DataFrame, data: pd.DataFrame,
                               fold_shap_path) -> pd.DataFrame:
    """Attach coverage plus the fold-wise rank statistics from the grouped-CV run."""
    fold = pd.read_csv(fold_shap_path)
    fold = fold[fold["level"] == "source"].set_index("feature")

    out = grouped.copy()
    out["feature_type"] = np.where(out["descriptor"].isin(NUMERIC_FEATURES),
                                   "numeric", "categorical")
    out["missingness"] = out["descriptor"].map(lambda f: float(data[f].isna().mean()))
    out["number_of_observations"] = out["descriptor"].map(
        lambda f: int(data[f].notna().sum()))
    for col, new in [("mean_rank", "mean_rank_across_folds"),
                     ("rank_sd", "rank_SD"),
                     ("mean_abs_shap", "foldwise_heldout_mean_abs_shap"),
                     ("shap_rank", "foldwise_heldout_rank")]:
        out[new] = out["descriptor"].map(
            lambda f: fold.loc[f, col] if f in fold.index else np.nan)
    n_folds = float(fold["n_folds"].iloc[0]) if "n_folds" in fold else np.nan
    out["top5_frequency"] = out["descriptor"].map(
        lambda f: fold.loc[f, "n_folds_top5"] / n_folds if f in fold.index else np.nan)
    out["display_name"] = out["descriptor"].map(lambda f: PLAIN.get(f, f))
    return out


def shap_directions(shap_df: pd.DataFrame, X_raw: pd.DataFrame,
                    features: list[str]) -> pd.DataFrame:
    """Spearman correlation between a descriptor's value and its own SHAP value."""
    rows = []
    for f in features:
        if f not in NUMERIC_FEATURES or f not in shap_df.columns:
            continue
        x = pd.to_numeric(X_raw[f], errors="coerce")
        ok = x.notna().values
        if ok.sum() < 8:
            continue
        rho = pd.Series(x[ok].values).corr(
            pd.Series(shap_df.loc[ok, f].values), method="spearman")
        rows.append({"descriptor": f, "n_used": int(ok.sum()),
                     "spearman_value_vs_shap": rho,
                     "direction": ("increasing" if rho > 0.15 else
                                   "decreasing" if rho < -0.15 else "non-monotone/flat")})
    return pd.DataFrame(rows)

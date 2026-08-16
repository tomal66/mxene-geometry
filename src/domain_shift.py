"""Stage F-2: quantify cross-paper heterogeneity and train/test domain shift.

Everything here is descriptive.  Nothing in this module claims a causal
explanation for the grouped-CV result; it measures how much of the variance in
the target and in each descriptor lives *between* publications rather than
within them, and how far each fold's test papers sit outside the descriptor
domain its training papers cover.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import TARGET
from .feature_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES


# --------------------------------------------------------------- variance split

def variance_decomposition(values: pd.Series, groups: pd.Series) -> dict:
    """One-way (paper) variance decomposition of a numeric variable.

    ``eta_squared`` is the share of total sum-of-squares attributable to
    between-paper differences.  ``icc1`` is the classical one-way
    intraclass-correlation estimate, which corrects for unequal group sizes and
    is the more conservative number of the two.
    """
    v = pd.to_numeric(values, errors="coerce")
    ok = v.notna()
    v, g = v[ok], groups[ok].astype(str)
    n, k = len(v), g.nunique()
    if n < 3 or k < 2:
        return {"n": n, "n_groups": k, "eta_squared": np.nan, "icc1": np.nan,
                "between_sd": np.nan, "within_sd": np.nan}

    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    sizes = g.value_counts()
    ss_between = float(sum(len(sub) * (sub.mean() - grand) ** 2 for _, sub in v.groupby(g)))
    ss_within = ss_total - ss_between
    df_b, df_w = k - 1, n - k
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan

    icc = np.nan
    if df_w > 0 and df_b > 0:
        ms_b, ms_w = ss_between / df_b, ss_within / df_w
        # average group size correction for unbalanced designs
        k0 = (n - (sizes ** 2).sum() / n) / df_b
        if k0 > 0 and (ms_b + (k0 - 1) * ms_w) != 0:
            icc = (ms_b - ms_w) / (ms_b + (k0 - 1) * ms_w)
    return {
        "n": n, "n_groups": k, "eta_squared": eta2, "icc1": icc,
        "between_sd": float(np.sqrt(max(ss_between / df_b, 0))) if df_b else np.nan,
        "within_sd": float(np.sqrt(max(ss_within / df_w, 0))) if df_w else np.nan,
    }


def heterogeneity_table(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Between-paper variance share for the target and every numeric descriptor."""
    rows = []
    for col in [TARGET] + [f for f in features if f in NUMERIC_FEATURES]:
        d = variance_decomposition(data[col], data["paper_id"])
        rows.append({"variable": col,
                     "role": "target" if col == TARGET else "descriptor", **d})
    # categorical descriptors: how concentrated is each paper on one level?
    for col in [f for f in features if f in CATEGORICAL_FEATURES]:
        sub = data[[col, "paper_id"]].dropna()
        if sub.empty:
            continue
        purity = sub.groupby("paper_id")[col].agg(
            lambda s: s.value_counts(normalize=True).iloc[0])
        rows.append({"variable": col, "role": "descriptor",
                     "n": len(sub), "n_groups": int(sub["paper_id"].nunique()),
                     "eta_squared": np.nan, "icc1": np.nan,
                     "between_sd": np.nan, "within_sd": np.nan,
                     "mean_within_paper_purity": float(purity.mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------- paper-level tables

def paper_summary(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for pid, g in data.groupby("paper_id"):
        r = {
            "paper_id": pid,
            "n_rows": len(g),
            "cap_mean": g[TARGET].mean(),
            "cap_median": g[TARGET].median(),
            "cap_sd": g[TARGET].std(),
            "cap_min": g[TARGET].min(),
            "cap_max": g[TARGET].max(),
            "cap_range": g[TARGET].max() - g[TARGET].min(),
            "n_descriptors_reported": int(sum(g[f].notna().any() for f in features)),
        }
        for f in features:
            if f in NUMERIC_FEATURES:
                r[f"{f}__median"] = g[f].median()
                r[f"{f}__n"] = int(g[f].notna().sum())
            else:
                vc = g[f].dropna()
                r[f"{f}__mode"] = vc.mode().iloc[0] if len(vc) else np.nan
                r[f"{f}__n_levels"] = int(vc.nunique())
        rows.append(r)
    return pd.DataFrame(rows).sort_values("cap_median", ascending=False).reset_index(drop=True)


def capacitance_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Compact paper-level capacitance table, ordered for plotting."""
    g = data.groupby("paper_id")[TARGET]
    out = pd.DataFrame({
        "paper_id": g.median().index,
        "n_rows": g.size().values,
        "cap_min": g.min().values,
        "cap_median": g.median().values,
        "cap_max": g.max().values,
        "cap_mean": g.mean().values,
        "cap_sd": g.std().values,
    }).sort_values("cap_median").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


# --------------------------------------------------------------- fold overlap

def _interval_overlap(tr: pd.Series, te: pd.Series) -> dict:
    tr, te = tr.dropna(), te.dropna()
    if len(tr) < 2 or len(te) < 1:
        return {"frac_test_inside_train_range": np.nan,
                "standardised_mean_diff": np.nan,
                "range_overlap_fraction": np.nan}
    lo, hi = tr.min(), tr.max()
    inside = float(((te >= lo) & (te <= hi)).mean())
    pooled = np.sqrt((tr.var(ddof=1) + te.var(ddof=1)) / 2) if len(te) > 1 else tr.std(ddof=1)
    smd = float((te.mean() - tr.mean()) / pooled) if pooled and pooled > 0 else np.nan
    tlo, thi = te.min(), te.max()
    inter = max(0.0, min(hi, thi) - max(lo, tlo))
    union = max(hi, thi) - min(lo, tlo)
    return {"frac_test_inside_train_range": inside,
            "standardised_mean_diff": smd,
            "range_overlap_fraction": float(inter / union) if union > 0 else np.nan}


def fold_domain_overlap(data: pd.DataFrame, features: list[str], splits) -> pd.DataFrame:
    """Per fold and per descriptor: how far the test papers sit outside the train domain."""
    rows = []
    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        tr, te = data.iloc[tr_idx], data.iloc[te_idx]
        for f in features:
            if f in NUMERIC_FEATURES:
                d = _interval_overlap(tr[f], te[f])
                rows.append({"fold": fold, "variable": f, "type": "numeric",
                             "n_train_reported": int(tr[f].notna().sum()),
                             "n_test_reported": int(te[f].notna().sum()),
                             "train_min": tr[f].min(), "train_max": tr[f].max(),
                             "test_min": te[f].min(), "test_max": te[f].max(), **d})
            else:
                seen = set(tr[f].dropna().astype(str))
                tev = te[f].dropna().astype(str)
                unseen = float((~tev.isin(seen)).mean()) if len(tev) else np.nan
                rows.append({"fold": fold, "variable": f, "type": "categorical",
                             "n_train_reported": int(tr[f].notna().sum()),
                             "n_test_reported": int(len(tev)),
                             "frac_test_unseen_level": unseen,
                             "n_train_levels": len(seen),
                             "n_test_levels": int(tev.nunique())})
        # target domain overlap
        d = _interval_overlap(tr[TARGET], te[TARGET])
        rows.append({"fold": fold, "variable": TARGET, "type": "target",
                     "n_train_reported": len(tr), "n_test_reported": len(te),
                     "train_min": tr[TARGET].min(), "train_max": tr[TARGET].max(),
                     "test_min": te[TARGET].min(), "test_max": te[TARGET].max(),
                     "test_target_sd": te[TARGET].std(),
                     "train_target_sd": tr[TARGET].std(), **d})
    return pd.DataFrame(rows)


def feature_range_overlap(data: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """How much of each descriptor's spread is realised inside a single paper."""
    rows = []
    for f in features:
        if f not in NUMERIC_FEATURES:
            continue
        s = data[f]
        if s.notna().sum() < 3:
            continue
        glob = s.max() - s.min()
        per_paper = data.groupby("paper_id")[f].agg(lambda x: x.max() - x.min())
        n_multi = int(data.groupby("paper_id")[f].apply(lambda x: x.notna().sum() > 1).sum())
        rows.append({
            "variable": f,
            "n_reported": int(s.notna().sum()),
            "n_papers_reporting": int(data.groupby("paper_id")[f].apply(lambda x: x.notna().any()).sum()),
            "n_papers_with_multiple_values": n_multi,
            "global_range": float(glob),
            "median_within_paper_range": float(per_paper.median(skipna=True)),
            "within_over_global_range": float(per_paper.median(skipna=True) / glob) if glob else np.nan,
        })
    return pd.DataFrame(rows).sort_values("within_over_global_range")


def fold_r2_context(fold_metrics: pd.DataFrame, overlap: pd.DataFrame) -> pd.DataFrame:
    """Pair each fold's score with the spread of its own test targets.

    R^2 is normalised by the variance of the held-out targets, so a fold whose
    test papers happen to agree with each other is scored against a very small
    denominator.  This table makes that visible instead of leaving it implicit.
    """
    tgt = overlap[overlap["type"] == "target"][
        ["fold", "test_target_sd", "train_target_sd",
         "frac_test_inside_train_range", "standardised_mean_diff"]]
    return fold_metrics.merge(tgt, on="fold", how="left")

"""Fidelity checks comparing real training data with generated synthetic data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.metrics import mutual_info_score


# ------------------------------------------------------------------ continuous

def continuous_stats(real: pd.Series, syn: pd.Series, name: str) -> dict:
    r = pd.to_numeric(real, errors="coerce").dropna().values
    s = pd.to_numeric(syn, errors="coerce").dropna().values
    out = {"variable": name, "type": "continuous", "n_real": len(r), "n_syn": len(s)}
    if len(r) < 2 or len(s) < 2:
        return out
    q = lambda a, p: float(np.percentile(a, p))
    out.update(
        real_mean=float(np.mean(r)), syn_mean=float(np.mean(s)),
        real_median=float(np.median(r)), syn_median=float(np.median(s)),
        real_std=float(np.std(r, ddof=1)), syn_std=float(np.std(s, ddof=1)),
        real_iqr=q(r, 75) - q(r, 25), syn_iqr=q(s, 75) - q(s, 25),
        real_min=float(np.min(r)), syn_min=float(np.min(s)),
        real_max=float(np.max(r)), syn_max=float(np.max(s)),
    )
    ks = ks_2samp(r, s)
    out["ks_statistic"] = float(ks.statistic)
    out["ks_pvalue"] = float(ks.pvalue)
    out["wasserstein"] = float(wasserstein_distance(r, s))
    scale = np.std(r, ddof=1)
    out["wasserstein_normalised"] = float(out["wasserstein"] / scale) if scale > 0 else np.nan
    out["mean_shift_in_real_sd"] = float((np.mean(s) - np.mean(r)) / scale) if scale > 0 else np.nan
    return out


def categorical_stats(real: pd.Series, syn: pd.Series, name: str) -> dict:
    cats = sorted(set(real.dropna().astype(str)) | set(syn.dropna().astype(str)))
    p = real.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).values
    q = syn.astype(str).value_counts(normalize=True).reindex(cats).fillna(0).values
    js = float(jensenshannon(p, q, base=2)) if len(cats) > 1 else 0.0
    return {
        "variable": name, "type": "categorical",
        "n_real": int(real.notna().sum()), "n_syn": int(syn.notna().sum()),
        "n_categories_real": int(real.dropna().nunique()),
        "n_categories_syn": int(syn.dropna().nunique()),
        "jensen_shannon_distance": 0.0 if np.isnan(js) else js,
        "total_variation_distance": float(0.5 * np.abs(p - q).sum()),
        "max_freq_abs_diff": float(np.abs(p - q).max()),
        "real_freqs": "; ".join(f"{c}={v:.3f}" for c, v in zip(cats, p)),
        "syn_freqs": "; ".join(f"{c}={v:.3f}" for c, v in zip(cats, q)),
    }


# ----------------------------------------------------------------- dependencies

def _numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce")


def correlation_agreement(real: pd.DataFrame, syn: pd.DataFrame,
                          num_cols: list[str]) -> tuple[dict, dict]:
    mats = {}
    res = {}
    for method in ("pearson", "spearman"):
        A = _numeric_frame(real, num_cols).corr(method=method)
        B = _numeric_frame(syn, num_cols).corr(method=method)
        B = B.reindex(index=A.index, columns=A.columns)
        D = (B - A).values
        iu = np.triu_indices_from(D, k=1)
        d = D[iu]
        d = d[np.isfinite(d)]
        res[f"{method}_frobenius_diff"] = float(np.sqrt(np.nansum(D ** 2)))
        res[f"{method}_mean_abs_diff"] = float(np.mean(np.abs(d))) if d.size else np.nan
        res[f"{method}_max_abs_diff"] = float(np.max(np.abs(d))) if d.size else np.nan
        a = A.values[iu]; b = B.values[iu]
        ok = np.isfinite(a) & np.isfinite(b)
        res[f"{method}_offdiag_corr_of_corrs"] = (
            float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else np.nan
        )
        mats[method] = (A, B)
    return res, mats


def _discretise(s: pd.Series, bins: int, edges=None):
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() == 0:
        return s.astype(str).values, None
    if edges is None:
        edges = np.unique(np.nanpercentile(v.dropna(), np.linspace(0, 100, bins + 1)))
        if len(edges) < 3:
            edges = np.array([v.min() - 1e-9, v.max() + 1e-9])
    return np.digitize(v.fillna(v.median()), edges[1:-1]).astype(str), edges


def mutual_information_matrix(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str],
                              bins: int = 5, edges_ref: dict | None = None):
    """Normalised pairwise mutual information over a mixed-type frame."""
    cols = num_cols + cat_cols
    disc, edges = {}, {}
    for c in cols:
        if c in num_cols:
            d, e = _discretise(df[c], bins, None if edges_ref is None else edges_ref.get(c))
            disc[c], edges[c] = d, e
        else:
            disc[c], edges[c] = df[c].astype(str).values, None
    M = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    for i, a in enumerate(cols):
        for b in cols[i:]:
            mi = mutual_info_score(disc[a], disc[b])
            ha = mutual_info_score(disc[a], disc[a])
            hb = mutual_info_score(disc[b], disc[b])
            denom = np.sqrt(ha * hb)
            v = float(mi / denom) if denom > 0 else 0.0
            M.loc[a, b] = M.loc[b, a] = v
    return M, edges


def mi_agreement(real: pd.DataFrame, syn: pd.DataFrame, num_cols: list[str],
                 cat_cols: list[str], bins: int = 5):
    A, edges = mutual_information_matrix(real, num_cols, cat_cols, bins)
    B, _ = mutual_information_matrix(syn, num_cols, cat_cols, bins, edges_ref=edges)
    B = B.reindex(index=A.index, columns=A.columns)
    D = (B - A).values
    iu = np.triu_indices_from(D, k=1)
    d = D[iu][np.isfinite(D[iu])]
    res = {
        "mi_frobenius_diff": float(np.sqrt(np.nansum(D ** 2))),
        "mi_mean_abs_diff": float(np.mean(np.abs(d))) if d.size else np.nan,
        "mi_max_abs_diff": float(np.max(np.abs(d))) if d.size else np.nan,
    }
    return res, A, B


def _eta_squared(values: pd.Series, groups: pd.Series) -> float:
    """Fraction of target variance explained by a categorical descriptor."""
    v = pd.to_numeric(values, errors="coerce")
    g = groups.astype(str)
    ok = v.notna()
    v, g = v[ok], g[ok]
    if len(v) < 3 or g.nunique() < 2:
        return np.nan
    grand = v.mean()
    ss_total = float(((v - grand) ** 2).sum())
    if ss_total == 0:
        return np.nan
    ss_between = float(sum(len(sub) * (sub.mean() - grand) ** 2 for _, sub in v.groupby(g)))
    return ss_between / ss_total


def target_relationships(real: pd.DataFrame, syn: pd.DataFrame, num_cols: list[str],
                         cat_cols: list[str], target: str) -> pd.DataFrame:
    """Descriptor-to-target association, real vs synthetic.

    This is the fidelity check that matters most for a downstream regressor:
    a generator can match every marginal perfectly and still destroy the
    descriptor-target relationship the model has to learn.
    """
    rows = []
    for c in num_cols:
        r_x, r_y = pd.to_numeric(real[c], errors="coerce"), pd.to_numeric(real[target], errors="coerce")
        s_x, s_y = pd.to_numeric(syn[c], errors="coerce"), pd.to_numeric(syn[target], errors="coerce")
        rp, sp = r_x.corr(r_y, method="pearson"), s_x.corr(s_y, method="pearson")
        rs, ss = r_x.corr(r_y, method="spearman"), s_x.corr(s_y, method="spearman")
        rows.append({"variable": c, "type": "continuous",
                     "real_pearson_with_target": rp, "syn_pearson_with_target": sp,
                     "pearson_abs_diff": abs(sp - rp) if pd.notna(sp) and pd.notna(rp) else np.nan,
                     "real_spearman_with_target": rs, "syn_spearman_with_target": ss,
                     "spearman_abs_diff": abs(ss - rs) if pd.notna(ss) and pd.notna(rs) else np.nan,
                     "sign_preserved": bool(pd.notna(rp) and pd.notna(sp) and np.sign(rp) == np.sign(sp))})
    for c in cat_cols:
        re_, se = _eta_squared(real[target], real[c]), _eta_squared(syn[target], syn[c])
        rows.append({"variable": c, "type": "categorical",
                     "real_eta_squared_on_target": re_, "syn_eta_squared_on_target": se,
                     "eta_squared_abs_diff": abs(se - re_) if pd.notna(se) and pd.notna(re_) else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ memorisation

def _gower_matrix(syn: pd.DataFrame, real: pd.DataFrame, num_cols: list[str],
                  cat_cols: list[str]) -> np.ndarray:
    """Scaled mixed-type distance: per-variable range-normalised L1 + category mismatch."""
    parts = []
    for c in num_cols:
        r = pd.to_numeric(real[c], errors="coerce").values.astype(float)
        s = pd.to_numeric(syn[c], errors="coerce").values.astype(float)
        rng = np.nanmax(r) - np.nanmin(r)
        rng = rng if rng > 0 else 1.0
        parts.append(np.abs(s[:, None] - r[None, :]) / rng)
    for c in cat_cols:
        r = real[c].astype(str).values
        s = syn[c].astype(str).values
        parts.append((s[:, None] != r[None, :]).astype(float))
    if not parts:
        return np.zeros((len(syn), len(real)))
    return np.nanmean(np.stack(parts, axis=0), axis=0)


def memorisation_check(syn: pd.DataFrame, real: pd.DataFrame, num_cols: list[str],
                       cat_cols: list[str], round_dp: int = 6) -> dict:
    cols = num_cols + cat_cols
    r_key = real[cols].round(round_dp).astype(str).agg("|".join, axis=1)
    s_key = syn[cols].round(round_dp).astype(str).agg("|".join, axis=1)
    exact_copies = int(s_key.isin(set(r_key)).sum())
    dup_within = int(s_key.duplicated().sum())

    D = _gower_matrix(syn, real, num_cols, cat_cols)
    nn = D.min(axis=1) if D.size else np.array([np.nan])
    # baseline: real-to-real nearest neighbour (excluding self)
    Drr = _gower_matrix(real, real, num_cols, cat_cols)
    np.fill_diagonal(Drr, np.inf)
    nn_rr = Drr.min(axis=1) if Drr.size else np.array([np.nan])

    return {
        "n_synthetic": len(syn),
        "n_real_train": len(real),
        "exact_duplicates_of_real": exact_copies,
        "exact_duplicate_fraction": exact_copies / max(1, len(syn)),
        "duplicates_within_synthetic": dup_within,
        "duplicate_within_fraction": dup_within / max(1, len(syn)),
        "nn_dist_mean": float(np.nanmean(nn)),
        "nn_dist_median": float(np.nanmedian(nn)),
        "nn_dist_p05": float(np.nanpercentile(nn, 5)),
        "nn_dist_min": float(np.nanmin(nn)),
        "real_to_real_nn_median": float(np.nanmedian(nn_rr)),
        "privacy_ratio_syn_vs_real_nn": float(np.nanmedian(nn) / np.nanmedian(nn_rr))
        if np.nanmedian(nn_rr) > 0 else np.nan,
        "n_syn_closer_than_real_p05": int(np.nansum(nn < np.nanpercentile(nn_rr, 5))),
    }

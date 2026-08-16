"""SHAP computation, cross-fold aggregation and rank-stability analysis."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr


def fold_shap_values(model, X_test: pd.DataFrame) -> np.ndarray:
    """TreeExplainer SHAP values for HELD-OUT rows."""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test, check_additivity=False)
    return np.asarray(sv)


def pool_folds(fold_records: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align per-fold SHAP matrices onto a union feature space.

    Returns ``(pooled_shap, pooled_X)`` with one row per held-out observation.
    """
    cols: list[str] = []
    for r in fold_records:
        for c in r["X"].columns:
            if c not in cols:
                cols.append(c)
    shap_parts, x_parts = [], []
    for r in fold_records:
        s = pd.DataFrame(r["shap"], columns=r["X"].columns, index=r["X"].index)
        shap_parts.append(s.reindex(columns=cols).fillna(0.0))
        x_parts.append(r["X"].reindex(columns=cols))
    return pd.concat(shap_parts), pd.concat(x_parts)


def importance_table(fold_records: list[dict], ohe_to_source: dict[str, str]) -> pd.DataFrame:
    """Per-encoded-feature mean|SHAP|, rank statistics across folds."""
    per_fold = {}
    for r in fold_records:
        s = pd.DataFrame(np.abs(r["shap"]), columns=r["X"].columns).mean(axis=0)
        per_fold[r["fold"]] = s
    M = pd.DataFrame(per_fold).fillna(0.0)          # features x folds
    ranks = M.rank(ascending=False, axis=0, method="min")

    out = pd.DataFrame(
        {
            "feature": M.index,
            "mean_abs_shap": M.mean(axis=1).values,
            "sd_abs_shap": M.std(axis=1).values,
            "mean_rank": ranks.mean(axis=1).values,
            "rank_sd": ranks.std(axis=1).values,
            "n_folds_top5": (ranks <= 5).sum(axis=1).values,
            "n_folds": M.shape[1],
        }
    )
    out["source_feature"] = out["feature"].map(lambda f: ohe_to_source.get(f, f))
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(1, "shap_rank", np.arange(1, len(out) + 1))
    out["level"] = "encoded"
    return out


def aggregate_to_source(enc: pd.DataFrame, fold_records: list[dict],
                        ohe_to_source: dict[str, str]) -> pd.DataFrame:
    """Aggregate one-hot columns back to the original variable.

    Aggregation is done on the |SHAP| values themselves (summed across the
    dummies of a variable **within each row**, then averaged), which is the
    correct group-level attribution rather than a sum of column means.
    """
    per_fold = {}
    for r in fold_records:
        src = [ohe_to_source.get(c, c) for c in r["X"].columns]
        A = pd.DataFrame(np.abs(r["shap"]), columns=r["X"].columns)
        grouped = A.T.groupby(pd.Index(src)).sum().T          # rows x source_feature
        per_fold[r["fold"]] = grouped.mean(axis=0)
    M = pd.DataFrame(per_fold).fillna(0.0)
    ranks = M.rank(ascending=False, axis=0, method="min")
    out = pd.DataFrame(
        {
            "feature": M.index,
            "mean_abs_shap": M.mean(axis=1).values,
            "sd_abs_shap": M.std(axis=1).values,
            "mean_rank": ranks.mean(axis=1).values,
            "rank_sd": ranks.std(axis=1).values,
            "n_folds_top5": (ranks <= 5).sum(axis=1).values,
            "n_folds": M.shape[1],
        }
    )
    out["source_feature"] = out["feature"]
    out = out.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out.insert(1, "shap_rank", np.arange(1, len(out) + 1))
    out["level"] = "source"
    return out


# ------------------------------------------------------------------ stability

def shap_stability(real: pd.DataFrame, aug: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compare two source-level SHAP importance tables."""
    a = real.set_index("feature")
    b = aug.set_index("feature")
    feats = sorted(set(a.index) | set(b.index))

    ra = a["shap_rank"].reindex(feats)
    rb = b["shap_rank"].reindex(feats)
    fill = len(feats) + 1
    ra_f, rb_f = ra.fillna(fill), rb.fillna(fill)

    sa = a["mean_abs_shap"].reindex(feats).fillna(0.0)
    sb = b["mean_abs_shap"].reindex(feats).fillna(0.0)
    na = sa / sa.sum() if sa.sum() else sa
    nb = sb / sb.sum() if sb.sum() else sb

    tab = pd.DataFrame(
        {
            "feature": feats,
            "real_rank": ra.values,
            "augmented_rank": rb.values,
            "rank_change": (rb_f - ra_f).values,
            "abs_rank_change": (rb_f - ra_f).abs().values,
            "real_mean_abs_shap": sa.values,
            "augmented_mean_abs_shap": sb.values,
            "real_norm_shap": na.values,
            "augmented_norm_shap": nb.values,
            "norm_shap_change": (nb - na).values,
        }
    ).sort_values("real_rank").reset_index(drop=True)

    rho, p = spearmanr(ra_f.values, rb_f.values)
    top_a3 = set(a.sort_values("shap_rank").index[:3])
    top_b3 = set(b.sort_values("shap_rank").index[:3])
    top_a5 = set(a.sort_values("shap_rank").index[:5])
    top_b5 = set(b.sort_values("shap_rank").index[:5])

    summary = {
        "spearman_rank_correlation": float(rho),
        "spearman_p_value": float(p),
        "top3_overlap": len(top_a3 & top_b3),
        "top3_overlap_fraction": len(top_a3 & top_b3) / max(1, len(top_a3)),
        "top5_overlap": len(top_a5 & top_b5),
        "top5_overlap_fraction": len(top_a5 & top_b5) / max(1, len(top_a5)),
        "mean_abs_rank_change": float(tab["abs_rank_change"].mean()),
        "max_abs_rank_change": float(tab["abs_rank_change"].max()),
        "mean_abs_norm_shap_change": float(tab["norm_shap_change"].abs().mean()),
        "n_features": len(feats),
    }
    return tab, summary

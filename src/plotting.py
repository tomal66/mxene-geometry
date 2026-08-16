"""Matplotlib figures (no interactive backend, no seaborn dependency)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C_REAL = "#1f77b4"
C_SYN = "#d62728"


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def predicted_vs_actual(pred: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for fold, g in pred.groupby("fold"):
        ax.scatter(g["y_true"], g["y_pred"], s=26, alpha=0.8, label=f"fold {fold}")
    lo = min(pred["y_true"].min(), pred["y_pred"].min())
    hi = max(pred["y_true"].max(), pred["y_pred"].max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="ideal")
    ax.set_xlabel("Measured capacitance (F/g)")
    ax.set_ylabel("Predicted capacitance (F/g)")
    ax.set_title(title)
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


def residual_plot(pred: pd.DataFrame, path: Path, title: str) -> None:
    res = pred["y_pred"] - pred["y_true"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    axes[0].scatter(pred["y_true"], res, s=24, alpha=0.8, color=C_REAL)
    axes[0].axhline(0, color="k", lw=1, ls="--")
    axes[0].set_xlabel("Measured capacitance (F/g)")
    axes[0].set_ylabel("Residual (pred - meas, F/g)")
    axes[0].set_title("Residuals vs measured")
    axes[1].hist(res, bins=18, color=C_REAL, alpha=0.85, edgecolor="white")
    axes[1].axvline(0, color="k", lw=1, ls="--")
    axes[1].set_xlabel("Residual (F/g)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual distribution")
    fig.suptitle(title, y=1.02)
    _save(fig, path)


def shap_bar(imp: pd.DataFrame, path: Path, title: str, top_n: int = 15) -> None:
    d = imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.4, 0.32 * len(d) + 1.3))
    err = d["sd_abs_shap"].fillna(0.0).values
    ax.barh(d["feature"], d["mean_abs_shap"], xerr=err, color="#4c72b0",
            error_kw=dict(lw=0.8, capsize=2, ecolor="#444"))
    ax.set_xlabel("mean |SHAP| on held-out papers (F/g)")
    ax.set_title(title)
    _save(fig, path)


def shap_beeswarm(shap_df: pd.DataFrame, X: pd.DataFrame, path: Path, title: str,
                  top_n: int = 12) -> None:
    order = shap_df.abs().mean(axis=0).sort_values(ascending=False).head(top_n).index[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(order) + 1.4))
    rng = np.random.default_rng(42)
    for i, feat in enumerate(order):
        sv = shap_df[feat].values
        xv = pd.to_numeric(X[feat], errors="coerce").values.astype(float)
        finite = np.isfinite(xv)
        if finite.sum() > 1 and np.nanmax(xv[finite]) > np.nanmin(xv[finite]):
            lo, hi = np.nanpercentile(xv[finite], [5, 95])
            norm = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
        else:
            norm = np.full_like(sv, 0.5, dtype=float)
        colors = np.where(np.isfinite(norm), norm, np.nan)
        y = i + rng.uniform(-0.16, 0.16, size=len(sv))
        sc = ax.scatter(sv, y, c=colors, cmap="coolwarm", s=13, alpha=0.85,
                        edgecolors="none", vmin=0, vmax=1)
        bad = ~np.isfinite(colors)
        if bad.any():
            ax.scatter(sv[bad], y[bad], c="#999999", s=13, alpha=0.7, edgecolors="none")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlabel("SHAP value (F/g)")
    ax.set_title(title)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.03)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["low", "high"])
    cb.set_label("feature value")
    _save(fig, path)


def shap_dependence(x, sv, path: Path, feature: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 3.4))
    ax.scatter(x, sv, s=26, alpha=0.85, color="#4c72b0", edgecolors="white", linewidths=0.4)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f"SHAP value for {feature} (F/g)")
    ax.set_title(f"Dependence: {feature}")
    _save(fig, path)


PRETTY_COND = {
    "A_real_only": "Real only",
    "B_real_plus_1.0x": "+1x synth",
    "C_real_plus_3.0x": "+3x synth",
    "D_real_plus_5.25x": "+5.25x synth",
}


def augmentation_bars(summary: pd.DataFrame, metric: str, path: Path,
                      ylabel: str, title: str, higher_is_better: bool,
                      fold_df: pd.DataFrame | None = None) -> None:
    """Condition means with fold-to-fold error bars and the individual folds on top."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    order = list(summary["condition"])
    x = np.arange(len(order))
    ax.bar(x, summary[f"{metric}_mean"], yerr=summary[f"{metric}_sd"].fillna(0),
           color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"][: len(order)],
           capsize=4, alpha=0.85, error_kw=dict(lw=1.0, ecolor="#333"))
    if fold_df is not None:
        rng = np.random.default_rng(42)
        for i, cond in enumerate(order):
            v = fold_df.loc[fold_df["condition"] == cond, metric].values
            ax.scatter(i + rng.uniform(-0.13, 0.13, len(v)), v, s=18, color="#222",
                       zorder=3, alpha=0.75,
                       label="individual folds" if i == 0 else None)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY_COND.get(c, c) for c in order], rotation=12, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if metric == "r2":
        ax.axhline(0, color="k", lw=0.8)
    arrow = "higher is better" if higher_is_better else "lower is better"
    ax.text(0.99, 0.97, arrow, transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="#666")
    _save(fig, path)


def fold_metric_lines(fold_df: pd.DataFrame, metric: str, path: Path, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    piv = fold_df.pivot_table(index="fold", columns="condition", values=metric)
    for c in piv.columns:
        ax.plot(piv.index, piv[c], marker="o", label=PRETTY_COND.get(c, c), lw=1.4, ms=5)
    ax.set_xlabel("outer grouped fold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(piv.index)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Per-fold {ylabel}")
    _save(fig, path)


def hist_real_vs_syn(real: pd.Series, syn: pd.Series, path: Path, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2),
                             gridspec_kw={"width_ratios": [2, 1]})
    r = pd.to_numeric(real, errors="coerce").dropna()
    s = pd.to_numeric(syn, errors="coerce").dropna()
    if len(r) and len(s):
        bins = np.histogram_bin_edges(np.concatenate([r, s]), bins=14)
        axes[0].hist(r, bins=bins, alpha=0.6, label="real (train)", color=C_REAL, density=True)
        axes[0].hist(s, bins=bins, alpha=0.6, label="synthetic", color=C_SYN, density=True)
    axes[0].set_xlabel(name)
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].boxplot([r, s], tick_labels=["real", "syn"], widths=0.55)
    axes[1].set_ylabel(name)
    fig.suptitle(f"Real vs synthetic: {name}", y=1.03)
    _save(fig, path)


def bar_real_vs_syn_categorical(real: pd.Series, syn: pd.Series, path: Path, name: str) -> None:
    cats = sorted(set(real.dropna().astype(str)) | set(syn.dropna().astype(str)))
    r = real.astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
    s = syn.astype(str).value_counts(normalize=True).reindex(cats).fillna(0)
    x = np.arange(len(cats))
    fig, ax = plt.subplots(figsize=(max(4.2, 0.9 * len(cats)), 3.4))
    ax.bar(x - 0.2, r.values, 0.4, label="real (train)", color=C_REAL)
    ax.bar(x + 0.2, s.values, 0.4, label="synthetic", color=C_SYN)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("relative frequency")
    ax.set_title(f"Category frequencies: {name}")
    ax.legend(fontsize=7, frameon=False)
    _save(fig, path)


def heatmap_pair(A: pd.DataFrame, B: pd.DataFrame, path: Path, title: str,
                 labels=("real", "synthetic"), vmin=-1, vmax=1, cmap="RdBu_r") -> None:
    cols = list(A.columns)
    fig, axes = plt.subplots(1, 3, figsize=(4.0 * 3, 3.8), layout="constrained")
    D = B.reindex(index=cols, columns=cols) - A
    ims = []
    for i, (ax, M, t, vv) in enumerate(zip(
        axes, [A, B.reindex(index=cols, columns=cols), D],
        [labels[0], labels[1], f"{labels[1]} - {labels[0]}"],
        [(vmin, vmax), (vmin, vmax), (-(vmax - vmin) / 2, (vmax - vmin) / 2)],
    )):
        im = ax.imshow(M.values, cmap=cmap, vmin=vv[0], vmax=vv[1])
        ims.append(im)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(cols if i == 0 else [""] * len(cols), fontsize=6.5)
        ax.set_title(t, fontsize=9)
        ax.grid(False)
    # one colour bar for the two matrices that share a scale, one for the delta
    fig.colorbar(ims[1], ax=axes[:2], fraction=0.035, pad=0.02, location="right")
    fig.colorbar(ims[2], ax=axes[2], fraction=0.05, pad=0.04, location="right")
    fig.suptitle(title)
    _save(fig, path)


def shap_rank_comparison(tab: pd.DataFrame, path: Path, title: str) -> None:
    d = tab.dropna(subset=["real_rank"]).sort_values("real_rank")
    fig, ax = plt.subplots(figsize=(5.6, 0.42 * len(d) + 1.5))
    for _, r in d.iterrows():
        if pd.isna(r["augmented_rank"]):
            continue
        ax.plot([0, 1], [r["real_rank"], r["augmented_rank"]], "-o", ms=5, lw=1.3,
                color="#4c72b0" if r["rank_change"] == 0 else "#c44e52")
        ax.text(-0.04, r["real_rank"], r["feature"], ha="right", va="center", fontsize=7)
        ax.text(1.04, r["augmented_rank"], r["feature"], ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.75, 1.75)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["real-only SHAP rank", "augmented SHAP rank"])
    ax.set_ylabel("rank (1 = most important)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    _save(fig, path)

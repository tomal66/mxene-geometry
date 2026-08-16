"""Publication-quality figures (vector PDF + PNG) for the LaTeX report.

House style: single-column width 3.4 in, double-column 7.0 in, 8 pt base type,
no titles (captions carry the message), no rainbow colormaps, no 3-D, every
axis labelled with units.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "mathtext.default": "regular",
})

W1, W2 = 3.4, 7.0          # single / double column width in inches

# colour-blind-safe sequential-neutral palette (Okabe-Ito derived)
BLUE, ORANGE, GREEN, RED = "#0072B2", "#E69F00", "#009E73", "#D55E00"
GREY, PURPLE = "#7A7A7A", "#CC79A7"
COND_COLORS = [BLUE, ORANGE, GREEN, RED]
SEQ = "viridis"
DIV = "RdBu_r"


def save(fig, outdir: Path, stem: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(pdf)
    fig.savefig(outdir / f"{stem}.png")
    plt.close(fig)
    return pdf


# ===========================================================================
# Figure 1 - dataset structure
# ===========================================================================

def fig_dataset_structure(cap_summary: pd.DataFrame, disposition: pd.Series,
                          outdir: Path, stem: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.7),
                             gridspec_kw={"width_ratios": [1.05, 1.5, 1.0]})

    # (a) row accounting
    ax = axes[0]
    kept = int(disposition.get("kept:exact", 0) + disposition.get("kept:approximate", 0))
    no_val = int(disposition.get("dropped:no_value_reported", 0))
    non_def = int(disposition.sum() - kept - no_val)
    bars = [("Usable", kept, GREEN), ("No value\nreported", no_val, GREY),
            ("Range or\ninequality", non_def, ORANGE)]
    for i, (lab, v, c) in enumerate(bars):
        ax.bar(i, v, color=c, width=0.68)
        ax.text(i, v + 2, str(v), ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.8)
    ax.set_ylabel("Corpus rows")
    ax.set_ylim(0, max(b[1] for b in bars) * 1.22)
    ax.text(-0.30, 1.06, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (b) capacitance range per paper
    ax = axes[1]
    d = cap_summary.sort_values("cap_median").reset_index(drop=True)
    yy = np.arange(len(d))
    ax.hlines(yy, d["cap_min"], d["cap_max"], color=GREY, lw=1.0, alpha=0.85)
    ax.scatter(d["cap_median"], yy, s=13, color=BLUE, zorder=3)
    ax.set_yticks([])
    ax.set_ylabel(f"Source paper (n = {len(d)})")
    ax.set_xlabel("Gravimetric capacitance (F g$^{-1}$)")
    ax.set_xscale("log")
    ax.text(-0.10, 1.06, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (c) observations per paper
    ax = axes[2]
    counts = cap_summary["n_rows"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=BLUE, width=0.7)
    ax.set_xlabel("Observations per paper")
    ax.set_ylabel("Number of papers")
    ax.set_xticks(sorted(counts.index))
    ax.tick_params(axis="x", labelsize=6.5)
    ax.text(-0.28, 1.06, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    fig.tight_layout(w_pad=1.6)
    return save(fig, outdir, stem)


# ===========================================================================
# Figure 2 - grouped cross-paper prediction
# ===========================================================================

def fig_grouped_prediction(pred: pd.DataFrame, fold_metrics: pd.DataFrame,
                           outdir: Path, stem: str) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.65),
                             gridspec_kw={"width_ratios": [1.15, 1.0, 1.0]})

    # (a) observed vs predicted
    ax = axes[0]
    lo = min(pred["y_true"].min(), pred["y_pred"].min()) * 0.7
    hi = max(pred["y_true"].max(), pred["y_pred"].max()) * 1.4
    ax.plot([lo, hi], [lo, hi], ls="--", color="k", lw=0.8, zorder=1)
    for f, g in pred.groupby("fold"):
        ax.scatter(g["y_true"], g["y_pred"], s=16, alpha=0.85, lw=0.3,
                   edgecolor="white", label=f"Fold {int(f)}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Measured $C_g$ (F g$^{-1}$)")
    ax.set_ylabel("Predicted $C_g$ (F g$^{-1}$)")
    ax.legend(frameon=False, fontsize=6, loc="upper left", handletextpad=0.2,
              borderpad=0.1, labelspacing=0.22)
    ax.set_aspect("equal")
    ax.text(-0.30, 1.06, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (b) residuals vs measured
    ax = axes[1]
    res = pred["y_pred"] - pred["y_true"]
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.scatter(pred["y_true"], res, s=15, color=BLUE, alpha=0.8, lw=0.3,
               edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("Measured $C_g$ (F g$^{-1}$)")
    ax.set_ylabel("Residual (F g$^{-1}$)")
    ax.text(-0.34, 1.06, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (c) per-fold R2
    ax = axes[2]
    fm = fold_metrics.sort_values("fold")
    ax.bar(fm["fold"], fm["r2"], color=[GREEN if v > 0 else RED for v in fm["r2"]],
           width=0.62)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(fm["r2"].mean(), color=GREY, ls=":", lw=1.0)
    ax.text(0.97, 0.94, f"mean $R^2$ = {fm['r2'].mean():.3f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.8, color="#444")
    ax.set_xticks(fm["fold"])
    ax.set_xlabel("Grouped fold (held-out papers)")
    ax.set_ylabel("$R^2$")
    ax.text(-0.34, 1.06, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    fig.tight_layout(w_pad=1.8)
    return save(fig, outdir, stem)


# ===========================================================================
# Figure 3 - augmentation comparison
# ===========================================================================

PRETTY = {"A_real_only": "Real\nonly", "B_real_plus_1.0x": "+1$\\times$",
          "C_real_plus_3.0x": "+3$\\times$", "D_real_plus_5.25x": "+5.25$\\times$"}


def fig_augmentation(summary: pd.DataFrame, fold_df: pd.DataFrame,
                     outdir: Path, stem: str) -> Path:
    s = summary.sort_values("ratio").reset_index(drop=True)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.6))
    rng = np.random.default_rng(0)
    for ax, metric, ylab in zip(
            axes, ["mae", "rmse", "r2"],
            ["MAE (F g$^{-1}$)", "RMSE (F g$^{-1}$)", "$R^2$"]):
        x = np.arange(len(s))
        ax.bar(x, s[f"{metric}_mean"], yerr=s[f"{metric}_sd"].fillna(0),
               color=COND_COLORS[: len(s)], width=0.64, alpha=0.9,
               error_kw=dict(lw=0.9, capsize=2.5, ecolor="#333"))
        for i, cond in enumerate(s["condition"]):
            v = fold_df.loc[fold_df["condition"] == cond, metric].values
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=9, color="#222",
                       alpha=0.8, zorder=4, lw=0)
        ax.set_xticks(x)
        ax.set_xticklabels([PRETTY.get(c, c) for c in s["condition"]], fontsize=6.6)
        ax.set_ylabel(ylab)
        if metric == "r2":
            ax.axhline(0, color="k", lw=0.8)
    for ax, tag in zip(axes, "abc"):
        ax.text(-0.32, 1.06, f"({tag})", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    axes[0].scatter([], [], s=9, color="#222", label="individual folds")
    axes[0].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(w_pad=1.7)
    return save(fig, outdir, stem)


# ===========================================================================
# Figure 4 - synthetic fidelity
# ===========================================================================

def fig_synthetic_fidelity(real: pd.DataFrame, syn: pd.DataFrame,
                           dist_stats: pd.DataFrame, mi_real: pd.DataFrame,
                           mi_syn: pd.DataFrame, target: str,
                           outdir: Path, stem: str) -> Path:
    fig = plt.figure(figsize=(W2, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=0.55, wspace=0.42)

    # (a) target marginal
    ax = fig.add_subplot(gs[0, 0])
    r = pd.to_numeric(real[target], errors="coerce").dropna()
    s = pd.to_numeric(syn[target], errors="coerce").dropna()
    bins = np.histogram_bin_edges(np.concatenate([r, s]), bins=16)
    ax.hist(r, bins=bins, density=True, color=BLUE, alpha=0.55, label="Real (train)")
    ax.hist(s, bins=bins, density=True, color=RED, alpha=0.55, label="Synthetic")
    ax.set_xlabel("$C_g$ (F g$^{-1}$)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=6)
    ax.text(-0.32, 1.08, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (b) interlayer marginal
    ax = fig.add_subplot(gs[0, 1])
    r = pd.to_numeric(real["interlayer_A"], errors="coerce").dropna()
    s = pd.to_numeric(syn["interlayer_A"], errors="coerce").dropna()
    bins = np.histogram_bin_edges(np.concatenate([r, s]), bins=16)
    ax.hist(r, bins=bins, density=True, color=BLUE, alpha=0.55)
    ax.hist(s, bins=bins, density=True, color=RED, alpha=0.55)
    ax.set_xlabel(r"Interlayer spacing ($\AA$)")
    ax.set_ylabel("Density")
    ax.text(-0.32, 1.08, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (c) KS / JS per variable
    ax = fig.add_subplot(gs[0, 2])
    ks = dist_stats.dropna(subset=["ks_statistic"]).groupby("variable")["ks_statistic"].mean()
    js = dist_stats.dropna(subset=["jensen_shannon_distance"]).groupby(
        "variable")["jensen_shannon_distance"].mean()
    allv = pd.concat([ks.rename("v"), js.rename("v")]).sort_values()
    kind = ["KS" if v in ks.index else "JS" for v in allv.index]
    ax.barh(np.arange(len(allv)), allv.values,
            color=[BLUE if k == "KS" else PURPLE for k in kind], height=0.62)
    ax.set_yticks(np.arange(len(allv)))
    ax.set_yticklabels([v.replace("_", " ") for v in allv.index], fontsize=6)
    ax.set_xlabel("Distance (KS or JS)")
    ax.set_xlim(0, max(0.3, allv.max() * 1.25))
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=4, label="KS (continuous)"),
                       Line2D([], [], color=PURPLE, lw=4, label="JS (categorical)")],
              frameon=False, fontsize=5.6, loc="lower right")
    ax.text(-0.52, 1.08, "(c)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (d-f) mutual information matrices
    cols = list(mi_real.columns)
    labs = [c.replace("_", " ") for c in cols]
    mats = [mi_real, mi_syn.reindex(index=cols, columns=cols),
            mi_syn.reindex(index=cols, columns=cols) - mi_real]
    titles = ["Real (train)", "Synthetic", "Difference"]
    for j, (M, t) in enumerate(zip(mats, titles)):
        ax = fig.add_subplot(gs[1, j])
        vlim = (0, 1) if j < 2 else (-0.5, 0.5)
        im = ax.imshow(M.values, cmap=SEQ if j < 2 else DIV,
                       vmin=vlim[0], vmax=vlim[1])
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=5.4)
        ax.set_yticks(range(len(cols)))
        ax.set_yticklabels(labs if j == 0 else [""] * len(cols), fontsize=5.4)
        ax.set_title(t, fontsize=7.5, pad=3)
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5.5)
        if j == 0:
            ax.text(-0.72, 1.14, "(d)", transform=ax.transAxes,
                    fontweight="bold", fontsize=9)
    return save(fig, outdir, stem)


# ===========================================================================
# Figure 5 - effect size vs configuration noise
# ===========================================================================

def fig_generator_uncertainty(summary: pd.DataFrame, ksens: pd.DataFrame,
                              per_seed: pd.DataFrame, outdir: Path, stem: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 1.0]})

    # (a) per-fold MAE across seeds, by condition
    ax = axes[0]
    conds = list(summary.sort_values("ratio")["condition"])
    base = summary.set_index("condition").loc["A_real_only", "mae_mean"]
    for i, cond in enumerate(conds):
        sub = per_seed[per_seed["condition"] == cond]
        if sub.empty:
            continue
        fmeans = sub.groupby("fold")["mae"].mean()
        ax.scatter(np.full(len(sub), i) + np.linspace(-0.16, 0.16, len(sub)),
                   sub["mae"], s=9, color=GREY, alpha=0.65, lw=0,
                   label="single generator seed" if i == 1 else None)
        ax.scatter(np.full(len(fmeans), i), fmeans, s=26, color=COND_COLORS[i],
                   zorder=4, lw=0.4, edgecolor="white",
                   label="fold mean over seeds" if i == 0 else None)
    ax.axhline(base, color="k", ls="--", lw=0.8)
    ax.text(0.02, base, " real-only mean", va="bottom", fontsize=6, color="#333",
            transform=ax.get_yaxis_transform())
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([PRETTY.get(c, c) for c in conds], fontsize=6.6)
    ax.set_ylabel("Fold MAE (F g$^{-1}$)")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.22)
    ax.legend(frameon=False, fontsize=6, loc="upper center", ncol=2,
              bbox_to_anchor=(0.5, 1.14), handletextpad=0.2, columnspacing=1.0)
    ax.text(-0.24, 1.12, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (b) effect size vs the two noise sources
    ax = axes[1]
    aug = summary[summary["ratio"] > 0]
    effect = abs(summary["mae_mean"].max() - summary["mae_mean"].min())
    seed_noise = aug["mae_sd_across_gen_seeds"].mean()
    k_noise = ksens[ksens["condition"] != "A_real_only"].groupby(
        "condition")["mae_mean"].agg(lambda x: x.max() - x.min()).mean()
    fold_noise = summary["mae_sd"].mean()
    items = [("Ratio effect\n(best vs worst)", effect, BLUE),
             ("Generator seed", seed_noise, ORANGE),
             ("BN degree $k$", k_noise, GREEN),
             ("Fold-to-fold SD", fold_noise, GREY)]
    for i, (lab, v, c) in enumerate(items):
        ax.barh(i, v, color=c, height=0.62)
        ax.text(v + 1.0, i, f"{v:.1f}", va="center", fontsize=6.8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([it[0] for it in items], fontsize=6.4)
    ax.invert_yaxis()
    ax.set_xlabel("MAE spread (F g$^{-1}$)")
    ax.set_xlim(0, max(i[1] for i in items) * 1.22)
    ax.text(-0.52, 1.06, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    fig.tight_layout(w_pad=1.9)
    return save(fig, outdir, stem)


# ===========================================================================
# Figures 6-8 - SHAP
# ===========================================================================

def fig_shap_importance(grouped: pd.DataFrame, label_fn, outdir: Path,
                        stem: str, top_n: int = 12) -> Path:
    d = grouped.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(W1, 0.235 * len(d) + 0.85))
    colors = [BLUE if t == "numeric" else PURPLE
              for t in d.get("feature_type", ["numeric"] * len(d))]
    ax.barh(np.arange(len(d)), d["mean_abs_shap"], color=colors, height=0.68)
    for i, (v, m) in enumerate(zip(d["mean_abs_shap"], d.get("missingness", [np.nan] * len(d)))):
        if not pd.isna(m):
            ax.text(v + 0.4, i, f"{m:.0%} miss.", va="center", fontsize=5.6, color="#666")
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels([label_fn(f) for f in d["descriptor"]], fontsize=6.6)
    ax.set_xlabel("mean $|$SHAP$|$ (F g$^{-1}$)")
    ax.set_xlim(0, d["mean_abs_shap"].max() * 1.32)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=4, label="continuous"),
                       Line2D([], [], color=PURPLE, lw=4, label="categorical")],
              frameon=False, fontsize=6, loc="lower right")
    return save(fig, outdir, stem)


def fig_shap_beeswarm(shap_df: pd.DataFrame, X: pd.DataFrame, ohe_map: dict,
                      label_fn, outdir: Path, stem: str, top_n: int = 12) -> Path:
    order = shap_df.abs().mean(axis=0).sort_values(ascending=False).head(top_n).index[::-1]
    fig, ax = plt.subplots(figsize=(W2 * 0.62, 0.27 * len(order) + 1.0))
    rng = np.random.default_rng(0)
    sc = None
    for i, feat in enumerate(order):
        sv = shap_df[feat].values
        xv = pd.to_numeric(X[feat], errors="coerce").values.astype(float)
        fin = np.isfinite(xv)
        if fin.sum() > 1 and np.nanmax(xv[fin]) > np.nanmin(xv[fin]):
            lo, hi = np.nanpercentile(xv[fin], [5, 95])
            norm = np.clip((xv - lo) / (hi - lo + 1e-12), 0, 1)
        else:
            norm = np.full_like(sv, 0.5, dtype=float)
        y = i + rng.uniform(-0.17, 0.17, size=len(sv))
        bad = ~np.isfinite(norm)
        sc = ax.scatter(sv[~bad], y[~bad], c=norm[~bad], cmap="coolwarm", s=7,
                        alpha=0.85, lw=0, vmin=0, vmax=1)
        if bad.any():
            ax.scatter(sv[bad], y[bad], c="#B0B0B0", s=7, alpha=0.7, lw=0)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(range(len(order)))
    names = []
    for f in order:
        src = ohe_map.get(f, f)
        names.append(label_fn(f) if src == f else
                     f"{label_fn(src)}: {f[len(src) + 1:].replace('_', ' ')}")
    ax.set_yticklabels(names, fontsize=6.2)
    ax.set_xlabel("SHAP value (F g$^{-1}$)")
    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.028)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.set_label("Descriptor value", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    ax.scatter([], [], c="#B0B0B0", s=7, label="not reported")
    ax.legend(frameon=False, fontsize=5.8, loc="lower right")
    return save(fig, outdir, stem)


def fig_shap_dependence(shap_df: pd.DataFrame, X_raw: pd.DataFrame,
                        feats: list[str], label_fn, directions: pd.DataFrame,
                        outdir: Path, stem: str) -> Path:
    feats = [f for f in feats if f in shap_df.columns][:4]
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(W2, 1.95))
    axes = np.atleast_1d(axes)
    dd = directions.set_index("descriptor")
    for ax, f, tag in zip(axes, feats, "abcd"):
        x = pd.to_numeric(X_raw[f], errors="coerce")
        ok = x.notna().values
        ax.axhline(0, color="k", lw=0.7, ls="--")
        ax.scatter(x[ok], shap_df.loc[ok, f], s=13, color=BLUE, alpha=0.8,
                   lw=0.3, edgecolor="white")
        if f in dd.index:
            ax.text(0.96, 0.94, r"$\rho$ = " + f"{dd.loc[f, 'spearman_value_vs_shap']:+.2f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                    color="#444")
        if f in ("scan_rate_mV_s", "current_density_A_g") and (x[ok] > 0).all():
            ax.set_xscale("log")
        ax.set_xlabel(label_fn(f), fontsize=6.8)
        ax.set_ylabel("SHAP (F g$^{-1}$)" if tag == "a" else "")
        ax.text(-0.22, 1.08, f"({tag})", transform=ax.transAxes,
                fontweight="bold", fontsize=9)
    fig.tight_layout(w_pad=1.5)
    return save(fig, outdir, stem)


def fig_shap_stability(stab: pd.DataFrame, summary: dict, label_fn,
                       outdir: Path, stem: str) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.8),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})

    # (a) rank slope chart
    ax = axes[0]
    d = stab.dropna(subset=["real_rank", "augmented_rank"]).sort_values("real_rank")
    for _, r in d.iterrows():
        col = GREY if r["rank_change"] == 0 else ORANGE
        ax.plot([0, 1], [r["real_rank"], r["augmented_rank"]], "-o", ms=4, lw=1.1,
                color=col, zorder=2)
        ax.text(-0.06, r["real_rank"], label_fn(r["feature"]), ha="right",
                va="center", fontsize=6.2)
        ax.text(1.06, r["augmented_rank"], f"{int(r['augmented_rank'])}", ha="left",
                va="center", fontsize=6.2)
    ax.set_xlim(-1.15, 1.35)
    ax.set_ylim(len(d) + 0.5, 0.5)
    ax.set_yticks(range(1, len(d) + 1))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Real only", "Augmented"], fontsize=7)
    ax.set_ylabel("SHAP rank (1 = most important)")
    ax.grid(axis="y", alpha=0.18)
    ax.text(-0.42, 1.06, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    # (b) normalised attribution share
    ax = axes[1]
    d2 = stab.sort_values("real_rank")
    x = np.arange(len(d2))
    ax.bar(x - 0.2, d2["real_norm_shap"], 0.4, color=BLUE, label="Real only")
    ax.bar(x + 0.2, d2["augmented_norm_shap"], 0.4, color=RED, label="Augmented")
    ax.set_xticks(x)
    ax.set_xticklabels([label_fn(f).split(" (")[0] for f in d2["feature"]],
                       rotation=28, ha="right", fontsize=6)
    ax.set_ylabel("Share of total mean $|$SHAP$|$")
    ax.legend(frameon=False, fontsize=6.2)
    txt = (r"Spearman $\rho$ = " + f"{summary['spearman_rank_correlation']:.2f}\n"
           f"top-3 overlap {int(summary['top3_overlap'])}/3\n"
           f"top-5 overlap {int(summary['top5_overlap'])}/5")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=6.0, color="#333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCC", lw=0.5))
    ax.text(-0.28, 1.06, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=9)

    fig.tight_layout(w_pad=2.4)
    return save(fig, outdir, stem)


# ===========================================================================
# Figure 9 - proposed physics-constrained model schematic
# ===========================================================================

def _box(ax, x, y, w, h, text, fc, ec, fontsize=6.6, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=0.9, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, zorder=3, fontweight=weight, linespacing=1.35)


def _arrow(ax, p1, p2, color="#555", style="-|>", lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=8,
                                 lw=lw, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1, zorder=1))


def fig_model_schematic(inputs: list[str], label_fn, outdir: Path, stem: str) -> Path:
    fig, ax = plt.subplots(figsize=(W2, 4.15))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

    # four columns: inputs | preprocessing | model | losses
    x0, w0 = 0.010, 0.215        # inputs
    x1, w1 = 0.258, 0.135        # preprocessing
    x2, w2 = 0.428, 0.185        # model
    x3, w3 = 0.645, 0.295        # losses

    inp = "\n".join("$\\bullet$ " + label_fn(f) for f in inputs)
    _box(ax, x0, 0.545, w0, 0.300,
         "MEASURED DESCRIPTORS\n(Category A)\n" + inp, "#EAF2FA", BLUE, fontsize=6.2)
    _box(ax, x0, 0.330, w0, 0.170,
         "CONTEXT COVARIATES\n(Category B)\n$\\bullet$ Composition family\n"
         "$\\bullet$ Layer morphology", "#F5EEF6", PURPLE, fontsize=6.2)
    _box(ax, x1, 0.475, w1, 0.225,
         "PREPROCESSING\n\nfold-local one-hot\n+ scaling\n\nmissingness\nindicators",
         "#F4F4F4", GREY, fontsize=5.9)
    _box(ax, x2, 0.590, w2, 0.185,
         "NEURAL PREDICTOR\n$f_\\theta(\\mathbf{x})$\nsoftplus output\n"
         "$\\Rightarrow\\ \\hat{C}_g>0$", "#E8F5EE", GREEN, fontsize=6.4)
    _box(ax, x2, 0.375, w2, 0.165,
         "AUXILIARY HEADS\n$\\hat{C}_A,\\ \\hat{C}_V$\nmeasured outputs,\nnever inputs",
         "#FDF1E5", ORANGE, fontsize=6.4)

    _box(ax, x3, 0.680, w3, 0.165,
         "DATA LOSS\n$\\mathcal{L}_{\\mathrm{data}}=\\mathrm{Huber}(\\hat{C}_g,C_g)$\n"
         "$+$ masked supervision on\nreported $C_A,\\ C_V$", "#EAF2FA", BLUE, fontsize=6.0)
    _box(ax, x3, 0.445, w3, 0.195,
         "PHYSICS-CLOSURE LOSS\n"
         "$\\mathcal{L}_{\\mathrm{phys}}=\\|\\hat{C}_A-\\hat{C}_g m_A\\|^2$\n"
         "$\\qquad+\\ \\|\\hat{C}_V-\\hat{C}_g m_A/t\\|^2$\n"
         "verified: median error $<1\\%$", "#E8F5EE", GREEN, fontsize=6.0)
    _box(ax, x3, 0.245, w3, 0.160,
         "BOUND / DOMAIN LOSS\n$\\hat{C}_g>0$; penalty outside\n"
         "the training descriptor hull", "#FDF1E5", ORANGE, fontsize=6.0)

    _box(ax, 0.238, 0.055, 0.560, 0.105,
         "TOTAL OBJECTIVE  $\\;\\mathcal{L}=\\lambda_{d}\\mathcal{L}_{\\mathrm{data}}"
         "+\\lambda_{p}\\mathcal{L}_{\\mathrm{phys}}"
         "+\\lambda_{b}\\mathcal{L}_{\\mathrm{bound}}"
         "+\\lambda_{r}\\|\\theta\\|^{2}$", "#FFFFFF", "#333", fontsize=7.4)

    cx1, cx2, cx3 = x1 + w1, x2 + w2, x3 + w3
    _arrow(ax, (x0 + w0, 0.695), (x1, 0.625))
    _arrow(ax, (x0 + w0, 0.415), (x1, 0.545))
    _arrow(ax, (cx1, 0.588), (x2, 0.682))
    _arrow(ax, (x2 + w2 / 2, 0.590), (x2 + w2 / 2, 0.540))
    _arrow(ax, (cx2, 0.700), (x3, 0.762))
    _arrow(ax, (cx2, 0.640), (x3, 0.575), color="#8A8A8A", ls=(0, (3, 2)))
    _arrow(ax, (cx2, 0.455), (x3, 0.525))
    _arrow(ax, (cx2, 0.620), (x3, 0.330), color="#8A8A8A", ls=(0, (3, 2)))

    # collector bus: the three loss terms sum into the total objective
    bus = 0.968
    ax.plot([bus, bus], [0.170, 0.762], color="#777", lw=0.9, zorder=0)
    for y in (0.762, 0.542, 0.325):
        ax.plot([cx3, bus], [y, y], color="#777", lw=0.9, zorder=0)
        ax.plot(bus, y, marker="o", ms=2.2, color="#777", zorder=1)
    _arrow(ax, (bus, 0.170), (0.802, 0.108), color="#777")

    ax.annotate("", xy=(x2 + w2 / 2, 0.375), xytext=(0.238 + 0.28, 0.160),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0, ls="--",
                                connectionstyle="arc3,rad=0.28"))
    ax.text(0.437, 0.245, "backpropagation", fontsize=6, color="#333",
            ha="center", style="italic")

    ax.text(0.5, 0.955, "Every split grouped by source paper; "
                        "no synthetic rows in any evaluation set",
            ha="center", fontsize=6.6, color="#666", style="italic")
    return save(fig, outdir, stem)

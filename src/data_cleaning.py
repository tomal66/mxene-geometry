"""Stage 1-3: dataset audit, target-leakage audit and target cleaning."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

from .config import RESULTS_DIR, TARGET, TARGET_RAW
from .parsing import norm_unit, parse_value

# ---------------------------------------------------------------------------
# 1. AUDIT
# ---------------------------------------------------------------------------

_TEXTY = ("reasoning", "authors", "title", "source_path", "notes")


def _is_categorical_like(s: pd.Series, max_unique: int = 30) -> bool:
    return s.dtype == object and s.nunique(dropna=True) <= max_unique


def audit_dataset(df: pd.DataFrame, dataset_path) -> dict:
    n_rows, n_cols = df.shape

    col_rows = []
    for c in df.columns:
        s = df[c]
        nn = int(s.notna().sum())
        col_rows.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "n_nonnull": nn,
                "n_missing": int(n_rows - nn),
                "missing_pct": round(100 * (n_rows - nn) / n_rows, 2),
                "n_unique": int(s.nunique(dropna=True)),
                "example": "" if nn == 0 else str(s.dropna().iloc[0])[:80],
            }
        )
    col_summary = pd.DataFrame(col_rows)
    missingness = col_summary[["column", "n_missing", "missing_pct"]].sort_values(
        "missing_pct", ascending=False
    )

    paper_id = build_paper_id(df)
    rows_per_paper = paper_id.value_counts()

    tgt = df[TARGET_RAW]
    kinds = Counter(parse_value(v).kind for v in tgt)

    # unit inconsistencies: how many distinct normalised units per measurement
    unit_cols = [c for c in df.columns if c.endswith("__unit")]
    unit_report = {}
    for uc in unit_cols:
        vals = [norm_unit(v) for v in df[uc].dropna()]
        vals = [v for v in vals if v]
        if vals:
            unit_report[uc] = Counter(vals)

    cat_cols = {
        c: df[c].value_counts(dropna=True).head(40).to_dict()
        for c in df.columns
        if _is_categorical_like(df[c]) and not any(t in c for t in _TEXTY)
    }

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "dataset_path": str(dataset_path),
        "col_summary": col_summary,
        "missingness": missingness,
        "n_papers": int(paper_id.nunique()),
        "rows_per_paper": rows_per_paper,
        "n_dup_full": int(df.duplicated().sum()),
        "n_dup_excl_meta": int(
            df.drop(columns=[c for c in df.columns if c.endswith(("__conf", "__src"))]).duplicated().sum()
        ),
        "target_kinds": kinds,
        "n_target_usable": sum(v for k, v in kinds.items() if k in {"exact", "approximate", "mean_pm_sd"}),
        "unit_report": unit_report,
        "categorical_values": cat_cols,
    }


def write_audit(audit: dict, df: pd.DataFrame) -> None:
    audit["col_summary"].to_csv(RESULTS_DIR / "00_column_summary.csv", index=False)
    audit["missingness"].to_csv(RESULTS_DIR / "00_missingness.csv", index=False)

    L = []
    A = L.append
    A("# 00 - Data audit\n")
    A(f"- Source file: `{audit['dataset_path']}`")
    A(f"- Rows: **{audit['n_rows']}**")
    A(f"- Columns: **{audit['n_cols']}**")
    A(f"- Unique source papers (`paper_id`): **{audit['n_papers']}**")
    A(f"- Fully duplicated rows: **{audit['n_dup_full']}**")
    A(f"- Duplicated rows ignoring `__conf`/`__src` metadata: **{audit['n_dup_excl_meta']}**")
    A("")
    rpp = audit["rows_per_paper"]
    A("## Rows per paper\n")
    A(f"- min {rpp.min()}, median {rpp.median():.0f}, mean {rpp.mean():.2f}, max {rpp.max()}")
    A("")
    A("| paper_id | n_rows |")
    A("| --- | ---: |")
    for k, v in rpp.items():
        A(f"| {k} | {v} |")
    A("")

    A("## Target availability\n")
    A(f"Raw `{TARGET_RAW}` non-null: {int(df[TARGET_RAW].notna().sum())} / {audit['n_rows']}")
    A("")
    A("| literal kind | n |")
    A("| --- | ---: |")
    for k, v in sorted(audit["target_kinds"].items(), key=lambda x: -x[1]):
        A(f"| {k} | {v} |")
    A("")
    A(f"**Usable single-valued capacitance rows: {audit['n_target_usable']}**")
    A("")

    A("## Column summary\n")
    A("| column | dtype | non-null | missing % | unique | example |")
    A("| --- | --- | ---: | ---: | ---: | --- |")
    for _, r in audit["col_summary"].iterrows():
        ex = str(r["example"]).replace("|", "\\|")
        A(f"| {r['column']} | {r['dtype']} | {r['n_nonnull']} | {r['missing_pct']} | {r['n_unique']} | {ex} |")
    A("")

    A("## Unit inconsistencies (normalised unit -> count)\n")
    A("Different spellings of the same unit (`F g-1`, `F/g`) are collapsed; what")
    A("remains are *genuinely different physical bases* that require care.\n")
    for uc, cnt in audit["unit_report"].items():
        flag = " **<- MIXED BASES**" if len(cnt) > 1 else ""
        A(f"- `{uc}`: {dict(cnt)}{flag}")
    A("")

    A("## Categorical / low-cardinality columns\n")
    for c, vals in audit["categorical_values"].items():
        A(f"### `{c}` ({len(vals)} shown)\n")
        for k, v in vals.items():
            A(f"- `{k}`: {v}")
        A("")

    (RESULTS_DIR / "00_data_audit.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# 2. LEAKAGE AUDIT
# ---------------------------------------------------------------------------

LEAKAGE_RULES: list[tuple[str, bool, str]] = [
    # (column, use_as_predictor, reason)
    ("source_file", False, "Identifier / grouping key only - not a physical descriptor."),
    ("source_path", False, "Filesystem identifier."),
    ("doi", False, "Publication identifier; used to build paper_id, never as a predictor."),
    ("year", False, "Publication metadata; a proxy for which paper a row came from (paper-level leakage under grouped CV)."),
    ("authors", False, "Publication metadata / identifier."),
    ("electrode_label", False, "Free-text within-paper label; acts as a row identifier."),
    ("role", False, "Annotation of the electrode's role in its own paper (primary/control/variant); assigned partly BECAUSE of the measured performance - post-hoc, not knowable before the experiment."),
    ("is_primary", False, "Same as `role`: the 'primary' electrode is usually the best-performing one, i.e. selected on the outcome."),
    ("mxene_formula", True, "Materials composition known before testing. (Near-constant here - see candidate table.)"),
    ("composition", True, "Materials composition known before testing."),
    ("synthesis_method", True, "Processing route known before testing."),
    ("layer_no", True, "Structural descriptor of the electrode material."),
    ("interlayer_spacing", True, "Structural descriptor (XRD (002)) measured on the electrode, independent of capacitance."),
    ("flake_size", True, "Morphological descriptor."),
    ("porosity", True, "Structural descriptor."),
    ("pore_diameter", True, "Structural descriptor."),
    ("tortuosity", False, "Completely empty in this corpus."),
    ("electrode_thickness", True, "Electrode geometry, set during fabrication."),
    ("mass_loading", True, "Electrode geometry, set during fabrication."),
    ("electrolyte", True, "Test condition, set before measurement (source of h2so4_M)."),
    ("ssa", True, "Textural descriptor from gas sorption, independent of the capacitance test."),
    ("scan_rate", True, "CV test condition, chosen by the experimenter."),
    ("current_density", True, "GCD test condition, chosen by the experimenter."),
    ("gravimetric_capacitance", False, "THIS IS THE TARGET."),
    ("volumetric_capacitance", False, "LEAKAGE: C_vol = C_grav x electrode density - a direct algebraic transform of the target."),
    ("areal_capacitance", False, "LEAKAGE: C_areal = C_grav x mass loading - a direct algebraic transform of the target."),
    ("formula_match", False, "Extraction-QA metadata about the LLM parse, not physics."),
    ("formula_reasoning", False, "Free-text extraction rationale; may quote the capacitance value verbatim."),
    ("synthesis_match", False, "Extraction-QA metadata."),
    ("synthesis_reasoning", False, "Free-text extraction rationale; may quote the capacitance value verbatim."),
    ("electrolyte_match", False, "Extraction-QA metadata."),
    ("electrolyte_reasoning", False, "Free-text extraction rationale; may quote the capacitance value verbatim."),
    ("overall", False, "Extraction-QA verdict (constant 'PASS')."),
    ("overall_reasoning", False, "Free-text extraction rationale; may quote the capacitance value verbatim."),
]


def leakage_audit(df: pd.DataFrame) -> pd.DataFrame:
    explicit = {c: (u, r) for c, u, r in LEAKAGE_RULES}
    rows = []
    for c in df.columns:
        if c in explicit:
            use, reason = explicit[c]
        elif c.endswith("__conf"):
            use, reason = False, "Extraction-confidence metadata about the target/descriptor, not a physical property; target-confidence in particular is annotated with knowledge of the reported value."
        elif c.endswith("__src"):
            use, reason = False, "Provenance string (table/figure/text location); pure metadata that can quote the value."
        elif c.endswith("__unit"):
            use, reason = False, "Unit string; consumed during normalisation, then discarded (a raw unit column would encode which paper a row came from)."
        else:
            use, reason = False, "Unclassified column - excluded conservatively."
        # anything derived from the target name is force-excluded
        if TARGET_RAW in c:
            use = False
            if c != TARGET_RAW:
                reason = f"Directly attached to the target column (`{TARGET_RAW}`)."
        rows.append({"column": c, "use_as_predictor": use, "reason": reason})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "00_leakage_audit.csv", index=False)
    return out


# ---------------------------------------------------------------------------
# 3. TARGET CLEANING
# ---------------------------------------------------------------------------

def build_paper_id(df: pd.DataFrame) -> pd.Series:
    """DOI when reliably available, else the source file name."""
    doi = df["doi"].astype(str).str.strip().str.lower()
    doi = doi.where(~doi.isin(["", "nan", "none", "n/a"]), other=np.nan)
    return doi.fillna(df["source_file"].astype(str)).rename("paper_id")


def clean_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (frame with target columns added, per-row disposition log)."""
    parsed = [parse_value(v) for v in df[TARGET_RAW]]
    units = [norm_unit(u) for u in df[f"{TARGET_RAW}__unit"]]

    vals, approx, status = [], [], []
    for p, u in zip(parsed, units):
        if p.kind == "missing":
            vals.append(np.nan); approx.append(np.nan); status.append("dropped:no_value_reported")
        elif not p.usable:
            vals.append(np.nan); approx.append(np.nan)
            status.append(f"dropped:{p.kind}")
        elif u and u != "f/g":
            vals.append(np.nan); approx.append(np.nan)
            status.append(f"dropped:unexpected_unit:{u}")
        elif p.value is not None and p.value <= 0:
            vals.append(np.nan); approx.append(np.nan)
            status.append("dropped:non_positive")
        else:
            vals.append(p.value); approx.append(p.is_approximate)
            status.append("kept:approximate" if p.is_approximate else "kept:exact")

    out = df.copy()
    out[TARGET] = vals
    out["target_is_approximate"] = approx
    out["paper_id"] = build_paper_id(df)

    log = pd.DataFrame(
        {
            "row_index": df.index,
            "paper_id": out["paper_id"],
            "raw_value": df[TARGET_RAW],
            "raw_unit": df[f"{TARGET_RAW}__unit"],
            "parsed_kind": [p.kind for p in parsed],
            "disposition": status,
            TARGET: vals,
        }
    )
    return out, log


def target_outlier_flags(s: pd.Series) -> pd.Series:
    """Flag capacitance values outside a physically sane band for MXene in H2SO4."""
    return (s < 20) | (s > 1500)

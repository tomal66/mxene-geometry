#!/usr/bin/env python3
"""
Filter MXene CSV rows by formula (Ti₃C₂Tₓ / Ti₃C₂) and synthesis (LiF + HCl).

Matching is rule-based — deterministic, fast, no GPU required.

Usage
-----
    python scripts/filter_mxene.py --csv 5.csv

Outputs  →  data/filtered/<stem>_filtered.csv / _ambiguous.csv / _reasoning.csv
"""

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd
from tqdm import tqdm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/filtered")

# Unicode subscript → ASCII digit/letter
_SUBSCRIPT_MAP = str.maketrans("₀₁₂₃₄₅₆₇₈₉ₓ", "0123456789x")


# ── rule-based checks ─────────────────────────────────────────────────────────

def _normalise(s: str) -> str:
    """NFKC normalisation + subscript collapse, lowercased."""
    return unicodedata.normalize("NFKC", s).translate(_SUBSCRIPT_MAP).lower()


def check_formula(formula: str) -> tuple[bool | None, str]:
    """
    Returns (match, reasoning).
    True  → primary formula is Ti₃C₂Tₓ or Ti₃C₂
    False → primary formula is something else
    None  → genuinely ambiguous (e.g. composite label, no clear primary)
    """
    norm = _normalise(formula)

    # Extract the primary formula: text before the first semicolon, slash,
    # comma, or "(secondary)" / "(minor)" qualifier
    primary = re.split(r";|/|,|\(secondary\)|\(minor\)|\(second", norm)[0].strip()

    ti3c2_pat = re.compile(r"ti3c2")

    if ti3c2_pat.search(primary):
        return True, f"Primary formula '{formula.strip()}' contains Ti₃C₂ (Ti₃C₂Tₓ or Ti₃C₂)."

    # If not in primary but present elsewhere (e.g. secondary), it still fails
    if ti3c2_pat.search(norm):
        return False, (
            f"Ti₃C₂ appears in '{formula.strip()}' but is not the primary formula."
        )

    if not norm.strip():
        return None, "Formula cell is empty."

    return False, f"Primary formula '{formula.strip()}' is not Ti₃C₂Tₓ or Ti₃C₂."


def check_synthesis(synthesis: str) -> tuple[bool | None, str]:
    """
    Returns (match, reasoning).
    True  → synthesis mentions both LiF and HCl
    False → one or both are absent
    None  → synthesis cell is empty
    """
    if not synthesis.strip():
        return None, "Synthesis cell is empty."

    lif_pat = re.compile(r"\blif\b|lithium[\s\-]?fluoride", re.IGNORECASE)
    hcl_pat = re.compile(r"\bhcl\b|hydrochloric[\s\-]?acid", re.IGNORECASE)

    has_lif = bool(lif_pat.search(synthesis))
    has_hcl = bool(hcl_pat.search(synthesis))

    if has_lif and has_hcl:
        return True, "Synthesis explicitly mentions both LiF and HCl."
    if not has_lif and not has_hcl:
        return False, "Synthesis mentions neither LiF nor HCl."
    if not has_lif:
        return False, "Synthesis mentions HCl but not LiF."
    return False, "Synthesis mentions LiF but not HCl."


def evaluate_row(formula: str, synthesis: str) -> dict:
    fm, fr = check_formula(formula)
    sm, sr = check_synthesis(synthesis)

    if fm is None or sm is None:
        overall = "AMBIGUOUS"
    elif fm and sm:
        overall = "PASS"
    else:
        overall = "FAIL"

    parts = []
    if fm is True and sm is True:
        parts.append("Passes both criteria.")
    else:
        if fm is False:
            parts.append("Formula does not match.")
        if sm is False:
            parts.append("Synthesis does not match.")
        if fm is None or sm is None:
            parts.append("One or more fields are empty.")

    return {
        "formula_match": fm,
        "formula_reasoning": fr,
        "synthesis_match": sm,
        "synthesis_reasoning": sr,
        "overall": overall,
        "overall_reasoning": " ".join(parts) or overall,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter MXene rows by formula and synthesis method.")
    p.add_argument("--csv", required=True, help="CSV filename inside data/raw/ (e.g. 5.csv)")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = RAW_DIR / args.csv
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from {csv_path}")

    required = {"Mxene formula", "Synthesis method"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing columns: {missing}\nFound: {list(df.columns)}")

    passed_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    reasoning_rows: list[dict] = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Filtering"):
        formula = str(row.get("Mxene formula", "")).strip()
        synthesis = str(row.get("Synthesis method", "")).strip()

        verdict = evaluate_row(formula, synthesis)

        reasoning_rows.append({"row_index": row_index, **verdict})

        base_row = {"row_index": row_index, **row.to_dict()}
        if verdict["overall"] == "PASS":
            passed_rows.append(base_row)
        elif verdict["overall"] == "AMBIGUOUS":
            ambiguous_rows.append(base_row)

    stem = csv_path.stem
    filtered_path = OUT_DIR / f"{stem}_filtered.csv"
    ambiguous_path = OUT_DIR / f"{stem}_ambiguous.csv"
    reasoning_path = OUT_DIR / f"{stem}_reasoning.csv"

    pd.DataFrame(passed_rows).to_csv(filtered_path, index=False)
    pd.DataFrame(ambiguous_rows).to_csv(ambiguous_path, index=False)
    pd.DataFrame(reasoning_rows).to_csv(reasoning_path, index=False)

    print(f"\n{'─'*50}")
    print(f"  PASS      : {len(passed_rows):>4}  →  {filtered_path}")
    print(f"  AMBIGUOUS : {len(ambiguous_rows):>4}  →  {ambiguous_path}")
    print(f"  FAIL      : {len(df) - len(passed_rows) - len(ambiguous_rows):>4}  (reasoning.csv only)")
    print(f"  Reasoning : {len(reasoning_rows):>4}  →  {reasoning_path}")
    print(f"{'─'*50}")


if __name__ == "__main__":
    main()

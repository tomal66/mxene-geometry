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


def check_electrolyte(electrolyte: str) -> tuple[bool | None, str]:
    """
    Returns (match, reasoning).
    True  → electrolyte mentions H₂SO₄ / sulfuric acid
    False → a different electrolyte is mentioned
    None  → electrolyte cell is empty
    """
    if not electrolyte.strip():
        return None, "Electrolyte cell is empty."

    # normalise collapses Unicode sub/superscripts (H₂SO₄ → h2so4, H²SO⁴ → h2so4)
    norm = _normalise(electrolyte)
    h2so4_pat = re.compile(r"h2so4|su[lp]phuric[\s\-]?acid", re.IGNORECASE)

    if h2so4_pat.search(norm):
        return True, "Electrolyte explicitly mentions H₂SO₄ (sulfuric acid)."
    return False, f"Electrolyte '{electrolyte.strip()}' is not H₂SO₄."


def check_synthesis(synthesis: str) -> tuple[bool | None, str]:
    """
    Returns (match, reasoning).
    True  → synthesis mentions both LiF and HCl
    False → one or both are absent
    None  → synthesis cell is empty
    """
    if not synthesis.strip():
        return None, "Synthesis cell is empty."

    norm = _normalise(synthesis)
    lif_pat = re.compile(r"\blif\b|lithium[\s\-]?fluoride")
    hcl_pat = re.compile(r"\bhcl\b|hydrochloric[\s\-]?acid")

    has_lif = bool(lif_pat.search(norm))
    has_hcl = bool(hcl_pat.search(norm))

    if has_lif and has_hcl:
        return True, "Synthesis explicitly mentions both LiF and HCl."
    if not has_lif and not has_hcl:
        return False, "Synthesis mentions neither LiF nor HCl."
    if not has_lif:
        return False, "Synthesis mentions HCl but not LiF."
    return False, "Synthesis mentions LiF but not HCl."


def evaluate_row(formula: str, synthesis: str, electrolyte: str) -> dict:
    fm, fr = check_formula(formula)
    sm, sr = check_synthesis(synthesis)
    em, er = check_electrolyte(electrolyte)

    if fm is None or sm is None or em is None:
        overall = "AMBIGUOUS"
    elif fm and sm and em:
        overall = "PASS"
    else:
        overall = "FAIL"

    parts = []
    if fm is True and sm is True and em is True:
        parts.append("Passes all criteria.")
    else:
        if fm is False:
            parts.append("Formula does not match.")
        if sm is False:
            parts.append("Synthesis does not match.")
        if em is False:
            parts.append("Electrolyte does not match.")
        if fm is None or sm is None or em is None:
            parts.append("One or more fields are empty.")

    return {
        "formula_match": fm,
        "formula_reasoning": fr,
        "synthesis_match": sm,
        "synthesis_reasoning": sr,
        "electrolyte_match": em,
        "electrolyte_reasoning": er,
        "overall": overall,
        "overall_reasoning": " ".join(parts) or overall,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter MXene rows by formula and synthesis method.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", help="CSV filename inside data/raw/ (e.g. 5.csv)")
    group.add_argument("--all", action="store_true", help="Process all CSVs in data/raw/ and write combined outputs")
    return p.parse_args()


_NULL_PATTERN = re.compile(
    r"not\s+reported|not\s+available|not\s+specified|not\s+applicable"
    r"|\bn/a\b|\bna\b|\bnone\b|\bunknown\b|\bunspecified\b",
    re.IGNORECASE,
)


def process_csv(csv_path: Path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Filter a single CSV and return (passed, failed, ambiguous, reasoning) row lists."""
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df = df.apply(lambda col: col.map(lambda v: "" if isinstance(v, str) and _NULL_PATTERN.search(v) else v))
    print(f"Loaded {len(df)} rows from {csv_path}")

    required = {"Mxene formula", "Synthesis method", "Electrolyte"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {csv_path} is missing columns: {missing}\nFound: {list(df.columns)}")

    search_id = csv_path.stem
    passed_rows: list[dict] = []
    failed_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    reasoning_rows: list[dict] = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc=f"Filtering {csv_path.name}"):
        formula = str(row.get("Mxene formula", "")).strip()
        synthesis = str(row.get("Synthesis method", "")).strip()
        electrolyte = str(row.get("Electrolyte", "")).strip()

        verdict = evaluate_row(formula, synthesis, electrolyte)

        reasoning_rows.append({"search_id": search_id, "row_index": row_index, **verdict})

        base_row = {"search_id": search_id, "row_index": row_index, **row.to_dict()}
        overall = verdict["overall"]
        if overall == "PASS":
            passed_rows.append(base_row)
        elif overall == "FAIL":
            failed_rows.append(base_row)
        else:
            ambiguous_rows.append(base_row)

    return passed_rows, failed_rows, ambiguous_rows, reasoning_rows


def save_and_report(
    passed_rows: list[dict],
    failed_rows: list[dict],
    ambiguous_rows: list[dict],
    reasoning_rows: list[dict],
    out_dir: Path,
    stem: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = out_dir / f"{stem}_filtered.csv"
    failed_path = out_dir / f"{stem}_failed.csv"
    ambiguous_path = out_dir / f"{stem}_ambiguous.csv"
    reasoning_path = out_dir / f"{stem}_reasoning.csv"

    pd.DataFrame(passed_rows).to_csv(filtered_path, index=False)
    pd.DataFrame(failed_rows).to_csv(failed_path, index=False)
    pd.DataFrame(ambiguous_rows).to_csv(ambiguous_path, index=False)
    pd.DataFrame(reasoning_rows).to_csv(reasoning_path, index=False)

    print(f"\n{'─'*50}")
    print(f"  PASS      : {len(passed_rows):>4}  →  {filtered_path}")
    print(f"  FAIL      : {len(failed_rows):>4}  →  {failed_path}")
    print(f"  AMBIGUOUS : {len(ambiguous_rows):>4}  →  {ambiguous_path}")
    print(f"  Reasoning : {len(reasoning_rows):>4}  →  {reasoning_path}")
    print(f"{'─'*50}")


def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        csv_files = sorted(RAW_DIR.glob("*.csv"))
        if not csv_files:
            sys.exit(f"ERROR: No CSV files found in {RAW_DIR}")
        print(f"Found {len(csv_files)} CSV file(s): {[f.name for f in csv_files]}")

        all_passed: list[dict] = []
        all_failed: list[dict] = []
        all_ambiguous: list[dict] = []
        all_reasoning: list[dict] = []

        for csv_path in csv_files:
            p, f, a, r = process_csv(csv_path)
            all_passed.extend(p)
            all_failed.extend(f)
            all_ambiguous.extend(a)
            all_reasoning.extend(r)

        save_and_report(all_passed, all_failed, all_ambiguous, all_reasoning, OUT_DIR / "combined", "combined")
    else:
        csv_path = RAW_DIR / args.csv
        if not csv_path.exists():
            sys.exit(f"ERROR: {csv_path} not found.")

        stem = csv_path.stem
        passed_rows, failed_rows, ambiguous_rows, reasoning_rows = process_csv(csv_path)
        save_and_report(passed_rows, failed_rows, ambiguous_rows, reasoning_rows, OUT_DIR / stem, stem)


if __name__ == "__main__":
    main()

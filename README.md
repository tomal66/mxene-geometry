# MXene Geometry

A pipeline for filtering MXene research literature entries by formula and synthesis method.
Matching is rule-based — deterministic, instant, no GPU required.

---

## Project structure

```
mxene-geometry/
├── data/
│   ├── raw/              # Input CSVs (one row = one literature entry)
│   └── filtered/         # Output CSVs written here after filtering
├── scripts/
│   └── filter_mxene.py   # LLM-based filtering script
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone

```bash
git clone <repo-url>
cd mxene-geometry
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Input CSV format

Place your CSV files in `data/raw/`. Each file must contain at least these two
columns (exact names, case-sensitive):

| Column | Description |
|---|---|
| `Mxene formula` | Chemical formula of the MXene, e.g. `Ti₃C₂Tₓ` |
| `Synthesis method` | Free-text description of the synthesis route |

All other columns are preserved as-is in the output files.

---

## Filtering criteria

Both criteria must be satisfied for a row to **PASS**.

| # | Criterion | Rule |
|---|---|---|
| 1 | **MXene formula** | Must be `Ti₃C₂Tₓ` or `Ti₃C₂` (all ASCII / unicode / subscript variants accepted). Any other primary stoichiometry (Ta₄C₃, Ti₂C, Mo₂C …) fails. |
| 2 | **Synthesis method** | Must explicitly mention **both** `LiF` (lithium fluoride) **and** `HCl` (hydrochloric acid). Mentioning only one, or using HF-based etchants without LiF, fails. |

---

## Running the filter

```bash
python scripts/filter_mxene.py --csv <filename>
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `--csv` | Yes | Filename inside `data/raw/`, e.g. `5.csv` |

### Examples

```bash
python scripts/filter_mxene.py --csv 5.csv
```

---

## Outputs

All output files are written to `data/filtered/` and prefixed with the input
filename stem (e.g. input `5.csv` → prefix `5_`).

| File | Contents |
|---|---|
| `<stem>_filtered.csv` | Rows that **PASS** both criteria. Includes all original columns plus `row_index`. |
| `<stem>_failed.csv` | Rows that **FAIL** one or both criteria. Includes all original columns plus `row_index`. |
| `<stem>_ambiguous.csv` | Rows where one or more fields are **empty** and a decision cannot be made. Includes all original columns plus `row_index`. |
| `<stem>_reasoning.csv` | Per-criterion reasoning for **every input row** (PASS, FAIL, and AMBIGUOUS). |

### `reasoning.csv` columns

| Column | Description |
|---|---|
| `row_index` | 0-based row position in the original input CSV |
| `formula_match` | `True` / `False` / `None` (uncertain) |
| `formula_reasoning` | One-sentence LLM explanation for the formula decision |
| `synthesis_match` | `True` / `False` / `None` (uncertain) |
| `synthesis_reasoning` | One-sentence LLM explanation for the synthesis decision |
| `overall` | `PASS`, `FAIL`, or `AMBIGUOUS` |
| `overall_reasoning` | One-sentence LLM summary of the overall decision |

`row_index` is consistent across all three files and can be used to join them
back to the original CSV.

---

## Troubleshooting

**Out of memory during model load**
Run with `--quantize` to enable 4-bit loading, which reduces peak VRAM from
~14 GB to ~6 GB.

**`JSON parsing failed` entries in reasoning.csv**
The model occasionally returns non-JSON text. These rows are automatically
marked `AMBIGUOUS` and routed to `<stem>_ambiguous.csv` for manual review.

**Missing column error**
Verify your CSV has columns named exactly `Mxene formula` and
`Synthesis method` (case-sensitive, with a space).

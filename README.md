# MXene Geometry

A pipeline for filtering MXene research literature entries using an LLM
([m3rg-iitd/llamat-3-chat](https://huggingface.co/m3rg-iitd/llamat-3-chat)),
a materials-science fine-tuned LLaMA-3 model.

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

> **GPU note:** the script uses `device_map="auto"` and will use any available
> CUDA GPU automatically. A GPU with at least 16 GB VRAM is recommended for
> full-precision inference; use `--quantize` (see below) to reduce this to ~6 GB.

### 4. HuggingFace access

The model is downloaded from HuggingFace on first run and cached locally.
If the model repository is gated, generate a token at
<https://huggingface.co/settings/tokens> and pass it via `--hf-token`.

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

| Argument | Required | Default | Description |
|---|---|---|---|
| `--csv` | Yes | — | Filename inside `data/raw/`, e.g. `5.csv` |
| `--batch-size` | No | `8` | Number of rows sent to the GPU per inference call |
| `--quantize` | No | off | Load model in 4-bit (bitsandbytes) — reduces VRAM from ~14 GB to ~6 GB |
| `--hf-token` | No | — | HuggingFace access token for gated model repositories |

### Examples

```bash
# Standard run
python scripts/filter_mxene.py --csv 5.csv

# Larger batch on a high-VRAM GPU (A100 80 GB)
python scripts/filter_mxene.py --csv 5.csv --batch-size 16

# Low-VRAM machine (4-bit quantisation)
python scripts/filter_mxene.py --csv 5.csv --quantize

# Gated model with HF token
python scripts/filter_mxene.py --csv 5.csv --hf-token hf_xxxxxxxxxxxxxxxx
```

---

## Outputs

All output files are written to `data/filtered/` and prefixed with the input
filename stem (e.g. input `5.csv` → prefix `5_`).

| File | Contents |
|---|---|
| `<stem>_filtered.csv` | Rows the LLM is **confident PASS** both criteria. Includes all original columns plus `row_index`. |
| `<stem>_ambiguous.csv` | Rows where the LLM is **uncertain** about one or both criteria. Includes all original columns plus `row_index`. |
| `<stem>_reasoning.csv` | Full LLM reasoning for **every input row** (PASS, FAIL, and AMBIGUOUS). |

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

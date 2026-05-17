#!/usr/bin/env python3
"""
Filter MXene CSV rows using m3rg-iitd/llamat-3-chat.

Usage
-----
    python scripts/filter_mxene.py --csv 5.csv
    python scripts/filter_mxene.py --csv 5.csv --batch-size 16
    python scripts/filter_mxene.py --csv 5.csv --quantize
    python scripts/filter_mxene.py --csv 5.csv --hf-token <token>

Criteria (both must hold to PASS)
----------------------------------
  1. MXene formula  : Ti₃C₂Tₓ or Ti₃C₂ (all unicode / ascii variants)
  2. Synthesis      : explicitly mentions BOTH LiF and HCl

Outputs  →  data/filtered/<stem>_filtered.csv / _ambiguous.csv / _reasoning.csv
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/filtered")
MODEL_ID = "m3rg-iitd/llamat-3-chat"

SYSTEM_PROMPT = (
    "You are an expert materials scientist specialising in MXene synthesis and "
    "characterisation. You evaluate research database entries against strict "
    "inclusion criteria. You respond ONLY with a valid JSON object — no preamble, "
    "no explanation outside the JSON."
)

USER_TEMPLATE = """Evaluate the MXene research entry below against two mandatory criteria.

━━━ CRITERION 1 — MXene Formula ━━━
Accept ONLY Ti₃C₂Tₓ  or  Ti₃C₂  as the primary MXene formula.
All ASCII / unicode / subscript variants are equivalent:
  Ti3C2Tx, Ti₃C₂Tₓ, Ti3C2, Ti₃C₂  →  all accepted.
A cell may list multiple MXenes (e.g. "Ti₃C₂Tₓ (primary); Ti₂CTₓ (secondary)").
It PASSES if Ti₃C₂Tₓ or Ti₃C₂ is labelled as PRIMARY, or is the only formula.
Any other stoichiometry as the primary entry (Ta₄C₃, Ti₂C, Mo₂C, Nb₂C, V₂C …) FAILS.

━━━ CRITERION 2 — Synthesis Method ━━━
The synthesis description must explicitly mention BOTH:
  • LiF  (lithium fluoride — the mild fluoride salt source)
  • HCl  (hydrochloric acid)
Mentioning only one of them, or using HF / other acids without LiF, FAILS this criterion.

━━━ Entry ━━━
MXene Formula    : {formula}
Synthesis Method : {synthesis}

━━━ Response format ━━━
Respond ONLY with this JSON and nothing else:
{{
  "formula_match":      true | false | null,
  "formula_reasoning":  "<one concise sentence>",
  "synthesis_match":    true | false | null,
  "synthesis_reasoning":"<one concise sentence>",
  "overall":            "PASS" | "FAIL" | "AMBIGUOUS",
  "overall_reasoning":  "<one concise sentence>"
}}

Rules for "overall":
  "PASS"      → formula_match is true  AND synthesis_match is true
  "FAIL"      → formula_match is false OR  synthesis_match is false  (neither is null)
  "AMBIGUOUS" → any field is null, or you cannot assess with confidence"""


# ── helpers ──────────────────────────────────────────────────────────────────

def build_prompt(formula: str, synthesis: str) -> str:
    return USER_TEMPLATE.format(
        formula=formula or "(empty)",
        synthesis=synthesis or "(empty)",
    )


def _fix_single_quotes(s: str) -> str:
    """
    Normalize a Python-style / mixed-quote dict string to valid JSON.
    Handles patterns the model produces:
      'key'  : value  →  "key": value
      'key"  : value  →  "key": value   (open single, close double)
    """
    # Strip special tokens that leak into output (ChatML / LLaMA-3 / Phi style)
    s = re.sub(r"<\|[^|>]+\|>", "", s)
    # Replace single-quoted keys (both 'k' and 'k" variants) with double-quoted keys
    s = re.sub(r"'(\w+)['\"](\s*:)", r'"\1"\2', s)
    return s


def _regex_fields(raw: str) -> dict | None:
    """Last-resort: extract individual fields with regex, no JSON parser."""
    bool_map = {"true": True, "false": False, "null": None, "none": None}
    result: dict = {}

    for key in ("formula_match", "synthesis_match"):
        m = re.search(rf"""['"]?{key}['"]?\s*:\s*(\w+)""", raw, re.IGNORECASE)
        if m:
            result[key] = bool_map.get(m.group(1).lower(), None)

    for key in ("formula_reasoning", "synthesis_reasoning", "overall_reasoning"):
        m = re.search(rf"""['"]?{key}['"]?\s*:\s*['"]([^'"]*?)['"]""", raw, re.IGNORECASE)
        result[key] = m.group(1) if m else ""

    m = re.search(r"""['"]?overall['"]?\s*:\s*['"]?(\w+)['"]?""", raw, re.IGNORECASE)
    if m:
        result["overall"] = m.group(1).upper()

    return result if len(result) >= 4 else None


def extract_json(raw: str) -> dict | None:
    """
    Parse the LLM response to a dict. Tries in order:
      1. Standard JSON (double quotes)
      2. After fixing single-quoted / mixed-quoted keys
      3. Regex field extraction (model-agnostic fallback)
    """
    raw = raw.strip()

    # Strip special tokens and markdown fences up front
    raw = re.sub(r"<\|[^|>]+\|>", "", raw)
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    # Isolate the first { … } block (take the earliest { and latest })
    start = raw.find("{")
    end = raw.rfind("}")
    candidate = raw[start : end + 1] if start != -1 and end > start else raw

    # Pass 1 — standard JSON
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Pass 2 — fix single / mixed quoting then retry
    try:
        return json.loads(_fix_single_quotes(candidate))
    except json.JSONDecodeError:
        pass

    # Pass 3 — regex extraction on the full raw string (not just candidate)
    return _regex_fields(raw)


def normalise_verdict(parsed: dict) -> dict:
    """Override 'overall' to be logically consistent with the per-criterion fields."""
    fm = parsed.get("formula_match")
    sm = parsed.get("synthesis_match")

    if fm is None or sm is None:
        parsed["overall"] = "AMBIGUOUS"
    elif fm is True and sm is True:
        parsed["overall"] = "PASS"
    elif fm is False or sm is False:
        parsed["overall"] = "FAIL"

    return parsed


def fallback_record(raw_output: str) -> dict:
    clean = re.sub(r"<\|[^|>]+\|>", "", raw_output).strip()
    return {
        "formula_match": None,
        "formula_reasoning": "All parsing strategies failed; could not assess.",
        "synthesis_match": None,
        "synthesis_reasoning": "All parsing strategies failed; could not assess.",
        "overall": "AMBIGUOUS",
        "overall_reasoning": f"Unparseable model output. Raw: {clean[:300]}",
    }


# ── model ────────────────────────────────────────────────────────────────────

def load_model(quantize: bool, hf_token: str | None):
    print(f"Loading tokenizer  : {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=hf_token,
        trust_remote_code=True,
        clean_up_tokenization_spaces=False,  # suppress BPE warning
    )

    # Left-padding is required for batched decoder-only generation.
    # Padding on the right would shift the start-of-generation position.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict = dict(
        token=hf_token,
        trust_remote_code=True,
        device_map="auto",
    )

    if quantize:
        print("Loading model      : 4-bit quantised (bitsandbytes)")
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        print("Loading model      : full precision")
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    print("Model ready.\n")
    return model, tokenizer


def call_model_batch(model, tokenizer, entries: list[tuple[str, str]]) -> list[str]:
    """
    Run inference on a batch of (formula, synthesis) pairs.
    Returns one decoded string per entry (new tokens only).
    """
    prompt_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(formula, synthesis)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for formula, synthesis in entries
    ]

    inputs = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,          # pad shorter prompts to the longest in the batch
        truncation=True,
        max_length=2048,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    # Slice off the shared input prefix; decode only new tokens per item
    return [
        tokenizer.decode(output_ids[i][input_len:], skip_special_tokens=True)
        for i in range(len(entries))
    ]


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter MXene rows with llamat-3-chat.")
    p.add_argument("--csv", required=True, help="CSV filename inside data/raw/ (e.g. 5.csv)")
    p.add_argument("--batch-size", type=int, default=8, help="Rows per inference batch (default: 8)")
    p.add_argument("--quantize", action="store_true", help="Load model in 4-bit (bitsandbytes)")
    p.add_argument("--hf-token", default=None, help="HuggingFace access token (if model is gated)")
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

    model, tokenizer = load_model(args.quantize, args.hf_token)
    print(f"Batch size         : {args.batch_size}\n")

    passed_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    reasoning_rows: list[dict] = []

    rows = list(df.iterrows())
    batches = [rows[i : i + args.batch_size] for i in range(0, len(rows), args.batch_size)]

    with tqdm(total=len(rows), desc="Evaluating rows") as pbar:
        for batch in batches:
            indices = [int(idx) for idx, _ in batch]
            entries = [
                (str(row.get("Mxene formula", "")).strip(),
                 str(row.get("Synthesis method", "")).strip())
                for _, row in batch
            ]

            raw_outputs = call_model_batch(model, tokenizer, entries)

            for (row_index, row), (formula, synthesis), raw_output in zip(
                batch, entries, raw_outputs
            ):
                parsed = extract_json(raw_output)
                verdict = normalise_verdict(parsed) if parsed is not None else fallback_record(raw_output)

                reasoning_rows.append({
                    "row_index": row_index,
                    "formula_match": verdict.get("formula_match"),
                    "formula_reasoning": verdict.get("formula_reasoning", ""),
                    "synthesis_match": verdict.get("synthesis_match"),
                    "synthesis_reasoning": verdict.get("synthesis_reasoning", ""),
                    "overall": verdict.get("overall", "AMBIGUOUS"),
                    "overall_reasoning": verdict.get("overall_reasoning", ""),
                })

                base_row = {"row_index": row_index, **row.to_dict()}
                overall = verdict.get("overall", "AMBIGUOUS")
                if overall == "PASS":
                    passed_rows.append(base_row)
                elif overall == "AMBIGUOUS":
                    ambiguous_rows.append(base_row)

            pbar.update(len(batch))

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

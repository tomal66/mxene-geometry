#!/usr/bin/env python3
"""
Filter MXene CSV rows using m3rg-iitd/llamat-3-chat.

Usage
-----
    python scripts/filter_mxene.py --csv 5.csv
    python scripts/filter_mxene.py --csv 5.csv --quantize          # 4-bit
    python scripts/filter_mxene.py --csv 5.csv --hf-token <token>  # gated model

Criteria (both must hold to PASS)
----------------------------------
  1. MXene formula  : Ti₃C₂Tₓ or Ti₃C₂ (all unicode / ascii variants)
  2. Synthesis      : explicitly mentions BOTH LiF and HCl

Outputs  →  data/filtered/
  filtered.csv   – rows the LLM is confident PASS
  ambiguous.csv  – rows the LLM is uncertain about
  reasoning.csv  – full per-row LLM reasoning (covers every input row)
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


def extract_json(raw: str) -> dict | None:
    """Parse JSON from LLM output; tolerates markdown fences and leading text."""
    raw = raw.strip()
    # Strip ```json ... ``` fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)

    # Try the whole string first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Find the first balanced { … } block
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None


def normalise_verdict(parsed: dict) -> dict:
    """Override 'overall' to be consistent with the individual match fields."""
    fm = parsed.get("formula_match")
    sm = parsed.get("synthesis_match")

    if fm is None or sm is None:
        parsed["overall"] = "AMBIGUOUS"
    elif fm is True and sm is True:
        parsed["overall"] = "PASS"
    elif fm is False or sm is False:
        parsed["overall"] = "FAIL"
    # else leave whatever the model returned (shouldn't happen)

    return parsed


def fallback_record(formula: str, synthesis: str, raw_output: str) -> dict:
    """Return an AMBIGUOUS record when JSON parsing fails entirely."""
    return {
        "formula_match": None,
        "formula_reasoning": "JSON parsing failed; could not assess.",
        "synthesis_match": None,
        "synthesis_reasoning": "JSON parsing failed; could not assess.",
        "overall": "AMBIGUOUS",
        "overall_reasoning": f"Model output was not valid JSON. Raw: {raw_output[:200]}",
    }


# ── model ────────────────────────────────────────────────────────────────────

def load_model(quantize: bool, hf_token: str | None):
    print(f"Loading tokenizer  : {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=hf_token,
        trust_remote_code=True,
    )

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
        kwargs["torch_dtype"] = torch.bfloat16  # saves memory, no quality loss on A100/H100

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    print("Model ready.\n")
    return model, tokenizer


def call_model(model, tokenizer, formula: str, synthesis: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(formula, synthesis)},
    ]

    # apply_chat_template handles special tokens for llamat-3-chat (LLaMA-3 base)
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,   # greedy — deterministic, no temperature needed
        )

    # Return only the newly generated tokens
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter MXene rows with llamat-3-chat.")
    p.add_argument("--csv", required=True, help="CSV filename inside data/raw/ (e.g. 5.csv)")
    p.add_argument("--quantize", action="store_true", help="Load model in 4-bit (bitsandbytes)")
    p.add_argument("--hf-token", default=None, help="HuggingFace access token (if model is gated)")
    return p.parse_args()


def main():
    args = parse_args()

    csv_path = RAW_DIR / args.csv
    if not csv_path.exists():
        sys.exit(f"ERROR: {csv_path} not found.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── load data ────────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    print(f"Loaded {len(df)} rows from {csv_path}\n")

    required = {"Mxene formula", "Synthesis method"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: CSV is missing columns: {missing}\nFound: {list(df.columns)}")

    # ── load model ───────────────────────────────────────────────────────────
    model, tokenizer = load_model(args.quantize, args.hf_token)

    # ── process rows ─────────────────────────────────────────────────────────
    passed_rows: list[dict] = []
    ambiguous_rows: list[dict] = []
    reasoning_rows: list[dict] = []

    for row_index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating rows"):
        formula = str(row.get("Mxene formula", "")).strip()
        synthesis = str(row.get("Synthesis method", "")).strip()

        raw_output = call_model(model, tokenizer, formula, synthesis)
        parsed = extract_json(raw_output)

        if parsed is None:
            verdict = fallback_record(formula, synthesis, raw_output)
        else:
            verdict = normalise_verdict(parsed)

        verdict["row_index"] = int(row_index)

        # Reasoning CSV: every row
        reasoning_rows.append({
            "row_index": row_index,
            "formula_match": verdict.get("formula_match"),
            "formula_reasoning": verdict.get("formula_reasoning", ""),
            "synthesis_match": verdict.get("synthesis_match"),
            "synthesis_reasoning": verdict.get("synthesis_reasoning", ""),
            "overall": verdict.get("overall", "AMBIGUOUS"),
            "overall_reasoning": verdict.get("overall_reasoning", ""),
        })

        # Route to filtered or ambiguous
        base_row = {"row_index": row_index, **row.to_dict()}
        overall = verdict.get("overall", "AMBIGUOUS")
        if overall == "PASS":
            passed_rows.append(base_row)
        elif overall == "AMBIGUOUS":
            ambiguous_rows.append(base_row)
        # FAIL rows appear only in reasoning.csv

    # ── write outputs ────────────────────────────────────────────────────────
    stem = csv_path.stem  # e.g. "5" from "5.csv"
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

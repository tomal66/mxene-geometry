"""Stage 4-6: standardised physical descriptors, H2SO4 molarity, paper grouping."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .parsing import convert, norm_text, norm_unit, parse_value

# ---------------------------------------------------------------------------
# numeric descriptors: (new column, raw column, unit column, canonical target)
# ---------------------------------------------------------------------------

NUMERIC_SPECS = [
    ("interlayer_A",            "interlayer_spacing",   "interlayer_spacing__unit",   "interlayer_A"),
    ("scan_rate_mV_s",          "scan_rate",            "scan_rate__unit",            "scan_rate_mV_s"),
    ("current_density_A_g",     "current_density",      "current_density__unit",      "current_density_A_g"),
    ("mass_loading_mg_cm2",     "mass_loading",         "mass_loading__unit",         "mass_loading_mg_cm2"),
    ("electrode_thickness_um",  "electrode_thickness",  "electrode_thickness__unit",  "electrode_thickness_um"),
    ("ssa_m2_g",                "ssa",                  "ssa__unit",                  "ssa_m2_g"),
    ("flake_size_um",           "flake_size",           "flake_size__unit",           "flake_size_um"),
    ("pore_diameter_nm",        "pore_diameter",        "pore_diameter__unit",        "pore_diameter_nm"),
]

# physically sane bands - values outside are FLAGGED (and logged), not silently dropped
CANONICAL_UNIT_LABEL = {
    "interlayer_A": "A (angstrom)",
    "scan_rate_mV_s": "mV/s",
    "current_density_A_g": "A/g",
    "mass_loading_mg_cm2": "mg/cm2",
    "electrode_thickness_um": "um",
    "ssa_m2_g": "m2/g",
    "flake_size_um": "um",
    "pore_diameter_nm": "nm",
}

PLAUSIBLE_BANDS = {
    "interlayer_A":           (5.0, 60.0),
    "scan_rate_mV_s":         (0.1, 100000.0),
    "current_density_A_g":    (0.01, 1000.0),
    "mass_loading_mg_cm2":    (0.01, 100.0),
    "electrode_thickness_um": (0.01, 5000.0),
    "ssa_m2_g":               (0.1, 3000.0),
    "flake_size_um":          (0.001, 1000.0),
    "pore_diameter_nm":       (0.1, 100000.0),
}


def build_numeric_descriptors(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame(index=df.index)
    records = []
    for new_col, raw_col, unit_col, target_unit in NUMERIC_SPECS:
        vals, approx = [], []
        reasons = []
        for raw, unit in zip(df[raw_col], df[unit_col]):
            p = parse_value(raw)
            if not p.usable:
                vals.append(np.nan); approx.append(np.nan); reasons.append(f"value:{p.kind}")
                continue
            v, why = convert(p.value, unit, target_unit)
            vals.append(v if v is not None else np.nan)
            approx.append(p.is_approximate if v is not None else np.nan)
            reasons.append("ok" if v is not None else why)
        s = pd.Series(vals, index=df.index, dtype=float)
        lo, hi = PLAUSIBLE_BANDS[new_col]
        implausible = s.notna() & ((s < lo) | (s > hi))
        out[new_col] = s
        out[f"{new_col}__approx"] = pd.Series(approx, index=df.index, dtype="object")
        out[f"{new_col}__implausible"] = implausible

        rc = pd.Series(reasons).value_counts().to_dict()
        records.append(
            {
                "standardised_column": new_col,
                "source_column": raw_col,
                "canonical_unit": CANONICAL_UNIT_LABEL[target_unit],
                "n_raw_nonnull": int(df[raw_col].notna().sum()),
                "n_standardised": int(s.notna().sum()),
                "n_approximate": int(pd.Series(approx).eq(True).sum()),
                "n_outside_plausible_band": int(implausible.sum()),
                "plausible_band": f"[{lo}, {hi}]",
                "rejection_reasons": "; ".join(f"{k}={v}" for k, v in sorted(rc.items()) if k != "ok"),
            }
        )
    return out, pd.DataFrame(records)


# ---------------------------------------------------------------------------
# H2SO4 electrolyte concentration  (test electrolyte only, never synthesis)
# ---------------------------------------------------------------------------

_MOLARITY_UNITS = r"(?:M|m|mol\s*/\s*L|mol\s*L-1|mol\s*dm-3|molar)"
_CONC_PAT = re.compile(rf"(\d+(?:\.\d+)?)\s*{_MOLARITY_UNITS}\b", re.I)
_ADDITIVE_PAT = re.compile(r"\+|\bKI\b|\bVOSO4\b|\bH3PO4\b|additive", re.I)
_GEL_PAT = re.compile(r"\bPVA\b|\bgel\b|poly\(vinyl alcohol\)|\bPAAm\b", re.I)


def extract_h2so4(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse molarity from the *electrolyte* field only.

    ``synthesis_method`` frequently contains acid concentrations (e.g. wet
    spinning in 98 wt% H2SO4) which are NOT the test electrolyte, so that field
    is deliberately never consulted here.
    """
    mol, gel, additive, status = [], [], [], []
    for raw in df["electrolyte"]:
        txt = norm_text(raw)
        low = txt.lower()
        if "h2so4" not in low and "sulfuric" not in low:
            mol.append(np.nan); gel.append(np.nan); additive.append(np.nan)
            status.append("not_h2so4"); continue

        is_gel = bool(_GEL_PAT.search(txt))
        has_add = bool(_ADDITIVE_PAT.search(txt))
        hits = _CONC_PAT.findall(txt)
        if len(hits) == 1:
            mol.append(float(hits[0])); status.append("parsed_single")
        elif len(hits) > 1:
            # e.g. "3 M H2SO4 + 0.3 M KI": take the concentration that sits
            # immediately before the H2SO4 token, otherwise refuse to guess.
            m = re.search(rf"(\d+(?:\.\d+)?)\s*{_MOLARITY_UNITS}\s*(?:[A-Za-z()/ ]{{0,25}})?H2SO4", txt, re.I)
            if m:
                mol.append(float(m.group(1))); status.append("parsed_multi_anchored_on_H2SO4")
            else:
                mol.append(np.nan); status.append("ambiguous_multiple_concentrations")
        else:
            mol.append(np.nan); status.append("no_concentration_reported")
        gel.append(is_gel); additive.append(has_add)

    out = pd.DataFrame(
        {"h2so4_M": mol, "electrolyte_is_gel": gel, "electrolyte_has_additive": additive},
        index=df.index,
    )
    log = (
        pd.DataFrame({"electrolyte": df["electrolyte"], "h2so4_M": mol, "status": status})
        .groupby(["electrolyte", "status"], dropna=False)["h2so4_M"]
        .agg(["first", "size"])
        .reset_index()
        .rename(columns={"first": "h2so4_M", "size": "n_rows"})
    )
    return out, log


# ---------------------------------------------------------------------------
# categorical families
# ---------------------------------------------------------------------------

def _layer_class(raw) -> float | str:
    t = norm_text(raw).lower()
    if not t or t == "nan":
        return np.nan
    if re.search(r"multilayer|multi-layer|multilayers|restack|orderly stacked|stacked layers", t):
        if not re.search(r"single|mono|few|delaminat", t):
            return "multilayer"
    if re.search(r"delaminat|exfoliat", t):
        return "delaminated"
    if re.search(r"single|mono|few|ultrathin|two-layer|double-layer|double layer|thin", t):
        return "single_or_few_layer"
    m = re.fullmatch(r"\d+", t)
    if m:
        n = int(m.group())
        return "single_or_few_layer" if n <= 3 else "multilayer"
    if re.search(r"multilayer|stacked", t):
        return "multilayer"
    return np.nan


_CARBON = r"cnt|swcnt|mwcnt|graphene|rgo|\bgo\b|carbon|cnf|graphite|cmk|nanotube"
_POLYMER = r"pedot|\bpva\b|pani|polyaniline|\bppy\b|polypyrrole|polymer|chitosan|pvdf|pdda|nafion|pmma"
_INORG = r"mno2|tio2|sio2|silica|mos2|fe3o4|nio|zno|v2o5|co3o4|nanoparticle"
_DOPED = r"n-doped|n doped|nitrogen-doped|s-doped|sulfur-doped|p-doped|\bn-ti3c2\b|heteroatom"
_POROUS = r"porous|macroporous|mesoporous|foam|aerogel|hydrogel|crumpled|template"


def _composition_family(raw) -> float | str:
    t = norm_text(raw).lower()
    if not t or t == "nan":
        return np.nan
    if re.search(_CARBON, t):
        return "mxene_carbon_composite"
    if re.search(_POLYMER, t):
        return "mxene_polymer_composite"
    if re.search(_INORG, t):
        return "mxene_inorganic_composite"
    if re.search(_DOPED, t):
        return "doped_mxene"
    if re.search(_POROUS, t):
        return "structured_pristine_mxene"
    if re.search(r"ti3c2", t):
        return "pristine_mxene"
    return "other"


def _synthesis_family(raw) -> float | str:
    t = norm_text(raw).lower()
    if not t or t == "nan":
        return np.nan
    if re.search(r"\blif\b.*\bhcl\b|\bhcl\b.*\blif\b|mild|minimally intensive", t):
        return "LiF_HCl_MILD"
    if re.search(r"nh4hf2|ammonium bifluoride", t):
        return "NH4HF2_etch"
    if re.search(r"\bhf\b|hydrofluoric", t):
        return "HF_etch"
    if re.search(r"electrochemical etch|electrochem", t):
        return "electrochemical_etch"
    if re.search(r"molten salt|lewis acid", t):
        return "molten_salt_etch"
    return "other_etch_route"


def build_categorical_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "layer_class": df["layer_no"].map(_layer_class),
            "composition_family": df["composition"].map(_composition_family),
            "synthesis_family": df["synthesis_method"].map(_synthesis_family),
        },
        index=df.index,
    )


NUMERIC_FEATURES = [s[0] for s in NUMERIC_SPECS] + ["h2so4_M"]
CATEGORICAL_FEATURES = [
    "layer_class",
    "composition_family",
    "synthesis_family",
    "electrolyte_is_gel",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    num, num_log = build_numeric_descriptors(df)
    elec, elec_log = extract_h2so4(df)
    cat = build_categorical_descriptors(df)
    feats = pd.concat([num, elec, cat], axis=1)
    return feats, {"numeric_log": num_log, "electrolyte_log": elec_log}

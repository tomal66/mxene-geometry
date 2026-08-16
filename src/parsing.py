"""Value / unit parsing primitives.

Every numeric extraction returns a :class:`ParsedValue` carrying the numeric
value *and* the kind of literal it came from, so the calling code can decide
what is defensible.  Ranges and inequalities are NEVER silently collapsed to a
midpoint -- they are returned with ``value=None`` and their own ``kind``.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

# ---------------------------------------------------------------- normalisation

_UNICODE_MAP = {
    "−": "-",   # minus sign
    "–": "-",   # en dash
    "—": "-",   # em dash
    "‒": "-",   # figure dash
    "·": ".",   # middle dot (used in mol*L-1)
    "≈": "~",   # almost equal
    "∼": "~",   # tilde operator
    "˜": "~",
    "µ": "u",   # micro sign
    "μ": "u",   # greek mu
    "Å": "A",   # angstrom sign
    "Å": "A",   # angstrom symbol
    "±": "+/-",
    "≤": "<=",
    "≥": ">=",
    " ": " ",
}


def norm_text(s) -> str:
    """Unicode-normalise a raw cell into plain ASCII-ish lowercase-safe text."""
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    for k, v in _UNICODE_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_UNIT_ALIASES = {
    "microns": "um", "micron": "um", "micrometer": "um", "micrometre": "um",
    "micrometers": "um", "angstrom": "a", "angstroms": "a", "aa": "a",
    "mol/l": "m", "moll": "m", "mol/dm3": "m", "molar": "m",
}


def norm_unit(s) -> str:
    """Normalise a unit string to a canonical ``numerator/denominator`` token.

    Negative exponents are folded into a denominator so that ``mg cm-2``,
    ``mg/cm2`` and ``mg cm**-2`` all collapse to ``mg/cm2``.
    """
    u = norm_text(s).lower()
    if not u or u in {"nan", "none"}:
        return ""
    u = u.replace("^", "").replace("**", "").rstrip(".")
    if "/" in u:                       # already in numerator/denominator form
        u = u.replace(" ", "").replace(".", "")
        return _UNIT_ALIASES.get(u, u)

    num, den = [], []
    for tok in u.split():
        tok = tok.strip(".")
        m = re.fullmatch(r"([a-z]+\d*)-(\d+)", tok)
        if m:
            base, exp = m.group(1), int(m.group(2))
            den.append(base if exp == 1 else f"{base}{exp}")
        elif tok:
            num.append(tok)
    numerator = "".join(num)
    numerator = _UNIT_ALIASES.get(numerator, numerator)
    if den:
        return f"{numerator}/{''.join(den)}"
    return numerator


# ---------------------------------------------------------------- value parsing

NUM = r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?"

_QUALITATIVE_TOKENS = (
    "several", "micrometer", "micron-sized", "few-layer", "sub-micron",
    "nanometer", "not reported", "n/a", "na", "various", "unspecified",
)
_INEQ_PAT = re.compile(
    r"(^|\s)(>|<|>=|<=|over|above|below|under|up to|at least|more than|less than|"
    r"larger than|smaller than|greater than|exceeding|beyond)(\s|$|\d)",
    re.I,
)
_RANGE_PAT = re.compile(
    rf"{NUM}\s*(?:-|to|~|and)\s*{NUM}", re.I
)
_APPROX_PAT = re.compile(r"(~|about|approx|approximately|around|ca\.|circa|nearly|almost|roughly)", re.I)
_PM_PAT = re.compile(rf"({NUM})\s*\+/-\s*({NUM})")
_MULTI_PAT = re.compile(r";|\)\s*(?:and|,)\s*\d|:\s*\d+(?:\.\d+)?\s*:")


@dataclass(frozen=True)
class ParsedValue:
    value: float | None
    kind: str          # exact | approximate | mean_pm_sd | range | inequality
                       # | multi | qualitative | missing | unparseable
    raw: str

    @property
    def usable(self) -> bool:
        return self.value is not None and self.kind in {"exact", "approximate", "mean_pm_sd"}

    @property
    def is_approximate(self) -> bool:
        return self.kind in {"approximate", "mean_pm_sd"}


def _to_float(tok: str) -> float | None:
    tok = tok.strip()
    # thousands separator "2,000" vs decimal comma "2,5"
    if re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+", tok):
        tok = tok.replace(",", "")
    else:
        tok = tok.replace(",", ".")
    try:
        return float(tok)
    except ValueError:
        return None


def parse_value(raw) -> ParsedValue:
    """Parse a single literature cell into a defensible scalar (or a reason not to)."""
    txt = norm_text(raw)
    if txt == "" or txt.lower() in {"nan", "none", "null", "-", "--"}:
        return ParsedValue(None, "missing", txt)

    low = txt.lower()

    # multi-phase / ratio style entries e.g. "12.62 (Ti3C2Tx); 3.88 (rGO)"
    if _MULTI_PAT.search(txt):
        return ParsedValue(None, "multi", txt)

    # mean +/- sd is a single defensible central value
    m = _PM_PAT.search(txt)
    if m:
        v = _to_float(m.group(1))
        return ParsedValue(v, "mean_pm_sd" if v is not None else "unparseable", txt)

    if _RANGE_PAT.search(txt):
        return ParsedValue(None, "range", txt)

    if _INEQ_PAT.search(low):
        return ParsedValue(None, "inequality", txt)

    nums = re.findall(NUM, txt)
    if not nums:
        return ParsedValue(None, "qualitative", txt)
    if len(nums) > 1:
        return ParsedValue(None, "multi", txt)

    if any(t in low for t in _QUALITATIVE_TOKENS) and not re.search(rf"^{_APPROX_PAT.pattern}?\s*{NUM}", low):
        return ParsedValue(None, "qualitative", txt)

    v = _to_float(nums[0])
    if v is None:
        return ParsedValue(None, "unparseable", txt)
    kind = "approximate" if _APPROX_PAT.search(low) else "exact"
    # a stray word beyond the approximation marker means we do not trust it
    residual = _APPROX_PAT.sub("", low)
    residual = re.sub(NUM, "", residual)
    residual = re.sub(r"[\s\(\)\[\].,;:%~=+/-]", "", residual)
    if residual and not residual.isalpha():
        return ParsedValue(None, "unparseable", txt)
    return ParsedValue(v, kind, txt)


# ---------------------------------------------------------------- unit factors
#
# Each entry maps a canonical target unit to {normalised source unit: factor}.
# A source unit that is absent is treated as *not convertible* and yields NaN
# rather than a guess.

UNIT_FACTORS: dict[str, dict[str, float]] = {
    # interlayer spacing -> Angstrom
    "interlayer_A": {"a": 1.0, "angstrom": 1.0, "nm": 10.0},
    # scan rate -> mV/s
    "scan_rate_mV_s": {"mv/s": 1.0, "mvs": 1.0, "v/s": 1000.0},
    # current density -> A/g  (area/volume bases are NOT convertible)
    "current_density_A_g": {"a/g": 1.0, "ma/g": 1e-3},
    # mass loading -> mg/cm2  ('mg' and 'wt. %' carry no area basis)
    "mass_loading_mg_cm2": {"mg/cm2": 1.0, "ug/cm2": 1e-3, "g/cm2": 1000.0},
    # electrode thickness -> um
    "electrode_thickness_um": {"um": 1.0, "nm": 1e-3, "mm": 1000.0, "cm": 1e4},
    # specific surface area -> m2/g
    "ssa_m2_g": {"m2/g": 1.0},
    # flake size -> um
    "flake_size_um": {"um": 1.0, "nm": 1e-3, "microns": 1.0, "micron": 1.0, "mm": 1000.0},
    # pore diameter -> nm
    "pore_diameter_nm": {"nm": 1.0, "um": 1000.0, "a": 0.1},
}


def convert(value: float | None, unit_raw, target: str) -> tuple[float | None, str]:
    """Convert ``value`` expressed in ``unit_raw`` into the canonical unit.

    Returns ``(converted_or_None, reason)``.  Unknown / incompatible units give
    ``None`` -- never a guessed conversion.
    """
    if value is None:
        return None, "no_value"
    u = norm_unit(unit_raw)
    table = UNIT_FACTORS[target]
    if u == "":
        return None, "missing_unit"
    if u not in table:
        return None, f"incompatible_unit:{u}"
    return value * table[u], "ok"

"""Stage F-4: what physics can actually be enforced with this corpus.

The point of this module is to answer a question honestly rather than to
manufacture a governing equation.  It checks, variable by variable, which terms
of the standard electrochemical capacitance relations are present in the corpus,
and it tests the algebraic identities that *are* available against the reported
values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .parsing import norm_unit, parse_value

# ---------------------------------------------------------------------------
# variables required by the textbook capacitance relations
# ---------------------------------------------------------------------------
#   Galvanostatic :  C_g = I * dt / (m * dV)      equivalently  C_g = j * dt / dV
#   CV integral   :  C_g = integral(I dV) / (m * nu * dV)
#   Areal closure :  C_A = C_g * m_A
#   Volumetric    :  C_V = C_g * rho ,  rho = m_A / t
PHYSICS_VARIABLES = [
    ("discharge_time_s", "dt", "Galvanostatic discharge time - required by C_g = I dt /(m dV)"),
    ("voltage_window_V", "dV", "Potential window - required by every capacitance relation"),
    ("current_A", "I", "Absolute discharge current - required by C_g = I dt /(m dV)"),
    ("current_density_A_g", "j", "Gravimetric current density - substitutes I/m in C_g = j dt / dV"),
    ("electrode_area_cm2", "A", "Geometric area - needed to move between areal and absolute quantities"),
    ("electrode_mass_g", "m", "Absolute active mass - needed by the galvanostatic relation"),
    ("mass_loading_mg_cm2", "m_A", "Areal mass loading - links gravimetric and areal capacitance"),
    ("electrode_thickness_um", "t", "Thickness - with m_A gives electrode density for the volumetric closure"),
    ("scan_rate_mV_s", "nu", "CV sweep rate - required by the CV-integral relation"),
    ("cv_current_integral", "int(I dV)", "Integrated CV current - required by the CV-integral relation"),
    ("testing_mode", "-", "Whether a row is GCD or CV - required to know which relation applies"),
    ("areal_capacitance_F_cm2", "C_A", "Auxiliary measured output, tied to C_g by C_A = C_g m_A"),
    ("volumetric_capacitance_F_cm3", "C_V", "Auxiliary measured output, tied to C_g by C_V = C_g m_A / t"),
    ("ssa_m2_g", "S", "Specific surface area - enters area-normalised double-layer arguments"),
    ("interlayer_A", "d", "Interlayer spacing - geometric input to any transport model"),
    ("h2so4_M", "c0", "Bulk electrolyte concentration - boundary condition for ion transport"),
]

# unit tables for the auxiliary capacitance outputs
_AREAL_TO_F_CM2 = {"f/cm2": 1.0, "mf/cm2": 1e-3}
_VOL_TO_F_CM3 = {"f/cm3": 1.0, "mf/cm3": 1e-3}
_ML_TO_MG_CM2 = {"mg/cm2": 1.0, "ug/cm2": 1e-3, "g/cm2": 1e3}
_TH_TO_UM = {"um": 1.0, "nm": 1e-3, "mm": 1e3, "cm": 1e4}


def _parsed(df: pd.DataFrame, col: str, unit_col: str | None,
            unit_map: dict | None) -> pd.Series:
    vals = pd.Series([parse_value(v).value for v in df[col]], index=df.index, dtype=float)
    if unit_col is None:
        return vals
    factors = pd.Series(
        [unit_map.get(norm_unit(u), np.nan) for u in df[unit_col]],
        index=df.index, dtype=float)
    return vals * factors


def build_auxiliary_outputs(raw: pd.DataFrame) -> pd.DataFrame:
    """Standardised areal / volumetric capacitance and the derived electrode density.

    These are **outputs**, not predictors: both are algebraic transforms of the
    target and are excluded from every feature set. They are standardised here
    only so the closure identities can be tested and, in a later model, used as
    auxiliary supervision.
    """
    out = pd.DataFrame(index=raw.index)
    out["areal_capacitance_F_cm2"] = _parsed(
        raw, "areal_capacitance", "areal_capacitance__unit", _AREAL_TO_F_CM2)
    out["volumetric_capacitance_F_cm3"] = _parsed(
        raw, "volumetric_capacitance", "volumetric_capacitance__unit", _VOL_TO_F_CM3)
    ml = _parsed(raw, "mass_loading", "mass_loading__unit", _ML_TO_MG_CM2)
    th = _parsed(raw, "electrode_thickness", "electrode_thickness__unit", _TH_TO_UM)
    out["mass_loading_mg_cm2"] = ml
    out["electrode_thickness_um"] = th
    # rho [g/cm3] = 10 * m_A [mg/cm2] / t [um]
    out["electrode_density_g_cm3"] = 10.0 * ml / th
    return out


def check_closure_identities(target: pd.Series, aux: pd.DataFrame,
                             paper_id: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test C_A = C_g m_A and C_V = C_g m_A / t against the reported values."""
    rows = []

    # ---- areal closure
    d = pd.DataFrame({
        "paper_id": paper_id,
        "C_g": target,
        "C_A_reported": aux["areal_capacitance_F_cm2"],
        "m_A": aux["mass_loading_mg_cm2"],
    }).dropna()
    d["C_A_predicted"] = d["C_g"] * d["m_A"] * 1e-3
    d["relative_error"] = (d["C_A_predicted"] - d["C_A_reported"]).abs() / d["C_A_reported"].abs()
    d["identity"] = "C_A = C_g * m_A"
    rows.append(d.rename(columns={"C_A_reported": "reported", "C_A_predicted": "predicted"})[
        ["identity", "paper_id", "C_g", "reported", "predicted", "relative_error"]])

    # ---- volumetric closure
    v = pd.DataFrame({
        "paper_id": paper_id,
        "C_g": target,
        "C_V_reported": aux["volumetric_capacitance_F_cm3"],
        "rho": aux["electrode_density_g_cm3"],
    }).dropna()
    v["C_V_predicted"] = v["C_g"] * v["rho"]
    v["relative_error"] = (v["C_V_predicted"] - v["C_V_reported"]).abs() / v["C_V_reported"].abs()
    v["identity"] = "C_V = C_g * m_A / t"
    rows.append(v.rename(columns={"C_V_reported": "reported", "C_V_predicted": "predicted"})[
        ["identity", "paper_id", "C_g", "reported", "predicted", "relative_error"]])

    detail = pd.concat(rows, ignore_index=True)

    summary = (
        detail.groupby("identity")
        .apply(lambda g: pd.Series({
            "n_testable_rows": len(g),
            "n_papers": g["paper_id"].nunique(),
            "median_relative_error": g["relative_error"].median(),
            "frac_within_1pct": (g["relative_error"] < 0.01).mean(),
            "frac_within_5pct": (g["relative_error"] < 0.05).mean(),
            "frac_within_20pct": (g["relative_error"] < 0.20).mean(),
            "n_gross_violations_gt50pct": int((g["relative_error"] > 0.50).sum()),
        }), include_groups=False)
        .reset_index()
    )
    return detail, summary


def physics_variable_availability(raw: pd.DataFrame, model: pd.DataFrame,
                                  aux: pd.DataFrame) -> pd.DataFrame:
    """Which terms of the standard capacitance relations this corpus actually has."""
    n = len(model)
    rows = []
    for name, symbol, role in PHYSICS_VARIABLES:
        if name in model.columns:
            series, unit_note = model[name], "normalised by the pipeline"
        elif name in aux.columns:
            series = aux.loc[model.index, name]
            unit_note = "normalised here (auxiliary output)"
        else:
            series, unit_note = None, "absent from the corpus"

        if series is None:
            cnt, frac = 0, 0.0
            usable = "no - variable is not present in the corpus at all"
        else:
            cnt = int(series.notna().sum())
            frac = cnt / n
            if frac >= 0.5:
                usable = "yes"
            elif frac > 0:
                usable = f"partial - only {frac:.0%} of rows"
            else:
                usable = "no - present as a column but never parseable"
        rows.append({
            "variable": name, "symbol": symbol,
            "availability_count": cnt,
            "availability_fraction": round(frac, 4),
            "unit_consistency": unit_note,
            "potential_physics_role": role,
            "usable_now": usable,
        })
    return pd.DataFrame(rows).sort_values("availability_fraction", ascending=False).reset_index(drop=True)


def model_class_assessment(avail: pd.DataFrame, closure: pd.DataFrame) -> pd.DataFrame:
    """Feasibility verdict for the three candidate model classes."""
    have = dict(zip(avail["variable"], avail["availability_fraction"]))
    missing_gcd = [v for v in ("discharge_time_s", "voltage_window_V", "current_A",
                               "electrode_mass_g") if have.get(v, 0) == 0]
    areal = closure[closure["identity"] == "C_A = C_g * m_A"].iloc[0]
    vol = closure[closure["identity"] == "C_V = C_g * m_A / t"].iloc[0]

    return pd.DataFrame([
        {
            "model_class": "A. Classical PDE/ODE PINN",
            "requirement": "A governing differential equation in space/time plus "
                           "initial and boundary conditions per observation",
            "evidence_in_dataset": (
                "Absent. The corpus records no time axis, no potential window, no "
                "discharge curve and no spatial coordinate: "
                f"{', '.join(missing_gcd)} are all 0% populated. Every row is a single "
                "scalar summary of an experiment, not a trajectory."),
            "verdict": "NOT JUSTIFIED",
            "confidence": "high",
        },
        {
            "model_class": "B. Physics-guided neural network",
            "requirement": "Physically motivated features, bounds, dimensional "
                           "consistency and optional monotonicity, without a PDE residual",
            "evidence_in_dataset": (
                "Supported. Positivity of capacitance, training-domain bounds and "
                "dimensionally meaningful inputs (interlayer spacing, molarity, "
                "sweep rate, mass loading) are all available. Monotonicity should "
                "NOT be imposed from SHAP alone - the corpus shows a d-spacing trend "
                "opposite to the naive physical expectation."),
            "verdict": "FEASIBLE",
            "confidence": "high",
        },
        {
            "model_class": "C. Hybrid physics-constrained multi-task regression",
            "requirement": "An exact algebraic relation among measured quantities "
                           "that the model can be forced to respect",
            "evidence_in_dataset": (
                f"Supported and measured. C_A = C_g*m_A reproduces the reported areal "
                f"capacitance to within 5% for {areal['frac_within_5pct']:.0%} of "
                f"{int(areal['n_testable_rows'])} testable rows (median relative error "
                f"{areal['median_relative_error']:.2%}); C_V = C_g*m_A/t holds to within "
                f"5% for {vol['frac_within_5pct']:.0%} of {int(vol['n_testable_rows'])} "
                f"rows (median {vol['median_relative_error']:.2%}). Areal and volumetric "
                "capacitance cannot be predictors - they are algebraic transforms of the "
                "target - but they can serve as auxiliary supervised outputs tied to the "
                "target by these identities."),
            "verdict": "RECOMMENDED",
            "confidence": "medium-high",
        },
    ])

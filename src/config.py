"""Global configuration for the MXene SHAP + Bayesian synthetic-data pipeline."""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
FIG_DEP_DIR = FIG_DIR / "shap_dependence"
FIG_SYN_DIR = FIG_DIR / "synthetic_vs_real"
CACHE_DIR = RESULTS_DIR / "_workdir"

DATASET_BASENAME = "h2so4_corpus_final.csv"

TARGET_RAW = "gravimetric_capacitance"
TARGET = "target_cap_F_g"

# Bayesian-network experiment grid
BN_K_VALUES = (1, 2, 3)
BN_K_DEFAULT = 2
BN_EPSILON = 0  # no differential-privacy noise: source data are already public literature
AUG_RATIOS = (0.0, 1.0, 3.0, 5.25)
AUG_LABELS = {
    0.0: "A_real_only",
    1.0: "B_real_plus_1.0x",
    3.0: "C_real_plus_3.0x",
    5.25: "D_real_plus_5.25x",
}

N_FOLDS_TARGET = 5

# Independent generator seeds per (fold, ratio) at the default k.  Re-drawing the
# synthetic set is itself a source of variance; running it several times turns
# "the difference looks like noise" into a measured quantity.
N_GEN_REPEATS = 3


def resolve_dataset(basename: str = DATASET_BASENAME) -> Path:
    """Locate the corpus CSV anywhere under the project root."""
    direct = PROJECT_ROOT / basename
    if direct.exists():
        return direct
    matches = sorted(
        p for p in PROJECT_ROOT.rglob(basename) if ".git" not in p.parts
    )
    if not matches:
        raise FileNotFoundError(f"Could not find {basename} under {PROJECT_ROOT}")
    return matches[0]


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, FIG_DIR, FIG_DEP_DIR, FIG_SYN_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

"""Bayesian-network synthetic data generation with DataSynthesizer.

The generator is fitted **inside a training fold only**.  Nothing from the
held-out papers ever reaches :func:`generate_synthetic`.

Compatibility note
------------------
DataSynthesizer 0.1.13 serialises Bayesian-network parent instances with
``str(list_of_numpy_scalars)``.  Under NumPy >= 2 that repr is
``[np.int64(3), ...]`` and ``DataGenerator.generate_encoded_dataset`` calls
``eval()`` on it in a module namespace that never imported NumPy, raising
``NameError: name 'np' is not defined``.  The smallest correct fix is to make
``np`` resolvable in that module's globals; the evaluated expression then
reconstructs exactly the intended integer indices.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import DataSynthesizer.DataGenerator as _dg_module
import DataSynthesizer.lib.PrivBayes as _pb_module

if not hasattr(_dg_module, "np"):          # <-- compatibility patch (documented above)
    _dg_module.np = np


class _SerialPool:
    """Drop-in replacement for ``multiprocessing.Pool`` inside PrivBayes.

    ``greedy_bayes`` opens a fresh ``Pool()`` on every iteration of its
    attribute loop.  On Windows (spawn start method) each of those costs a full
    interpreter start per core, which dwarfs the actual work when the network
    has a handful of attributes and a few dozen rows - a single fold took
    minutes almost entirely in process startup.  ``pool.map`` is an ordered map
    over a pure function, so evaluating it serially returns *bit-identical*
    results; only the wall-clock cost changes.
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, iterable):
        return [fn(x) for x in iterable]

    def close(self):
        pass

    def join(self):
        pass


_pb_module.Pool = lambda *a, **kw: _SerialPool()

from DataSynthesizer.DataDescriber import DataDescriber  # noqa: E402
from DataSynthesizer.DataGenerator import DataGenerator  # noqa: E402
from DataSynthesizer.datatypes.utils.DataType import DataType  # noqa: E402


class _StrSafeDataDescriber(DataDescriber):
    """DataDescriber that honours a declared STRING datatype when reading the CSV.

    ``DataDescriber.read_dataset_from_csv`` lets pandas infer dtypes, so a
    categorical column whose levels look numeric (``1.0``, ``2.0``) comes back
    as float64.  ``StringAttribute`` then builds its bin domain from
    ``data_dropna.astype(str)`` while ``encode_values_into_bin_idx`` looks up the
    *unconverted* float values, raising ``KeyError: 2.0``.  Reading the declared
    STRING columns as strings in the first place removes the inconsistency
    without touching any modelling logic.
    """

    def read_dataset_from_csv(self, file_name=None):
        str_cols = [a for a, t in (self.attr_to_datatype or {}).items() if t is DataType.STRING]
        self.df_input = pd.read_csv(
            file_name, skipinitialspace=True, na_values=self.null_values,
            dtype={c: str for c in str_cols},
        )


def describe_and_generate(
    train: pd.DataFrame,
    n_synth: int,
    k: int,
    epsilon: float,
    categorical_flags: dict[str, bool],
    workdir: Path,
    tag: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Fit a Bayesian network on ``train`` and sample ``n_synth`` rows."""
    workdir.mkdir(parents=True, exist_ok=True)
    train_csv = workdir / f"{tag}_train.csv"
    desc_json = workdir / f"{tag}_description.json"
    syn_csv = workdir / f"{tag}_synthetic.csv"

    # Everything flagged categorical is handed to DataSynthesizer as a STRING
    # attribute.  Booleans and discrete floats otherwise trip DataDescriber's
    # bin-index encoder (`KeyError: False`) because it re-reads the CSV with
    # pandas' own dtype inference, which disagrees with the declared DataType.
    original_dtypes = {c: train[c].dtype for c in train.columns}
    out = train.copy()
    dtypes = {}
    for c in out.columns:
        if categorical_flags.get(c, False):
            out[c] = out[c].astype(str)
            dtypes[c] = DataType.STRING
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            dtypes[c] = DataType.FLOAT
    out.to_csv(train_csv, index=False)

    describer = _StrSafeDataDescriber(category_threshold=1)  # explicit flags govern, not the heuristic
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        describer.describe_dataset_in_correlated_attribute_mode(
            dataset_file=str(train_csv),
            k=k,
            epsilon=epsilon,
            attribute_to_datatype=dtypes,
            attribute_to_is_categorical=categorical_flags,
            attribute_to_is_candidate_key={c: False for c in train.columns},
            seed=seed,
        )
    describer.save_dataset_description_to_file(str(desc_json))

    generator = DataGenerator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generator.generate_dataset_in_correlated_attribute_mode(
            n_synth, str(desc_json), seed=seed
        )
    generator.save_synthetic_data(str(syn_csv))

    syn = pd.read_csv(syn_csv, dtype=str)
    # restore the dtypes the modelling code expects
    for c in syn.columns:
        dt = original_dtypes.get(c)
        if dt is None:
            continue
        if pd.api.types.is_bool_dtype(dt):
            syn[c] = syn[c].map({"True": True, "False": False, "true": True, "false": False})
        elif pd.api.types.is_numeric_dtype(dt):
            syn[c] = pd.to_numeric(syn[c], errors="coerce")
        else:
            syn[c] = syn[c].astype(object)
    syn = syn[[c for c in train.columns if c in syn.columns]]

    desc = json.loads(desc_json.read_text(encoding="utf-8"))
    meta = {
        "bayesian_network": desc.get("bayesian_network", []),
        "k": k,
        "epsilon": epsilon,
        "n_train": len(train),
        "n_requested": n_synth,
        "description_file": str(desc_json),
    }
    return syn, meta


# --------------------------------------------------------------- validity gate

def physical_filter(
    syn: pd.DataFrame,
    train: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target: str,
    non_negative: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop synthetic rows that are physically impossible or out-of-domain.

    Domain bounds come from the TRAINING fold only.
    """
    syn = syn.copy()
    n0 = len(syn)
    reject_reason = pd.Series([""] * n0, index=syn.index)

    def mark(mask: pd.Series, reason: str) -> None:
        newly = mask & (reject_reason == "")
        reject_reason.loc[newly] = reason

    # 1. hard physical impossibilities
    if target in syn.columns:
        mark(~pd.to_numeric(syn[target], errors="coerce").gt(0).fillna(False),
             "target_non_positive_or_nan")
    for c in numeric_cols:
        if c in syn.columns:
            v = pd.to_numeric(syn[c], errors="coerce")
            mark(v.isna(), f"{c}_nan")
            if c in non_negative:
                mark(v.le(0).fillna(False), f"{c}_non_positive")
            else:
                mark(v.lt(0).fillna(False), f"{c}_negative")

    # 2. categorical labels unseen in the training fold
    for c in categorical_cols:
        if c in syn.columns:
            allowed = set(train[c].astype(str).unique())
            mark(~syn[c].astype(str).isin(allowed), f"{c}_invalid_category")

    # 3. numeric extrapolation beyond the training-fold domain
    for c in numeric_cols + [target]:
        if c in syn.columns and c in train.columns:
            lo = float(pd.to_numeric(train[c], errors="coerce").min())
            hi = float(pd.to_numeric(train[c], errors="coerce").max())
            v = pd.to_numeric(syn[c], errors="coerce")
            mark((v < lo).fillna(False), f"{c}_below_train_min")
            mark((v > hi).fillna(False), f"{c}_above_train_max")

    keep = reject_reason == ""
    counts = reject_reason[~keep].value_counts()
    log = (
        counts.rename_axis("reason").reset_index(name="n_rejected")
        if not counts.empty
        else pd.DataFrame({"reason": ["none_rejected"], "n_rejected": [0]})
    )
    log["n_generated"] = n0
    log["n_kept"] = int(keep.sum())
    return syn.loc[keep].reset_index(drop=True), log


def generate_filtered(
    train: pd.DataFrame,
    n_target: int,
    k: int,
    epsilon: float,
    categorical_flags: dict[str, bool],
    numeric_cols: list[str],
    categorical_cols: list[str],
    target: str,
    workdir: Path,
    tag: str,
    seed: int = 42,
    oversample: float = 1.15,
    max_rounds: int = 4,
    non_negative: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Generate, filter, and top up until ``n_target`` valid rows are available."""
    kept_parts, logs = [], []
    meta = {}
    n_have = 0
    for r in range(max_rounds):
        need = n_target - n_have
        if need <= 0:
            break
        ask = int(np.ceil(need * oversample)) + 10
        syn, meta = describe_and_generate(
            train, ask, k, epsilon, categorical_flags, workdir,
            f"{tag}_r{r}", seed=seed + r,
        )
        kept, log = physical_filter(syn, train, numeric_cols, categorical_cols,
                                    target, non_negative)
        log.insert(0, "round", r)
        logs.append(log)
        kept_parts.append(kept)
        n_have += len(kept)

    all_kept = pd.concat(kept_parts, ignore_index=True) if kept_parts else pd.DataFrame(columns=train.columns)
    all_kept = all_kept.head(n_target).reset_index(drop=True)
    log_df = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=["round", "reason", "n_rejected"])
    meta["n_valid_returned"] = len(all_kept)
    meta["n_valid_requested"] = n_target
    return all_kept, log_df, meta

# MXene capacitance: SHAP + Bayesian-network synthetic data pipeline

Predicts **gravimetric capacitance (F/g)** of Ti3C2Tx MXene electrodes in H2SO4
from a literature-derived corpus, explains the model with SHAP, augments the
training data with a **DataSynthesizer** Bayesian network fitted *inside each
training fold*, and measures whether the augmentation helps on unseen real
papers.

## Reproduce everything

```bash
# 1. environment (Python 3.11-3.13, CPU only)
python -m venv .venv
.venv\Scripts\activate         # Windows
# source .venv/bin/activate    # Linux / macOS
pip install -r requirements.txt

# 2. run the whole pipeline (~3-5 min on a laptop CPU, single-threaded)
python run_pipeline.py
```

Everything lands in `results/`. The run is deterministic: `RANDOM_STATE = 42`
seeds Python `random`, NumPy, XGBoost, the cross-validation splitters and
DataSynthesizer's describer/generator.

The corpus is located automatically: `run_pipeline.py` looks for
`h2so4_corpus_final.csv` in the project root and, failing that, searches the
project recursively (it currently resolves to `terra/h2so4_corpus_final.csv`).

## Layout

```
run_pipeline.py                 stage orchestration, CV loops, experiment grid
src/config.py                   paths, seeds, experiment grid constants
src/parsing.py                  value/unit parsing primitives (ranges are never
                                collapsed to midpoints)
src/data_cleaning.py            dataset audit, target-leakage audit, target cleaning
src/feature_engineering.py      standardised descriptors, H2SO4 molarity, families
src/modeling.py                 grouped CV, fold-local one-hot encoding, XGBoost
src/shap_analysis.py            TreeExplainer, cross-fold aggregation, rank stability
src/bayesian_synthesis.py       DataSynthesizer wrapper + physical-validity filter
src/synthetic_validation.py     KS / Wasserstein / JS / correlation / MI / NN checks
src/plotting.py                 matplotlib figures
src/reporting.py                assembles results/FINAL_REPORT.md
```

## Stage order

| # | stage | key outputs |
|---|---|---|
| 0 | environment capture | `environment.txt` |
| 1 | dataset audit | `00_data_audit.md`, `00_missingness.csv`, `00_column_summary.csv` |
| 2 | target-leakage audit | `00_leakage_audit.csv` |
| 3 | target cleaning | `01_cleaned_model_data.csv`, `01_target_disposition_log.csv` |
| 4 | descriptor standardisation | `01_preprocessing_log.md` |
| 5 | real-only grouped-CV XGBoost | `02_real_only_*.csv`, `figures/real_only_*.png` |
| 6 | SHAP on real data | `03_real_shap_importance.csv`, `figures/real_shap_*.png`, `figures/shap_dependence/` |
| 7 | BN candidate descriptors | `04_bn_candidate_features.csv`, `04_bn_excluded_columns.csv` |
| 8 | BN feature selection + cohort | `04_selected_bn_features.json`, `04_bn_real_cohort.csv`, `04_bn_selection_trace.csv` |
| 9 | fold-wise generation + fidelity | `05_*.csv` (rejection log, BN structures, distribution stats, MI, target relationships, nearest-neighbour), `figures/synthetic_vs_real/` |
| 10 | real vs augmented comparison | `06_augmentation_summary.csv`, `06_augmentation_fold_metrics.csv`, `06_augmentation_per_seed_metrics.csv`, `06_bn_k_sensitivity.csv`, `figures/augmentation_*.png` |
| 11 | paired statistics | `06_statistical_comparison.csv` |
| 12 | augmented SHAP + stability | `07_*.csv`, `figures/augmented_shap_*.png`, `figures/shap_rank_comparison.png` |
| 13 | PINN descriptor table | `08_pinn_descriptor_recommendation.csv` |
| 14 | final report | `FINAL_REPORT.md` |

`results/_workdir/` holds the per-fold DataSynthesizer inputs, learned
Bayesian-network description JSONs and raw synthetic draws, so every generation
step is inspectable after the fact.

## Experiment grid

* Bayesian network: DataSynthesizer **correlated-attribute mode**, `epsilon = 0`
  (public literature data, no DP noise needed), `k in {1, 2, 3}` with `k = 2`
  as the default reported condition.
* Augmentation ratios: `A` real only, `B` +1.0x, `C` +3.0x, `D` +5.25x synthetic
  (D reproduces the 32 -> 200 ratio of the FCDI study that motivated the design).
* Each augmented condition is generated `N_GEN_REPEATS = 3` times per fold with
  independent generator seeds, and the fold score is the mean over those draws.
  The within-fold spread across seeds is reported as an explicit noise floor, so
  a difference between ratios can be compared against the cost of simply
  re-drawing the same configuration.
* Evaluation: `GroupKFold` on `paper_id`; the test fold is always the same set
  of **real** held-out papers for all four conditions. Hyper-parameters are tuned
  once per fold on the real training papers and then held fixed across the four
  conditions, so tuning cannot confound the comparison.

## Rules the code enforces

* No synthetic rows ever enter a test fold.
* The Bayesian network, the one-hot encoder, the hyper-parameter search and the
  synthetic-domain bounds are all fitted on training papers only.
* No random row splitting - rows from one paper never straddle train and test.
* No target-derived predictors (volumetric and areal capacitance, extraction
  confidence/provenance metadata and free-text rationales are all excluded).
* Ranges (`300-350`) and inequalities (`over 200`) are dropped, never converted
  to midpoints, and every dropped row is logged.
* Synthetic values are clipped to the training-fold support; rejections are
  counted by reason in `05_synthetic_rejection_log.csv`.

## Known compatibility fixes

Both are documented inline in `src/bayesian_synthesis.py` and in
`results/environment.txt`:

1. **DataSynthesizer 0.1.13 + NumPy >= 2** - `DataGenerator.generate_encoded_dataset`
   `eval()`s parent-instance strings that NumPy 2 renders as `[np.int64(...)]`,
   in a module that never imports NumPy (`NameError: name 'np' is not defined`).
   Fixed by injecting `np` into that module's globals.
2. **Numeric-looking categorical levels** - `DataDescriber` re-reads the CSV with
   pandas dtype inference, so a declared STRING attribute whose levels look like
   numbers (`1.0`, `2.0`) comes back as float and the bin-index lookup raises
   `KeyError: 2.0`. Fixed with a `DataDescriber` subclass that reads declared
   STRING columns as strings.

And one performance patch (same file, results unchanged):

3. **`greedy_bayes` process pools** - PrivBayes opens a fresh
   `multiprocessing.Pool()` on every iteration of its attribute loop. Under the
   Windows spawn start method that costs one interpreter start per core per
   iteration and dominates the runtime on a network with a handful of attributes
   and a few dozen rows (~90 s per fit, versus ~1.3 s serial). `Pool.map` is an
   ordered map over a pure function, so a serial stand-in returns bit-identical
   results.

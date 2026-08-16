# LaTeX report project

**Machine-Learning Descriptor Analysis and Bayesian Data Augmentation for MXene
Supercapacitor Capacitance Prediction: Implications for Physics-Guided Modelling**

Self-contained LaTeX project. `main.pdf` is committed alongside the sources.

## Build

```bash
make                 # auto-detects latexmk -> tectonic -> pdflatex
make latexmk         # latexmk -pdf main.tex
make tectonic        # tectonic -X compile main.tex
make manual          # pdflatex / bibtex / pdflatex / pdflatex
make clean           # remove auxiliary files
```

The committed PDF was produced with **Tectonic 0.15.0** on Windows. Tectonic is a
single self-contained binary that downloads the TeX packages it needs on first
run, which makes it the least painful option if no TeX distribution is installed:

```bash
# Windows
curl -L -o tectonic.zip https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip
unzip tectonic.zip && ./tectonic.exe -X compile main.tex
```

## Regenerating everything from the pipeline

Sections, tables and figures are **generated**, not hand-written. Do not edit
`sections/*.tex` or `tables/*.tex` directly — they are overwritten. To rebuild
from the stored results:

```bash
cd ..
python run_pipeline.py        # stage 1: CV, SHAP, Bayesian augmentation  (~3 min)
python run_final_stage.py     # stage 2: domain shift, final SHAP, PINN, figures, tables
python build_report.py        # emits sections/*.tex and main.tex
cd latex_report && make
```

Or, from this directory, `make regenerate` runs the last three steps.

Every number in the prose is interpolated at generation time from a CSV under
`../results/` or `../results_final/`. Nothing is transcribed by hand, so the
report cannot drift out of step with the pipeline.

## Layout

```
main.tex              preamble, title, section includes, nomenclature
references.bib        method and software references (see BIBLIOGRAPHY_TODO.md)
sections/             00_abstract .. 13_conclusion   (generated)
tables/               9 booktabs tables               (generated)
figures/              10 vector PDF figures           (generated)
Makefile              build targets
```

### Sections

| File | Content |
|---|---|
| `00_abstract.tex` | ~300-word abstract |
| `01_introduction.tex` | problem, obstacles, contributions |
| `02_dataset.tex` | corpus, target cleaning, unit standardisation, leakage control |
| `03_methods.tex` | XGBoost, paper-grouped CV, the two-models distinction |
| `04_bayesian_augmentation.tex` | DataSynthesizer setup, anti-leakage protocol, grid |
| `05_grouped_results.tex` | cross-paper performance + variance decomposition |
| `06_augmentation_results.tex` | four conditions, why the gain is not robust |
| `07_synthetic_fidelity.tex` | fidelity audit and its interpretation |
| `08_shap.tex` | final real-only SHAP, directions, confounding caveat |
| `09_shap_stability.tex` | rank stability under augmentation |
| `10_pinn_design.tex` | physics availability, model-class verdict, proposed loss |
| `11_discussion.tex` | rows vs information, four purposes, reporting practice |
| `12_limitations.tex` | limitations |
| `13_conclusion.tex` | seven numbered conclusions |

### Figures

All figures are vector PDF at double-column width (7 in), 8 pt type, no titles
(captions carry the message), colour-blind-safe palette, no rainbow colormaps.
Masters live in `../results_final/figures/` in both PDF and 400 dpi PNG.

## Known state

- The PDF compiles clean: no undefined references, no BibTeX errors.
- One deliberate placeholder appears in the reference list — see
  `BIBLIOGRAPHY_TODO.md`. It is visible in the PDF by design so it cannot be
  missed before circulation.

# Bibliography: outstanding items

Nothing in `references.bib` was invented. This file records what could **not** be
verified from files in this repository and therefore must be supplied by hand
before the report is circulated.

## 1. The motivating FCDI study — REQUIRED

`references.bib` contains a placeholder entry:

```
@misc{fcdi_augmentation, ... PLACEHOLDER ... }
```

**Status:** no flow-capacitive-deionisation paper exists anywhere in this
repository. A recursive search for `*fcdi*` and `*flow*capacit*` returned
nothing, and the corpus CSV covers only MXene/H2SO4 supercapacitor articles.

**What the text claims about it** (Sec. 1, Sec. 11): a study that expanded 32
experimental observations to 200 using DataSynthesizer's correlated-attribute
mode and reported improved model performance. That description came from the
project brief, not from a file that can be checked here.

**Action:** replace the placeholder with the real citation. The placeholder is
deliberately rendered in the PDF as `[CITATION TO BE SUPPLIED]` so it cannot be
overlooked.

## 2. Source publications of the corpus — OPTIONAL

The corpus contains **48 unique DOIs** across 49 source files. The extraction
schema stores `doi`, `year` and an abbreviated author string (e.g. `Luo et al.`)
but **no article titles and no journal names**, so complete BibTeX entries
cannot be reconstructed from local data.

They are not cited individually in the current report, which refers to the
corpus collectively. If a per-source reference list is wanted for the supporting
information, the DOIs are exported to:

```
results_final/corpus_source_dois.csv
```

and can be resolved in bulk via the Crossref API
(`https://api.crossref.org/works/{doi}`) or `doi2bib`. Do not hand-type them.

## 3. Entries that ARE verified

The remaining 14 entries are standard method and software references whose
metadata is well established (XGBoost, SHAP, TreeSHAP, DataSynthesizer,
PrivBayes, scikit-learn, NumPy, SciPy, Matplotlib, Raissi et al. on PINNs, and
four MXene references). Volume, page and DOI fields should still be spot-checked
against the publisher record before submission to a journal, as is normal
practice.

## 4. Citations in the text

Only these keys are cited: `naguib2011`, `ghidiu2014`, `lukatskaya2017`,
`alhabeb2017`, `fcdi_augmentation`, `datasynthesizer2017`, `privbayes2017`,
`xgboost2016`, `shap2017`, `treeshap2020`, `pinn2019`, `sklearn2011`. No entry
is cited that is not used, and no claim in the text is attributed to a source
that was not consulted.

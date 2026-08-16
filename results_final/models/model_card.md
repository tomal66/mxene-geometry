# Final real-only interpretation model

**Purpose.** Characterise multivariate descriptor associations across the whole compiled corpus. This model is NOT an estimate of predictive generalisation; that role belongs to the paper-grouped cross-validation in `results/02_real_only_fold_metrics.csv`.

- Estimator: XGBoost regressor, seed 42
- Training rows: 118 (all usable real observations; no synthetic rows)
- Papers represented: 40
- Descriptors: 12 (9 numeric, 3 categorical -> 21 encoded columns)
- Dropped as constant in this corpus: ['synthesis_family']
- Hyper-parameters (paper-grouped inner search): {'subsample': 0.85, 'reg_lambda': 3.0, 'reg_alpha': 0.1, 'n_estimators': 150, 'min_child_weight': 2, 'max_depth': 4, 'learning_rate': 0.02, 'colsample_bytree': 0.6}
- In-sample fit (NOT a generalisation estimate): R2 = 0.624, MAE = 45.0 F/g

Target leakage control: volumetric and areal capacitance, all extraction confidence/provenance metadata, free-text rationales and publication identifiers are excluded from the feature set (see `results/00_leakage_audit.csv`).
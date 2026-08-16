"""Grouped cross-validation, leakage-safe encoding and XGBoost fitting."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import RANDOM_STATE

MISSING_CAT = "__missing__"


# --------------------------------------------------------------------- splits

def make_group_kfold(groups: pd.Series, target_folds: int = 5) -> tuple[GroupKFold, int, str]:
    n_groups = int(pd.Series(groups).nunique())
    n = min(target_folds, n_groups)
    reason = (
        f"{n} folds used: {n_groups} unique papers available, target was {target_folds}."
        if n < target_folds
        else f"{n} folds used ({n_groups} unique papers available)."
    )
    return GroupKFold(n_splits=n), n, reason


# ------------------------------------------------------------------- encoding

@dataclass
class FoldEncoder:
    """One-hot encoder for categoricals, fitted on TRAIN ROWS ONLY."""

    cat_cols: list[str]
    num_cols: list[str]
    ohe: OneHotEncoder | None = None
    feature_names_: list[str] = field(default_factory=list)
    ohe_to_source_: dict[str, str] = field(default_factory=dict)

    def fit(self, X: pd.DataFrame) -> "FoldEncoder":
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if self.cat_cols:
            self.ohe.fit(X[self.cat_cols].astype(object).fillna(MISSING_CAT).astype(str))
            enc_names = list(self.ohe.get_feature_names_out(self.cat_cols))
        else:
            enc_names = []
        self.feature_names_ = list(self.num_cols) + enc_names
        self.ohe_to_source_ = {n: n for n in self.num_cols}
        for n in enc_names:
            src = next(c for c in self.cat_cols if n.startswith(c + "_"))
            self.ohe_to_source_[n] = src
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = [X[self.num_cols].astype(float).reset_index(drop=True)] if self.num_cols else []
        if self.cat_cols:
            arr = self.ohe.transform(X[self.cat_cols].astype(object).fillna(MISSING_CAT).astype(str))
            parts.append(pd.DataFrame(arr, columns=self.ohe.get_feature_names_out(self.cat_cols)))
        out = pd.concat(parts, axis=1)
        out.index = X.index
        return out[self.feature_names_]


# -------------------------------------------------------------------- metrics

def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan
    return {"r2": r2, "mae": float(mean_absolute_error(y_true, y_pred)), "rmse": rmse,
            "n_test": int(len(y_true))}


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    agg = df.groupby(by, dropna=False).agg(
        n_folds=("r2", "size"),
        r2_mean=("r2", "mean"), r2_sd=("r2", "std"), r2_median=("r2", "median"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"), mae_median=("mae", "median"),
        rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"), rmse_median=("rmse", "median"),
    ).reset_index()
    return agg


# ------------------------------------------------------------------- xgboost

BASE_PARAMS = dict(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbosity=0,
)

PARAM_SPACE = {
    "n_estimators": [150, 250, 400, 600],
    "max_depth": [2, 3, 4],
    "learning_rate": [0.02, 0.05, 0.08, 0.12],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "min_child_weight": [1, 2, 4],
    "reg_lambda": [1.0, 3.0, 10.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}

DEFAULT_PARAMS = dict(
    n_estimators=400, max_depth=3, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.8, min_child_weight=2,
    reg_lambda=3.0, reg_alpha=0.0,
)


def tune_xgb(X: pd.DataFrame, y: np.ndarray, groups, n_iter: int = 30,
             seed: int = RANDOM_STATE) -> dict:
    """Inner-loop hyper-parameter search. Only ever called on training rows."""
    n_groups = int(pd.Series(groups).nunique())
    if n_groups < 3 or len(y) < 20:
        return dict(DEFAULT_PARAMS)
    inner = GroupKFold(n_splits=min(3, n_groups))
    search = RandomizedSearchCV(
        XGBRegressor(**BASE_PARAMS),
        PARAM_SPACE,
        n_iter=n_iter,
        cv=inner,
        scoring="neg_mean_absolute_error",
        random_state=seed,
        n_jobs=1,
        refit=False,
        error_score="raise",
    )
    search.fit(X, y, groups=groups)
    return dict(search.best_params_)


def fit_xgb(X: pd.DataFrame, y: np.ndarray, params: dict, seed: int = RANDOM_STATE) -> XGBRegressor:
    p = dict(BASE_PARAMS)
    p.update(params)
    p["random_state"] = seed
    model = XGBRegressor(**p)
    model.fit(X, y)
    return model

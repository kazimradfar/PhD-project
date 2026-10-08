"""Leakage-safe preprocessing steps. Every step is fitted on the training fold only, because the
whole sklearn Pipeline is refitted inside each cross-validation / bootstrap iteration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from .combat import ComBat


class Imputer(BaseEstimator, TransformerMixin):
    """Chained-equations imputation (sklearn IterativeImputer) returning a DataFrame.

    Non-numeric pass-through columns (e.g. the batch label) are left untouched. For a full
    multiple-imputation analysis, run the pipeline with several ``random_state`` values and pool.
    """

    def __init__(self, passthrough: tuple = ("batch",), random_state: int = 0, max_iter: int = 10):
        self.passthrough = passthrough
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, X: pd.DataFrame, y=None):
        self.cols_ = [c for c in X.columns if c not in self.passthrough]
        self.imp_ = IterativeImputer(random_state=self.random_state, max_iter=self.max_iter, sample_posterior=False)
        self.imp_.fit(X[self.cols_])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        out[self.cols_] = self.imp_.transform(X[self.cols_])
        return out


class Harmonize(BaseEstimator, TransformerMixin):
    """Apply ComBat to radiomic columns (prefix-based); drops the batch column afterwards."""

    def __init__(self, radiomic_prefixes: tuple = ("wb_", "vwm_", "sd_", "range_"), batch_col: str = "batch",
                 enabled: bool = True, min_batch_size: int = 20):
        self.radiomic_prefixes = radiomic_prefixes
        self.batch_col = batch_col
        self.enabled = enabled
        self.min_batch_size = min_batch_size

    def _rad(self, X):
        return [c for c in X.columns if c.startswith(self.radiomic_prefixes)]

    def fit(self, X: pd.DataFrame, y=None):
        self.rad_ = self._rad(X)
        self.cb_ = None
        if self.enabled and self.batch_col in X and self.rad_ and X[self.batch_col].nunique() > 1:
            self.cb_ = ComBat(batch_col=self.batch_col, min_batch_size=self.min_batch_size).fit(X[self.rad_ + [self.batch_col]])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        if self.cb_ is not None:
            out[self.rad_] = self.cb_.transform(X[self.rad_ + [self.batch_col]])
        return out.drop(columns=[self.batch_col], errors="ignore")


class VarianceCorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove near-constant radiomic features, then one of each pair with |Spearman rho| > threshold
    (the one with the higher mean absolute correlation with all other features)."""

    def __init__(self, radiomic_prefixes: tuple = ("wb_", "vwm_", "sd_", "range_"), threshold: float = 0.90,
                 min_rel_std: float = 1e-6):
        self.radiomic_prefixes = radiomic_prefixes
        self.threshold = threshold
        self.min_rel_std = min_rel_std

    def fit(self, X: pd.DataFrame, y=None):
        rad = [c for c in X.columns if c.startswith(self.radiomic_prefixes)]
        R = X[rad]
        sd = R.std()
        scale = R.abs().mean().replace(0, 1)
        keep = [c for c in rad if sd[c] / scale[c] > self.min_rel_std]
        if len(keep) > 1:
            corr = R[keep].rank().corr().abs().to_numpy().copy()
            np.fill_diagonal(corr, 0)
            mean_corr = corr.mean(axis=0)
            alive = np.ones(len(keep), bool)
            for i, j in sorted(zip(*np.where(np.triu(corr) > self.threshold)), key=lambda ij: -corr[ij]):
                if alive[i] and alive[j]:
                    alive[i if mean_corr[i] >= mean_corr[j] else j] = False
            keep = [c for c, a in zip(keep, alive) if a]
        self.drop_ = [c for c in rad if c not in keep]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(columns=self.drop_, errors="ignore")


class RadiomicSelector(BaseEstimator, TransformerMixin):
    """LASSO selection of radiomic features (Cox for survival, logistic for binary outcomes).

    mode = "score": replace selected radiomic features by one radiomic score (linear predictor);
    mode = "select": keep the selected radiomic features themselves (for tree models / SVMs).
    Non-radiomic columns (reference predictors) are passed through unchanged.
    """

    def __init__(self, task: str = "survival", mode: str = "score", alpha: float = 0.15,
                 radiomic_prefixes: tuple = ("wb_", "vwm_", "sd_", "range_"), max_features: int = 10):
        self.task = task
        self.mode = mode
        self.alpha = alpha
        self.radiomic_prefixes = radiomic_prefixes
        self.max_features = max_features

    def fit(self, X: pd.DataFrame, y):
        self.rad_ = [c for c in X.columns if c.startswith(self.radiomic_prefixes)]
        self.selected_, self.coef_ = [], np.array([])
        if not self.rad_:
            return self
        self.scaler_ = StandardScaler().fit(X[self.rad_])
        Z = self.scaler_.transform(X[self.rad_])
        if self.task == "survival":
            from sksurv.linear_model import CoxnetSurvivalAnalysis
            m = CoxnetSurvivalAnalysis(l1_ratio=1.0, alphas=[self.alpha], max_iter=100000)
            m.fit(Z, y)
            coef = m.coef_.ravel()
        else:
            from .models import logistic
            m = logistic(1.0, C=10.0 / max(self.alpha * len(y), 1e-6))
            m.fit(Z, y)
            coef = m.coef_.ravel()
        order = np.argsort(-np.abs(coef))
        idx = [i for i in order if abs(coef[i]) > 0][: self.max_features]
        self.selected_ = [self.rad_[i] for i in idx]
        self.coef_ = coef[idx]
        self.sel_idx_ = idx
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        other = X.drop(columns=self.rad_, errors="ignore")
        if not self.selected_:
            if self.mode == "score":
                other = other.assign(rad_score=0.0)
            return other
        if self.mode == "select":
            return pd.concat([other, X[self.selected_]], axis=1)
        Z = self.scaler_.transform(X[self.rad_])[:, self.sel_idx_]
        return other.assign(rad_score=Z @ self.coef_)


class ToArray(BaseEstimator, TransformerMixin):
    """Standardize and convert to a float array (keeps feature names for explanations)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.columns_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.scaler_.transform(X[self.columns_])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_)

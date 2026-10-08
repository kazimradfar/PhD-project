"""ComBat harmonization with separate fit/transform (parametric empirical Bayes; Johnson et al. 2007).

Fitting on the training fold and applying the stored parameters to the test fold avoids
information leakage (proposal: Data analysis, reproducibility and harmonization). Batches unseen during fitting cannot be harmonized.

Model for feature g, batch i, sample j:
    y_ijg = alpha_g + X_ij beta_g + gamma_ig + delta_ig * eps_ijg
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def _ab_prior(delta_hat: np.ndarray) -> tuple[float, float]:
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    s2 = max(s2, 1e-12)
    return (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2


class ComBat(BaseEstimator, TransformerMixin):
    """sklearn-style ComBat. ``X`` must be a DataFrame containing ``batch_col`` (and covariates).

    The batch and covariate columns are removed from the output.
    """

    def __init__(self, batch_col: str = "batch", covariates: tuple = (), min_batch_size: int = 2,
                 max_iter: int = 1000, tol: float = 1e-6):
        self.batch_col = batch_col
        self.covariates = covariates
        self.min_batch_size = min_batch_size
        self.max_iter = max_iter
        self.tol = tol

    def _design(self, X: pd.DataFrame, batches) -> np.ndarray:
        B = np.column_stack([(X[self.batch_col].to_numpy() == b).astype(float) for b in batches])
        if self.covariates:
            C = X[list(self.covariates)].to_numpy(dtype=float)
            return np.column_stack([B, C])
        return B

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.features_ = [c for c in X.columns if c != self.batch_col and c not in self.covariates]
        Y = X[self.features_].to_numpy(dtype=float)  # n x g
        self.batches_ = sorted(X[self.batch_col].astype(str).unique())
        X[self.batch_col] = X[self.batch_col].astype(str)
        counts = X[self.batch_col].value_counts()
        small = counts[counts < self.min_batch_size]
        if len(small):
            raise ValueError(f"ComBat: batches below minimum size {self.min_batch_size}: {small.to_dict()}")
        n, g = Y.shape
        D = self._design(X, self.batches_)
        coef, *_ = np.linalg.lstsq(D, Y, rcond=None)           # (nb + nc) x g
        nb = len(self.batches_)
        n_i = np.array([counts[b] for b in self.batches_], dtype=float)
        self.alpha_ = (n_i / n) @ coef[:nb]                     # grand mean per feature
        self.beta_ = coef[nb:] if self.covariates else np.zeros((0, g))
        fitted = D @ coef
        self.var_ = ((Y - fitted) ** 2).sum(axis=0) / n
        self.var_[self.var_ <= 0] = 1e-12
        Z = self._standardize(X, Y)
        self.gamma_, self.delta_ = {}, {}
        gamma_hat = np.vstack([Z[X[self.batch_col].to_numpy() == b].mean(axis=0) for b in self.batches_])
        delta_hat = np.vstack([Z[X[self.batch_col].to_numpy() == b].var(axis=0, ddof=1) for b in self.batches_])
        delta_hat[delta_hat <= 0] = 1e-12
        for bi, b in enumerate(self.batches_):
            zb = Z[X[self.batch_col].to_numpy() == b]
            nbi = zb.shape[0]
            g_bar, t2 = gamma_hat[bi].mean(), max(gamma_hat[bi].var(ddof=1), 1e-12)
            a, bb = _ab_prior(delta_hat[bi])
            g_new, d_new = gamma_hat[bi].copy(), delta_hat[bi].copy()
            for _ in range(self.max_iter):
                g_upd = (nbi * t2 * gamma_hat[bi] + d_new * g_bar) / (nbi * t2 + d_new)
                ss = ((zb - g_upd) ** 2).sum(axis=0)
                d_upd = (bb + 0.5 * ss) / (nbi / 2.0 + a - 1.0)
                change = max(np.max(np.abs(g_upd - g_new) / (np.abs(g_new) + 1e-12)),
                             np.max(np.abs(d_upd - d_new) / (np.abs(d_new) + 1e-12)))
                g_new, d_new = g_upd, d_upd
                if change < self.tol:
                    break
            self.gamma_[b], self.delta_[b] = g_new, d_new
        return self

    def _standardize(self, X: pd.DataFrame, Y: np.ndarray) -> np.ndarray:
        mean = np.tile(self.alpha_, (len(X), 1))
        if self.covariates:
            mean = mean + X[list(self.covariates)].to_numpy(dtype=float) @ self.beta_
        return (Y - mean) / np.sqrt(self.var_)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[self.batch_col] = X[self.batch_col].astype(str)
        unseen = set(X[self.batch_col]) - set(self.batches_)
        if unseen:
            raise ValueError(f"ComBat: batches not seen during fit cannot be harmonized: {unseen}")
        Y = X[self.features_].to_numpy(dtype=float)
        Z = self._standardize(X, Y)
        out = np.empty_like(Z)
        bvals = X[self.batch_col].to_numpy()
        for b in self.batches_:
            m = bvals == b
            out[m] = (Z[m] - self.gamma_[b]) / np.sqrt(self.delta_[b])
        mean = np.tile(self.alpha_, (len(X), 1))
        if self.covariates:
            mean = mean + X[list(self.covariates)].to_numpy(dtype=float) @ self.beta_
        res = out * np.sqrt(self.var_) + mean
        return pd.DataFrame(res, columns=self.features_, index=X.index)

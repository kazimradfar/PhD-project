"""Validation: nested cross-validation, bootstrap optimism correction, temporal split
(proposal: Validation strategy)."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


def survival_scorer(tau: float):
    """GridSearchCV scorer: Uno's C truncated at tau, using the training-fold censoring distribution
    approximated by the evaluated fold itself (sksurv requires y_train; we pass the fold's y)."""
    from sksurv.metrics import concordance_index_ipcw

    def score(estimator, X, y):
        risk = estimator.predict(X)
        try:
            return concordance_index_ipcw(y, y, risk, tau=tau)[0]
        except ValueError:
            return np.nan
    return score


def event_strata(y) -> np.ndarray:
    if y.dtype.names:
        return y[y.dtype.names[0]].astype(int)
    return np.asarray(y).astype(int)


def tuned(pipeline, param_grid: dict, scoring, inner_folds: int = 5, seed: int = 0):
    """Wrap a pipeline in an inner-CV grid search (the tuning is then repeated in every outer split)."""
    cv = StratifiedKFold(inner_folds, shuffle=True, random_state=seed)
    return GridSearchCV(pipeline, param_grid, scoring=scoring, cv=cv, refit=True, error_score=np.nan, n_jobs=1)


def nested_cv(estimator, X: pd.DataFrame, y, metric: Callable, outer_folds: int = 5, repeats: int = 1,
              seed: int = 0) -> pd.DataFrame:
    """Outer-loop performance of an estimator (which may itself be a GridSearchCV)."""
    rows = []
    strata = event_strata(y)
    for r in range(repeats):
        cv = StratifiedKFold(outer_folds, shuffle=True, random_state=seed + r)
        for k, (tr, te) in enumerate(cv.split(X, strata)):
            est = clone(estimator).fit(X.iloc[tr], y[tr])
            rows.append({"repeat": r, "fold": k, **metric(est, X.iloc[tr], y[tr], X.iloc[te], y[te])})
    return pd.DataFrame(rows)


def bootstrap_optimism(estimator, X: pd.DataFrame, y, metric: Callable, n_boot: int = 200, seed: int = 0,
                       verbose: bool = False) -> dict:
    """Harrell's bootstrap optimism correction of the whole modelling pipeline.

    ``metric(est, X_train, y_train, X_eval, y_eval) -> dict`` of performance measures.
    Returns apparent, mean optimism and optimism-corrected values for each measure, plus the
    per-replicate table.
    """
    rng = np.random.default_rng(seed)
    full = clone(estimator).fit(X, y)
    apparent = metric(full, X, y, X, y)
    rows = []
    n = len(X)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(event_strata(y[idx]))) < 2:
            continue
        try:
            est = clone(estimator).fit(X.iloc[idx], y[idx])
            boot = metric(est, X.iloc[idx], y[idx], X.iloc[idx], y[idx])
            orig = metric(est, X.iloc[idx], y[idx], X, y)
        except Exception as exc:  # a replicate can fail (e.g., no events in a stratum)
            if verbose:
                print(f"bootstrap {b} failed: {exc}")
            continue
        rows.append({**{f"boot_{k}": v for k, v in boot.items()}, **{f"orig_{k}": v for k, v in orig.items()}})
    reps = pd.DataFrame(rows)
    res = {"n_boot_ok": len(reps)}
    for k, v in apparent.items():
        opt = (reps[f"boot_{k}"] - reps[f"orig_{k}"]).mean()
        res[f"{k}_apparent"] = v
        res[f"{k}_optimism"] = float(opt)
        res[f"{k}_corrected"] = float(v - opt)
    return {"summary": res, "replicates": reps, "final_model": full}


def temporal_split(df: pd.DataFrame, date_col: str = "date_cycle1", train_fraction: float = 0.70):
    """Earlier ``train_fraction`` of patients (by treatment date) vs the most recent remainder."""
    order = pd.to_datetime(df[date_col]).sort_values().index
    cut = int(round(len(order) * train_fraction))
    return order[:cut], order[cut:]


def leave_one_center_out(df: pd.DataFrame, center_col: str = "center"):
    for c in sorted(df[center_col].unique()):
        yield c, df.index[df[center_col] != c], df.index[df[center_col] == c]

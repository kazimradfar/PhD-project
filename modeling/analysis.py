"""Shared building blocks for steps 9-13: estimators, metrics and the H2 comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from .metrics import binary_metrics, harrell_c, td_auc, uno_c
from .models import PARAM_GRIDS, build_pipeline, classifier, survival_estimator
from .validation import survival_scorer, tuned


def make_estimator(task: str, model_name: str, predictor_set: str, cfg: dict, harmonize: bool, tune: bool = True):
    """Leakage-safe pipeline (impute -> ComBat -> filter -> LASSO -> scale -> model), optionally tuned
    by inner cross-validation. Regression models get the radiomic SCORE, the other models the selected
    radiomic FEATURES (proposal: Variables)."""
    seed = cfg["random_seed"]
    if task == "survival":
        est, mode, grid_key = survival_estimator(model_name, seed), ("score" if model_name == "coxnet" else "select"), model_name
    else:
        est = classifier(model_name, seed)
        mode = "score" if model_name == "logistic" else "select"
        grid_key = {"logistic": "logistic", "rf": "rf_clf", "xgb": "xgb", "svm": "svm_clf"}[model_name]
    pipe = build_pipeline(est, task=task, radiomics_mode=mode, harmonize=harmonize and predictor_set != "reference",
                          corr_threshold=cfg["corr_threshold"], lasso_alpha=cfg["lasso_alpha"], random_state=seed,
                          min_batch_size=cfg["min_batch_size"])
    if not tune:
        return pipe
    grid = dict(PARAM_GRIDS.get(grid_key, {}))
    if task == "binary" and model_name == "xgb":
        grid = {"model__learning_rate": [0.01, 0.05], "model__max_depth": [1, 2, 3]}
    scoring = survival_scorer(cfg["truncation_months"]) if task == "survival" else "neg_log_loss"
    return tuned(pipe, grid, scoring, cfg["inner_folds"], seed) if grid else pipe


def metric_fn(task: str, cfg: dict):
    tau, times = cfg["truncation_months"], cfg["td_auc_months"]

    def surv(est, Xtr, ytr, Xte, yte):
        r = est.predict(Xte)
        out = {"uno_c": uno_c(ytr, yte, r, tau), "harrell_c": harrell_c(yte, r)}
        out.update(td_auc(ytr, yte, r, times))
        return out

    def binm(est, Xtr, ytr, Xte, yte):
        p = est.predict_proba(Xte)[:, 1]
        if len(np.unique(yte)) < 2:
            return {"auc": np.nan, "brier": float(np.mean((p - yte) ** 2))}
        return binary_metrics(yte, p)

    return surv if task == "survival" else binm


def cox_partial_loglik(X: np.ndarray, y, beta: np.ndarray) -> float:
    """Breslow partial log-likelihood of a Cox model (used for likelihood-ratio tests)."""
    e, t = y[y.dtype.names[0]].astype(bool), y[y.dtype.names[1]].astype(float)
    eta = X @ beta
    ll = 0.0
    for i in np.where(e)[0]:
        at_risk = t >= t[i]
        ll += eta[i] - np.log(np.exp(eta[at_risk]).sum())
    return float(ll)


def delta_c_bootstrap(Xr: pd.DataFrame, Xc: pd.DataFrame, y, est_ref, est_comb, tau: float, n_boot: int, seed: int) -> dict:
    """Paired bootstrap of the optimism-corrected difference in Uno's C (combined - reference).

    Both pipelines are refitted on the same bootstrap sample; optimism = mean(bootstrap - original)
    difference; the percentile interval is centred on the corrected estimate.
    """
    fr, fc = clone(est_ref).fit(Xr, y), clone(est_comb).fit(Xc, y)
    c_ref, c_comb = uno_c(y, y, fr.predict(Xr), tau), uno_c(y, y, fc.predict(Xc), tau)
    apparent = c_comb - c_ref
    rng = np.random.default_rng(seed)
    opt, orig = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            br, bc = clone(est_ref).fit(Xr.iloc[i], y[i]), clone(est_comb).fit(Xc.iloc[i], y[i])
            db = uno_c(y[i], y[i], bc.predict(Xc.iloc[i]), tau) - uno_c(y[i], y[i], br.predict(Xr.iloc[i]), tau)
            do = uno_c(y[i], y, bc.predict(Xc), tau) - uno_c(y[i], y, br.predict(Xr), tau)
        except Exception:
            continue
        opt.append(db - do)
        orig.append(do)
    corrected = apparent - float(np.mean(opt))
    lo, hi = np.percentile(np.asarray(orig) - np.mean(orig) + corrected, [2.5, 97.5])
    return {"c_reference_apparent": c_ref, "c_combined_apparent": c_comb, "delta_c_apparent": apparent,
            "delta_c_corrected": corrected, "ci_low": float(lo), "ci_high": float(hi), "n_boot_ok": len(opt)}


def delta_auc_bootstrap(Xr, Xc, y, est_ref, est_comb, n_boot: int, seed: int) -> dict:
    """Same as delta_c_bootstrap for the binary endpoint (AUC)."""
    fr, fc = clone(est_ref).fit(Xr, y), clone(est_comb).fit(Xc, y)
    auc = lambda m, X, yy: roc_auc_score(yy, m.predict_proba(X)[:, 1])
    apparent = auc(fc, Xc, y) - auc(fr, Xr, y)
    rng = np.random.default_rng(seed)
    opt, orig = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            br, bc = clone(est_ref).fit(Xr.iloc[i], y[i]), clone(est_comb).fit(Xc.iloc[i], y[i])
            db = auc(bc, Xc.iloc[i], y[i]) - auc(br, Xr.iloc[i], y[i])
            do = auc(bc, Xc, y) - auc(br, Xr, y)
        except Exception:
            continue
        opt.append(db - do)
        orig.append(do)
    corrected = apparent - float(np.mean(opt))
    lo, hi = np.percentile(np.asarray(orig) - np.mean(orig) + corrected, [2.5, 97.5])
    return {"delta_auc_apparent": apparent, "delta_auc_corrected": corrected, "ci_low": float(lo),
            "ci_high": float(hi), "n_boot_ok": len(opt)}

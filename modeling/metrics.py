"""Performance measures (proposal: Data analysis, model performance)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import (concordance_index_censored, concordance_index_ipcw, cumulative_dynamic_auc,
                            integrated_brier_score)
from sksurv.nonparametric import kaplan_meier_estimator


def _et(y):
    return y[y.dtype.names[0]].astype(bool), y[y.dtype.names[1]].astype(float)


def uno_c(y_train, y_test, risk, tau: float) -> float:
    """Uno's C truncated at tau (same time unit as y). Higher risk = earlier event."""
    return float(concordance_index_ipcw(y_train, y_test, risk, tau=tau)[0])


def harrell_c(y, risk) -> float:
    e, t = _et(y)
    return float(concordance_index_censored(e, t, risk)[0])


def td_auc(y_train, y_test, risk, times) -> dict:
    e, t = _et(y_test)
    times = [x for x in times if t[e].min() < x < t.max()]
    if not times:
        return {}
    aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, risk, times)
    return {**{f"auc_{x:g}": float(a) for x, a in zip(times, aucs)}, "auc_mean": float(mean_auc)}


def ibs(y_train, y_test, surv_funcs, times) -> float:
    """Integrated Brier score (IPCW); ``surv_funcs`` from model.predict_survival_function."""
    e, t = _et(y_test)
    tr_e, tr_t = _et(y_train)
    lo, hi = max(t.min(), tr_t.min()), min(t.max(), tr_t.max())
    times = np.asarray([x for x in times if lo < x < hi])
    if len(times) < 2:
        return np.nan
    preds = np.vstack([[fn(x) for x in times] for fn in surv_funcs])
    return float(integrated_brier_score(y_train, y_test, preds, times))


def calibration_slope_survival(y, linear_predictor) -> float:
    """Slope of a Cox model refitted on the model's linear predictor (ideal = 1)."""
    lp = np.asarray(linear_predictor, float).reshape(-1, 1)
    return float(CoxPHSurvivalAnalysis().fit(lp, y).coef_[0])


def calibration_at_time(y, surv_prob_at_t: np.ndarray, t: float, groups: int = 5) -> pd.DataFrame:
    """Predicted vs Kaplan-Meier observed survival at time t, by risk group (quantiles)."""
    e, time = _et(y)
    q = pd.qcut(surv_prob_at_t, groups, labels=False, duplicates="drop")
    rows = []
    for g in np.unique(q):
        m = q == g
        km_t, km_s = kaplan_meier_estimator(e[m], time[m])
        obs = km_s[km_t <= t][-1] if np.any(km_t <= t) else 1.0
        rows.append({"group": int(g), "n": int(m.sum()), "predicted": float(np.mean(surv_prob_at_t[m])), "observed": float(obs)})
    return pd.DataFrame(rows)


def net_benefit_survival(y, risk_at_t: np.ndarray, t: float, thresholds=np.arange(0.05, 0.81, 0.05)) -> pd.DataFrame:
    """Decision-curve analysis for a time-to-event outcome at time t (Kaplan-Meier based).

    risk_at_t = predicted probability of the event by time t. For threshold pt:
    NB = P(high) * (1 - S_high(t)) - P(high) * S_high(t) * pt / (1 - pt).
    """
    e, time = _et(y)
    n = len(time)

    def surv_at(mask):
        if mask.sum() == 0:
            return 1.0
        km_t, km_s = kaplan_meier_estimator(e[mask], time[mask])
        return km_s[km_t <= t][-1] if np.any(km_t <= t) else 1.0

    s_all = surv_at(np.ones(n, bool))
    rows = []
    for pt in thresholds:
        high = risk_at_t >= pt
        p_high = high.mean()
        s_high = surv_at(high)
        nb = p_high * (1 - s_high) - p_high * s_high * pt / (1 - pt)
        nb_all = (1 - s_all) - s_all * pt / (1 - pt)
        rows.append({"threshold": float(pt), "net_benefit_model": float(nb), "net_benefit_treat_all": float(nb_all),
                     "net_benefit_treat_none": 0.0})
    return pd.DataFrame(rows)


def binary_metrics(y_true, prob) -> dict:
    """AUC, Brier score, calibration intercept and slope for a binary outcome."""
    from sklearn.linear_model import LogisticRegression
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(prob, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    slope_m = LogisticRegression(C=1e10, max_iter=1000).fit(logit, y_true)
    # calibration-in-the-large: intercept with logit as offset
    from scipy.optimize import minimize_scalar
    def nll(a):
        z = a + logit.ravel()
        return np.sum(np.log1p(np.exp(z)) - y_true * z)
    intercept = minimize_scalar(nll).x
    return {"auc": float(roc_auc_score(y_true, p)), "brier": float(np.mean((p - y_true) ** 2)),
            "cal_intercept": float(intercept), "cal_slope": float(slope_m.coef_[0, 0])}

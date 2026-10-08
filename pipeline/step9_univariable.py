"""Step 9 (H1, specific objective 1): univariable association of each reproducible radiomic feature
with OS, PFS, recurrence (cause-specific) and initial response, with Benjamini-Hochberg FDR.

    python -m pipeline.step9_univariable

Features are standardized, so hazard / odds ratios are per 1 SD. ComBat is applied on the whole
cohort here because this is a descriptive association analysis (no prediction is evaluated).
Writes results/h1_univariable.csv.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

from modeling.combat import ComBat
from modeling.data import harmonize_enabled, load_config, radiomic_columns
from pipeline.common import RESULTS_DIR, log, setup_logging


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    n = np.sum(~np.isnan(p))
    order = np.argsort(np.where(np.isnan(p), np.inf, p))
    q = np.full_like(p, np.nan)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        if np.isnan(p[i]):
            continue
        k = n - rank
        prev = min(prev, p[i] * n / k)
        q[i] = prev
    return q


def cox_uni(x, event, time):
    """Wald test of a 1-variable Cox model (Newton-Raphson on the Breslow partial likelihood)."""
    order = np.argsort(-time)
    x, event, time = x[order], event[order], time[order]
    b = 0.0
    for _ in range(50):
        w = np.exp(b * x)
        s0, s1, s2 = np.cumsum(w), np.cumsum(w * x), np.cumsum(w * x * x)
        # ties: risk set = all with time >= t_i
        idx = np.searchsorted(-time, -time, side="right") - 1
        s0, s1, s2 = s0[idx], s1[idx], s2[idx]
        grad = np.sum(event * (x - s1 / s0))
        info = np.sum(event * (s2 / s0 - (s1 / s0) ** 2))
        if info <= 0:
            return np.nan, np.nan
        step = grad / info
        b += step
        if abs(step) < 1e-8:
            break
    se = 1 / np.sqrt(info)
    return b, se


def logit_uni(x, y):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=1e8, max_iter=1000).fit(x.reshape(-1, 1), y)
    p = m.predict_proba(x.reshape(-1, 1))[:, 1]
    info = np.sum(p * (1 - p) * x * x) - np.sum(p * (1 - p) * x) ** 2 / np.sum(p * (1 - p))
    return m.coef_[0, 0], 1 / np.sqrt(info)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = pd.read_csv(RESULTS_DIR / "analysis_table.csv")
    rad = radiomic_columns(df, cfg)
    R = df[rad].copy()
    if harmonize_enabled(df, cfg):
        R = ComBat(batch_col="batch", min_batch_size=cfg["min_batch_size"]).fit_transform(
            pd.concat([R.fillna(R.median()), df[["batch"]]], axis=1))
    Z = (R - R.mean()) / R.std(ddof=0).replace(0, np.nan)

    outcomes = {"os": ("os_event", "os_months"), "pfs": ("pfs_event", "pfs_months"),
                "recurrence_cs": ("recur_event", "recur_months")}
    rows = []
    for f in rad:
        x_all = Z[f].to_numpy()
        for name, (ev, tm) in outcomes.items():
            d = df[[ev, tm]].assign(x=x_all).dropna()
            if name == "recurrence_cs":
                d[ev] = (d[ev] == 1).astype(int)   # cause-specific: death = censored
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b, se = cox_uni(d["x"].to_numpy(), d[ev].to_numpy().astype(bool), d[tm].to_numpy(float))
            rows.append({"feature": f, "outcome": name, "n": len(d), "events": int(d[ev].sum()),
                         "effect": "HR per SD", "estimate": np.exp(b), "ci_low": np.exp(b - 1.96 * se),
                         "ci_high": np.exp(b + 1.96 * se), "p": 2 * norm.sf(abs(b / se)) if se == se else np.nan})
        d = df[["response_12w"]].assign(x=x_all).dropna()
        if d["response_12w"].nunique() == 2:
            b, se = logit_uni(d["x"].to_numpy(), d["response_12w"].astype(int).to_numpy())
            rows.append({"feature": f, "outcome": "response", "n": len(d), "events": int(d["response_12w"].sum()),
                         "effect": "OR per SD", "estimate": np.exp(b), "ci_low": np.exp(b - 1.96 * se),
                         "ci_high": np.exp(b + 1.96 * se), "p": 2 * norm.sf(abs(b / se))})
    res = pd.DataFrame(rows)
    res["q_fdr"] = res.groupby("outcome")["p"].transform(lambda s: pd.Series(bh(s.to_numpy()), index=s.index))
    res = res.sort_values(["outcome", "p"])
    res.to_csv(RESULTS_DIR / "h1_univariable.csv", index=False)
    for o, g in res.groupby("outcome"):
        sig = g[g["q_fdr"] < 0.05]
        log.info("%-14s %3d features tested, %3d with FDR < 0.05%s", o, len(g), len(sig),
                 f" (top: {sig.iloc[0]['feature']})" if len(sig) else "")
    log.info("H1 is supported if at least one feature has FDR < 0.05 for at least one outcome")


if __name__ == "__main__":
    main()

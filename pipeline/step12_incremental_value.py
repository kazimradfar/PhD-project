"""Step 12 (H2, specific objective 3): does adding the radiomic features improve on the reference model?

    python -m pipeline.step12_incremental_value [--n-boot 200]

For OS and PFS (penalized Cox, radiomic score):
  * optimism-corrected difference in Uno's C (combined - reference) with a paired bootstrap 95% CI;
    H2 (OS) is supported if the corrected gain is >= 0.03 and the CI excludes 0;
  * likelihood-ratio test of the nested Cox models (reference vs reference + radiomic score);
  * decision-curve analysis at 12 months.
For initial response: optimism-corrected difference in AUC (penalized logistic regression).
Also exports results/analysis_recurrence.csv for the competing-risk analysis in R/.
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis

from modeling.analysis import cox_partial_loglik, delta_auc_bootstrap, delta_c_bootstrap, make_estimator
from modeling.data import choose_reference, get_xy, harmonize_enabled, load_config, radiomic_columns, transform_reference
from modeling.metrics import net_benefit_survival
from modeling.transformers import Harmonize, Imputer, RadiomicSelector
from pipeline.common import RESULTS_DIR, log, setup_logging


def radiomic_score(X: pd.DataFrame, y, task: str, cfg: dict, harm: bool) -> np.ndarray:
    """Radiomic score fitted on the full data (used only for the LRT and the R export)."""
    Xi = Imputer(random_state=cfg["random_seed"]).fit_transform(X)
    Xh = Harmonize(enabled=harm, min_batch_size=cfg["min_batch_size"]).fit_transform(Xi)
    sel = RadiomicSelector(task=task, mode="score", alpha=cfg["lasso_alpha"]).fit(Xh, y)
    return sel.transform(Xh)["rad_score"].to_numpy()


def lrt(Xr: pd.DataFrame, rad: np.ndarray, y) -> dict:
    A = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(Xr))
    B = np.column_stack([A, (rad - rad.mean()) / (rad.std() or 1)])
    m0, m1 = CoxPHSurvivalAnalysis().fit(A, y), CoxPHSurvivalAnalysis().fit(B, y)
    stat = 2 * (cox_partial_loglik(B, y, m1.coef_) - cox_partial_loglik(A, y, m0.coef_))
    return {"lrt_chi2": stat, "lrt_df": 1, "lrt_p": float(chi2.sf(max(stat, 0), 1)),
            "note": "radiomic score selected on the same data: p-value is optimistic"}


def risk_at(pipe, X, t):
    sf = pipe.predict_survival_function(X)
    return np.array([1 - fn(t) for fn in sf])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = pd.read_csv(RESULTS_DIR / "analysis_table.csv")
    ref, _ = choose_reference(df, cfg)
    harm = harmonize_enabled(df, cfg)
    nb, seed, tau, t_dca = args.n_boot or cfg["n_bootstrap"], cfg["random_seed"], cfg["truncation_months"], cfg["dca_month"]
    rows, dca = [], []
    for ep in ("os", "pfs"):
        Xr, y, task, d = get_xy(df, ep, "reference", ref, cfg)
        Xc, _, _, _ = get_xy(df, ep, "combined", ref, cfg)
        er = make_estimator(task, "coxnet", "reference", cfg, harm, tune=False)
        ec = make_estimator(task, "coxnet", "combined", cfg, harm, tune=False)
        res = delta_c_bootstrap(Xr, Xc, y, er, ec, tau, nb, seed)
        res.update(lrt(Xr, radiomic_score(Xc, y, task, cfg, harm), y))
        if ep == "os":
            res["h2_supported"] = bool(res["delta_c_corrected"] >= cfg["delta_c_min"] and res["ci_low"] > 0)
        rows.append({"endpoint": ep, "metric": "Uno C", **res})
        log.info("%s: delta C corrected = %.3f (95%% CI %.3f to %.3f); LRT p = %.3g", ep,
                 res["delta_c_corrected"], res["ci_low"], res["ci_high"], res["lrt_p"])
        fr, fc = er.fit(Xr, y), ec.fit(Xc, y)
        for name, r in (("reference", risk_at(fr, Xr, t_dca)), ("combined", risk_at(fc, Xc, t_dca))):
            nbf = net_benefit_survival(y, r, t_dca)
            dca.append(nbf.assign(endpoint=ep, model=name))
    Xr, y, task, _ = get_xy(df, "response", "reference", ref, cfg)
    Xc, _, _, _ = get_xy(df, "response", "combined", ref, cfg)
    res = delta_auc_bootstrap(Xr, Xc, y, make_estimator(task, "logistic", "reference", cfg, harm, tune=False),
                              make_estimator(task, "logistic", "combined", cfg, harm, tune=False), nb, seed)
    rows.append({"endpoint": "response", "metric": "AUC", **res})
    log.info("response: delta AUC corrected = %.3f (95%% CI %.3f to %.3f)", res["delta_auc_corrected"], res["ci_low"], res["ci_high"])

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "h2_incremental_value.csv", index=False)
    pd.concat(dca).to_csv(RESULTS_DIR / "h2_decision_curve.csv", index=False)

    # ---- export for the competing-risk analysis of recurrence (R/recurrence_competing_risks.R)
    r = df[df["response_12w"] == 1].dropna(subset=["recur_event", "recur_months"]).reset_index(drop=True)
    if len(r) >= 20 and (r["recur_event"] == 1).sum() >= 5:
        from sksurv.util import Surv
        cols = ref + radiomic_columns(r, cfg) + ["batch"]
        X = transform_reference(r[cols], cfg)
        y_cs = Surv.from_arrays(event=(r["recur_event"] == 1).to_numpy(), time=r["recur_months"].astype(float).to_numpy())
        out = transform_reference(r[["patient_id", "recur_months", "recur_event", *ref]], cfg)
        out["rad_score"] = radiomic_score(X, y_cs, "survival", cfg, harm)
        out.to_csv(RESULTS_DIR / "analysis_recurrence.csv", index=False)
        log.info("Exported %d responders for the Fine-Gray analysis (Rscript R/recurrence_competing_risks.R)", len(out))
    else:
        log.warning("Too few responders/recurrences for the competing-risk analysis")


if __name__ == "__main__":
    main()

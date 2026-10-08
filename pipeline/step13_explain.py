"""Step 13 (H3): which radiomic features matter most, and are the explanations stable and plausible?

    python -m pipeline.step13_explain --endpoint os [--model xgb] [--n-boot 50]

* SHAP values of the combined model (global importance and per-patient explanations);
* stability: how often each radiomic feature is selected across bootstrap refits (H3 needs >= 70%)
  and the rank correlation of SHAP importance between refits;
* plausibility: observed direction of effect vs the prespecified expected direction (config).
Writes results/h3_shap_<endpoint>.csv, h3_importance_<endpoint>.csv, h3_stability_<endpoint>.csv.
LIME / SurvLIME and SurvSHAP(t) for selected patients: see notebooks (optional packages 'lime', 'survshap').
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone

from modeling.analysis import make_estimator
from modeling.data import choose_reference, get_xy, harmonize_enabled, load_config
from modeling.explain import direction_check, global_importance, importance_stability, shap_values
from pipeline.common import RESULTS_DIR, log, setup_logging


def selected_radiomics(pipe) -> list[str]:
    return list(pipe.named_steps["select"].selected_)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--endpoint", default="os")
    ap.add_argument("--model", default=None, help="default: xgb (fast exact TreeSHAP); any model of the config works")
    ap.add_argument("--n-boot", type=int, default=50)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = pd.read_csv(RESULTS_DIR / "analysis_table.csv")
    ref, _ = choose_reference(df, cfg)
    harm = harmonize_enabled(df, cfg)
    X, y, task, d = get_xy(df, args.endpoint, "combined", ref, cfg)
    model = args.model or "xgb"
    est = make_estimator(task, model, "combined", cfg, harm, tune=False)
    fit = clone(est).fit(X, y)
    sv = shap_values(fit, X, seed=cfg["random_seed"])
    sv.insert(0, "patient_id", d["patient_id"].to_numpy())
    imp = global_importance(sv.drop(columns="patient_id"))
    sv.to_csv(RESULTS_DIR / f"h3_shap_{args.endpoint}.csv", index=False)

    rng = np.random.default_rng(cfg["random_seed"])
    sel_count, imps = {}, []
    for _ in range(args.n_boot):
        i = rng.integers(0, len(y), len(y))
        try:
            f = clone(est).fit(X.iloc[i], y[i])
        except Exception:
            continue
        for s in selected_radiomics(f):
            sel_count[s] = sel_count.get(s, 0) + 1
        s_b = shap_values(f, X.iloc[np.unique(i)[:80]], background_size=30, seed=cfg["random_seed"])
        imps.append(global_importance(s_b))
    freq = pd.Series(sel_count, dtype=float) / max(len(imps), 1)
    stab = importance_stability(imps) if len(imps) > 1 else pd.DataFrame()
    table = pd.DataFrame({"shap_mean_abs": imp}).join(freq.rename("selection_frequency"), how="left")
    table["is_radiomic"] = table.index.str.startswith(tuple(cfg["radiomic_prefixes"]))
    table.to_csv(RESULTS_DIR / f"h3_importance_{args.endpoint}.csv")
    stab.to_csv(RESULTS_DIR / f"h3_stability_{args.endpoint}.csv")

    Z = pd.DataFrame(fit[:-1].transform(X), columns=fit.named_steps["scale"].columns_)
    dirs = direction_check(sv.drop(columns="patient_id"), Z, cfg.get("expected_directions", {}))
    dirs.to_csv(RESULTS_DIR / f"h3_direction_{args.endpoint}.csv", index=False)

    rad = table[table["is_radiomic"]].sort_values("shap_mean_abs", ascending=False)
    stable = rad[rad["selection_frequency"].fillna(0) >= cfg["h3_min_selection_frequency"]]
    log.info("Most important features (%s, %s): %s", args.endpoint, model, ", ".join(imp.index[:5]))
    log.info("Radiomic features selected in >= %.0f%% of bootstrap refits: %d %s", 100 * cfg["h3_min_selection_frequency"],
             len(stable), list(stable.index[:5]))
    if not stab.empty:
        log.info("Mean pairwise Spearman of SHAP importance across refits: %.2f", stab["mean_pairwise_spearman"].iloc[0])
    log.info("H3 also needs the review of the leading features by two nuclear medicine physicians")


if __name__ == "__main__":
    main()

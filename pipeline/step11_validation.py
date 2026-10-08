"""Step 11: internal validation of one model family (bootstrap optimism correction of the whole
pipeline), temporal validation (TRIPOD 2b, earlier 70% vs most recent 30%) and leave-one-center-out.

    python -m pipeline.step11_validation --endpoint os --model coxnet [--n-boot 200]

Writes results/validation_<endpoint>_<model>.csv.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone

from modeling.analysis import make_estimator, metric_fn
from modeling.data import choose_reference, get_xy, harmonize_enabled, load_config
from modeling.validation import bootstrap_optimism
from pipeline.common import RESULTS_DIR, log, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--endpoint", default="os")
    ap.add_argument("--model", default="coxnet")
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--no-tune", action="store_true")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = pd.read_csv(RESULTS_DIR / "analysis_table.csv")
    ref, _ = choose_reference(df, cfg)
    harm = harmonize_enabled(df, cfg)
    n_boot = args.n_boot or cfg["n_bootstrap"]
    rows = []
    for ps in cfg["predictor_sets"]:
        X, y, task, d = get_xy(df, args.endpoint, ps, ref, cfg)
        est = make_estimator(task, args.model, ps, cfg, harm, tune=not args.no_tune)
        mf = metric_fn(task, cfg)
        bo = bootstrap_optimism(est, X, y, mf, n_boot=n_boot, seed=cfg["random_seed"])
        rows.append({"predictor_set": ps, "analysis": "bootstrap", "n": len(y), **bo["summary"]})

        tr = np.where(d["temporal_set"] == "development")[0]
        te = np.where(d["temporal_set"] == "validation")[0]
        if len(te):
            # ComBat cannot harmonize scanners that are absent from the development period
            est_t = make_estimator(task, args.model, ps, cfg,
                                   harm and set(d["batch"].iloc[te]) <= set(d["batch"].iloc[tr]), tune=not args.no_tune)
            fit = clone(est_t).fit(X.iloc[tr], y[tr])
            rows.append({"predictor_set": ps, "analysis": "temporal", "n_dev": len(tr), "n_val": len(te),
                         **mf(fit, X.iloc[tr], y[tr], X.iloc[te], y[te])})
        for c in sorted(d["center"].unique()):
            tr_c, te_c = np.where(d["center"] != c)[0], np.where(d["center"] == c)[0]
            if len(tr_c) < 20 or len(te_c) < 10:
                continue
            est_c = make_estimator(task, args.model, ps, cfg, False, tune=not args.no_tune)  # unseen center: no ComBat
            try:
                fit = clone(est_c).fit(X.iloc[tr_c], y[tr_c])
                rows.append({"predictor_set": ps, "analysis": f"leave_out_{c}", "n_dev": len(tr_c), "n_val": len(te_c),
                             **mf(fit, X.iloc[tr_c], y[tr_c], X.iloc[te_c], y[te_c])})
            except Exception as exc:
                log.warning("leave-one-center-out (%s) failed: %s", c, exc)
        log.info("%s done", ps)
    out = pd.DataFrame(rows)
    f = RESULTS_DIR / f"validation_{args.endpoint}_{args.model}.csv"
    out.to_csv(f, index=False)
    key = [c for c in ("predictor_set", "analysis", "n", "n_val", "uno_c_corrected", "uno_c", "auc_corrected", "auc")
           if c in out]
    print(out[key].round(3).to_string(index=False))
    n_val = int((df["temporal_set"] == "validation").sum())
    ev = int(df.loc[df["temporal_set"] == "validation", "os_event"].sum())
    if ev < 100:
        log.info("Temporal validation set has %d OS events (< 100): report it as exploratory (proposal)", ev)
    log.info("Saved %s (temporal validation n = %d)", f.name, n_val)


if __name__ == "__main__":
    main()

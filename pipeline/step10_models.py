"""Step 10 (specific objective 2): repeated nested cross-validation of all model families with the
three predictor sets (reference, radiomics only, combined) for OS, PFS and initial response.

    python -m pipeline.step10_models                 # full analysis (slow: hours)
    python -m pipeline.step10_models --quick         # 1 repeat, no tuning (to check that it runs)

Radiomics-only models show the predictive value of the radiomic features by themselves; combined
models show how much they add to the reference variables. Writes results/models_cv.csv.
"""
import argparse

import pandas as pd

from modeling.analysis import make_estimator, metric_fn
from modeling.data import choose_reference, get_xy, harmonize_enabled, load_config
from modeling.validation import nested_cv
from pipeline.common import RESULTS_DIR, log, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--quick", action="store_true", help="1 repeat, no inner tuning")
    ap.add_argument("--endpoints", nargs="*", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = pd.read_csv(RESULTS_DIR / "analysis_table.csv")
    ref, _ = choose_reference(df, cfg)
    harm = harmonize_enabled(df, cfg)
    rows = []
    for ep in args.endpoints or cfg["endpoints"]:
        for ps in cfg["predictor_sets"]:
            X, y, task, _ = get_xy(df, ep, ps, ref, cfg)
            names = cfg["survival_models"] if task == "survival" else cfg["binary_models"]
            for m in names:
                est = make_estimator(task, m, ps, cfg, harm, tune=not args.quick)
                res = nested_cv(est, X, y, metric_fn(task, cfg), cfg["outer_folds"],
                                1 if args.quick else cfg["cv_repeats"], cfg["random_seed"])
                summ = res.drop(columns=["repeat", "fold"]).agg(["mean", "std"]).T
                for metric, r in summ.iterrows():
                    rows.append({"endpoint": ep, "predictor_set": ps, "model": m, "metric": metric,
                                 "mean": r["mean"], "sd": r["std"], "n": len(y)})
                main_metric = "uno_c" if task == "survival" else "auc"
                log.info("%-8s %-10s %-8s %s = %.3f", ep, ps, m, main_metric, summ.loc[main_metric, "mean"])
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "models_cv.csv", index=False)
    log.info("Saved results/models_cv.csv")


if __name__ == "__main__":
    main()

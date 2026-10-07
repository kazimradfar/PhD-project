"""Step 6 - Inter-reader reproducibility filter (ICC).

For the random subset of 30 patients whose masks were corrected independently by two readers,
ICC(2,1) - two-way random effects, absolute agreement, single measure (Koo & Li 2016) - is
computed for every patient-level feature. Features with ICC < 0.75 are excluded from modelling.

Run steps 3-5 with --mask reader1 AND --mask reader2 first.

Output: results/icc_pet_bw<bw>.csv            (ICC with 95 % CI for every feature)
        results/reproducible_features_pet_bw<bw>.txt   (features kept for modelling)

Usage:
    python -m pipeline.step6_reproducibility
    python -m pipeline.step6_reproducibility --bin-width 0.25 --threshold 0.75
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.common import RESULTS_DIR, log, results_dir, setup_logging


def icc_a1(y: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """ICC(A,1) with McGraw & Wong (1996) confidence interval. y: (n subjects, k raters)."""
    n, k = y.shape
    grand = y.mean()
    ss_r = k * ((y.mean(axis=1) - grand) ** 2).sum()
    ss_c = n * ((y.mean(axis=0) - grand) ** 2).sum()
    ss_e = ((y - grand) ** 2).sum() - ss_r - ss_c
    msr, msc, mse = ss_r / (n - 1), ss_c / (k - 1), ss_e / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom <= 0:
        return np.nan, np.nan, np.nan
    icc = (msr - mse) / denom
    if icc >= 1 or mse == 0:
        return float(icc), np.nan, np.nan

    a = k * icc / (n * (1 - icc))
    b = 1 + k * icc * (n - 1) / (n * (1 - icc))
    v = (a * msc + b * mse) ** 2 / ((a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1)))
    f_s = stats.f.ppf(1 - alpha / 2, n - 1, v)
    f_i = stats.f.ppf(1 - alpha / 2, v, n - 1)
    lower = n * (msr - f_s * mse) / (f_s * (k * msc + (k * n - k - n) * mse) + n * msr)
    upper = n * (f_i * msr - mse) / (k * msc + (k * n - k - n) * mse + n * f_i * msr)
    return float(icc), float(lower), float(upper)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin-width", type=float, default=0.5)
    ap.add_argument("--modality", default="pet", choices=["pet", "ct"])
    ap.add_argument("--threshold", type=float, default=0.75)
    args = ap.parse_args()
    setup_logging()

    tag = f"{args.modality}_bw{args.bin_width:g}"
    r1 = pd.read_csv(results_dir("reader1") / f"patient_features_{tag}.csv").set_index("patient_id")
    r2 = pd.read_csv(results_dir("reader2") / f"patient_features_{tag}.csv").set_index("patient_id")
    common = r1.index.intersection(r2.index)
    if len(common) < 10:
        log.warning("Only %d patients segmented by both readers - ICC will be very uncertain", len(common))
    cols = [c for c in r1.columns.intersection(r2.columns) if pd.api.types.is_numeric_dtype(r1[c])]

    rows = []
    for c in cols:
        y = np.column_stack([r1.loc[common, c].to_numpy(float), r2.loc[common, c].to_numpy(float)])
        y = y[~np.isnan(y).any(axis=1)]
        icc, lo, hi = icc_a1(y) if len(y) >= 3 and np.ptp(y) > 0 else (np.nan, np.nan, np.nan)
        rows.append({"feature": c, "n": len(y), "icc": icc, "ci_lower": lo, "ci_upper": hi})
    res = pd.DataFrame(rows)
    res["reproducible"] = res["icc"] >= args.threshold

    res.to_csv(RESULTS_DIR / f"icc_{tag}.csv", index=False)
    keep = res.loc[res["reproducible"], "feature"]
    (RESULTS_DIR / f"reproducible_features_{tag}.txt").write_text("\n".join(keep) + "\n")

    rad = res[res["feature"].str.startswith(("wb_", "vwm_", "sd_", "range_"))]
    log.info("%d patients, %d radiomic features: %d with ICC >= %.2f (%.0f %%)", len(common), len(rad),
             rad["reproducible"].sum(), args.threshold, 100 * rad["reproducible"].mean() if len(rad) else 0)
    for c in ("suvmean", "psma_tv_ml"):
        if c in set(res["feature"]):
            r = res.set_index("feature").loc[c]
            log.info("  reference variable %-10s ICC %.3f (%.3f-%.3f)", c, r.icc, r.ci_lower, r.ci_upper)


if __name__ == "__main__":
    main()

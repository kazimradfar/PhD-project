"""Step 5 - Patient-level imaging feature table (no outcome data).

Lesion features are summarised per patient in the three pre-specified ways of the proposal:
  * wb_    : features of the whole-body union of all lesions        (from step 4, level wholebody)
  * vwm_   : volume-weighted mean of each feature across lesions
  * sd_    : standard deviation of each feature between lesions     (>= 2 lesions)
  * range_ : range (max - min) of each feature between lesions      (>= 2 lesions)
Texture features are summarised only over lesions that have them (>= 64 voxels).
Conventional PET parameters (step 3) are added without prefix.

Imaging and outcome data are kept apart on purpose (blinding / pre-registration):
outcomes are built in step 7 and joined only in the modelling phase.

Output: results/<mask>/patient_features_pet_bw<bw>.csv

Usage:
    python -m pipeline.step5_build_dataset                      # reader1, bw 0.5
    python -m pipeline.step5_build_dataset --mask reader2
    python -m pipeline.step5_build_dataset --bin-width 0.25
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

from pipeline.common import MASK_FILES, log, results_dir, setup_logging

FEATURE_PREFIXES = ("original_", "log-", "wavelet-", "square_", "exponential_", "gradient_")


def _feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def aggregate_lesions(lesions: pd.DataFrame) -> pd.DataFrame:
    """Volume-weighted mean, SD and range of every lesion feature, per patient."""
    feats = _feature_cols(lesions)
    out = {}
    for pid, g in lesions.groupby("patient_id"):
        x = g[feats].to_numpy(dtype=float)
        w = g["lesion_volume_ml"].to_numpy(dtype=float)[:, None]
        valid = ~np.isnan(x)
        n = valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns (no textured lesion)
            vwm = np.where(n >= 1, np.nansum(np.where(valid, x * w, 0), axis=0) / np.where(valid, w, 0).sum(axis=0), np.nan)
            sd = np.where(n >= 2, np.nanstd(x, axis=0, ddof=1), np.nan)
            rng = np.where(n >= 2, np.nanmax(x, axis=0) - np.nanmin(x, axis=0), np.nan)
        row = {}
        for prefix, vals in (("vwm_", vwm), ("sd_", sd), ("range_", rng)):
            row.update({prefix + f: v for f, v in zip(feats, vals)})
        row["n_lesions_with_texture"] = int(g["has_texture"].sum())
        out[pid] = row
    return pd.DataFrame.from_dict(out, orient="index")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mask", choices=list(MASK_FILES), default="reader1")
    ap.add_argument("--bin-width", type=float, default=0.5)
    ap.add_argument("--modality", default="pet", choices=["pet", "ct"])
    args = ap.parse_args()
    setup_logging()

    rdir = results_dir(args.mask)
    tag = f"{args.modality}_bw{args.bin_width:g}"
    conv_f, wb_f, les_f = (rdir / "conventional_patient.csv", rdir / f"radiomics_{args.modality}_wholebody_bw{args.bin_width:g}.csv",
                           rdir / f"radiomics_{args.modality}_lesion_bw{args.bin_width:g}.csv")
    for f in (conv_f, wb_f, les_f):
        if not f.exists():
            raise SystemExit(f"{f} missing - run steps 3 and 4 with the same --mask / --bin-width first")

    conv = pd.read_csv(conv_f).set_index("patient_id")
    wb = pd.read_csv(wb_f)
    wb = wb.set_index("patient_id")[_feature_cols(wb)].add_prefix("wb_")
    agg = aggregate_lesions(pd.read_csv(les_f))

    data = pd.concat([conv, wb, agg], axis=1).copy()
    data.index.name = "patient_id"
    out = rdir / f"patient_features_{tag}.csv"
    data.reset_index().to_csv(out, index=False)
    n_rad = sum(c.startswith(("wb_", "vwm_", "sd_", "range_")) for c in data.columns)
    log.info("Wrote %s: %d patients, %d radiomic + %d conventional columns",
             out, len(data), n_rad, conv.shape[1])


if __name__ == "__main__":
    main()

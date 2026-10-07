"""Step 5 - Build the patient-level dataset for machine learning.

Merges, per patient:
  * conventional PET parameters                  (prefix: none)
  * inter-lesion heterogeneity of SUVmax / volume (prefix: het_)
  * whole-body PET / CT radiomics                (prefix: wbpet_, wbct_)
  * radiomics of the hottest and largest lesion  (prefix: hot_, big_)
  * clinical data and the response label          (data/clinical.csv, optional)

Response label (PCWG3): PSA50 = PSA decline >= 50 % from baseline, computed from
`psa_baseline` and `psa_followup` (e.g. 12 weeks or after 2 cycles) if both columns exist.

Output: results/patient_level_dataset.csv

Usage:
    python -m pipeline.step5_build_dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline.common import DATA_DIR, RESULTS_DIR, log, setup_logging

FEATURE_PREFIXES = ("original_", "log-", "wavelet-")


def _read(name: str) -> pd.DataFrame | None:
    f = RESULTS_DIR / name
    if not f.exists() or f.stat().st_size == 0:
        return None
    df = pd.read_csv(f)
    return df if not df.empty else None


def _features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    return df.set_index("patient_id")[cols].add_prefix(prefix)


def heterogeneity(lesions: pd.DataFrame) -> pd.DataFrame:
    g = lesions.groupby("patient_id")
    out = pd.DataFrame({
        "het_suvmax_cv": g["suvmax"].std() / g["suvmax"].mean(),
        "het_suvmax_range": g["suvmax"].max() - g["suvmax"].min(),
        "het_suvmax_min": g["suvmax"].min(),
        "het_volume_cv": g["volume_ml"].std() / g["volume_ml"].mean(),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clinical", type=Path, default=DATA_DIR / "clinical.csv")
    args = ap.parse_args()
    setup_logging()

    conv = _read("conventional_patient.csv")
    if conv is None:
        raise SystemExit("Run step 3 first (results/conventional_patient.csv missing)")
    parts = [conv.set_index("patient_id").drop(columns=["mask_used"], errors="ignore")]

    les = _read("conventional_lesion.csv")
    if les is not None:
        parts.append(heterogeneity(les))

    for name, prefix in (("radiomics_pet_wholebody.csv", "wbpet_"), ("radiomics_ct_wholebody.csv", "wbct_")):
        df = _read(name)
        if df is not None:
            parts.append(_features(df, prefix))

    rl = _read("radiomics_pet_lesion.csv")
    if rl is not None:
        hottest = rl.loc[rl.groupby("patient_id")["lesion_suvmax"].idxmax()]
        largest = rl.loc[rl.groupby("patient_id")["lesion_volume_ml"].idxmax()]
        parts += [_features(hottest, "hot_"), _features(largest, "big_")]

    data = pd.concat(parts, axis=1).copy()
    data.index.name = "patient_id"

    if args.clinical.exists():
        clin = pd.read_csv(args.clinical).set_index("patient_id")
        if {"psa_baseline", "psa_followup"} <= set(clin.columns):
            clin["psa_change_pct"] = 100 * (clin["psa_followup"] - clin["psa_baseline"]) / clin["psa_baseline"]
            clin["psa50_response"] = (clin["psa_change_pct"] <= -50).astype("Int64")
            clin.loc[clin["psa_change_pct"].isna(), "psa50_response"] = pd.NA
        data = clin.join(data, how="right").copy()
    else:
        log.warning("No %s - dataset has no clinical data / response label yet", args.clinical)

    out = RESULTS_DIR / "patient_level_dataset.csv"
    data.reset_index().to_csv(out, index=False)
    n_rad = sum(c.startswith(("wbpet_", "wbct_", "hot_", "big_")) for c in data.columns)
    log.info("Wrote %s: %d patients x %d columns (%d radiomics features)", out.name, len(data), data.shape[1], n_rad)


if __name__ == "__main__":
    main()

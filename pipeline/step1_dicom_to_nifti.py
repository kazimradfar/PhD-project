"""Step 1 - Convert baseline PSMA PET/CT DICOM to NIfTI, PET in SUVbw.

Input : data/raw/<patient_id>/PET/*.dcm and data/raw/<patient_id>/CT/*.dcm
Output: data/nifti/<patient_id>/PET_SUV.nii.gz, CT.nii.gz
        results/acquisition_info.csv  (injected activity, uptake time, scanner ... for QC / harmonisation)

Usage:
    python -m pipeline.step1_dicom_to_nifti
    python -m pipeline.step1_dicom_to_nifti --patients RLT001 RLT002
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import SimpleITK as sitk

from pipeline.common import DATA_DIR, RESULTS_DIR, list_patients, log, read_dicom_series, setup_logging, suv_bw_factor


def convert_patient(pid: str, raw_dir: Path, out_dir: Path) -> dict:
    pdir = raw_dir / pid
    odir = out_dir / pid
    odir.mkdir(parents=True, exist_ok=True)

    pet_bqml, pet_headers = read_dicom_series(pdir / "PET")
    factor, info = suv_bw_factor(pet_headers)
    pet_suv = sitk.Cast(pet_bqml * factor, sitk.sitkFloat32)
    sitk.WriteImage(pet_suv, str(odir / "PET_SUV.nii.gz"), useCompression=True)

    ct_dir = pdir / "CT"
    if ct_dir.exists():
        ct, _ = read_dicom_series(ct_dir)
        sitk.WriteImage(sitk.Cast(ct, sitk.sitkInt16), str(odir / "CT.nii.gz"), useCompression=True)
    else:
        log.warning("%s: no CT folder, skipping CT", pid)

    stats = sitk.StatisticsImageFilter()
    stats.Execute(pet_suv)
    log.info("%s: %.0f MBq, uptake %.1f min, %.1f kg, image SUVmax %.1f",
             pid, info["injected_MBq"], info["uptake_min"], info["weight_kg"], stats.GetMaximum())
    return {"patient_id": pid, **info}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=DATA_DIR / "raw")
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR / "nifti")
    ap.add_argument("--patients", nargs="*")
    args = ap.parse_args()
    setup_logging()

    rows = []
    for pid in args.patients or list_patients(args.raw_dir):
        try:
            rows.append(convert_patient(pid, args.raw_dir, args.out_dir))
        except Exception as exc:  # keep going, report at the end
            log.error("%s: FAILED - %s", pid, exc)
            rows.append({"patient_id": pid, "error": str(exc)})

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "acquisition_info.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()

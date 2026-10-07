"""Step 3 - Conventional PSMA PET parameters (the clinical baseline model).

Every radiomics model must be compared against these simple, established predictors.

Reference model (fixed in the proposal): whole-body SUVmean + PSMA-TV (+ LDH, Hb from the CRF).
Descriptive: SUVmax, SUVpeak, TL-PSMA (= PSMA-TV x SUVmean), number of lesions, liver SUVmean,
                tumour-to-liver ratio.
Lesion level  : SUVmax, SUVpeak, SUVmean, volume, TL-PSMA, centroid.

Input : data/nifti/<pid>/PET_SUV.nii.gz, data/masks/<pid>/tumor_mask_<mask>.nii.gz
Output: results/<mask>/conventional_patient.csv, results/<mask>/conventional_lesion.csv

Usage:
    python -m pipeline.step3_conventional_metrics                  # reader1 (main analysis)
    python -m pipeline.step3_conventional_metrics --mask auto      # sensitivity: uncorrected masks
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from pipeline.common import (DATA_DIR, MASK_FILES, label_lesions, list_patients, load_tumor_mask, log,
                             results_dir, setup_logging, voxel_volume_ml)


def suvpeak_map(pet: sitk.Image, volume_ml: float = 1.0) -> np.ndarray:
    """Mean SUV in a 1 mL sphere (diameter ~1.24 cm) centred on every voxel (PERCIST definition)."""
    r_mm = (3 * volume_ml * 1000 / (4 * np.pi)) ** (1 / 3)
    sx, sy, sz = pet.GetSpacing()
    rz, ry, rx = (int(np.ceil(r_mm / s)) for s in (sz, sy, sx))
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    kernel = ((zz * sz) ** 2 + (yy * sy) ** 2 + (xx * sx) ** 2) <= r_mm ** 2
    kernel = kernel / kernel.sum()
    return ndimage.convolve(sitk.GetArrayFromImage(pet), kernel, mode="constant")


def patient_metrics(pid: str, args) -> tuple[dict, list[dict]]:
    pet = sitk.ReadImage(str(args.nifti_dir / pid / "PET_SUV.nii.gz"), sitk.sitkFloat32)
    mask = load_tumor_mask(args.mask_dir / pid, args.mask)
    labels, lesions = label_lesions(mask, pet)

    pet_arr = sitk.GetArrayViewFromImage(pet)
    lab_arr = sitk.GetArrayViewFromImage(labels)
    peak = suvpeak_map(pet)
    vox_ml = voxel_volume_ml(pet)

    for d in lesions:
        sel = lab_arr == d["lesion_id"]
        d["suvpeak"] = float(peak[sel].max())
        d["tl_psma"] = d["suvmean"] * d["volume_ml"]
        d["patient_id"] = pid

    tumor = lab_arr > 0
    vals = pet_arr[tumor]
    row = {"patient_id": pid, "n_lesions": len(lesions)}
    if vals.size:
        row.update({
            "suvmax": float(vals.max()),
            "suvpeak": float(peak[tumor].max()),
            "suvmean": float(vals.mean()),
            "psma_tv_ml": float(vals.size * vox_ml),
            "tl_psma": float(vals.sum() * vox_ml),  # = PSMA-TV x SUVmean
        })

    liver_f = args.mask_dir / pid / "liver_ref.nii.gz"
    if liver_f.exists():
        liver = sitk.GetArrayViewFromImage(sitk.ReadImage(str(liver_f))) > 0
        row["liver_suvmean"] = float(pet_arr[liver].mean())
        if vals.size:
            row["tumor_to_liver_ratio"] = row["suvmean"] / row["liver_suvmean"]
            row["frac_lesions_below_liver"] = float(np.mean([d["suvmax"] < row["liver_suvmean"] for d in lesions]))

    log.info("%s: %d lesions, SUVmax %.1f, PSMA-TV %.1f mL, TL-PSMA %.0f", pid, len(lesions),
             row.get("suvmax", np.nan), row.get("psma_tv_ml", 0), row.get("tl_psma", 0))
    return row, lesions


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nifti-dir", type=Path, default=DATA_DIR / "nifti")
    ap.add_argument("--mask-dir", type=Path, default=DATA_DIR / "masks")
    ap.add_argument("--mask", choices=list(MASK_FILES), default="reader1")
    ap.add_argument("--patients", nargs="*")
    args = ap.parse_args()
    setup_logging()

    patients, lesions = [], []
    for pid in args.patients or list_patients(args.mask_dir):
        if not (args.mask_dir / pid / MASK_FILES[args.mask]).exists():
            log.warning("%s: no %s mask, skipped", pid, args.mask)
            continue
        try:
            row, les = patient_metrics(pid, args)
            patients.append(row)
            lesions.extend(les)
        except Exception as exc:
            log.error("%s: FAILED - %s", pid, exc)

    out_dir = results_dir(args.mask)
    pd.DataFrame(patients).to_csv(out_dir / "conventional_patient.csv", index=False)
    les_df = pd.DataFrame(lesions)
    if not les_df.empty:
        les_df[["cx_mm", "cy_mm", "cz_mm"]] = pd.DataFrame(les_df.pop("centroid_mm").tolist(), index=les_df.index)
        les_df = les_df[["patient_id", "lesion_id"] + [c for c in les_df if c not in ("patient_id", "lesion_id")]]
    les_df.to_csv(out_dir / "conventional_lesion.csv", index=False)
    log.info("Wrote %s/conventional_patient.csv and conventional_lesion.csv (%d patients)", out_dir, len(patients))


if __name__ == "__main__":
    main()

"""Step 2 - Automatic whole-body tumour segmentation on [68Ga]Ga-PSMA-11 PET.

Two sources of the automatic mask (--method):
  * nnunet    : prediction of the nnU-Net model trained on PSMA-PET-CT-Lesions (main method of
                the proposal). Run nnU-Net first (see tools/prepare_nnunet_input.py) and put the
                predictions in data/nnunet_pred/<pid>.nii.gz.
  * threshold : SUV >= 3 (Seifert et al.) - fallback / for testing before nnU-Net is ready.

In both cases physiological PSMA uptake is removed with TotalSegmentator organ masks
(salivary and lacrimal glands, kidneys, spleen, bowel, bladder). Inside the liver only
uptake above liver SUVmean + 3 SD is kept, so liver metastases are not lost.
Components smaller than --min-volume mL are dropped.

Then (outside this script):
  1. Reader 1 corrects every mask in 3D Slicer, a senior nuclear-medicine physician reviews
     it -> save as data/masks/<pid>/tumor_mask_reader1.nii.gz  (main analysis)
  2. Reader 2 independently corrects the auto masks of 30 random patients
     (tools/select_icc_subset.py) -> data/masks/<pid>/tumor_mask_reader2.nii.gz

Input : data/nifti/<pid>/PET_SUV.nii.gz, data/totalseg/<pid>/*.nii.gz, [data/nnunet_pred/<pid>.nii.gz]
Output: data/masks/<pid>/tumor_mask_auto.nii.gz, liver_ref.nii.gz, results/segmentation_summary.csv

Usage:
    python -m pipeline.step2_segment_lesions --method nnunet
    python -m pipeline.step2_segment_lesions --method threshold --threshold 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

from pipeline.common import (DATA_DIR, MASK_FILES, RESULTS_DIR, label_lesions, list_patients, log,
                             resample_to_reference, setup_logging)

# TotalSegmentator structure names with physiological PSMA uptake / excretion.
# Salivary glands come from the "head_glands_cavities" task; the rest from the default "total" task.
PHYSIOLOGICAL_ORGANS = [
    "kidney_left", "kidney_right", "spleen", "urinary_bladder", "small_bowel", "duodenum", "colon", "stomach",
    "parotid_gland_left", "parotid_gland_right", "submandibular_gland_left", "submandibular_gland_right",
    "lacrimal_gland_left", "lacrimal_gland_right",  # only if your organ model provides them
]
LIVER = "liver"


def _load_organ(totalseg_dir: Path, name: str, ref: sitk.Image) -> sitk.Image | None:
    f = totalseg_dir / f"{name}.nii.gz"
    if not f.exists():
        return None
    return resample_to_reference(sitk.ReadImage(str(f), sitk.sitkUInt8), ref, is_mask=True) > 0


def liver_reference(liver: sitk.Image, pet: sitk.Image, erode_mm: float = 10.0) -> tuple[sitk.Image, float, float]:
    """Eroded liver VOI (avoids partial volume at the edges); returns mask, SUVmean, SUVsd."""
    radius = [max(1, int(round(erode_mm / s))) for s in pet.GetSpacing()]
    core = sitk.BinaryErode(liver, radius)
    vals = sitk.GetArrayViewFromImage(pet)[sitk.GetArrayViewFromImage(core) > 0]
    if vals.size < 50:
        core = liver
        vals = sitk.GetArrayViewFromImage(pet)[sitk.GetArrayViewFromImage(liver) > 0]
    # robust: drop the hottest 5 % (possible metastases) before computing the reference
    vals = vals[vals <= np.percentile(vals, 95)]
    return core, float(vals.mean()), float(vals.std())


def segment_patient(pid: str, args) -> dict:
    pet = sitk.ReadImage(str(args.nifti_dir / pid / "PET_SUV.nii.gz"), sitk.sitkFloat32)
    odir = args.mask_dir / pid
    odir.mkdir(parents=True, exist_ok=True)
    tdir = args.totalseg_dir / pid

    threshold = args.threshold
    liver_mean = liver_sd = np.nan
    liver = _load_organ(tdir, LIVER, pet) if tdir.exists() else None
    if liver is not None:
        core, liver_mean, liver_sd = liver_reference(liver, pet)
        sitk.WriteImage(core, str(odir / "liver_ref.nii.gz"), useCompression=True)
        if args.liver_factor and args.method == "threshold":
            threshold = max(threshold, args.liver_factor * liver_mean)

    if args.method == "nnunet":
        pred_f = args.nnunet_dir / f"{pid}.nii.gz"
        if not pred_f.exists():
            raise FileNotFoundError(f"{pred_f} not found - run nnU-Net inference first")
        pred = sitk.ReadImage(str(pred_f), sitk.sitkUInt8)
        mask = resample_to_reference(pred, pet, is_mask=True) > 0
    else:
        mask = pet >= threshold

    excluded = []
    if tdir.exists():
        dilate = [max(1, int(round(args.organ_margin / s))) for s in pet.GetSpacing()]
        for organ in PHYSIOLOGICAL_ORGANS:
            om = _load_organ(tdir, organ, pet)
            if om is None:
                continue
            om = sitk.BinaryDilate(om, dilate)
            mask = mask & sitk.Not(om)
            excluded.append(organ)
        if liver is not None:
            in_liver = mask & liver & (pet >= liver_mean + 3 * liver_sd)
            mask = (mask & sitk.Not(liver)) | in_liver
    else:
        log.warning("%s: no TotalSegmentator masks - physiological uptake is NOT removed, review carefully", pid)

    labels, lesions = label_lesions(sitk.Cast(mask, sitk.sitkUInt8), pet, args.min_volume)
    tumor = sitk.Cast(labels > 0, sitk.sitkUInt8)
    sitk.WriteImage(tumor, str(odir / MASK_FILES["auto"]), useCompression=True)

    log.info("%s: %s, %d lesions, PSMA-TV %.1f mL", pid, args.method, len(lesions),
             sum(d["volume_ml"] for d in lesions))
    return {"patient_id": pid, "method": args.method,
            "threshold_suv": threshold if args.method == "threshold" else np.nan,
            "liver_suvmean": liver_mean, "liver_suvsd": liver_sd,
            "n_lesions": len(lesions), "psma_tv_ml": sum(d["volume_ml"] for d in lesions),
            "excluded_organs": ";".join(excluded)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nifti-dir", type=Path, default=DATA_DIR / "nifti")
    ap.add_argument("--totalseg-dir", type=Path, default=DATA_DIR / "totalseg")
    ap.add_argument("--mask-dir", type=Path, default=DATA_DIR / "masks")
    ap.add_argument("--method", choices=["nnunet", "threshold"], default="nnunet")
    ap.add_argument("--nnunet-dir", type=Path, default=DATA_DIR / "nnunet_pred")
    ap.add_argument("--threshold", type=float, default=3.0, help="fixed SUV threshold (default 3)")
    ap.add_argument("--liver-factor", type=float, default=None,
                    help="threshold method only: threshold = max(threshold, factor x liver SUVmean)")
    ap.add_argument("--min-volume", type=float, default=0.5, help="minimum lesion volume in mL (default 0.5)")
    ap.add_argument("--organ-margin", type=float, default=5.0, help="dilation of organ masks in mm (default 5)")
    ap.add_argument("--patients", nargs="*")
    args = ap.parse_args()
    setup_logging()

    rows = []
    for pid in args.patients or list_patients(args.nifti_dir):
        try:
            rows.append(segment_patient(pid, args))
        except Exception as exc:
            log.error("%s: FAILED - %s", pid, exc)
            rows.append({"patient_id": pid, "error": str(exc)})
    RESULTS_DIR.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "segmentation_summary.csv", index=False)


if __name__ == "__main__":
    main()

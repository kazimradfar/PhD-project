"""Step 4 - Radiomics feature extraction with PyRadiomics (IBSI-compliant).

Two levels of analysis:
  * whole-body : all lesions together as one VOI (total tumour burden phenotype)
  * lesion     : every lesion separately (needed for heterogeneity / hottest-lesion analysis)

Modalities: PET (SUV) and, optionally, the low-dose CT (mask resampled to the CT grid).

Input : data/nifti/<pid>/PET_SUV.nii.gz, CT.nii.gz, data/masks/<pid>/tumor_mask[_reviewed].nii.gz
Output: results/radiomics_<modality>_<level>.csv   (one row per patient or per lesion)

Usage:
    python -m pipeline.step4_extract_radiomics                       # PET, whole-body + lesion
    python -m pipeline.step4_extract_radiomics --modalities PET CT --levels wholebody
    python -m pipeline.step4_extract_radiomics --lesion-min-volume 1.0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor

from pipeline.common import (DATA_DIR, PROJECT_ROOT, RESULTS_DIR, label_lesions, list_patients, load_tumor_mask,
                             log, resample_to_reference, setup_logging)

PARAMS = {"PET": PROJECT_ROOT / "configs" / "pet_params.yaml", "CT": PROJECT_ROOT / "configs" / "ct_params.yaml"}
IMAGE_FILE = {"PET": "PET_SUV.nii.gz", "CT": "CT.nii.gz"}


def _clean(result: dict, keep_diagnostics: bool) -> dict:
    out = {}
    for k, v in result.items():
        if k.startswith("diagnostics_") and not keep_diagnostics:
            continue
        try:
            arr = np.asarray(v)
            # a complex result (e.g. degenerate shape of a tiny VOI) is invalid -> NaN
            out[k] = float("nan") if np.iscomplexobj(arr) else float(arr)
        except (TypeError, ValueError):
            out[k] = str(v)
    return out


def extract_patient(pid: str, modality: str, levels: list[str], extractor, args) -> dict[str, list[dict]]:
    image_f = args.nifti_dir / pid / IMAGE_FILE[modality]
    if not image_f.exists():
        raise FileNotFoundError(image_f)
    image = sitk.ReadImage(str(image_f), sitk.sitkFloat32)
    pet = image if modality == "PET" else sitk.ReadImage(str(args.nifti_dir / pid / "PET_SUV.nii.gz"), sitk.sitkFloat32)

    mask, _ = load_tumor_mask(args.mask_dir / pid)
    labels, lesions = label_lesions(mask, pet)
    if modality != "PET":  # masks are defined on the PET grid
        labels = resample_to_reference(labels, image, is_mask=True)

    rows: dict[str, list[dict]] = {lvl: [] for lvl in levels}
    if "wholebody" in levels and lesions:
        wb = sitk.Cast(labels > 0, sitk.sitkUInt8)
        res = extractor.execute(image, wb, label=1)
        rows["wholebody"].append({"patient_id": pid, **_clean(res, args.diagnostics)})

    if "lesion" in levels:
        for d in lesions:
            if d["volume_ml"] < args.lesion_min_volume:
                continue
            try:
                res = extractor.execute(image, labels, label=d["lesion_id"])
            except ValueError as exc:  # e.g. lesion too small after resampling
                log.debug("%s lesion %d skipped: %s", pid, d["lesion_id"], exc)
                continue
            rows["lesion"].append({"patient_id": pid, "lesion_id": d["lesion_id"],
                                   "lesion_suvmax": d["suvmax"], "lesion_volume_ml": d["volume_ml"],
                                   **_clean(res, args.diagnostics)})
    log.info("%s %s: wholebody=%d, lesions=%d/%d", pid, modality, len(rows.get("wholebody", [])),
             len(rows.get("lesion", [])), len(lesions))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nifti-dir", type=Path, default=DATA_DIR / "nifti")
    ap.add_argument("--mask-dir", type=Path, default=DATA_DIR / "masks")
    ap.add_argument("--modalities", nargs="+", default=["PET"], choices=["PET", "CT"])
    ap.add_argument("--levels", nargs="+", default=["wholebody", "lesion"], choices=["wholebody", "lesion"])
    ap.add_argument("--lesion-min-volume", type=float, default=1.0,
                    help="only lesions >= this volume (mL) get lesion-level texture features (default 1.0)")
    ap.add_argument("--diagnostics", action="store_true", help="keep PyRadiomics diagnostics_* columns")
    ap.add_argument("--patients", nargs="*")
    args = ap.parse_args()
    setup_logging()
    radiomics.setVerbosity(logging.ERROR)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    logging.getLogger("pykwalify").setLevel(logging.ERROR)

    patients = args.patients or list_patients(args.mask_dir)
    RESULTS_DIR.mkdir(exist_ok=True)
    for modality in args.modalities:
        extractor = featureextractor.RadiomicsFeatureExtractor(str(PARAMS[modality]))
        log.info("%s: enabled image types %s", modality, list(extractor.enabledImagetypes))
        collected: dict[str, list[dict]] = {lvl: [] for lvl in args.levels}
        for pid in patients:
            try:
                for lvl, rows in extract_patient(pid, modality, args.levels, extractor, args).items():
                    collected[lvl].extend(rows)
            except Exception as exc:
                log.error("%s %s: FAILED - %s", pid, modality, exc)
        for lvl, rows in collected.items():
            out = RESULTS_DIR / f"radiomics_{modality.lower()}_{lvl}.csv"
            df = pd.DataFrame(rows)
            df.to_csv(out, index=False)
            n_feat = sum(c.startswith(("original_", "log-", "wavelet-")) for c in df.columns)
            log.info("Wrote %s (%d rows, %d features)", out.name, len(df), n_feat)


if __name__ == "__main__":
    main()

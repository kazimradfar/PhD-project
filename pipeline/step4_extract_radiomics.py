"""Step 4 - IBSI-standardised radiomics extraction with PyRadiomics.

As pre-specified in the proposal:
  * features are computed for every lesion (shape, first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM);
  * texture features only for lesions with >= 64 voxels after resampling - smaller lesions get
    only shape (burden) and first-order (intensity) features;
  * the whole-body union of all lesions is also extracted (first-order + texture; shape of a
    union of disconnected lesions has no geometric meaning and is not computed);
  * fixed bin width 0.5 SUV (0.25 and 1.0 in sensitivity analysis: --bin-width).

Input : data/nifti/<pid>/PET_SUV.nii.gz [CT.nii.gz], data/masks/<pid>/tumor_mask_<mask>.nii.gz
Output: results/<mask>/radiomics_<modality>_<level>_bw<bin width>.csv

Usage:
    python -m pipeline.step4_extract_radiomics                          # reader1, PET, bw 0.5
    python -m pipeline.step4_extract_radiomics --mask reader2           # ICC subset
    python -m pipeline.step4_extract_radiomics --bin-width 0.25         # sensitivity analysis
    python -m pipeline.step4_extract_radiomics --modalities PET CT
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from radiomics import featureextractor

from pipeline.common import (DATA_DIR, MASK_FILES, PROJECT_ROOT, label_lesions, list_patients, load_tumor_mask, log,
                             resample_to_reference, results_dir, setup_logging)

PARAMS = {"PET": PROJECT_ROOT / "configs" / "pet_params.yaml", "CT": PROJECT_ROOT / "configs" / "ct_params.yaml"}
IMAGE_FILE = {"PET": "PET_SUV.nii.gz", "CT": "CT.nii.gz"}
FEATURE_PREFIXES = ("original_", "log-", "wavelet-", "square_", "exponential_", "gradient_")


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


def build_extractors(modality: str, bin_width: float | None) -> dict:
    """full: all classes; small: shape + first-order; union: first-order + texture."""
    full = featureextractor.RadiomicsFeatureExtractor(str(PARAMS[modality]))
    if bin_width is not None:
        full.settings["binWidth"] = bin_width
    small, union = copy.deepcopy(full), copy.deepcopy(full)
    small.enabledFeatures = {k: v for k, v in full.enabledFeatures.items() if k in ("shape", "firstorder")}
    union.enabledFeatures = {k: v for k, v in full.enabledFeatures.items() if k != "shape"}
    return {"full": full, "small": small, "union": union}


def resampled_voxel_ml(extractor, image: sitk.Image) -> float:
    spacing = extractor.settings.get("resampledPixelSpacing") or image.GetSpacing()
    return float(np.prod(spacing)) / 1000.0


def extract_patient(pid: str, modality: str, extractors: dict, args) -> dict[str, list[dict]]:
    image_f = args.nifti_dir / pid / IMAGE_FILE[modality]
    if not image_f.exists():
        raise FileNotFoundError(image_f)
    image = sitk.ReadImage(str(image_f), sitk.sitkFloat32)
    pet = image if modality == "PET" else sitk.ReadImage(str(args.nifti_dir / pid / "PET_SUV.nii.gz"), sitk.sitkFloat32)

    mask = load_tumor_mask(args.mask_dir / pid, args.mask)
    labels, lesions = label_lesions(mask, pet)
    if modality != "PET":  # masks are defined on the PET grid
        labels = resample_to_reference(labels, image, is_mask=True)

    rows: dict[str, list[dict]] = {lvl: [] for lvl in args.levels}
    if "wholebody" in args.levels and lesions:
        union = sitk.Cast(labels > 0, sitk.sitkUInt8)
        res = extractors["union"].execute(image, union, label=1)
        rows["wholebody"].append({"patient_id": pid, **_clean(res, args.diagnostics)})

    if "lesion" in args.levels:
        vox_ml = resampled_voxel_ml(extractors["full"], image)
        for d in lesions:
            n_vox = d["volume_ml"] / vox_ml
            has_texture = n_vox >= args.min_texture_voxels
            try:
                res = extractors["full" if has_texture else "small"].execute(image, labels, label=d["lesion_id"])
            except ValueError as exc:  # e.g. lesion too small after resampling
                log.debug("%s lesion %d skipped: %s", pid, d["lesion_id"], exc)
                continue
            rows["lesion"].append({"patient_id": pid, "lesion_id": d["lesion_id"], "lesion_suvmax": d["suvmax"],
                                   "lesion_volume_ml": d["volume_ml"], "n_voxels_resampled": round(n_vox, 1),
                                   "has_texture": has_texture, **_clean(res, args.diagnostics)})
    n_tex = sum(r["has_texture"] for r in rows.get("lesion", []))
    log.info("%s %s: %d lesions (%d with texture)", pid, modality, len(lesions), n_tex)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nifti-dir", type=Path, default=DATA_DIR / "nifti")
    ap.add_argument("--mask-dir", type=Path, default=DATA_DIR / "masks")
    ap.add_argument("--mask", choices=list(MASK_FILES), default="reader1")
    ap.add_argument("--modalities", nargs="+", default=["PET"], choices=["PET", "CT"])
    ap.add_argument("--levels", nargs="+", default=["wholebody", "lesion"], choices=["wholebody", "lesion"])
    ap.add_argument("--bin-width", type=float, default=None,
                    help="override binWidth of the config (PET: 0.25 / 0.5 / 1.0 SUV)")
    ap.add_argument("--min-texture-voxels", type=int, default=64,
                    help="texture features only for lesions with >= N voxels after resampling (default 64)")
    ap.add_argument("--diagnostics", action="store_true", help="keep PyRadiomics diagnostics_* columns")
    ap.add_argument("--patients", nargs="*")
    args = ap.parse_args()
    setup_logging()
    radiomics.setVerbosity(logging.ERROR)
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    logging.getLogger("pykwalify").setLevel(logging.ERROR)

    patients = [p for p in (args.patients or list_patients(args.mask_dir))
                if (args.mask_dir / p / MASK_FILES[args.mask]).exists()]
    out_dir = results_dir(args.mask)
    log.info("%d patients with %s masks", len(patients), args.mask)
    for modality in args.modalities:
        extractors = build_extractors(modality, args.bin_width)
        bw = extractors["full"].settings["binWidth"]
        log.info("%s: binWidth %s, resampling %s, image types %s", modality, bw,
                 extractors["full"].settings.get("resampledPixelSpacing"), list(extractors["full"].enabledImagetypes))
        collected: dict[str, list[dict]] = {lvl: [] for lvl in args.levels}
        for pid in patients:
            try:
                for lvl, rows in extract_patient(pid, modality, extractors, args).items():
                    collected[lvl].extend(rows)
            except Exception as exc:
                log.error("%s %s: FAILED - %s", pid, modality, exc)
        for lvl, rows in collected.items():
            out = out_dir / f"radiomics_{modality.lower()}_{lvl}_bw{bw:g}.csv"
            df = pd.DataFrame(rows)
            df.to_csv(out, index=False)
            n_feat = sum(c.startswith(FEATURE_PREFIXES) for c in df.columns)
            log.info("Wrote %s (%d rows, %d features)", out, len(df), n_feat)


if __name__ == "__main__":
    main()

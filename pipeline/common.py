"""Shared helpers for the Lu-PSMA radiomics pipeline."""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk

log = logging.getLogger("rlt_radiomics")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def list_patients(root: Path) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir() and not p.name.startswith("."))


# --------------------------------------------------------------------------- DICOM I/O

def read_dicom_series(folder: Path) -> tuple[sitk.Image, list[pydicom.Dataset]]:
    """Read a single-series DICOM folder into a float32 SimpleITK image.

    Slices are sorted along the slice normal and the per-slice RescaleSlope /
    RescaleIntercept is applied (PET slices often have different slopes, which
    a plain series reader may not handle correctly).
    """
    datasets = []
    for f in sorted(Path(folder).rglob("*")):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(f, force=True)
        except Exception:
            continue
        if "PixelData" in ds and "ImagePositionPatient" in ds:
            datasets.append(ds)
    if not datasets:
        raise FileNotFoundError(f"No DICOM images found in {folder}")

    uids = {ds.SeriesInstanceUID for ds in datasets}
    if len(uids) > 1:
        raise ValueError(f"{folder} contains {len(uids)} series; keep one series per folder")

    row, col = (np.array(datasets[0].ImageOrientationPatient[:3], float),
                np.array(datasets[0].ImageOrientationPatient[3:], float))
    normal = np.cross(row, col)
    datasets.sort(key=lambda d: float(np.dot(normal, np.array(d.ImagePositionPatient, float))))

    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))
        slices.append(arr)
    volume = np.stack(slices, axis=0)  # (z, y, x)

    positions = [np.array(d.ImagePositionPatient, float) for d in datasets]
    if len(positions) > 1:
        gaps = np.diff([np.dot(normal, p) for p in positions])
        dz = float(np.median(gaps))
        if np.ptp(gaps) > 0.01 * abs(dz):
            warnings.warn(f"Non-uniform slice spacing in {folder} (min {gaps.min():.3f}, max {gaps.max():.3f} mm)")
    else:
        dz = float(getattr(datasets[0], "SliceThickness", 1.0))

    px = [float(v) for v in datasets[0].PixelSpacing]  # [row spacing, col spacing]
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing((px[1], px[0], dz))
    img.SetOrigin(tuple(positions[0]))
    img.SetDirection(tuple(np.column_stack([row, col, normal]).ravel()))
    return img, datasets


# --------------------------------------------------------------------------- SUV

def _parse_dt(date: str | None, time: str | None) -> datetime | None:
    if not date or not time:
        return None
    time = str(time).split(".")[0].ljust(6, "0")
    return datetime.strptime(f"{date}{time}", "%Y%m%d%H%M%S")


def _parse_full_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    value = str(value).split(".")[0].split("+")[0].split("-")[0]
    return datetime.strptime(value[:14].ljust(14, "0"), "%Y%m%d%H%M%S")


def suv_bw_factor(datasets: list[pydicom.Dataset]) -> tuple[float, dict]:
    """Return the factor converting Bq/mL to body-weight SUV, plus the values used.

    SUVbw = C(Bq/mL) * weight(g) / (injected activity(Bq) * 2^(-dt / T1/2))
    with dt = scan start - injection time (QIBA / Kinahan & Fletcher 2010).
    """
    ds = datasets[0]
    units = str(getattr(ds, "Units", "")).upper()
    if units != "BQML":
        raise ValueError(f"PET Units are '{units}', expected BQML (check vendor SUV conversion)")
    corrected = str(getattr(ds, "CorrectedImage", ""))
    if "DECY" not in corrected or "ATTN" not in corrected:
        warnings.warn(f"CorrectedImage={corrected}: image may not be decay/attenuation corrected")
    decay_ref = str(getattr(ds, "DecayCorrection", "START")).upper()
    if decay_ref != "START":
        raise ValueError(f"DecayCorrection={decay_ref} is not supported (expected START)")

    rp = ds.RadiopharmaceuticalInformationSequence[0]
    dose = float(rp.RadionuclideTotalDose)
    half_life = float(rp.RadionuclideHalfLife)
    weight_kg = float(ds.PatientWeight)

    injection = _parse_full_dt(getattr(rp, "RadiopharmaceuticalStartDateTime", None)) or _parse_dt(
        ds.SeriesDate, getattr(rp, "RadiopharmaceuticalStartTime", None))

    # Earliest acquisition time is more robust than SeriesTime (some vendors rewrite SeriesTime).
    acq_times = [t for t in (_parse_dt(getattr(d, "AcquisitionDate", None) or d.SeriesDate,
                                       getattr(d, "AcquisitionTime", None)) for d in datasets) if t]
    scan_start = min(acq_times) if acq_times else _parse_dt(ds.SeriesDate, ds.SeriesTime)
    series_start = _parse_dt(ds.SeriesDate, ds.SeriesTime)
    if series_start and scan_start and series_start < scan_start:
        scan_start = series_start

    if injection is None or scan_start is None:
        raise ValueError("Could not determine injection or scan time")
    delta_s = (scan_start - injection).total_seconds()
    if not 0 < delta_s < 6 * 3600:
        warnings.warn(f"Uptake time {delta_s / 60:.1f} min looks unusual - check the date/time tags")

    decayed_dose = dose * 2 ** (-delta_s / half_life)
    factor = weight_kg * 1000.0 / decayed_dose
    info = {
        "pet_date": scan_start.date().isoformat(),
        "radiopharmaceutical": str(getattr(rp, "Radiopharmaceutical", "")),
        "injected_MBq": dose / 1e6,
        "half_life_s": half_life,
        "weight_kg": weight_kg,
        "uptake_min": delta_s / 60.0,
        "suv_factor": factor,
        # technical covariates (ComBat batches / sensitivity analyses, never predictors)
        "institution": str(getattr(ds, "InstitutionName", "")),
        "station": str(getattr(ds, "StationName", "")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "scanner": str(getattr(ds, "ManufacturerModelName", "")),
        "reconstruction": str(getattr(ds, "ReconstructionMethod", "")),
        "recon_filter": str(getattr(ds, "ConvolutionKernel", "")),
        "matrix": f"{ds.Rows}x{ds.Columns}",
        "voxel_mm": "x".join(f"{v:.2f}" for v in (*map(float, ds.PixelSpacing), float(ds.SliceThickness))),
    }
    return factor, info


# --------------------------------------------------------------------------- geometry

def resample_to_reference(image: sitk.Image, reference: sitk.Image, is_mask: bool, default: float = 0.0) -> sitk.Image:
    interp = sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear
    return sitk.Resample(image, reference, sitk.Transform(), interp, default, image.GetPixelID())


def voxel_volume_ml(image: sitk.Image) -> float:
    return float(np.prod(image.GetSpacing())) / 1000.0


# --------------------------------------------------------------------------- masks

MASK_FILES = {
    "auto": "tumor_mask_auto.nii.gz",        # step 2 output (nnU-Net or threshold), uncorrected
    "reader1": "tumor_mask_reader1.nii.gz",  # corrected by reader 1 + senior review (main analysis)
    "reader2": "tumor_mask_reader2.nii.gz",  # independent correction by reader 2 (ICC subset)
}


def load_tumor_mask(mask_dir: Path, mask: str = "reader1") -> sitk.Image:
    """Load a binary tumour mask: ``auto``, ``reader1`` or ``reader2`` (see MASK_FILES)."""
    f = Path(mask_dir) / MASK_FILES[mask]
    if not f.exists():
        raise FileNotFoundError(f"{f} not found")
    return sitk.ReadImage(str(f), sitk.sitkUInt8) > 0


def results_dir(mask: str) -> Path:
    """Results of steps 3-6 are kept separate per mask version (main analysis vs. sensitivity)."""
    d = RESULTS_DIR / mask
    d.mkdir(parents=True, exist_ok=True)
    return d


def label_lesions(mask: sitk.Image, pet: sitk.Image, min_volume_ml: float = 0.0) -> tuple[sitk.Image, list[dict]]:
    """Split a binary tumour mask into 3D-connected lesions (26-connectivity).

    Labels are ordered by SUVmax (label 1 = hottest lesion). Lesions smaller
    than ``min_volume_ml`` are dropped.
    """
    if (mask.GetSize() != pet.GetSize() or not np.allclose(mask.GetSpacing(), pet.GetSpacing(), atol=1e-3)
            or not np.allclose(mask.GetOrigin(), pet.GetOrigin(), atol=1e-2)):
        # e.g. a mask exported from 3D Slicer with another reference volume
        warnings.warn("Mask geometry differs from PET - resampling mask to the PET grid (check the export)")
        mask = resample_to_reference(sitk.Cast(mask, sitk.sitkUInt8), pet, is_mask=True)
    mask = sitk.Cast(mask > 0, sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(mask, True)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc, pet)
    vox_ml = voxel_volume_ml(pet)

    lesions = []
    for lab in stats.GetLabels():
        vol = stats.GetNumberOfPixels(lab) * vox_ml
        if vol < min_volume_ml:
            continue
        lesions.append({"old": lab, "suvmax": stats.GetMaximum(lab), "suvmean": stats.GetMean(lab),
                        "volume_ml": vol, "centroid_mm": stats.GetCentroid(lab)})
    lesions.sort(key=lambda d: d["suvmax"], reverse=True)

    mapping = {d["old"]: i + 1 for i, d in enumerate(lesions)}
    cc_arr = sitk.GetArrayViewFromImage(cc)
    lut = np.zeros(int(cc_arr.max()) + 1, dtype=np.uint16)
    for old, new in mapping.items():
        lut[old] = new
    out = sitk.GetImageFromArray(lut[cc_arr])
    out.CopyInformation(mask)
    for i, d in enumerate(lesions):
        d["lesion_id"] = i + 1
        del d["old"]
    return out, lesions

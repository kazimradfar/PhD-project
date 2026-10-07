"""Create synthetic PSMA PET/CT DICOM patients to test the pipeline end-to-end.

The phantom has a body cylinder, liver, kidneys, bladder and a few "lesions"
(bone/lymph-node-like spheres). Values are written in Bq/mL with the DICOM tags
needed for SUVbw, so step 1 is tested exactly as with real data.

Usage:
    python tools/make_synthetic_data.py --n 3 --out data/raw
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

PET_SOP = "1.2.840.10008.5.1.4.1.1.128"
CT_SOP = "1.2.840.10008.5.1.4.1.1.2"


def sphere(shape, center, radius_vox):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    return ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= radius_vox ** 2


def phantom(rng, shape):
    """Return SUV and HU volumes (z, y, x) and the organ masks (TotalSegmentator names)."""
    z, y, x = shape
    zz, yy, xx = np.ogrid[:z, :y, :x]
    body = np.broadcast_to(((yy - y / 2) / (y * 0.38)) ** 2 + ((xx - x / 2) / (x * 0.45)) ** 2 <= 1, shape)
    suv = np.where(body, 0.6, 0.0)
    hu = np.where(body, 30.0, -1000.0)
    organs = {"liver": ((z * .70, y * .45, x * .35), 7, 5.0, 60),
              "kidney_right": ((z * .55, y * .60, x * .35), 3, 25.0, 30),
              "kidney_left": ((z * .55, y * .60, x * .65), 3, 25.0, 30),
              "urinary_bladder": ((z * .15, y * .50, x * .50), 3, 40.0, 10)}
    masks = {}
    for name, (c, r, s, h) in organs.items():
        m = masks[name] = sphere(shape, c, r)
        suv[m], hu[m] = s, h
    for _ in range(rng.integers(3, 7)):  # lesions
        c = (rng.uniform(.2, .9) * z, rng.uniform(.3, .7) * y, rng.uniform(.25, .75) * x)
        r = rng.uniform(1.5, 4.0)
        m = sphere(shape, c, r) & body
        core = sphere(shape, c, r * 0.5)
        suv[m] = rng.uniform(6, 30)
        suv[core & m] *= rng.uniform(1.0, 1.6)  # heterogeneity
        hu[m] = rng.choice([45.0, 600.0])        # node-like or sclerotic bone
    suv = suv * rng.gamma(20, 1 / 20, size=shape)  # noise
    hu = hu + rng.normal(0, 15, size=shape)
    return suv.astype(np.float32), hu.astype(np.float32), masks


def write_series(folder: Path, vol, spacing, modality, study_uid, info):
    folder.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    for k in range(vol.shape[0]):
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = PET_SOP if modality == "PT" else CT_SOP
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.PatientID, ds.PatientName = modality, info["pid"], info["pid"]
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study_uid, series_uid
        ds.SeriesDate = ds.AcquisitionDate = info["scan"].strftime("%Y%m%d")
        ds.SeriesTime = ds.AcquisitionTime = info["scan"].strftime("%H%M%S")
        ds.InstanceNumber = k + 1
        ds.ImagePositionPatient = [0.0, 0.0, float(k * spacing[2])]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [spacing[1], spacing[0]]
        ds.SliceThickness = spacing[2]
        ds.Rows, ds.Columns = vol.shape[1], vol.shape[2]
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
        sl = vol[k]
        if modality == "PT":
            ds.PixelRepresentation = 0
            slope = max(float(sl.max()), 1.0) / 32000.0  # per-slice slope, like real scanners
            ds.RescaleSlope, ds.RescaleIntercept = slope, 0
            ds.PixelData = np.round(sl / slope).astype(np.uint16).tobytes()
            ds.Units, ds.DecayCorrection, ds.CorrectedImage = "BQML", "START", ["DECY", "ATTN", "SCAT"]
            ds.PatientWeight = info["weight"]
            rp = Dataset()
            rp.Radiopharmaceutical = "Ga-68 PSMA-11"
            rp.RadionuclideTotalDose = info["dose"]
            rp.RadionuclideHalfLife = 4062.6  # Ga-68, s
            rp.RadiopharmaceuticalStartTime = info["inj"].strftime("%H%M%S")
            ds.RadiopharmaceuticalInformationSequence = [rp]
            ds.Manufacturer = "Synthetic"
        else:
            ds.PixelRepresentation = 1
            ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
            ds.PixelData = np.clip(np.round(sl + 1024), -32768, 32767).astype(np.int16).tobytes()
        ds.save_as(folder / f"{modality}_{k:04d}.dcm", enforce_file_format=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    ap.add_argument("--totalseg", type=Path, default=Path("data/totalseg"),
                    help="also write synthetic organ masks here")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    shape, spacing = (40, 64, 64), (4.0, 4.0, 4.0)  # (z,y,x) voxels; (x,y,z) mm
    for i in range(args.n):
        pid = f"SYN{i + 1:03d}"
        weight, dose = float(rng.uniform(65, 95)), float(rng.uniform(120e6, 180e6))
        inj = datetime(2026, 1, 10, 9, 0, 0)
        scan = inj + timedelta(minutes=float(rng.uniform(55, 70)))
        info = dict(pid=pid, weight=weight, dose=dose, inj=inj, scan=scan)
        suv, hu, organs = phantom(rng, shape)
        decayed = dose * 2 ** (-(scan - inj).total_seconds() / 4062.6)
        bqml = suv * decayed / (weight * 1000)
        study = generate_uid()
        write_series(args.out / pid / "PET", bqml, spacing, "PT", study, info)
        write_series(args.out / pid / "CT", hu, spacing, "CT", study, info)
        if args.totalseg:  # mimic TotalSegmentator output (one NIfTI per structure)
            tdir = args.totalseg / pid
            tdir.mkdir(parents=True, exist_ok=True)
            for name, m in organs.items():
                img = sitk.GetImageFromArray(m.astype(np.uint8))
                img.SetSpacing(spacing)
                sitk.WriteImage(img, str(tdir / f"{name}.nii.gz"))
        print(f"{pid}: weight {weight:.0f} kg, {dose / 1e6:.0f} MBq, uptake {(scan - inj).seconds / 60:.0f} min")


if __name__ == "__main__":
    main()

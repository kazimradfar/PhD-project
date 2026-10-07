"""Write PET/CT in nnU-Net input format for whole-body lesion inference.

nnU-Net expects one file per channel: <case>_0000.nii.gz, <case>_0001.nii.gz, ... on the same grid.
Here channel 0 = CT resampled to the PET grid, channel 1 = PET in SUV - the SAME order and the
same preprocessing must be used when the model is trained on PSMA-PET-CT-Lesions.

Usage:
    python tools/prepare_nnunet_input.py --out data/nnunet_input
    nnUNetv2_predict -i data/nnunet_input -o data/nnunet_pred -d <DatasetID> -c 3d_fullres
    python -m pipeline.step2_segment_lesions --method nnunet
"""

import argparse
import sys
from pathlib import Path

import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.common import resample_to_reference  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--nifti-dir", type=Path, default=ROOT / "data" / "nifti")
ap.add_argument("--out", type=Path, default=ROOT / "data" / "nnunet_input")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

for pdir in sorted(p for p in args.nifti_dir.iterdir() if p.is_dir()):
    pet = sitk.ReadImage(str(pdir / "PET_SUV.nii.gz"), sitk.sitkFloat32)
    ct = sitk.ReadImage(str(pdir / "CT.nii.gz"), sitk.sitkFloat32)
    ct_on_pet = resample_to_reference(ct, pet, is_mask=False, default=-1024.0)
    sitk.WriteImage(ct_on_pet, str(args.out / f"{pdir.name}_0000.nii.gz"), useCompression=True)
    sitk.WriteImage(pet, str(args.out / f"{pdir.name}_0001.nii.gz"), useCompression=True)
    print(pdir.name)

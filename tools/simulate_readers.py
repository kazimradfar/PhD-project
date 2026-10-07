"""DEMO ONLY - simulate reader corrections on synthetic data to test steps 3-6.

reader1 = auto mask; reader2 = auto mask randomly dilated/eroded by one voxel per lesion.
Never use this on real patients: real masks come from manual correction in 3D Slicer.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--mask-dir", type=Path, default=ROOT / "data" / "masks")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = np.random.default_rng(args.seed)

for pdir in sorted(p for p in args.mask_dir.iterdir() if p.is_dir()):
    auto = pdir / "tumor_mask_auto.nii.gz"
    shutil.copy(auto, pdir / "tumor_mask_reader1.nii.gz")
    m = sitk.ReadImage(str(auto), sitk.sitkUInt8)
    cc = sitk.ConnectedComponent(m)
    out = sitk.Image(m.GetSize(), sitk.sitkUInt8)
    out.CopyInformation(m)
    for lab in range(1, int(sitk.GetArrayViewFromImage(cc).max()) + 1):
        les = cc == lab
        op = rng.choice(["keep", "dilate", "erode"], p=[0.4, 0.4, 0.2])
        if op == "dilate":
            les = sitk.BinaryDilate(les, [1, 1, 1])
        elif op == "erode":
            eroded = sitk.BinaryErode(les, [1, 1, 1])
            les = eroded if sitk.GetArrayViewFromImage(eroded).any() else les
        out = out | les
    sitk.WriteImage(out, str(pdir / "tumor_mask_reader2.nii.gz"), useCompression=True)
    print(pdir.name)

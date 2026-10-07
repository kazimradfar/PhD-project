"""Randomly select the patients whose masks reader 2 corrects independently (ICC subset).

The seed is fixed and the list is saved, so the selection is reproducible and can be
reported. Reader 2 must start from tumor_mask_auto.nii.gz, NOT from reader 1's mask.

Usage:
    python tools/select_icc_subset.py --n 30 --seed 2026
"""

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--mask-dir", type=Path, default=ROOT / "data" / "masks")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--out", type=Path, default=ROOT / "results" / "icc_subset.txt")
args = ap.parse_args()

patients = sorted(p.name for p in args.mask_dir.iterdir() if (p / "tumor_mask_auto.nii.gz").exists())
chosen = sorted(np.random.default_rng(args.seed).choice(patients, size=min(args.n, len(patients)), replace=False))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text("\n".join(chosen) + "\n")
print(f"{len(chosen)} of {len(patients)} patients selected (seed {args.seed}) -> {args.out}")

"""Run the whole pipeline on synthetic data with one command (Windows, macOS, Linux).

    python tools/run_demo.py            # create synthetic data and run steps 1-7
    python tools/run_demo.py --phase2   # phase 2 (steps 8-13) on 220 synthetic patients (tables only)
    python tools/run_demo.py --clean    # delete all demo data and results

Each step is printed before it runs, so you can see which command does what and
copy the same commands later for your real data.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO_PATHS = ["data/raw", "data/nifti", "data/masks", "data/totalseg", "data/crf.csv", "results"]

STEPS = [
    ("0  Synthetic PET/CT DICOM + CRF", ["tools/make_synthetic_data.py", "--n", "12"]),
    ("1  DICOM -> NIfTI + SUV", ["-m", "pipeline.step1_dicom_to_nifti"]),
    ("2  Automatic segmentation (threshold, demo)", ["-m", "pipeline.step2_segment_lesions", "--method", "threshold"]),
    ("2b Simulated reader 1 / reader 2 masks (demo only)", ["tools/simulate_readers.py"]),
]
for m in ("reader1", "reader2"):
    STEPS += [
        (f"3  Conventional PET metrics [{m}]", ["-m", "pipeline.step3_conventional_metrics", "--mask", m]),
        (f"4  Radiomics extraction [{m}]", ["-m", "pipeline.step4_extract_radiomics", "--mask", m]),
        (f"5  Patient-level features [{m}]", ["-m", "pipeline.step5_build_dataset", "--mask", m]),
    ]
STEPS += [
    ("6  Inter-reader ICC filter", ["-m", "pipeline.step6_reproducibility"]),
    ("7  Endpoints from CRF", ["-m", "pipeline.step7_outcomes", "--data-lock", "2026-06-30"]),
]


PHASE2_STEPS = [
    ("0  Synthetic phase-1 tables (220 patients)", ["tools/make_synthetic_table.py", "--n", "220"]),
    ("8  Analysis table", ["-m", "pipeline.step8_analysis_table"]),
    ("9  H1: univariable associations", ["-m", "pipeline.step9_univariable"]),
    ("10 Models, nested CV (quick)", ["-m", "pipeline.step10_models", "--quick"]),
    ("11 Validation (bootstrap, temporal, leave-one-center-out)",
     ["-m", "pipeline.step11_validation", "--endpoint", "os", "--model", "coxnet", "--n-boot", "20", "--no-tune"]),
    ("12 H2: incremental value of radiomics", ["-m", "pipeline.step12_incremental_value", "--n-boot", "30"]),
    ("13 H3: explainability", ["-m", "pipeline.step13_explain", "--endpoint", "os", "--n-boot", "10"]),
]


def clean() -> None:
    for p in DEMO_PATHS:
        path = ROOT / p
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    print("Demo data and results removed.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--phase2", action="store_true", help="run the phase-2 demo (steps 8-13)")
    args = ap.parse_args()
    if args.clean:
        clean()
        return
    steps = STEPS
    if args.phase2:
        if (ROOT / "results" / "outcomes.csv").exists():
            sys.exit("results/ is not empty. Run 'python tools/run_demo.py --clean' first (demo data only!).")
        steps = PHASE2_STEPS
    elif (ROOT / "data" / "raw").exists() and any((ROOT / "data" / "raw").iterdir()):
        sys.exit("data/raw is not empty. The demo only runs on an empty data folder "
                 "(run 'python tools/run_demo.py --clean' first - this deletes data/raw!).")

    for title, cmd in steps:
        print(f"\n{'=' * 70}\nSTEP {title}\n$ python {' '.join(cmd)}\n{'=' * 70}", flush=True)
        if subprocess.run([sys.executable, *cmd], cwd=ROOT).returncode != 0:
            sys.exit(f"Step '{title}' failed - read the error above.")
    print("\nDone. Results are in the 'results' folder. Next: open notebooks/explore_results.ipynb")


if __name__ == "__main__":
    main()

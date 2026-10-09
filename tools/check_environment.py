"""Check that every package needed for phase 1 and phase 2 is installed.

    python tools/check_environment.py

Run it inside the project environment (.venv activated). Prints OK / MISSING for each package,
grouped by phase, and the command to install what is missing. Nothing is installed or changed.
"""
import importlib
import platform
import shutil
import subprocess
import sys

# (import name, pip name, what it is used for)
CORE = [
    ("numpy", "numpy", "arrays"), ("scipy", "scipy", "statistics"), ("pandas", "pandas", "tables"),
    ("yaml", "PyYAML", "config files"), ("SimpleITK", "SimpleITK", "image I/O and processing"),
    ("pydicom", "pydicom", "reading DICOM"), ("matplotlib", "matplotlib", "figures"),
]
PHASE1 = [
    ("radiomics", "pyradiomics", "step 4: radiomic features (see requirements-radiomics.txt)"),
    ("pywt", "PyWavelets", "needed by PyRadiomics"),
]
PHASE1_OPTIONAL = [
    ("nnunetv2", "nnunetv2", "step 2: nnU-Net segmentation (GPU workstation)"),
    ("totalsegmentator", "TotalSegmentator", "step 2: organ masks for physiological uptake"),
]
PHASE2 = [
    ("sklearn", "scikit-learn", "pipelines, cross-validation"), ("sksurv", "scikit-survival", "survival models, C-index"),
    ("xgboost", "xgboost", "XGBoost models"), ("shap", "shap", "step 13: SHAP explanations"),
]
TOOLS = [
    ("pytest", "pytest", "tests"), ("ipykernel", "ipykernel", "running the notebooks in VS Code"),
]
R_PACKAGES = ["survival", "prodlim", "riskRegression", "cmprsk", "randomForestSRC"]


def check(group):
    missing = []
    for mod, pip_name, use in group:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "")
            print(f"  OK       {pip_name:<18} {ver:<10} {use}")
        except Exception as exc:  # ImportError or a broken install
            print(f"  MISSING  {pip_name:<18} {'':<10} {use}   ({type(exc).__name__})")
            missing.append(pip_name)
    return missing


def main() -> None:
    print(f"Python {platform.python_version()}  ({sys.executable})")
    if ".venv" not in sys.executable:
        print("  NOTE: this is not the project's .venv - activate it first:  .venv\\Scripts\\activate")
    if sys.version_info >= (3, 13):
        print("  NOTE: Python 3.13+ - PyRadiomics may not build; Python 3.10 or 3.11 is the safest choice")

    report = {}
    for title, group in (("Core (both phases)", CORE), ("Phase 1 - images and radiomics", PHASE1),
                         ("Phase 2 - modelling", PHASE2), ("Notebooks and tests", TOOLS)):
        print(f"\n{title}")
        report[title] = check(group)
    print("\nPhase 1 - optional, only on the segmentation workstation")
    check(PHASE1_OPTIONAL)

    print("\nR (recurrence with competing risks, optional)")
    rscript = shutil.which("Rscript")
    if not rscript:
        print("  MISSING  R / Rscript not found on PATH (install R from cran.r-project.org)")
    else:
        code = "cat(sapply(c(%s), requireNamespace, quietly=TRUE))" % ",".join(f'"{p}"' for p in R_PACKAGES)
        try:
            out = subprocess.run([rscript, "-e", code], capture_output=True, text=True, timeout=120).stdout.split()
            for p, ok in zip(R_PACKAGES, out):
                print(f"  {'OK     ' if ok == 'TRUE' else 'MISSING'}  R package {p}")
        except Exception as exc:
            print(f"  could not run Rscript: {exc}")

    print("\n" + "=" * 70)
    core, p1, p2, tools = (report[k] for k in report)
    print(f"Phase 1 ready: {'YES' if not (core or p1) else 'NO'}")
    print(f"Phase 2 ready: {'YES' if not (core or p2) else 'NO'}")
    need = core + p2 + tools
    if need:
        print("\nInstall the missing phase-2 / core packages with:\n  pip install -r requirements.txt")
    if p1:
        print("\nInstall PyRadiomics (phase 1) with:\n  pip install docopt-ng pykwalify\n"
              '  pip install --no-deps "pyradiomics @ git+https://github.com/AIM-Harvard/pyradiomics.git"\n'
              "  (if it asks for 'Microsoft Visual C++ 14.0', install Build Tools for Visual Studio first)")


if __name__ == "__main__":
    main()

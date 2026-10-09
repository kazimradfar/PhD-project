# CLAUDE.md

PhD thesis code (Medical Physics, TUMS). Radiomic features from pre-treatment [⁶⁸Ga]Ga-PSMA-11 PET/CT
are used to predict outcomes of [¹⁷⁷Lu]Lu-PSMA-617 radioligand therapy in mCRPC: OS, PFS, recurrence
(death as competing event) and response at week 12. Every change must serve this goal and the
registered analysis plan. The full walkthrough is in `docs/PROJECT_GUIDE_FA.md`.

## Language

- Talk to the user in **Persian**. Code, comments, docstrings, commit messages and config keys stay in **English**.
- The user is a medical physicist, not a software engineer: explain what a change means for the
  analysis, not only what the code does.

## Hard rules (never break these)

1. **Patient data never leaves the machine.** Do not read, print, summarize, commit or upload anything
   under `data/` or `results/`, or any `.dcm` / `.nii` / `.nii.gz` file, even if it is de-identified.
   `.claude/settings.json` denies file reads there; do not get around it with shell or Python
   commands that print file contents (`cat`, `head`, `df.head()`, `print(df)` ...). When a script runs
   on real data, show only aggregate output (counts, shapes, column names, error messages), never
   patient-level rows. If unsure whether a folder holds real data, ask first.
2. **Imaging features are fixed before outcomes are linked.** Phase 1 (`step1`-`step6`) must not
   read outcome data. Phase 2 (`step8`-`step13`) starts only after OSF registration.
3. **No leakage.** Every learning preprocessing step (scaling, ComBat, correlation filter, LASSO
   selection, imputation) is fitted inside each training fold only.
4. **`configs/modeling.yaml` is pre-registered.** Do not change values after results have been seen.
   Open decisions are marked `DECISION NEEDED` and are made with the supervisors, not by Claude.
5. **Demo numbers have no clinical meaning.** Never present synthetic-data results as findings.

## Fixed analysis design (from the proposal)

- Reference model: whole-body SUVmean, PSMA-TV, LDH, haemoglobin. If one is > 30 % missing it is
  replaced by time since diagnosis.
- Final regression model: 1 radiomic score + 4 reference variables = 5 parameters (Riley sample size,
  about 200 patients).
- H1: univariable associations with Benjamini-Hochberg FDR. H2: added value over the reference model,
  ΔC ≥ 0.03, LRT, DCA. H3: SHAP and stability (selected in ≥ 70 % of bootstrap samples).
- Validation: repeated nested CV, bootstrap optimism correction, temporal 70/30 split,
  leave-one-center-out.

## Layout

| Path | Contents |
|---|---|
| `pipeline/step1`-`step6` | Phase 1: DICOM → SUV, segmentation (nnU-Net + TotalSegmentator + two-reader correction), conventional PET metrics, IBSI radiomics (PyRadiomics, bin width 0.5 SUV, texture only for lesions ≥ 64 voxels), patient-level aggregation (`wb_`, `vwm_`, `sd_`, `range_`), ICC ≥ 0.75 filter |
| `pipeline/step7_outcomes.py` | Endpoints from the CRF and temporal split |
| `pipeline/step8`-`step13`, `modeling/` | Phase 2: analysis table, H1, models (Cox-LASSO, RSF, XGBoost, SVM) for reference / radiomics / combined sets, validation, H2, H3 |
| `R/recurrence_competing_risks.R` | Recurrence: Fine-Gray, cause-specific Cox, competing-risk RSF |
| `configs/` | `pet_params.yaml`, `ct_params.yaml`, `modeling.yaml` |
| `tools/` | Demos, synthetic data, environment check, nnU-Net input preparation, ICC subset selection |
| `notebooks/` | `explore_results.ipynb` (phase 1), `phase2_results.ipynb` (phase 2) |
| `tests/` | pytest suite |

## Commands

```
python tools/check_environment.py      # which phases are ready on this machine
python -m pytest tests -q              # run after every code change
python tools/run_demo.py               # phase 1 demo, 12 synthetic patients (needs PyRadiomics)
python tools/run_demo.py --phase2      # phase 2 demo, 220 synthetic patients
python tools/run_demo.py --clean       # delete demo data and results (demo only!)
```

The demos refuse to run if `data/raw` or `results/` already hold data. Never run `--clean` when
real data may be present.

## Environment

- Windows laptop, VS Code, Python 3.13, virtual environment `.venv`.
- PyRadiomics is installed from GitHub (`requirements-radiomics.txt`); the PyPI release is broken.
- The laptop GPU (MX330, 2 GB) is too small for nnU-Net / TotalSegmentator; segmentation runs on a
  GPU workstation at the hospital or university.
- R is optional (only for the competing-risks analysis).

## Working conventions

- Small, focused commits on the working branch (`phase2-modeling` until it is merged into the default branch).
- Add or update a test for every bug fix, ideally one that compares against reference values.
- Keep the Persian guides in `docs/` in sync when a step's behaviour or a command changes.

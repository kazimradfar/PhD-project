# Data layout (not tracked by git)

```
data/
├── raw/<patient_id>/PET/*.dcm      # baseline 68Ga/18F-PSMA PET (attenuation corrected)
├── raw/<patient_id>/CT/*.dcm       # low-dose CT of the same session
├── totalseg/<patient_id>/*.nii.gz  # optional TotalSegmentator organ masks (CT)
├── nifti/<patient_id>/             # output of step 1
└── masks/<patient_id>/             # output of step 2 (after manual review in 3D Slicer)
```

Use pseudonymised IDs only (e.g. `RLT001`). Do not push patient data to GitHub.

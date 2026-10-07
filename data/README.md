# Data layout (not tracked by git — keep on the in-house workstation only)

```
data/
├── raw/<pid>/PET/*.dcm          # baseline [68Ga]Ga-PSMA-11 PET (AC), <= 8 weeks before cycle 1
├── raw/<pid>/CT/*.dcm           # low-dose CT of the same session
├── nifti/<pid>/                 # step 1: PET_SUV.nii.gz, CT.nii.gz
├── totalseg/<pid>/*.nii.gz      # TotalSegmentator organ masks (physiological uptake, liver reference)
├── nnunet_input/                # tools/prepare_nnunet_input.py
├── nnunet_pred/<pid>.nii.gz     # nnU-Net lesion predictions
├── masks/<pid>/
│   ├── tumor_mask_auto.nii.gz      # step 2 (uncorrected)
│   ├── tumor_mask_reader1.nii.gz   # corrected by reader 1 + senior review (main analysis)
│   ├── tumor_mask_reader2.nii.gz   # independent correction, ICC subset (30 patients)
│   └── liver_ref.nii.gz
└── crf.csv                      # case-report form (step 7)
```

Use pseudonymised IDs only (e.g. `RLT001`); the linkage key stays with the PI inside the hospital.

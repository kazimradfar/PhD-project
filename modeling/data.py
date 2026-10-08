"""Assemble the phase-2 analysis table from the phase-1 results and the CRF.

Inputs (all produced by steps 1-7, or by tools/make_synthetic_table.py for the demo):
    results/<mask>/patient_features_<tag>.csv   radiomic + conventional PET features per patient
    results/reproducible_features_<tag>.txt     features with ICC >= 0.75 (step 6)
    results/outcomes.csv                         endpoints and temporal split (step 7)
    results/acquisition_info.csv                 scanner / technical covariates (step 1)
    data/crf.csv                                 laboratory values (ldh, hb), optional date_diagnosis
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.common import DATA_DIR, PROJECT_ROOT, RESULTS_DIR

log = logging.getLogger("rlt_radiomics")

ENDPOINTS = {
    # name: (event column, time column, task)
    "os": ("os_event", "os_months", "survival"),
    "pfs": ("pfs_event", "pfs_months", "survival"),
    "response": ("response_12w", None, "binary"),
}


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load(Path(path or PROJECT_ROOT / "configs" / "modeling.yaml").read_text())


def radiomic_columns(df: pd.DataFrame, cfg: dict) -> list[str]:
    return [c for c in df.columns if c.startswith(tuple(cfg["radiomic_prefixes"]))]


def choose_reference(df: pd.DataFrame, cfg: dict) -> tuple[list[str], list[str]]:
    """Apply the prespecified missing-data rule to the reference variables.

    Returns (reference predictors actually used, log messages).
    """
    used, notes = [], []
    for v in cfg["reference_predictors"]:
        frac = df[v].isna().mean() if v in df else 1.0
        if frac > cfg["max_missing_fraction"]:
            b = cfg["backup_predictor"]
            if b not in df or df[b].isna().mean() > cfg["max_missing_fraction"]:
                raise ValueError(f"Reference variable '{v}' is {frac:.0%} missing and the backup '{b}' "
                                 "is not available (add 'date_diagnosis' to the CRF)")
            if b in used:
                raise ValueError(f"More than one reference variable needs the backup '{b}'")
            notes.append(f"{v}: {frac:.0%} missing > {cfg['max_missing_fraction']:.0%} -> replaced by {b}")
            used.append(b)
        else:
            notes.append(f"{v}: {frac:.0%} missing (kept; imputed inside resampling)")
            used.append(v)
    return used, notes


def build_analysis_table(cfg: dict, results: Path = RESULTS_DIR, crf_path: Path = DATA_DIR / "crf.csv") -> pd.DataFrame:
    tag, mask = cfg["feature_tag"], cfg["mask"]
    feats = pd.read_csv(results / mask / f"patient_features_{tag}.csv")
    keep = set(Path(results / f"reproducible_features_{tag}.txt").read_text().split())
    rad = radiomic_columns(feats, cfg)
    dropped = [c for c in rad if c not in keep]
    feats = feats.drop(columns=dropped)
    log.info("Radiomic features: %d reproducible kept, %d removed (ICC < threshold)", len(rad) - len(dropped), len(dropped))

    out = pd.read_csv(results / "outcomes.csv")
    crf = pd.read_csv(crf_path)
    lab_cols = [c for c in ("ldh", "hb", "date_diagnosis") if c in crf]
    df = out.merge(feats, on="patient_id", how="inner").merge(crf[["patient_id", *lab_cols]], on="patient_id", how="left")

    acq = results / "acquisition_info.csv"
    if acq.exists():
        a = pd.read_csv(acq)[["patient_id", "scanner"]]
        df = df.merge(a, on="patient_id", how="left")
    if "date_diagnosis" in df:
        df["time_since_diagnosis_months"] = (pd.to_datetime(df["date_cycle1"]) - pd.to_datetime(df["date_diagnosis"])).dt.days / 30.4375

    # inclusion criteria from step 7
    for c in ("eligible_pet_window", "eligible_followup"):
        if c in df:
            n0 = len(df)
            df = df[df[c].astype(str).str.lower() == "true"]
            if len(df) < n0:
                log.info("Excluded %d patients (%s)", n0 - len(df), c)

    bc = cfg["batch_column"]
    df["batch"] = df[bc].fillna("unknown").astype(str) if bc in df else "single"
    return df.reset_index(drop=True)


def transform_reference(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    for v in cfg.get("log_transform", []):
        if v in df:
            df[v] = np.log(df[v].astype(float).clip(lower=1e-3))
    return df


def get_xy(df: pd.DataFrame, endpoint: str, predictor_set: str, reference: list[str], cfg: dict):
    """(X, y, task, rows) for one endpoint and predictor set.

    predictor_set: 'reference' (4 clinical/PET variables), 'radiomics' (reproducible radiomic features
    only) or 'combined' (reference + radiomics). The batch column is added whenever radiomic features
    are present, so ComBat can be fitted inside each training fold.
    """
    from sksurv.util import Surv
    ev, tm, task = ENDPOINTS[endpoint]
    d = df.dropna(subset=[ev] + ([tm] if tm else []))
    rad = radiomic_columns(d, cfg)
    cols = {"reference": reference, "radiomics": rad, "combined": reference + rad}[predictor_set]
    if predictor_set != "reference":
        cols = cols + ["batch"]
    X = transform_reference(d[cols], cfg).reset_index(drop=True)
    if task == "survival":
        y = Surv.from_arrays(event=d[ev].astype(bool).to_numpy(), time=d[tm].astype(float).to_numpy())
    else:
        y = d[ev].astype(int).to_numpy()
    return X, y, task, d.reset_index(drop=True)


def harmonize_enabled(df: pd.DataFrame, cfg: dict) -> bool:
    counts = df["batch"].value_counts()
    ok = len(counts) > 1 and counts.min() >= cfg["min_batch_size"]
    if len(counts) > 1 and not ok:
        log.warning("ComBat disabled: smallest batch has %d patients (< %d)", counts.min(), cfg["min_batch_size"])
    return ok

"""Step 7 - Endpoints from the case-report form (independent of imaging features).

Definitions follow the proposal (time zero = first [177Lu]Lu-PSMA-617 cycle):
  * OS (primary)       : death from any cause; alive -> censored at date_last_followup.
  * PFS                : first of PSA progression (PCWG3), radiographic progression (RECIP 1.0 /
                         PCWG3), unequivocal clinical progression, or death  -> date_progression
                         is the first progression date of any type (adjudicated, blinded).
  * Initial response   : PSA decline >= 50 % at 12 weeks OR RECIP 1.0 CR/PR at 12 weeks.
                         Death or progression before 12 weeks = non-responder.
  * Recurrence         : responders only, from the 12-week landmark to progression;
                         death without recurrence = competing event (event code 2).

Also checks eligibility (PET <= 8 weeks before cycle 1; >= 6 months possible follow-up)
and assigns the temporal-validation split (most recent ~30 % of patients by date of cycle 1).

Input : data/crf.csv with columns
        patient_id, center, date_pet, date_cycle1, date_last_followup, date_death,
        date_progression, psa_baseline, psa_12w, recip_12w (CR/PR/SD/PD or empty)
        Dates as YYYY-MM-DD; empty = not occurred / not available.
Output: results/outcomes.csv

Usage:
    python -m pipeline.step7_outcomes --data-lock 2027-06-30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.common import DATA_DIR, RESULTS_DIR, log, setup_logging

DAYS_PER_MONTH = 365.25 / 12
LANDMARK_DAYS = 84  # 12 weeks
DATE_COLS = ["date_pet", "date_cycle1", "date_last_followup", "date_death", "date_progression"]


def build_outcomes(crf: pd.DataFrame, data_lock: pd.Timestamp, temporal_fraction: float) -> pd.DataFrame:
    df = crf.copy()
    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    t0 = df["date_cycle1"]
    last = df[["date_last_followup", "date_death"]].max(axis=1)

    out = pd.DataFrame({"patient_id": df["patient_id"], "center": df.get("center")})

    # eligibility
    pet_gap = (t0 - df["date_pet"]).dt.days
    out["pet_to_cycle1_days"] = pet_gap
    out["eligible_pet_window"] = pet_gap.between(0, 56)
    out["eligible_followup"] = t0 <= data_lock - pd.DateOffset(months=6)

    # OS
    out["os_event"] = df["date_death"].notna().astype(int)
    out["os_months"] = (last - t0).dt.days / DAYS_PER_MONTH

    # PFS
    pfs_date = df[["date_progression", "date_death"]].min(axis=1)
    out["pfs_event"] = pfs_date.notna().astype(int)
    out["pfs_months"] = (pfs_date.fillna(last) - t0).dt.days / DAYS_PER_MONTH

    # Initial response at the 12-week landmark
    psa_change = 100 * (df["psa_12w"] - df["psa_baseline"]) / df["psa_baseline"]
    out["psa_change_12w_pct"] = psa_change
    recip = df.get("recip_12w", pd.Series(index=df.index, dtype=object)).astype("string").str.upper().str.strip()
    early_event = (pfs_date - t0).dt.days < LANDMARK_DAYS
    responder = (psa_change <= -50) | recip.isin(["CR", "PR"])
    assessed = psa_change.notna() | recip.isin(["CR", "PR", "SD", "PD"])
    response = pd.Series(pd.NA, index=df.index, dtype="Int64")
    response[assessed] = responder[assessed].astype(int)
    response[early_event.fillna(False)] = 0
    out["response_12w"] = response

    # Recurrence after response (competing risk), time from landmark
    landmark = t0 + pd.Timedelta(days=LANDMARK_DAYS)
    resp = (response == 1).fillna(False)
    prog_after = df["date_progression"].notna()
    death_wo_prog = df["date_death"].notna() & ~prog_after
    event = np.select([prog_after, death_wo_prog], [1, 2], default=0)
    end = df["date_progression"].fillna(df["date_death"]).fillna(last)
    out["recur_event"] = pd.Series(event, index=df.index).where(resp)  # 0 censored, 1 recurrence, 2 death
    out["recur_months"] = ((end - landmark).dt.days / DAYS_PER_MONTH).where(resp)
    bad = resp & (out["recur_months"] < 0)
    if bad.any():
        log.warning("Responders with progression before the landmark (check CRF): %s",
                    ", ".join(out.loc[bad, "patient_id"]))
        out.loc[bad, ["recur_event", "recur_months"]] = np.nan

    # Temporal validation split (TRIPOD 2b): most recent fraction of patients by cycle-1 date
    order = t0.rank(method="first")
    n_dev = int(round((1 - temporal_fraction) * t0.notna().sum()))
    out["temporal_set"] = np.where(order > n_dev, "validation", "development")
    out["date_cycle1"] = t0.dt.date
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crf", type=Path, default=DATA_DIR / "crf.csv")
    ap.add_argument("--data-lock", required=True, help="data-lock date YYYY-MM-DD")
    ap.add_argument("--temporal-fraction", type=float, default=0.30)
    args = ap.parse_args()
    setup_logging()

    out = build_outcomes(pd.read_csv(args.crf), pd.Timestamp(args.data_lock), args.temporal_fraction)
    RESULTS_DIR.mkdir(exist_ok=True)
    out.to_csv(RESULTS_DIR / "outcomes.csv", index=False)

    eligible = out["eligible_pet_window"] & out["eligible_followup"]
    log.info("%d patients, %d eligible (PET window + follow-up)", len(out), eligible.sum())
    log.info("OS events %d | PFS events %d | responders %s/%s | recurrences %d (competing deaths %d)",
             out["os_event"].sum(), out["pfs_event"].sum(), int((out["response_12w"] == 1).sum()),
             int(out["response_12w"].notna().sum()), int((out["recur_event"] == 1).sum()),
             int((out["recur_event"] == 2).sum()))
    log.info("Temporal split: %s", out["temporal_set"].value_counts().to_dict())


if __name__ == "__main__":
    main()

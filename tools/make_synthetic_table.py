"""Synthetic phase-1 OUTPUT tables for testing phase 2 (steps 8-13). No images, no patient data.

    python tools/make_synthetic_table.py --n 220

Writes the same files that steps 1-7 produce (results/outcomes.csv, results/reader1/patient_features_*.csv,
results/reproducible_features_*.txt, results/acquisition_info.csv) plus data/crf.csv. The outcomes
depend on SUVmean, PSMA-TV, hemoglobin and on a latent 'heterogeneity' that only some radiomic
features capture, so radiomics truly adds information in this simulation.
Refuses to run if results/outcomes.csv already exists (use --force only for demo data).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAG = "pet_bw0.5"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=220)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    res, data = ROOT / "results", ROOT / "data"
    if (res / "outcomes.csv").exists() and not args.force:
        sys.exit("results/outcomes.csv exists - refusing to overwrite (real data?). Use --force for demo data only.")
    (res / "reader1").mkdir(parents=True, exist_ok=True)
    data.mkdir(exist_ok=True)

    rng = np.random.default_rng(args.seed)
    n = args.n
    pid = [f"SYN{i:03d}" for i in range(1, n + 1)]
    center = rng.choice(["IKH", "SH"], n, p=[0.6, 0.4])
    scanner = np.where(center == "IKH", rng.choice(["Biograph", "Discovery"], n, p=[0.7, 0.3]), "Discovery")
    c1 = pd.Timestamp("2019-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 6 * 365, n)), "D")
    dx = c1 - pd.to_timedelta(rng.lognormal(4.0, 0.5, n) * 30.4, "D")

    suvmean = rng.lognormal(2.2, 0.35, n)
    tv = rng.lognormal(5.0, 1.0, n)
    hb = rng.normal(11.8, 1.5, n)
    ldh = rng.lognormal(5.6, 0.4, n)
    het = rng.normal(0, 1, n)                      # latent heterogeneity, seen only by radiomics
    shift = {"Biograph": 0.0, "Discovery": 0.7}
    scale = {"Biograph": 1.0, "Discovery": 1.3}
    feats = {"n_lesions": rng.poisson(np.clip(tv / 20, 1, 60)) + 1, "suvmax": suvmean * rng.uniform(2, 4, n),
             "suvmean": suvmean, "psma_tv_ml": tv, "tl_psma": suvmean * tv}
    rad_names = []
    for k in range(80):
        prefix = ["wb_", "vwm_", "sd_", "range_"][k % 4]
        name = f"{prefix}original_feat{k:02d}"
        base = 0.8 * het + rng.normal(0, 0.6, n) if k < 6 else rng.normal(0, 1, n)
        if 6 <= k < 12:
            base = np.log(tv) / np.log(tv).std() + rng.normal(0, 0.3, n)
        feats[name] = base * np.array([scale[s] for s in scanner]) + np.array([shift[s] for s in scanner])
        rad_names.append(name)
    pd.DataFrame({"patient_id": pid, **feats}).to_csv(res / "reader1" / f"patient_features_{TAG}.csv", index=False)
    keep = ["n_lesions", "suvmax", "suvmean", "psma_tv_ml", "tl_psma"] + [r for i, r in enumerate(rad_names) if i % 5 != 4]
    (res / f"reproducible_features_{TAG}.txt").write_text("\n".join(keep) + "\n")

    lp = (-0.3 * (np.log(suvmean) - np.log(suvmean).mean()) / np.log(suvmean).std()
          + 0.5 * (np.log(tv) - np.log(tv).mean()) / np.log(tv).std() - 0.3 * (hb - 11.8) / 1.5 + 0.6 * het)
    t_death = rng.exponential(15 * np.exp(-lp))
    t_prog = np.minimum(rng.exponential(7 * np.exp(-lp)), t_death)
    t_cens = rng.uniform(6, 40, n)
    os_m, os_e = np.minimum(t_death, t_cens), (t_death <= t_cens).astype(int)
    pfs_m, pfs_e = np.minimum(t_prog, t_cens), (t_prog <= t_cens).astype(int)
    resp = ((rng.random(n) < 1 / (1 + np.exp(-(0.2 - 0.9 * lp)))) & (pfs_m > 2.8)).astype(int)
    rec_e, rec_m = np.full(n, np.nan), np.full(n, np.nan)
    for i in np.where(resp == 1)[0]:
        lm = 12 / 4.345
        if pfs_e[i] and t_prog[i] < t_death[i] and pfs_m[i] > lm:
            rec_e[i], rec_m[i] = 1, pfs_m[i] - lm
        elif os_e[i] and os_m[i] > lm:
            rec_e[i], rec_m[i] = 2, os_m[i] - lm
        else:
            rec_e[i], rec_m[i] = 0, max(os_m[i] - lm, 0.1)
    order = np.argsort(c1)
    temporal = np.where(np.argsort(order) >= int(0.7 * n), "validation", "development")
    pd.DataFrame({"patient_id": pid, "center": center, "pet_to_cycle1_days": rng.integers(5, 50, n),
                  "eligible_pet_window": True, "eligible_followup": True, "os_event": os_e, "os_months": os_m,
                  "pfs_event": pfs_e, "pfs_months": pfs_m, "psa_change_12w_pct": np.nan, "response_12w": resp,
                  "recur_event": rec_e, "recur_months": rec_m, "temporal_set": temporal,
                  "date_cycle1": c1.strftime("%Y-%m-%d")}).to_csv(res / "outcomes.csv", index=False)
    ldh_obs = ldh.copy()
    ldh_obs[rng.random(n) < 0.10] = np.nan
    pd.DataFrame({"patient_id": pid, "center": center, "date_cycle1": c1.strftime("%Y-%m-%d"),
                  "date_diagnosis": dx.strftime("%Y-%m-%d"), "ldh": ldh_obs, "hb": hb}).to_csv(data / "crf.csv", index=False)
    pd.DataFrame({"patient_id": pid, "scanner": scanner, "institution": center}).to_csv(res / "acquisition_info.csv", index=False)
    print(f"Synthetic phase-1 tables for {n} patients written to results/ and data/crf.csv")


if __name__ == "__main__":
    main()

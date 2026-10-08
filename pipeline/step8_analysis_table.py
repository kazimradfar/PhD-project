"""Step 8: assemble the analysis table and check it against the prespecified plan.

    python -m pipeline.step8_analysis_table

Run this only AFTER the analysis plan is registered (OSF): from here on, imaging features are
linked to outcomes. Writes results/analysis_table.csv and results/analysis_table_report.txt.
"""
import argparse

from modeling.data import build_analysis_table, choose_reference, harmonize_enabled, load_config, radiomic_columns
from pipeline.common import RESULTS_DIR, log, setup_logging


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    df = build_analysis_table(cfg)
    ref, notes = choose_reference(df, cfg)
    n_ev = int(df["os_event"].sum())
    lines = [f"Patients: {len(df)}", f"OS events: {n_ev}", f"PFS events: {int(df['pfs_event'].sum())}",
             f"Responders at 12 weeks: {int(df['response_12w'].sum())}/{int(df['response_12w'].notna().sum())}",
             f"Reproducible radiomic features: {len(radiomic_columns(df, cfg))}",
             "Reference model:", *[f"  {n}" for n in notes], f"  -> used: {', '.join(ref)}",
             f"Batches ({cfg['batch_column']}): {df['batch'].value_counts().to_dict()}",
             f"ComBat: {'on' if harmonize_enabled(df, cfg) else 'off'}",
             f"Final regression model: 1 radiomic score + {len(ref)} reference variables = {len(ref) + 1} parameters "
             f"(compare with the Riley calculation in the proposal: about 200 patients for 5 parameters)"]
    if len(df) < 200:
        lines.append(f"WARNING: {len(df)} patients < 200 assumed in the sample-size calculation; report the larger uncertainty")
    df.to_csv(RESULTS_DIR / "analysis_table.csv", index=False)
    (RESULTS_DIR / "analysis_table_report.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        log.info(line)


if __name__ == "__main__":
    main()

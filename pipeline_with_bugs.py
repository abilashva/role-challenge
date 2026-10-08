"""
Prenatal Imaging Data Pipeline
Origin Medical Research Lab -- Clinical Data Engineering

Ingests patient/study metadata and fetal biometry measurements, cleans and
normalizes key fields, merges the two sources into one analysis-ready
table, and produces a summary report + a visualization comparing clinics.

Run:
    python pipeline_with_bugs.py

Inputs (same folder):
    patient_studies.csv
    fetal_biometry.csv

Output:
    analysis_ready.csv
    clinic_summary.png
"""

import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDIES_PATH = "patient_studies.csv"
BIOMETRY_PATH = "fetal_biometry.csv"


def load_data(studies_path, biometry_path):
    studies = pd.read_csv(studies_path)
    biometry = pd.read_csv(biometry_path)
    return studies, biometry

NOTES_STATS = {"repaired": 0, "unreadable": 0}
def parse_notes(notes_raw):
    """
    Some `notes` values contain a JSON blob with delivery context
    (twin pregnancy, fetal presentation). Extract it into a dict;
    plain free-text notes or missing notes just return {}.
    """
    if pd.isna(notes_raw):
        return {}
    text = str(notes_raw).strip()
    if not text.startswith("{"):
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass


    fixed = text
    if not fixed.endswith("}"):
        fixed += "}"
    fixed = re.sub(r",\s*}", "}", fixed)

    try:
        result = json.loads(fixed)
        NOTES_STATS["repaired"] += 1
        return result
    except json.JSONDecodeError:
        NOTES_STATS["unreadable"] += 1
        return {}


def normalize_gestational_age(ga_raw):
    """
    Convert the many gestational-age formats we receive from clinics into
    decimal weeks:
        "24w3d"    -> 24.43
        "24.4"     -> 24.4
        "26 weeks" -> 26.0
    """
    if pd.isna(ga_raw):
        return np.nan

    ga_str = str(ga_raw).strip().lower()

    match = re.match(r"(\d+)w(\d+)d", ga_str)
    if match:
        weeks, days = int(match.group(1)), int(match.group(2))
        return weeks + days / 7  # convert extra days into a fraction of a week

    match = re.match(r"(\d+)\s*weeks?", ga_str)
    if match:
        return float(match.group(1))

    try:
        value = float(ga_str)
    except ValueError:
        return np.nan

    if value > 45:
        return value / 7
    return value


def normalize_device_manufacturer(name_raw):
    """Collapse manufacturer name variants down to a canonical label."""
    if pd.isna(name_raw):
        return "Unknown"
    name = str(name_raw).strip().lower()
    if "ge" in name:
        return "GE Healthcare"
    if "philips" in name:
        return "Philips"
    if "samsung" in name:
        return "Samsung"
    if "canon" in name:
        return "Canon Medical"
    return name.title()


def clean_studies(studies):
    studies = studies.copy()
    studies["gestational_age_weeks"] = studies["gestational_age_at_scan"].apply(
        normalize_gestational_age
    )
    studies["delivery_context"] = studies["notes"].apply(parse_notes)
    studies["scan_date_parsed"] = pd.to_datetime(studies["scan_date"], errors="coerce")
    return studies


def clean_biometry(biometry):
    biometry = biometry.copy()
    biometry["device_manufacturer_clean"] = biometry["device_manufacturer"].apply(
        normalize_device_manufacturer
    )
    return biometry


def merge_datasets(studies, biometry):
    """Join clinical metadata with the corresponding imaging measurements."""
    merged = studies.merge(biometry, on="study_id", how="inner")
    return merged


def flag_high_risk(merged, clinic_thresholds):
    """
    Flag a study as high risk if its gestational age falls below the
    clinic-specific threshold for early anomaly review.

    NOTE: this currently runs nightly against ~450 studies. Ops wants to
    point it at the full historical archive (~500K studies) next quarter.
    """
    flags = []
    for i in range(len(merged)):
        row = merged.iloc[i]
        threshold = None
        for clinic_row in clinic_thresholds.itertuples():
            if clinic_row.clinic_id == row["clinic_id"]:
                threshold = clinic_row.min_ga_weeks
                break
        is_flagged = (
            pd.notna(row["gestational_age_weeks"])
            and threshold is not None
            and row["gestational_age_weeks"] < threshold
        )
        flags.append(is_flagged)
    merged = merged.copy()
    merged["early_review_flag"] = flags
    return merged


def summarize_by_clinic(merged):
    summary = (
        merged.groupby("clinic_id")
        .agg(
            n_studies=("study_id", "count"),
            avg_gestational_age=("gestational_age_weeks", "mean"),
            avg_bpd_mm=("bpd_mm", "mean"),
            pct_flagged=("early_review_flag", "mean"),
        )
        .reset_index()
    )
    return summary


def plot_clinic_summary(summary, out_path="clinic_summary.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(summary["clinic_id"], summary["avg_gestational_age"])
    ax.set_ylabel("Avg. gestational age at scan (weeks)")
    ax.set_title("Average Gestational Age at Scan by Clinic")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    studies, biometry = load_data(STUDIES_PATH, BIOMETRY_PATH)
    print(f"Loaded {len(studies)} study records and {len(biometry)} biometry records.")

    studies = clean_studies(studies)
    print(f"Notes repaired: {NOTES_STATS['repaired']}, unreadable: {NOTES_STATS['unreadable']}")
    biometry = clean_biometry(biometry)

    merged = merge_datasets(studies, biometry)
    print(f"Merged dataset has {len(merged)} rows.")

    clinic_thresholds = pd.DataFrame(
        {
            "clinic_id": ["CLN-01", "CLN-02", "CLN-03", "CLN-04", "CLN-05", "CLN-06", "CLN-07"],
            "min_ga_weeks": [18, 20, 18, 22, 20, 18, 20],
        }
    )
    merged = flag_high_risk(merged, clinic_thresholds)

    summary = summarize_by_clinic(merged)
    print("\nClinic summary:")
    print(summary.to_string(index=False))

    merged.to_csv("analysis_ready.csv", index=False)
    plot_clinic_summary(summary)
    print("\nWrote analysis_ready.csv and clinic_summary.png")


if __name__ == "__main__":
    main()

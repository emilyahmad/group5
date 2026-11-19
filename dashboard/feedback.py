"""Clinician validation log (TICICDS fig. 4: 'Do you agree? Which actions did you take?').

Stored as a CSV so the team can analyse agreement / override rates later.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

FEEDBACK_CSV = Path(__file__).resolve().parent / "feedback" / "validations.csv"
FIELDS = ["timestamp", "image_id", "model_score", "model_band", "agreement",
          "actions", "comment", "reviewer_role"]
AGREEMENT = ["Agree", "Disagree", "Unsure"]
ACTIONS = [
    "Reviewed film myself",
    "Discussed with radiologist / physician",
    "Ordered lateral or repeat imaging",
    "Ordered labs (CBC, CRP, cultures)",
    "Started / adjusted antibiotics",
    "No action",
]
ROLES = ["Physician", "Resident / fellow", "Nurse / NP / PA", "Radiologist", "Other"]


def record(row: dict) -> None:
    FEEDBACK_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not FEEDBACK_CSV.exists()
    with FEEDBACK_CSV.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"timestamp": datetime.now().isoformat(timespec="seconds"), **row})


def load() -> pd.DataFrame:
    if not FEEDBACK_CSV.exists():
        return pd.DataFrame(columns=FIELDS)
    return pd.read_csv(FEEDBACK_CSV)


def latest_by_image() -> dict[str, dict]:
    df = load()
    if df.empty:
        return {}
    return df.sort_values("timestamp").groupby("image_id").tail(1).set_index("image_id").to_dict("index")

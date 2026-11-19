"""CheXpert / CheXlocalize loader.

Expected download (Stanford AIMI → "CheXlocalize"), unzipped anywhere under
`data/chexlocalize/` (or set CHEXLOCALIZE_DIR):

    data/chexlocalize/
      CheXpert/val/patient64541/study1/view1_frontal.jpg ...
      CheXpert/test/...
      CheXpert/val_labels.csv, CheXpert/test_labels.csv
      gt_segmentations_val.json, gt_segmentations_test.json

Files are discovered by name, so the exact nesting does not matter.

CheXlocalize has no *Pneumonia* masks. We use the union of the radiologist
masks for Consolidation and Airspace (Lung) Opacity as the closest reference
region for pneumonia, and say so in the UI.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .model import PreparedImage, orig_to_display

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "data" / "chexlocalize"
GT_PATHOLOGIES = ("Consolidation", "Airspace Opacity", "Lung Opacity")
LABEL_COLS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity",
    "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis",
    "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]
_ID_RE = re.compile(r"(patient\d+)[/\\](study\d+)[/\\](view\d+_\w+?)\.(?:jpg|png)$", re.I)


def data_root() -> Path:
    return Path(os.environ.get("CHEXLOCALIZE_DIR", DEFAULT_ROOT))


def label_csvs(root: Path) -> list[Path]:
    """CheXlocalize names them val_labels.csv / test_labels.csv; CheXpert itself
    (incl. the Kaggle 'small' copy) uses valid.csv / test.csv."""
    names = {"val_labels.csv", "test_labels.csv", "valid.csv", "val.csv", "test.csv"}
    return sorted(p for p in root.rglob("*.csv") if p.name.lower() in names)


def available(root: Path | None = None) -> bool:
    root = root or data_root()
    return root.exists() and bool(label_csvs(root)) and bool(_image_index(root))


def diagnose(root: Path | None = None) -> str:
    """Human-readable explanation of what is missing under `root`."""
    root = root or data_root()
    if not root.exists():
        return f"{root} does not exist."
    n_img, csvs = len(_image_index(root)), label_csvs(root)
    gts = sorted(p.name for p in root.rglob("gt_segmentations_*.json"))
    lines = [f"Looked in {root}:",
             f"  X-ray images (patientXXXXX/studyN/viewN_*.jpg): {n_img}",
             f"  label CSVs (val_labels.csv / test_labels.csv / valid.csv): {[c.name for c in csvs] or 'none'}",
             f"  radiologist segmentations: {gts or 'none'}"]
    if not n_img or not csvs:
        lines.append("Missing the CheXpert images and/or label CSVs. These come in the "
                     "'CheXpert' folder of the Stanford AIMI CheXlocalize download (val/, test/, "
                     "val_labels.csv, test_labels.csv), separate from the annotation JSONs. "
                     "See dashboard/README.md.")
    return "\n".join(lines)


def _image_index(root: Path) -> dict[str, Path]:
    idx = {}
    for p in root.rglob("*"):
        if p.suffix.lower() in {".jpg", ".png"}:
            m = _ID_RE.search(p.as_posix())
            if m:
                idx["_".join(m.groups())] = p
    return idx


def load_index(root: Path | None = None) -> pd.DataFrame:
    """One row per frontal image with metadata and reference labels (`label_*`)."""
    root = root or data_root()
    images = _image_index(root)
    frames = []
    for csv in label_csvs(root):
        split = "test" if "test" in csv.name.lower() else "val"
        df = pd.read_csv(csv)
        path_col = next(c for c in df.columns if c.lower() == "path")
        ids = df[path_col].astype(str).str.extract(_ID_RE)
        df["id"] = ids[0] + "_" + ids[1] + "_" + ids[2]
        df["patient"], df["study"], df["view"] = ids[0], ids[1], ids[2]
        df["split"] = split
        frames.append(df.dropna(subset=["id"]))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[df["view"].str.contains("frontal", case=False)]
    df["image_path"] = df["id"].map(lambda i: str(images[i]) if i in images else None)
    df = df.dropna(subset=["image_path"]).drop_duplicates("id")

    out = pd.DataFrame({
        "id": df["id"], "split": df["split"], "patient": df["patient"],
        "study": df["study"], "image_path": df["image_path"],
        "sex": df.get("Sex", pd.Series(index=df.index, dtype=object)),
        "age": pd.to_numeric(df.get("Age", pd.Series(index=df.index, dtype=float)), errors="coerce"),
        "ap_pa": df.get("AP/PA", pd.Series(index=df.index, dtype=object)),
    })
    for c in LABEL_COLS:
        if c in df.columns:
            out[f"label_{c}"] = pd.to_numeric(df[c], errors="coerce")
    gt_ids = set()
    for split in ("val", "test"):
        gt_ids |= {k for k, v in _gt_json(str(root), split).items() if _has_pneumonia_gt(v)}
    out["has_gt"] = out["id"].isin(gt_ids)
    return out.reset_index(drop=True)


# ---------------------------------------------------------------- ground truth
@lru_cache(maxsize=4)
def _gt_json(root: str, split: str) -> dict:
    hits = sorted(Path(root).rglob(f"gt_segmentations_{split}*.json"))
    return json.loads(hits[0].read_text()) if hits else {}


def _has_pneumonia_gt(entry: dict) -> bool:
    return any(_decode(entry[p]).any() for p in GT_PATHOLOGIES if p in entry)


def _decode(rle: dict) -> np.ndarray:
    from pycocotools import mask as mask_utils

    rle = dict(rle)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode()
    return mask_utils.decode(rle).astype(bool)


def gt_mask(image_id: str, split: str, prep: PreparedImage, root: Path | None = None) -> np.ndarray | None:
    """Union of Consolidation + Airspace Opacity masks in the display frame, or None."""
    entry = _gt_json(str(root or data_root()), split).get(image_id)
    if not entry:
        return None
    union = None
    for p in GT_PATHOLOGIES:
        if p in entry:
            m = _decode(entry[p])
            union = m if union is None else (union | m)
    if union is None or not union.any():
        return None
    # masks are at the original acquisition size; same aspect as our image, so resizing aligns them
    return orig_to_display(union, prep)

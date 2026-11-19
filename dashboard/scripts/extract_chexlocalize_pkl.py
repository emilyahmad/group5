"""Extract X-ray images + labels from CheXlocalize's gradcam_maps_val/*.pkl files.

The Stanford AIMI CheXlocalize release ships the CheXpert *validation* X-rays
only inside these pickles (no .jpg / .csv). Each pickle holds:
    cxr_img   float tensor [3, 320, 320] in [0, 1]  (image stretched to a square)
    cxr_dims  original (width, height)
    task, gt  one finding and its radiologist label (0/1)
    map, prob the dataset authors' own Grad-CAM map and score (not used here)

This script writes, under data/chexlocalize/CheXpert/:
    val/patientX/studyY/viewZ_frontal.jpg   image restored to its original aspect ratio
    val_labels.csv                          one row per image, one column per finding

Pickles can execute code when loaded, so they are read with a restricted
unpickler that only allows the three tensor-building globals these files use,
and loads nested tensor bytes with torch's weights_only loader.

Usage:  python dashboard/scripts/extract_chexlocalize_pkl.py [--all-views]
"""
from __future__ import annotations

import argparse
import collections
import io
import pickle
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch._utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xai.dataset import data_root  # noqa: E402

NAME_RE = re.compile(r"^(patient\d+)_(study\d+)_(view\d+_(?:frontal|lateral))_(.+)_map\.pkl$")
# CheXlocalize task name -> CheXpert label column used by the dashboard
TASK_TO_LABEL = {"Airspace Opacity": "Lung Opacity"}


def _safe_load_from_bytes(b: bytes):
    return torch.load(io.BytesIO(b), weights_only=True, map_location="cpu")


class _SafeUnpickler(pickle.Unpickler):
    ALLOWED = {
        ("collections", "OrderedDict"): collections.OrderedDict,
        ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
        ("torch.storage", "_load_from_bytes"): _safe_load_from_bytes,
    }

    def find_class(self, module, name):
        try:
            return self.ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}") from None


def safe_load(path: Path) -> dict:
    with path.open("rb") as f:
        return _SafeUnpickler(f).load()


def restore_aspect(img01: torch.Tensor, dims) -> np.ndarray:
    """[3,320,320] float -> uint8 grayscale at the original aspect ratio.

    `cxr_dims` is (width, height): verified visually (text and devices only
    look undistorted this way) and it matches the [rows, cols] of the
    radiologist masks. Short side is kept at 320 px, since upsampling adds no detail.
    """
    gray = (img01[0].numpy().clip(0, 1) * 255).astype(np.uint8)
    w, h = int(dims[0]), int(dims[1])
    scale = gray.shape[0] / min(w, h)
    return cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_CUBIC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--all-views", action="store_true", help="also extract lateral views")
    args = ap.parse_args()
    root = args.root or data_root()
    pkls = sorted(root.rglob("gradcam_maps_val/*_map.pkl"))
    if not pkls:
        sys.exit(f"No gradcam_maps_val/*.pkl under {root}")

    out_dir = root / "CheXpert"
    labels: dict[str, dict] = collections.defaultdict(dict)
    written = 0
    for i, p in enumerate(pkls, 1):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        patient, study, view, _ = m.groups()
        if "lateral" in view and not args.all_views:
            continue
        d = safe_load(p)
        rel = f"val/{patient}/{study}/{view}.jpg"
        row = labels[rel]
        row["Path"] = rel
        row[TASK_TO_LABEL.get(d["task"], d["task"])] = int(d["gt"])
        img_path = out_dir / rel
        if not img_path.exists():
            img_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(img_path), restore_aspect(d["cxr_img"], d["cxr_dims"]),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            row["Original Width"], row["Original Height"] = map(int, d["cxr_dims"])
            row["Frontal/Lateral"] = "Lateral" if "lateral" in view else "Frontal"
            written += 1
        if i % 200 == 0:
            print(f"  {i}/{len(pkls)} pickles read", flush=True)

    df = pd.DataFrame(labels.values())
    cols = ["Path", "Frontal/Lateral", "Original Width", "Original Height"]
    df = df[cols + sorted(c for c in df.columns if c not in cols)]
    df.to_csv(out_dir / "val_labels.csv", index=False)
    print(f"Wrote {written} new images and {len(df)} label rows to {out_dir}")
    print("Positives per finding:", {c: int(df[c].sum()) for c in df.columns if c not in cols})


if __name__ == "__main__":
    main()

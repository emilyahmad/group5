"""Batch-run the model over CheXlocalize and cache cohort-level results.

Outputs (dashboard/cache/):
  predictions.parquet  one row per frontal image: metadata, reference labels,
                       all model probabilities (prob_*), OOD score, Grad-CAM
                       share outside the lungs, and per-method IoU / pointing
                       hit where radiologist masks exist.
  metrics.json         AUROC per finding + localization summary per method.

Usage:
  python dashboard/scripts/precompute.py                 # Grad-CAM + IG
  python dashboard/scripts/precompute.py --methods gradcam,ig,occlusion
  python dashboard/scripts/precompute.py --limit 20      # quick test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xai import dataset as D  # noqa: E402
from xai import model as M  # noqa: E402
from xai import regions as R  # noqa: E402
from xai import reliability as Rel  # noqa: E402
from xai import saliency as S  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "cache"
AUROC_FINDINGS = {"Pneumonia": "Pneumonia", "Consolidation": "Consolidation",
                  "Lung Opacity": "Lung Opacity", "Effusion": "Pleural Effusion",
                  "Atelectasis": "Atelectasis", "Edema": "Edema"}  # model name -> label name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None, help="CheXlocalize folder")
    ap.add_argument("--methods", default="gradcam,ig")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=CACHE)
    args = ap.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    torch.set_grad_enabled(True)

    root = args.root or D.data_root()
    idx = D.load_index(root)
    if idx.empty:
        sys.exit(D.diagnose(root))
    if args.limit:
        # keep the GT cases in a small sample so localization metrics aren't empty
        idx = pd.concat([idx[idx.has_gt].head(args.limit // 2),
                         idx[~idx.has_gt].head(args.limit)]).head(args.limit)
    print(f"{len(idx)} frontal images ({int(idx.has_gt.sum())} with pneumonia-related masks)")

    clf, seg, ae = M.load_classifier(), M.load_segmenter(), M.load_autoencoder()
    t_idx = M.target_index(clf, M.PNEUMONIA)
    rows, t0 = [], time.time()
    for i, row in enumerate(idx.to_dict("records"), 1):
        img = M.prepare(M.read_image(row["image_path"]))
        row.update({f"prob_{k}": v for k, v in M.predict(clf, img.x).items()})
        row["ood"] = Rel.ood_score(ae, img.x)
        zones = R.zone_masks(R.lung_masks(seg, img.x_seg))
        gt = (D.gt_mask(row["id"], row["split"], img.crop_box, img.orig_shape, root)
              if row["has_gt"] else None)
        for m in methods:
            if m != "gradcam" and gt is None:
                continue  # slower methods only where we can score them
            sal = S.compute(m, clf, img.x, t_idx)
            if m == "gradcam":
                row["gradcam_outside"] = R.zone_contributions(sal, zones)[R.OUTSIDE]
            if gt is not None:
                row[f"iou_{m}"] = S.iou(S.binarize(sal), gt)
                row[f"hit_{m}"] = S.pointing_hit(sal, gt)
        rows.append(row)
        if i % 25 == 0 or i == len(idx):
            print(f"  {i}/{len(idx)}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out / "predictions.parquet", index=False)
    metrics = compute_metrics(df, methods)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def compute_metrics(df: pd.DataFrame, methods: list[str]) -> dict:
    out = {"n_images": int(len(df)), "auroc": {}, "localization": {},
           "generated": time.strftime("%Y-%m-%d"),
           "model": "torchxrayvision densenet121-res224-all",
           "gt_definition": "union of radiologist Consolidation + Airspace Opacity masks (CheXlocalize)",
           "binarization": "saliency >= 0.5 of its 99.5th percentile"}
    for model_name, label in AUROC_FINDINGS.items():
        col = f"label_{label}"
        if col not in df or f"prob_{model_name}" not in df:
            continue
        sub = df[df[col].isin([0, 1])]
        npos = int((sub[col] == 1).sum())
        entry = {"n": int(len(sub)), "n_pos": npos, "auroc": None}
        if 0 < npos < len(sub):
            entry["auroc"] = round(float(roc_auc_score(sub[col], sub[f"prob_{model_name}"])), 3)
        out["auroc"][model_name] = entry
    if "label_Pneumonia" not in df and {"label_Consolidation", "label_Lung Opacity"} <= set(df.columns):
        # CheXlocalize has no pneumonia label: score the pneumonia output against the
        # closest radiographic proxy, and say so wherever it is shown.
        proxy = ((df["label_Consolidation"] == 1) | (df["label_Lung Opacity"] == 1)).astype(int)
        npos = int(proxy.sum())
        out["auroc"]["Pneumonia"] = {
            "n": int(len(df)), "n_pos": npos, "proxy": "consolidation or lung opacity label",
            "auroc": round(float(roc_auc_score(proxy, df["prob_Pneumonia"])), 3) if 0 < npos < len(df) else None}
    for m in methods:
        if f"iou_{m}" in df:
            s = df[f"iou_{m}"].dropna()
            out["localization"][m] = {"n": int(len(s)),
                                      "mean_iou": round(float(s.mean()), 3) if len(s) else None,
                                      "hit_rate": round(float(df[f"hit_{m}"].dropna().astype(float).mean()), 3) if len(s) else None}
    return out


if __name__ == "__main__":
    main()

"""Batch-run the model over CheXlocalize and cache cohort-level results.

Outputs (dashboard/cache/):
  predictions.parquet  one row per frontal image: metadata, reference labels,
                       all model probabilities (prob_*), OOD score, Grad-CAM
                       share outside the lungs, and per-method IoU / pointing
                       hit where radiologist masks exist.
  metrics.json         AUROC per finding + localization summary per method.

Usage:
  python dashboard/scripts/precompute.py                 # Grad-CAM + IG + Occlusion, 2 threads
  python dashboard/scripts/precompute.py --methods gradcam,ig,occlusion
  python dashboard/scripts/precompute.py --limit 20      # quick test
  python dashboard/scripts/precompute.py --threads 0     # all CPU cores: faster, but hot/loud
  python dashboard/scripts/precompute.py --methods gradcam,ig   # skip occlusion: ~3x faster
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from xai import dataset as D  # noqa: E402
from xai import evaluation as E  # noqa: E402
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
    ap.add_argument("--methods", default="gradcam,ig,occlusion")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=CACHE)
    ap.add_argument("--threads", type=int, default=2,
                    help="CPU threads for PyTorch. Default 2 keeps a laptop cool and quiet "
                         "(slower); use 0 for all cores.")
    args = ap.parse_args()
    methods = [m for m in args.methods.split(",") if m]
    torch.set_grad_enabled(True)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

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
        lungs = R.lung_masks(seg, img)
        zones = R.zone_masks(lungs)
        gt = D.gt_mask(row["id"], row["split"], img, root) if row["has_gt"] else None
        if gt is not None:
            # baselines a heatmap has to beat: a random peak, and "just highlight both lungs"
            row["gt_area_frac"] = float(gt.mean())
            row["iou_lungs_baseline"] = S.iou(lungs["Left"] | lungs["Right"], gt)
        for m in methods:
            if m != "gradcam" and gt is None:
                continue  # slower methods only where we can score them
            sal = S.compute(m, clf, img, t_idx)
            if m == "gradcam":
                row["gradcam_outside"] = R.zone_contributions(sal, zones)[R.OUTSIDE]
            if gt is not None:
                row[f"iou_{m}"] = S.iou(S.binarize(sal), gt)
                row[f"hit_{m}"] = float(S.pointing_hit(sal, gt))
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
    out = {"n_images": int(len(df)), "auroc": {}, "specificity": {}, "localization": {},
           "generated": time.strftime("%Y-%m-%d"),
           "model": "torchxrayvision densenet121-res224-all",
           "input_mode": M.INPUT_MODE,
           "gt_definition": "union of radiologist Consolidation + Airspace Opacity masks (CheXlocalize)",
           "localization_target": "model's Pneumonia output",
           "binarization": "saliency >= 0.5 of its 99.5th percentile",
           "ci": "95%: bootstrap (AUROC, IoU), Wilson (rates)"}
    # discrimination: each model output vs its own radiologist label
    for model_name, label in AUROC_FINDINGS.items():
        col = f"label_{label}"
        if col in df and f"prob_{model_name}" in df:
            sub = df[df[col].isin([0, 1])]
            out["auroc"][model_name] = E.auroc_ci(sub[col], sub[f"prob_{model_name}"])
    proxy = E.proxy_label(df)
    if "label_Pneumonia" not in df and proxy is not None:
        # no pneumonia labels: score the pneumonia output against the closest stand-in
        out["auroc"]["Pneumonia"] = {**E.auroc_ci(proxy, df["prob_Pneumonia"]), "proxy": E.PROXY_NAME}
    # specificity: does the PNEUMONIA score also "detect" other conditions?
    for col in [c for c in df.columns if c.startswith("label_")]:
        lab = col[6:]
        if lab in ("No Finding",) or df[col].isin([0, 1]).sum() < 20:
            continue
        sub = df[df[col].isin([0, 1])]
        r = E.auroc_ci(sub[col], sub["prob_Pneumonia"], n_boot=500)
        if r["auroc"] is not None and r["n_pos"] >= 5:
            out["specificity"][lab] = r
    # localization (pneumonia output vs radiologist outline) + baselines
    if "gt_area_frac" in df:
        g = df.dropna(subset=["gt_area_frac"])
        out["localization"]["_chance"] = {"hit": E.mean_ci(g["gt_area_frac"])}
        out["localization"]["_lungs"] = {"iou": E.mean_ci(g["iou_lungs_baseline"])}
    for m in methods:
        if f"iou_{m}" in df:
            hits = df[f"hit_{m}"].dropna()
            p, lo, hi = E.wilson(int(hits.sum()), int(len(hits)))
            out["localization"][m] = {"iou": E.mean_ci(df[f"iou_{m}"]),
                                      "hit": {"n": int(len(hits)), "mean": round(p, 3),
                                              "lo": round(lo, 3), "hi": round(hi, 3)}}
    return out


if __name__ == "__main__":
    main()

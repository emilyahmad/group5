"""Translate pixel heatmaps into clinical vocabulary: lung zones.

Clinicians describe chest X-ray findings by zone ("right lower zone
consolidation"), not by pixel. We segment the lungs with torchxrayvision's
PSPNet, split each lung into upper / mid / lower thirds of its own height,
and report what share of the attribution falls in each zone and outside
the lungs (a warning sign: the model may be using devices, labels or
positioning rather than lung tissue).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .model import PreparedImage, to_display

LEVELS = ["upper", "mid", "lower"]
# PSPNet target names are anatomical (patient's) left/right. On a standard
# PA/AP film the patient's right lung appears on the viewer's LEFT.
LUNGS = {"Right Lung": "Right", "Left Lung": "Left"}
ZONES = [f"{side} {lvl}" for side in ("Right", "Left") for lvl in LEVELS]
OUTSIDE = "Outside lungs"


@torch.no_grad()
def lung_masks(seg_model: torch.nn.Module, prep: PreparedImage) -> dict[str, np.ndarray]:
    out = seg_model(prep.x_seg)[0]
    masks = {}
    for target, side in LUNGS.items():
        logits = out[seg_model.targets.index(target)].numpy()
        m = (to_display((logits > 0).astype(np.uint8), prep, cv2.INTER_NEAREST) > 0.5).astype(np.uint8)
        masks[side] = _largest_component(m)
    return masks


def _largest_component(m: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return m.astype(bool)
    keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == keep


def zone_masks(lungs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    zones = {}
    for side, m in lungs.items():
        rows = np.where(m.any(axis=1))[0]
        if rows.size == 0:
            for lvl in LEVELS:
                zones[f"{side} {lvl}"] = np.zeros_like(m)
            continue
        cuts = np.linspace(rows.min(), rows.max() + 1, 4).astype(int)
        for i, lvl in enumerate(LEVELS):
            band = np.zeros_like(m)
            band[cuts[i]:cuts[i + 1]] = True
            zones[f"{side} {lvl}"] = m & band
    return zones


def zone_contributions(sal: np.ndarray, zones: dict[str, np.ndarray]) -> dict[str, float]:
    """Signed net attribution per zone, as a share of the total net effect.

    Absolute values sum to 1, so bars read like the TICICDS contribution chart
    ("+30% reason for increased likelihood, -16% reason for decreased").
    """
    lung_any = np.zeros(sal.shape, dtype=bool)
    net = {}
    for z in ZONES:
        net[z] = float(sal[zones[z]].sum())
        lung_any |= zones[z]
    net[OUTSIDE] = float(sal[~lung_any].sum())
    total = sum(abs(v) for v in net.values())
    return {z: (v / total if total else 0.0) for z, v in net.items()}


def draw_zones(rgb: np.ndarray, zones: dict[str, np.ndarray], color=(255, 255, 255)) -> np.ndarray:
    out = rgb.copy()
    for m in zones.values():
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 1, lineType=cv2.LINE_AA)
    return out


def describe_zone(z: str) -> str:
    return f"{z.split()[0].lower()} {z.split()[1]} zone" if z != OUTSIDE else "area outside the lungs"

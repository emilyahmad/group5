"""'What the model can't see' — limits and per-image warnings.

Implements the checklist item from the team notes: missing inputs,
out-of-distribution warnings and known limits, shown next to every prediction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

STATIC_LIMITS = [
    "Sees **one frontal image only**: no lateral view, no prior films for comparison.",
    "Has **no clinical context**: no symptoms, vitals (SpO₂, temperature, RR), labs "
    "(WBC, CRP) or history. Pneumonia is a clinical diagnosis; this score is image evidence only.",
    "Cannot distinguish pneumonia from other causes of airspace opacity "
    "(oedema, haemorrhage, atelectasis, aspiration) on imaging alone.",
    "Trained on adult public datasets (NIH, CheXpert, MIMIC, PadChest, RSNA, …); "
    "not validated for children or for this hospital's scanners.",
    "Heatmaps show where the model looked, **not** a lesion outline. In the Saporta et al. "
    "benchmark every saliency method localized findings worse than radiologists, "
    "and worst for small or diffuse findings.",
]


@dataclass
class Flag:
    level: str  # "warn" | "info"
    text: str


@torch.no_grad()
def ood_score(ae: torch.nn.Module, x: torch.Tensor) -> float:
    """Mean absolute reconstruction error of the xrv autoencoder.

    Images unlike the training data (wrong body part, heavy artefacts,
    unusual processing) reconstruct poorly.
    """
    out = ae(x)["out"]
    return float((out - x).abs().mean())


def per_image_flags(meta: dict, ood: float, cohort_ood: np.ndarray | None,
                    outside_share: float, cohort_outside: np.ndarray | None = None,
                    ood_pct_warn: float = 95, outside_pct_warn: float = 90) -> list[Flag]:
    flags: list[Flag] = []
    if cohort_ood is not None and len(cohort_ood) > 10:
        pct = float((cohort_ood < ood).mean() * 100)
        if pct >= ood_pct_warn:
            flags.append(Flag("warn", f"**Unusual image** — less typical than {pct:.0f}% of the "
                                      "cohort (autoencoder check). Interpret the score with caution."))
        else:
            flags.append(Flag("info", f"Image looks typical for the training data "
                                      f"(atypicality percentile {pct:.0f})."))
    view = str(meta.get("ap_pa") or "").upper()
    if view == "AP":
        flags.append(Flag("warn", "**AP (often portable) film.** Projection and positioning "
                                  "differ from PA; opacities and heart size are harder to judge."))
    # Grad-CAM is coarse and the lung segmentation excludes retrocardiac /
    # sub-diaphragmatic lung, so most films put a large share "outside". Only warn
    # when this film is unusual relative to the cohort, to avoid alert fatigue.
    if cohort_outside is not None and len(cohort_outside) > 10:
        unusual = (cohort_outside < outside_share).mean() * 100 >= outside_pct_warn
    else:
        unusual = outside_share > 0.85
    if unusual:
        flags.append(Flag("warn", f"**{outside_share:.0%} of the model's attention is outside the "
                                  "lung fields**, more than for most films. Check whether it is "
                                  "focusing on devices, labels or edges rather than lung."))
    age = meta.get("age")
    if age is not None and not np.isnan(age) and age < 18:
        flags.append(Flag("warn", "**Paediatric patient.** The model was trained mostly on adults."))
    return flags

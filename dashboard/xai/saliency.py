"""Saliency methods for a single pathology output.

Each method returns a *signed* float map in the display frame (true aspect):
positive = evidence FOR the target finding, negative = evidence AGAINST.
Grad-CAM is non-negative by construction.

Method choice follows Saporta et al. 2022 ("Benchmarking saliency methods for
chest X-ray interpretation"): Grad-CAM localized best of the methods they
tested, so it is the default; Integrated Gradients and Occlusion are offered
so users can cross-check ("adversarial verification").
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
from captum.attr import IntegratedGradients, Occlusion

from .model import PreparedImage, logit_head, to_display

METHODS = {
    "gradcam": "Grad-CAM",
    "ig": "Integrated Gradients",
    "occlusion": "Occlusion",
}
METHOD_HELP = {
    "gradcam": "Which regions of the last feature layer drove the score. Coarse but "
               "the most anatomically faithful method in the Saporta et al. benchmark.",
    "ig": "Pixel-level attribution accumulated along a path from a black image. "
          "Sharper, but noisier; smoothed here for readability.",
    "occlusion": "Blanks out one 32×32 patch at a time and measures how much the "
                 "pneumonia score drops. Slowest, but the easiest to interpret: "
                 "'hiding this region lowers the score by X'.",
}




def gradcam(model: torch.nn.Module, x: torch.Tensor, idx: int) -> np.ndarray:
    feats = model.features(x.clone())
    logit = logit_head(model, feats)[0, idx]
    (grads,) = torch.autograd.grad(logit, feats)
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(1))[0].detach().numpy()
    return np.clip(cam, 0, None)


def _logit_fn(model, idx):
    def f(inp):
        return logit_head(model, model.features(inp))[:, idx]
    return f


def integrated_gradients(model: torch.nn.Module, x: torch.Tensor, idx: int,
                         steps: int = 32) -> np.ndarray:
    ig = IntegratedGradients(_logit_fn(model, idx))
    baseline = torch.full_like(x, -1024.0)  # black image in xrv's [-1024, 1024] range
    attr = ig.attribute(x.clone().requires_grad_(True), baselines=baseline,
                        n_steps=steps, internal_batch_size=8)
    a = attr[0, 0].detach().numpy()
    return cv2.GaussianBlur(a, (0, 0), sigmaX=4)  # pixel-level IG is speckled; smooth for display


def occlusion(model: torch.nn.Module, x: torch.Tensor, idx: int,
              window: int = 32, stride: int = 16) -> np.ndarray:
    @torch.no_grad()
    def prob(inp):
        return model(inp)[:, idx]

    occ = Occlusion(prob)
    attr = occ.attribute(x, sliding_window_shapes=(1, window, window),
                         strides=(1, stride, stride), baselines=0.0,
                         perturbations_per_eval=32)
    return attr[0, 0].numpy()


def compute(method: str, model: torch.nn.Module, prep: PreparedImage, idx: int) -> np.ndarray:
    """Saliency for output `idx`, mapped onto the display frame."""
    fn = {"gradcam": gradcam, "ig": integrated_gradients, "occlusion": occlusion}[method]
    m = to_display(fn(model, prep.x, idx), prep)
    return np.clip(m, 0, None) if method == "gradcam" else m


# ---------------------------------------------------------------- display helpers
def positive_unit(m: np.ndarray) -> np.ndarray:
    """Positive part scaled to [0, 1] (used for overlays and IoU)."""
    pos = np.clip(m, 0, None)
    hi = np.percentile(pos, 99.5)
    return np.clip(pos / hi, 0, 1) if hi > 0 else pos


def overlay(gray: np.ndarray, sal: np.ndarray, alpha: float = 0.5,
            floor: float = 0.2) -> np.ndarray:
    """Blend a heatmap over the X-ray. Weak attributions fade out so anatomy stays visible."""
    unit = positive_unit(sal)
    heat = cv2.applyColorMap((unit * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[..., ::-1]
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    a = (alpha * np.clip((unit - floor) / (1 - floor), 0, 1))[..., None]
    return (base * (1 - a) + heat * a).astype(np.uint8)


def draw_contours(rgb: np.ndarray, mask: np.ndarray, color=(0, 220, 255), thickness=2) -> np.ndarray:
    out = rgb.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def binarize(sal: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return positive_unit(sal) >= thresh


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    return float(np.logical_and(pred, gt).sum() / union) if union else float("nan")


def pointing_hit(sal: np.ndarray, gt: np.ndarray) -> bool:
    """'Hit' if the single most salient pixel falls inside the ground truth."""
    y, x = np.unravel_index(np.argmax(sal), sal.shape)
    return bool(gt[y, x])

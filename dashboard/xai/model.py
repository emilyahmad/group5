"""Model loading, image I/O and preprocessing.

Every spatial map in the dashboard (heatmaps, lung masks, ground-truth masks)
lives in one shared "model frame": the center-cropped square of the original
X-ray, resized to DISPLAY x DISPLAY. That keeps overlays aligned no matter
which method or resolution produced them.
"""
from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchxrayvision as xrv
from PIL import Image

warnings.filterwarnings("ignore", module="torchxrayvision")

PNEUMONIA = "Pneumonia"
# Model outputs shown as "related findings": radiographic signs a clinician
# would weigh alongside a pneumonia call.
RELATED = ["Consolidation", "Lung Opacity", "Infiltration", "Effusion", "Atelectasis", "Edema"]
FRIENDLY = {
    "Lung Opacity": "Lung opacity",
    "Effusion": "Pleural effusion",
    "Infiltration": "Infiltrate",
    "Pleural_Thickening": "Pleural thickening",
    "Enlarged Cardiomediastinum": "Enlarged cardiomediastinum",
}
MODEL_RES = 224
SEG_RES = 512
DISPLAY = 512


def friendly(name: str) -> str:
    return FRIENDLY.get(name, name.replace("_", " "))


@dataclass
class PreparedImage:
    display: np.ndarray      # uint8 [DISPLAY, DISPLAY], center-cropped
    x: torch.Tensor          # [1, 1, 224, 224], xrv-normalized ([-1024, 1024])
    x_seg: torch.Tensor      # [1, 1, 512, 512], xrv-normalized
    crop_box: tuple[int, int, int, int]  # (top, left, size, size) in original pixels
    orig_shape: tuple[int, int]


# ---------------------------------------------------------------- models
def load_classifier() -> torch.nn.Module:
    """DenseNet-121 trained on 8 public CXR datasets (incl. CheXpert train, RSNA).

    Outputs are calibrated so 0.5 == the pathology's operating threshold.
    """
    return xrv.models.DenseNet(weights="densenet121-res224-all").eval()


def load_segmenter() -> torch.nn.Module:
    return xrv.baseline_models.chestx_det.PSPNet().eval()


def load_autoencoder() -> torch.nn.Module:
    return xrv.autoencoders.ResNetAE(weights="101-elastic").eval()


# ---------------------------------------------------------------- image I/O
def read_image(src: str | Path | bytes, name: str = "") -> np.ndarray:
    """Return a 2-D uint8 grayscale array from a path or uploaded bytes (PNG/JPG/DICOM)."""
    suffix = Path(name or ("" if isinstance(src, bytes) else str(src))).suffix.lower()
    if suffix in {".dcm", ".dicom"}:
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(src) if isinstance(src, bytes) else src)
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr
        arr = (arr - arr.min()) / max(arr.max() - arr.min(), 1e-6) * 255
        return arr.astype(np.uint8)
    img = Image.open(io.BytesIO(src) if isinstance(src, bytes) else src).convert("L")
    return np.asarray(img, dtype=np.uint8)


def center_crop_box(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    s = min(h, w)
    return (h - s) // 2, (w - s) // 2, s, s


def to_frame(arr: np.ndarray, box: tuple[int, int, int, int], size: int = DISPLAY,
             interp: int = cv2.INTER_AREA) -> np.ndarray:
    """Crop `arr` (same shape as the original image) with `box` and resize to the frame."""
    t, l, h, w = box
    return cv2.resize(arr[t:t + h, l:l + w], (size, size), interpolation=interp)


def _normalize(gray: np.ndarray, res: int) -> torch.Tensor:
    resized = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32)
    return torch.from_numpy(xrv.datasets.normalize(resized, 255))[None, None]


def prepare(gray: np.ndarray) -> PreparedImage:
    box = center_crop_box(gray.shape)
    square = to_frame(gray, box, size=max(DISPLAY, SEG_RES))
    return PreparedImage(
        display=cv2.resize(square, (DISPLAY, DISPLAY), interpolation=cv2.INTER_AREA),
        x=_normalize(square, MODEL_RES),
        x_seg=_normalize(square, SEG_RES),
        crop_box=box,
        orig_shape=gray.shape,
    )


# ---------------------------------------------------------------- inference
@torch.no_grad()
def predict(model: torch.nn.Module, x: torch.Tensor) -> dict[str, float]:
    out = model(x)[0]
    return {p: float(v) for p, v in zip(model.pathologies, out) if p}


def logit_head(model: torch.nn.Module, feats: torch.Tensor) -> torch.Tensor:
    """Classifier head on top of `model.features` output, returning raw logits.

    Mirrors DenseNet.features2 + classifier but without the in-place ReLU, so
    gradients w.r.t. `feats` (needed for Grad-CAM) are well defined.
    """
    pooled = F.adaptive_avg_pool2d(F.relu(feats), 1).flatten(1)
    return model.classifier(pooled)


def target_index(model: torch.nn.Module, name: str) -> int:
    return list(model.pathologies).index(name)

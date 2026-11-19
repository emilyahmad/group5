"""Model loading, image I/O and preprocessing.

Two coordinate frames:
  * model frame   — the square the networks see (224 for the classifier,
                    512 for the lung segmenter), built per INPUT_MODE.
  * display frame — the WHOLE film at its true aspect ratio, long side DISPLAY.
Every spatial map (heatmaps, lung masks, radiologist masks) is shown in the
display frame; `to_display` maps model-frame maps there exactly, so clinicians
always see undistorted anatomy whatever the model was fed.

INPUT_MODE was chosen by measurement on CheXlocalize validation (202 frontal
films, mostly landscape). AUROC per finding, center-crop / pad / stretch:
  consolidation .846/.863/.890 · lung opacity .859/.873/.879 · effusion .844/.851/.864
  atelectasis .804/.827/.832 · edema .859/.858/.859
Center-cropping (torchxrayvision's default) cut off up to ~18% of the lung
fields on wide films, so it is not offered.
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


INPUT_MODE = "stretch"  # "stretch" (best measured) | "pad"
# Bump whenever preprocessing or cached data shapes change: it is part of every
# Streamlit cache key, so a running server never serves results from older code.
PIPELINE_VERSION = f"3-{INPUT_MODE}"


@dataclass
class PreparedImage:
    display: np.ndarray      # uint8 [H_d, W_d], whole film at true aspect, long side DISPLAY
    x: torch.Tensor          # [1, 1, 224, 224], xrv-normalized ([-1024, 1024])
    x_seg: torch.Tensor      # [1, 1, 512, 512], xrv-normalized
    orig_shape: tuple[int, int]
    mode: str = INPUT_MODE

    @property
    def display_shape(self) -> tuple[int, int]:
        return self.display.shape[:2]


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


def display_size(shape: tuple[int, int]) -> tuple[int, int]:
    """(width, height) of the display frame for an image of `shape` (h, w)."""
    h, w = shape
    k = DISPLAY / max(h, w)
    return max(1, round(w * k)), max(1, round(h * k))


def _pad_box(shape: tuple[int, int]) -> tuple[int, int, int, int]:
    """Where the film sits inside its zero-padded square: (top, left, h, w) as pixels of a side-max square."""
    h, w = shape
    s = max(h, w)
    return (s - h) // 2, (s - w) // 2, h, w


def _model_square(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "stretch":
        return gray  # resized straight to a square below (aspect is distorted, nothing is lost)
    if mode == "pad":
        t, l, h, w = _pad_box(gray.shape)
        out = np.zeros((max(gray.shape),) * 2, np.uint8)
        out[t:t + h, l:l + w] = gray
        return out
    raise ValueError(mode)


def _normalize(gray: np.ndarray, res: int) -> torch.Tensor:
    resized = cv2.resize(gray, (res, res), interpolation=cv2.INTER_AREA).astype(np.float32)
    return torch.from_numpy(xrv.datasets.normalize(resized, 255))[None, None]


def prepare(gray: np.ndarray, mode: str = INPUT_MODE) -> PreparedImage:
    sq = _model_square(gray, mode)
    return PreparedImage(
        display=cv2.resize(gray, display_size(gray.shape), interpolation=cv2.INTER_AREA),
        x=_normalize(sq, MODEL_RES), x_seg=_normalize(sq, SEG_RES),
        orig_shape=gray.shape, mode=mode)


def to_display(m: np.ndarray, prep: PreparedImage, interp: int = cv2.INTER_CUBIC) -> np.ndarray:
    """Map a square model-frame map (any resolution) onto the display frame."""
    w_d, h_d = display_size(prep.orig_shape)
    m = m.astype(np.float32)
    if prep.mode == "pad":
        n = m.shape[0]
        s = max(prep.orig_shape)
        t, l, h, w = (round(v * n / s) for v in _pad_box(prep.orig_shape))
        m = m[t:t + max(h, 1), l:l + max(w, 1)]
    return cv2.resize(m, (w_d, h_d), interpolation=interp)


def orig_to_display(m: np.ndarray, prep: PreparedImage) -> np.ndarray:
    """Map a mask in original-image pixels onto the display frame."""
    w_d, h_d = display_size(prep.orig_shape)
    return cv2.resize(m.astype(np.uint8), (w_d, h_d), interpolation=cv2.INTER_NEAREST).astype(bool)


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

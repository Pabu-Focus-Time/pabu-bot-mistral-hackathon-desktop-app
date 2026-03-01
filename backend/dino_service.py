"""
DINOv3 Image Similarity Service

Singleton service that computes scroll-resistant cosine similarity between
consecutive screenshots using DINOv2/v3 patch embeddings. Used as a pre-filter
to avoid expensive LLM calls when the screen hasn't changed.

Adapted from: image_comparison_metrics/slow/DINOv3/screen_change_trigger.py
"""

import os
import io
import base64
import logging
import threading
from typing import Optional, Tuple

import numpy as np
import torch
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)

# ── Settings ──────────────────────────────────────────────────────────────────

MODEL_ID = os.environ.get(
    "DINO_MODEL_ID", "facebook/dinov2-small-imagenet1k-1-layer"
)
HF_TOKEN = os.environ.get("HF_TOKEN")

SCREEN_SCALE = float(os.environ.get("DINO_SCREEN_SCALE", "0.5"))
TOKEN_STRIDE_PREFER = int(os.environ.get("DINO_TOKEN_STRIDE_PREFER", "16"))
SCROLL_K = int(os.environ.get("DINO_SCROLL_K", "16"))
CHANGE_THRESHOLD = float(os.environ.get("DINO_TRIGGER_THRESH", "0.10"))


# ── Core Functions (from screen_change_trigger.py) ────────────────────────────

_mean = [0.485, 0.456, 0.406]
_std = [0.229, 0.224, 0.225]
_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=_mean, std=_std),
])


def _preprocess(img_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert RGB numpy array to normalized tensor on device."""
    pil_img = Image.fromarray(img_rgb)
    return _transform(pil_img).unsqueeze(0).to(device)  # (1, 3, H, W)


def _infer_stride_and_special_tokens(
    H: int, W: int, T: int, prefer_stride: int
) -> Tuple[int, int, int, int]:
    """Figure out patch grid dimensions from token count."""
    stride_candidates = [prefer_stride, 32, 16, 14, 8]
    seen = set()
    stride_candidates = [
        s for s in stride_candidates
        if isinstance(s, int) and s > 0 and not (s in seen or seen.add(s))
    ]

    for stride in stride_candidates:
        if H % stride != 0 or W % stride != 0:
            continue
        rows, cols = H // stride, W // stride
        n_patches = rows * cols
        for n_special in range(0, 33):
            if T - n_special == n_patches:
                return stride, n_special, rows, cols
    return prefer_stride, 1, H // prefer_stride, W // prefer_stride


def _extract_Xn(
    model: torch.nn.Module, tensor: torch.Tensor, prefer_stride: int
) -> Tuple[torch.Tensor, int, int, int]:
    """Extract L2-normalized patch embeddings from model output."""
    with torch.no_grad():
        out = model(pixel_values=tensor)
    hs = out.last_hidden_state.squeeze(0)  # (T, D)
    T, D = int(hs.shape[0]), int(hs.shape[1])
    H, W = int(tensor.shape[2]), int(tensor.shape[3])
    stride, n_special, rows, cols = _infer_stride_and_special_tokens(
        H, W, T, prefer_stride=prefer_stride
    )
    n_patches = rows * cols
    X = hs[n_special: n_special + n_patches, :].reshape(n_patches, D)
    Xn = X / (torch.linalg.norm(X, dim=1, keepdim=True) + 1e-8)
    return Xn, rows, cols, stride


def _dino_col_change(
    prev_Xn: torch.Tensor,
    curr_Xn: torch.Tensor,
    rows: int,
    cols: int,
    k_rows: int,
) -> float:
    """
    Scroll-resistant cosine change metric.
    Returns 0.0 = identical, higher = more changed. NaN if invalid.
    """
    if prev_Xn is None or curr_Xn is None:
        return float("nan")
    if prev_Xn.shape != curr_Xn.shape or prev_Xn.shape[0] != rows * cols:
        return float("nan")

    k_rows = max(0, int(k_rows))
    prev = prev_Xn.reshape(rows, cols, -1)
    curr = curr_Xn.reshape(rows, cols, -1)

    best = torch.full(
        (rows, cols), float("-inf"), device=curr.device, dtype=curr.dtype
    )
    for delta in range(-k_rows, k_rows + 1):
        if delta >= 0:
            r_curr0, r_curr1 = 0, rows - delta
            r_prev0, r_prev1 = delta, rows
        else:
            r_curr0, r_curr1 = -delta, rows
            r_prev0, r_prev1 = 0, rows + delta

        if r_curr1 <= r_curr0:
            continue

        sim = (curr[r_curr0:r_curr1] * prev[r_prev0:r_prev1]).sum(dim=-1)
        best[r_curr0:r_curr1] = torch.maximum(best[r_curr0:r_curr1], sim)

    mean_best = float(best.mean().detach().cpu().item())
    return float(1.0 - mean_best)


# ── DinoService Class ─────────────────────────────────────────────────────────


class DinoService:
    """
    Thread-safe singleton service for DINOv3 image similarity.

    Supports multiple channels (e.g., "desktop", "robot") so each source
    maintains its own cached embeddings and focus state independently.

    Usage:
        service = DinoService()
        service.load_model()  # Call once at startup
        result = service.compare(base64_image_string, channel="desktop")
        result = service.compare(base64_image_string, channel="robot")
    """

    DEFAULT_CHANNEL = "desktop"

    def __init__(self):
        self.model = None
        self.device = None
        # Per-channel state: {channel: value}
        self._prev_Xn: dict = {}
        self._prev_shape: dict = {}  # {channel: (rows, cols, stride)}
        self._cached_focus_state: dict = {}  # {channel: dict}
        self._lock = threading.Lock()
        self._loaded = False

    def load_model(self) -> None:
        """Load the DINOv3 model. Call once at server startup."""
        if self._loaded:
            return

        # Detect device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"Loading DINOv3 model '{MODEL_ID}' on {self.device}...")

        from transformers import AutoModel

        trust = bool(str(MODEL_ID).startswith("facebook/dino")) or os.path.isdir(
            str(MODEL_ID)
        )

        try:
            if HF_TOKEN:
                try:
                    self.model = (
                        AutoModel.from_pretrained(
                            MODEL_ID, trust_remote_code=trust, token=HF_TOKEN
                        )
                        .to(self.device)
                        .eval()
                    )
                except TypeError:
                    self.model = (
                        AutoModel.from_pretrained(
                            MODEL_ID,
                            trust_remote_code=trust,
                            use_auth_token=HF_TOKEN,
                        )
                        .to(self.device)
                        .eval()
                    )
            else:
                self.model = (
                    AutoModel.from_pretrained(MODEL_ID, trust_remote_code=trust)
                    .to(self.device)
                    .eval()
                )

            self._loaded = True
            logger.info(f"DINOv3 model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load DINOv3 model: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _decode_base64_image(self, image_b64: str) -> np.ndarray:
        """Decode base64 PNG to RGB numpy array."""
        image_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil_img)

    def _prepare_image(self, img_rgb: np.ndarray) -> torch.Tensor:
        """Crop to 32-multiples, optionally downscale, return tensor."""
        H0, W0 = img_rgb.shape[:2]

        # Crop to nearest multiple of 32
        Hc = (H0 // 32) * 32
        Wc = (W0 // 32) * 32
        if Hc == 0 or Wc == 0:
            raise ValueError(f"Image too small after cropping: {H0}x{W0}")

        img_crop = img_rgb[:Hc, :Wc]

        # Optional downscale (keep multiples of 32)
        if 0 < SCREEN_SCALE < 1.0:
            import cv2

            new_w = max(32, int(Wc * SCREEN_SCALE) // 32 * 32)
            new_h = max(32, int(Hc * SCREEN_SCALE) // 32 * 32)
            if new_w != Wc or new_h != Hc:
                img_crop = cv2.resize(
                    img_crop, (new_w, new_h), interpolation=cv2.INTER_AREA
                )

        return _preprocess(np.ascontiguousarray(img_crop, dtype=np.uint8), self.device)

    def compare(self, image_b64: str, channel: str = None) -> dict:
        """
        Compare a base64-encoded image against the previous one for a given channel.

        Args:
            image_b64: Base64-encoded image (PNG or JPEG).
            channel: Source channel ("desktop", "robot", etc.). Each channel
                     maintains its own cached embeddings independently.

        Returns:
            {
                "changed": bool,          # True if content changed above threshold
                "similarity_score": float, # 0.0 = identical, higher = more changed
                "is_first_frame": bool,    # True if this is the first frame (no comparison)
                "threshold": float,        # The threshold used
            }
        """
        if channel is None:
            channel = self.DEFAULT_CHANNEL

        if not self._loaded:
            return {
                "changed": True,
                "similarity_score": 1.0,
                "is_first_frame": True,
                "threshold": CHANGE_THRESHOLD,
                "error": "Model not loaded",
            }

        with self._lock:
            try:
                # Decode and prepare
                img_rgb = self._decode_base64_image(image_b64)
                tensor = self._prepare_image(img_rgb)

                # Extract embeddings
                curr_Xn, rows, cols, stride = _extract_Xn(
                    self.model, tensor, prefer_stride=TOKEN_STRIDE_PREFER
                )
                curr_shape = (int(rows), int(cols), int(stride))

                # Compare with previous frame for this channel
                prev_Xn = self._prev_Xn.get(channel)
                prev_shape = self._prev_shape.get(channel)

                if prev_Xn is None or prev_shape != curr_shape:
                    # First frame or shape changed for this channel
                    change = float("nan")
                    is_first = True
                else:
                    change = _dino_col_change(
                        prev_Xn, curr_Xn,
                        rows=rows, cols=cols, k_rows=SCROLL_K,
                    )
                    is_first = False

                # Update state for this channel
                self._prev_Xn[channel] = curr_Xn
                self._prev_shape[channel] = curr_shape

                # Determine if changed
                if is_first or not np.isfinite(change):
                    return {
                        "changed": True,
                        "similarity_score": 1.0,
                        "is_first_frame": True,
                        "threshold": CHANGE_THRESHOLD,
                    }

                changed = change > CHANGE_THRESHOLD

                logger.debug(
                    f"DINO [{channel}] similarity: {change:.4f} "
                    f"(threshold={CHANGE_THRESHOLD}, changed={changed})"
                )

                return {
                    "changed": changed,
                    "similarity_score": round(change, 4),
                    "is_first_frame": False,
                    "threshold": CHANGE_THRESHOLD,
                }

            except Exception as e:
                logger.error(f"DINOv3 [{channel}] comparison error: {e}")
                # On error, assume changed (so LLM still gets called)
                return {
                    "changed": True,
                    "similarity_score": 1.0,
                    "is_first_frame": False,
                    "threshold": CHANGE_THRESHOLD,
                    "error": str(e),
                }

    def get_cached_focus(self, channel: str = None) -> Optional[dict]:
        """Get the last cached LLM focus state for a channel."""
        if channel is None:
            channel = self.DEFAULT_CHANNEL
        return self._cached_focus_state.get(channel)

    def set_cached_focus(self, focus_state: dict, channel: str = None) -> None:
        """Cache the latest LLM focus state result for a channel."""
        if channel is None:
            channel = self.DEFAULT_CHANNEL
        self._cached_focus_state[channel] = focus_state

    def reset(self, channel: str = None) -> None:
        """Reset state for a specific channel, or all channels if None."""
        with self._lock:
            if channel is None:
                # Reset all channels
                self._prev_Xn.clear()
                self._prev_shape.clear()
                self._cached_focus_state.clear()
            else:
                # Reset only the specified channel
                self._prev_Xn.pop(channel, None)
                self._prev_shape.pop(channel, None)
                self._cached_focus_state.pop(channel, None)

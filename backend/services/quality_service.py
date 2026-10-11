"""
quality_service.py — Image quality validation for the facial profiling pipeline.

Decoding strategy (why Pillow instead of cv2.imdecode):
  - cv2.imdecode does NOT apply EXIF orientation on all platforms; the
    separate _fix_exif_rotation() parser was fragile and could double-rotate.
  - Pillow's ImageOps.exif_transpose() handles all 8 EXIF orientations in
    one reliable call, then we convert to a BGR NumPy array for OpenCV.

All numeric thresholds are collected in __init__ parameters so they can be
re-calibrated from a single call-site without touching logic.
"""
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ── Safe pixel limit: refuse images > 40 MP (prevents zip-bomb decompression) ──
Image.MAX_IMAGE_PIXELS = 40_000_000

# ── Allowed image magic bytes (JPEG and PNG only) ────────────────────────────
_MAGIC = {
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG":      "PNG",
}
_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB hard limit


class QualityService:
    """
    Validates image quality attributes: resolution, sharpness (Laplacian
    variance), and brightness.

    All threshold values live in __init__ so they can be calibrated later
    without touching any logic.
    """

    def __init__(
        self,
        min_resolution: int = 300,
        blur_threshold: float = 50.0,   # Relaxed from 100 → 50 for full-body shots
        min_brightness: float = 30.0,   # Relaxed from 40 → 30 for darker shots
        max_brightness: float = 245.0,
    ):
        self.min_resolution = min_resolution
        self.blur_threshold = blur_threshold
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def validate_image_quality(self, image_bytes: bytes) -> dict:
        """
        Validates the quality of an image supplied as raw bytes.

        Checks (in order):
          0. Non-empty bytes, file-size limit, magic-byte format check.
          1. Decode via Pillow with EXIF auto-rotation (rotation applied once).
          2. Resolution: width and height must both be ≥ min_resolution.
          3. Blur: Laplacian variance must be ≥ blur_threshold.
          4. Brightness: mean pixel intensity must be in [min_brightness, max_brightness].

        Returns:
            dict with keys:
                "is_valid": bool
                "reason":   str   (human-readable message for the client)
                "image":    np.ndarray | None  (BGR OpenCV array if decoding succeeded)
        """
        try:
            # ── Pre-decode guards ─────────────────────────────────────────────
            if not image_bytes:
                return self._fail("Provided image bytes are empty.")

            if len(image_bytes) > _MAX_FILE_BYTES:
                mb = len(image_bytes) / 1_048_576
                return self._fail(
                    f"Image file is too large ({mb:.1f} MB). "
                    f"Maximum allowed size is {_MAX_FILE_BYTES // 1_048_576} MB."
                )

            fmt = self._check_magic(image_bytes)
            if fmt is None:
                return self._fail(
                    "Unsupported image format. Please upload a JPEG or PNG file."
                )

            # ── Decode + EXIF rotation (exactly once, via Pillow) ─────────────
            try:
                pil_img = Image.open(BytesIO(image_bytes))
                pil_img = ImageOps.exif_transpose(pil_img)  # handles all 8 orientations
                pil_img = pil_img.convert("RGB")
            except Exception as decode_err:
                logger.warning("Pillow decode failed: %s", decode_err)
                return self._fail(
                    "Failed to decode image. The file may be corrupted or truncated."
                )

            # RGB → BGR for OpenCV
            img: np.ndarray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # ── 1. Resolution check ────────────────────────────────────────────
            h, w = img.shape[:2]
            if w < self.min_resolution or h < self.min_resolution:
                return {
                    "is_valid": False,
                    "reason": (
                        f"Image resolution too low ({w}×{h} px). "
                        f"Minimum required: {self.min_resolution}×{self.min_resolution} px."
                    ),
                    "image": img,
                }

            # ── 2. Blur check (Laplacian variance on grayscale) ───────────────
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            if laplacian_var < self.blur_threshold:
                return {
                    "is_valid": False,
                    "reason": (
                        f"Image is too blurry (sharpness score: {laplacian_var:.1f}, "
                        f"minimum required: {self.blur_threshold:.0f}). "
                        "Please provide a sharper, well-focused photo."
                    ),
                    "image": img,
                }

            # ── 3. Brightness check ───────────────────────────────────────────
            brightness = float(np.mean(gray))
            if brightness < self.min_brightness:
                return {
                    "is_valid": False,
                    "reason": (
                        f"Image is too dark (brightness: {brightness:.1f}, "
                        f"minimum: {self.min_brightness:.0f}). "
                        "Please improve lighting and retake."
                    ),
                    "image": img,
                }
            if brightness > self.max_brightness:
                return {
                    "is_valid": False,
                    "reason": (
                        f"Image is overexposed (brightness: {brightness:.1f}, "
                        f"maximum: {self.max_brightness:.0f}). "
                        "Please reduce lighting and retake."
                    ),
                    "image": img,
                }

            # ── All checks passed ─────────────────────────────────────────────
            return {
                "is_valid": True,
                "reason": "Image quality validation passed.",
                "image": img,
            }

        except Exception as exc:
            logger.exception("Unexpected error in QualityService.validate_image_quality")
            return self._fail(f"An unexpected error occurred during image validation.")

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_magic(data: bytes) -> str | None:
        """Returns 'JPEG' or 'PNG' if the magic bytes match, else None."""
        for magic, fmt in _MAGIC.items():
            if data[: len(magic)] == magic:
                return fmt
        return None

    @staticmethod
    def _fail(reason: str) -> dict:
        return {"is_valid": False, "reason": reason, "image": None}

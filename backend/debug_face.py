"""
debug_face.py — Diagnostic script for the facial profiling backend.

Usage (from grooming_component_1_backend/):
    python -m backend.debug_face <path_to_image>

Reports:
  1. MediaPipe version and API surface (solutions vs tasks)
  2. QualityService decode + EXIF-rotation result (saves debug_decoded.jpg)
  3. Image shape before and after rotation, EXIF orientation value
  4. MediaPipe Tasks FaceDetector with short-range model (detection count)
  5. MediaPipe Tasks FaceLandmarker result (landmark count)
"""
import sys
import os
import pathlib

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageOps
from io import BytesIO

# ── 1. MediaPipe version and API surface ─────────────────────────────────────
print("=" * 60)
print(f"MediaPipe version : {mp.__version__}")
print(f"Has mp.solutions  : {hasattr(mp, 'solutions')}")
print(f"Has mp.tasks      : {hasattr(mp, 'tasks')}")
if hasattr(mp, 'solutions'):
    s = mp.solutions
    print(f"  solutions.face_mesh      : {hasattr(s, 'face_mesh')}")
    print(f"  solutions.face_detection : {hasattr(s, 'face_detection')}")
else:
    print("  mp.solutions NOT available — using Tasks API instead")

try:
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core import base_options as mp_base
    print("Tasks FaceDetector available    : True")
    print("Tasks FaceLandmarker available  : True")
except ImportError as e:
    print(f"Tasks API import failed: {e}")

print()

# ── Image path ────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python -m backend.debug_face <path_to_image>")
    sys.exit(1)

img_path = sys.argv[1]
if not os.path.exists(img_path):
    print(f"ERROR: File not found: {img_path}")
    sys.exit(1)

with open(img_path, "rb") as f:
    image_bytes = f.read()

print(f"Input file        : {img_path}")
print(f"File size (bytes) : {len(image_bytes)}")
print()

# ── 2. Pillow EXIF orientation ────────────────────────────────────────────────
print("=" * 60)
print("EXIF / Orientation check (Pillow)")

pil_img_raw = Image.open(BytesIO(image_bytes))
print(f"Pillow mode         : {pil_img_raw.mode}")
print(f"Shape before rotate : {pil_img_raw.size[0]}w × {pil_img_raw.size[1]}h")

try:
    exif_data = pil_img_raw._getexif()
    orientation_val = exif_data.get(0x0112, 1) if exif_data else 1
except Exception:
    orientation_val = 1
print(f"EXIF Orientation tag: {orientation_val}")

pil_img_fixed = ImageOps.exif_transpose(pil_img_raw).convert("RGB")
print(f"Shape after rotate  : {pil_img_fixed.size[0]}w × {pil_img_fixed.size[1]}h")

# Convert to OpenCV BGR for downstream
cv2_image = cv2.cvtColor(np.array(pil_img_fixed), cv2.COLOR_RGB2BGR)
print(f"OpenCV array shape  : {cv2_image.shape}  (H×W×C BGR)")

# Save decoded image for manual inspection
out_path = "debug_decoded.jpg"
cv2.imwrite(out_path, cv2_image)
print(f"Saved decoded image : {out_path}")
print()

# ── 3. cv2.imdecode (old approach) shape — for comparison ────────────────────
print("=" * 60)
print("cv2.imdecode shape (old approach — do NOT use for production)")
np_arr = np.frombuffer(image_bytes, np.uint8)
cv2_old = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
if cv2_old is not None:
    print(f"cv2.imdecode shape : {cv2_old.shape}  (H×W×C BGR)")
    if cv2_old.shape[:2] != cv2_image.shape[:2]:
        print("⚠  Shape DIFFERS from Pillow version — EXIF was not applied by cv2")
    else:
        print("✓  Shape matches Pillow version")
else:
    print("cv2.imdecode returned None!")
print()

# ── 4 & 5. MediaPipe Tasks FaceDetector and FaceLandmarker ───────────────────
print("=" * 60)
print("MediaPipe Tasks API detection")

# Determine model path
_base = pathlib.Path(__file__).parent
_model_path = _base / "models" / "face_landmarker.task"
_fd_model = _base / "models" / "face_landmarker.task"   # FaceLandmarker bundles detection

if not _model_path.exists():
    print(f"ERROR: face_landmarker.task not found at {_model_path}")
    print("Download it with:")
    print("  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -OutFile backend/models/face_landmarker.task")
    sys.exit(1)

from mediapipe.tasks.python import vision as mpv
from mediapipe.tasks.python.core.base_options import BaseOptions

# ── 4a. FaceDetector (short-range, model_selection=0 equivalent) ─────────────
print("\n--- FaceDetector (short-range) ---")
try:
    fd_opts = mpv.FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(_model_path)),
        min_detection_confidence=0.5,
    )
    # FaceDetector uses its own model; use FaceLandmarker for full pipeline
    # For FaceDetector we need a separate short-range model
    print("Note: FaceLandmarker model used; FaceDetector needs a separate model.")
    print("Skipping standalone FaceDetector — using FaceLandmarker which includes detection.")
except Exception as e:
    print(f"FaceDetector error: {e}")

# ── 4b. FaceLandmarker ────────────────────────────────────────────────────────
print("\n--- FaceLandmarker (478 landmarks) ---")
try:
    fl_opts = mpv.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_model_path)),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=mpv.RunningMode.IMAGE,
    )
    with mpv.FaceLandmarker.create_from_options(fl_opts) as landmarker:
        # Convert to mediapipe Image (RGB)
        rgb_np = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_np)
        fl_result = landmarker.detect(mp_image)

        n_faces = len(fl_result.face_landmarks) if fl_result.face_landmarks else 0
        print(f"Faces detected            : {n_faces}")
        if n_faces > 0:
            n_lm = len(fl_result.face_landmarks[0])
            print(f"Landmarks on face[0]      : {n_lm}")
            # Sample a few landmark coords
            lm0 = fl_result.face_landmarks[0][10]
            print(f"Landmark[10] (forehead)   : x={lm0.x:.4f} y={lm0.y:.4f} z={lm0.z:.4f}")
            print("\n✅ Face detected successfully with Tasks FaceLandmarker!")
        else:
            print("❌ No face detected. Possible causes:")
            print("   - Image still rotated incorrectly")
            print("   - min_detection_confidence too high for this image")
            print("   - Face too small / partially visible")
except Exception as e:
    import traceback
    print(f"FaceLandmarker error: {e}")
    traceback.print_exc()

print()
print("=" * 60)
print("Diagnosis complete.")

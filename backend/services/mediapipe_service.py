"""
mediapipe_service.py — MediaPipe Tasks FaceLandmarker-based face validation.

Why Tasks API instead of mp.solutions:
  MediaPipe ≥ 0.10.x (and the installed 1.0.1) dropped the legacy
  mp.solutions namespace entirely.  The Tasks API (mp.tasks.python.vision)
  is the only supported interface and uses a compiled .task model bundle.

  The FaceLandmarker model (face_landmarker.task) performs both face
  detection and 478-landmark extraction in a single forward pass, so a
  separate FaceDetector step is not needed.

Threading safety:
  FaceLandmarker objects are NOT thread-safe.  Every call to .detect() is
  wrapped in a threading.Lock so concurrent FastAPI requests cannot race.

Threshold constants:
  All confidence / angle thresholds live in __init__ parameters so they
  can be adjusted from a single call-site without touching logic.
"""
import logging
import os
import threading

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mpv
from mediapipe.tasks.python.core.base_options import BaseOptions

logger = logging.getLogger(__name__)


class MediaPipeService:
    """
    Validates that a BGR OpenCV image contains exactly one clearly visible,
    roughly frontal face and extracts 10 tabular geometric features.

    Detection pipeline:
      1. FaceLandmarker.detect() on the full image.
      2. If no landmarks found, retry on a centre-cropped region (handles
         images where the face already fills the frame and FaceLandmarker
         struggles with the edge padding).
      3. Pose validation via pitch / yaw / roll angles.
      4. Tabular feature extraction for downstream classifiers.
    """

    # Fraction of image to use for fallback centre crop (0.85 = 85 % of each dim)
    _CENTRE_CROP_RATIO: float = 0.85

    def __init__(
        self,
        model_path: str | None = None,
        num_faces: int = 1,
        min_detection_confidence: float = 0.5,   # Threshold for all confidence gates
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 30.0,
        roll_threshold: float = 30.0,
    ):
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self._lock = threading.Lock()

        # ── Resolve model path ────────────────────────────────────────────────
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "face_landmarker.task")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {model_path}\n"
                "Download it with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                "-OutFile backend/models/face_landmarker.task"
            )

        # ── Build FaceLandmarker (IMAGE running mode for single-shot inference) ─
        try:
            options = mpv.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=mpv.RunningMode.IMAGE,
                num_faces=num_faces,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mpv.FaceLandmarker.create_from_options(options)
            logger.info("FaceLandmarker initialised from %s", model_path)
        except Exception as exc:
            # Fail loudly so the server refuses to start rather than silently
            # returning "No face detected" on every request.
            raise RuntimeError(
                f"Failed to initialise FaceLandmarker: {exc}"
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def validate_and_extract_landmarks(self, cv2_image: np.ndarray) -> dict:
        """
        Runs the full detection pipeline on a BGR OpenCV image.

        Returns:
            {"is_valid": False, "reason": "<user-friendly message>"}
            or
            {"is_valid": True, "features": [float, ...], "reason": "Valid face"}
        """
        if cv2_image is None or not isinstance(cv2_image, np.ndarray) or cv2_image.size == 0:
            return {"is_valid": False, "reason": "Invalid or empty image provided."}

        try:
            # ── Attempt 1: full image ─────────────────────────────────────────
            landmarks = self._detect_landmarks(cv2_image)

            # ── Attempt 2: centre crop (handles extreme close-ups) ────────────
            if landmarks is None:
                crop = self._centre_crop(cv2_image)
                if crop is not None:
                    landmarks = self._detect_landmarks(crop)

            if landmarks is None:
                return {
                    "is_valid": False,
                    "reason": (
                        "No face detected in the image. "
                        "Please ensure your face is clearly visible, well-lit, "
                        "and looking directly at the camera."
                    ),
                }

            # ── Pose validation ───────────────────────────────────────────────
            pitch, yaw, roll = self._estimate_pose_angles(landmarks)
            logger.debug("Pose angles — pitch=%.1f yaw=%.1f roll=%.1f", pitch, yaw, roll)

            if abs(yaw) > self.yaw_threshold:
                return {
                    "is_valid": False,
                    "reason": "Please turn your head to face the camera directly (too much left/right rotation).",
                }
            if abs(pitch) > self.pitch_threshold:
                return {
                    "is_valid": False,
                    "reason": "Please tilt your chin to face the camera directly (too much up/down tilt).",
                }
            if abs(roll) > self.roll_threshold:
                return {
                    "is_valid": False,
                    "reason": "Please keep your head straight (too much side tilt).",
                }

            # ── Feature extraction ────────────────────────────────────────────
            features = self._extract_tabular_features(landmarks)
            return {"is_valid": True, "features": features, "reason": "Valid face"}

        except Exception as exc:
            # Log the real traceback server-side; send a generic message to the client.
            logger.exception("Unexpected error in MediaPipeService.validate_and_extract_landmarks")
            return {
                "is_valid": False,
                "reason": "Face validation failed due to an internal processing error.",
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_landmarks(self, cv2_image: np.ndarray):
        """
        Runs FaceLandmarker on a BGR image.
        Returns the first face's landmark list, or None if nothing detected.
        Thread-safe via self._lock.
        """
        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            result = self._landmarker.detect(mp_image)

        if result and result.face_landmarks and len(result.face_landmarks) > 0:
            return result.face_landmarks[0]   # NormalizedLandmark list
        return None

    def _centre_crop(self, cv2_image: np.ndarray) -> np.ndarray | None:
        """Returns the central _CENTRE_CROP_RATIO fraction of the image."""
        h, w = cv2_image.shape[:2]
        ratio = self._CENTRE_CROP_RATIO
        dx = int(w * (1 - ratio) / 2)
        dy = int(h * (1 - ratio) / 2)
        crop = cv2_image[dy: h - dy, dx: w - dx]
        if crop.size == 0:
            return None
        return crop

    def _estimate_pose_angles(self, landmarks) -> tuple[float, float, float]:
        """
        Estimates Pitch, Yaw, and Roll in degrees from 3-D normalised landmarks.
        Reference points:
          33  → left eye outer corner
          263 → right eye outer corner
          10  → forehead / hairline centre
          152 → chin tip
        """
        l_eye   = landmarks[33]
        r_eye   = landmarks[263]
        forehead = landmarks[10]
        chin    = landmarks[152]

        dx_eye = r_eye.x - l_eye.x
        dy_eye = r_eye.y - l_eye.y
        dz_eye = r_eye.z - l_eye.z

        roll  = float(np.degrees(np.arctan2(dy_eye, dx_eye)))
        yaw   = float(np.degrees(np.arctan2(dz_eye, dx_eye)))
        pitch = float(np.degrees(np.arctan2(chin.z - forehead.z, chin.y - forehead.y)))

        return pitch, yaw, roll

    def _extract_tabular_features(self, landmarks) -> list[float]:
        """
        Derives 10 normalised geometric ratios for downstream classifiers.
        Landmark indices follow the canonical 478-point FaceMesh topology.
        """
        def dist(a, b):
            return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2))

        def angle_deg(p1, vertex, p2):
            v1 = np.array([p1.x - vertex.x, p1.y - vertex.y, p1.z - vertex.z])
            v2 = np.array([p2.x - vertex.x, p2.y - vertex.y, p2.z - vertex.z])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                return 0.0
            return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

        forehead  = landmarks[10]
        chin      = landmarks[152]
        nose      = landmarks[1]
        l_jaw     = landmarks[172]
        r_jaw     = landmarks[397]
        l_cheek   = landmarks[234]
        r_cheek   = landmarks[454]
        l_temple  = landmarks[54]
        r_temple  = landmarks[284]
        l_eye     = landmarks[33]
        r_eye     = landmarks[263]

        face_h       = dist(forehead, chin) + 1e-9
        jaw_w        = dist(l_jaw, r_jaw)
        cheek_w      = dist(l_cheek, r_cheek)
        forehead_w   = dist(l_temple, r_temple)
        interocular  = dist(l_eye, r_eye)

        f1  = jaw_w / face_h
        f2  = cheek_w / face_h
        f3  = cheek_w / (jaw_w + 1e-9)
        f4  = forehead_w / (jaw_w + 1e-9)
        f5  = angle_deg(l_jaw, chin, r_jaw)
        f6  = dist(nose, chin) / face_h
        f7  = angle_deg(l_cheek, l_jaw, chin)
        f8  = angle_deg(r_cheek, r_jaw, chin)
        f9  = interocular / (cheek_w + 1e-9)
        f10 = dist(forehead, nose) / face_h

        return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

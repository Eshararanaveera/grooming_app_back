"""
mediapipe_service.py — Two-stage MediaPipe face detection + landmark extraction.

Stage A: Face detection with FaceDetector (blaze_face_short_range.tflite)
  - Detects faces in full image
  - Rejects: 0 faces, multiple faces, face too small
  
Stage B: FaceLandmarker on face crop
  - Crops detected face with margin
  - Extracts 478 landmarks
  - Validates pose angles

Threading safety:
  Both FaceDetector and FaceLandmarker are protected by threading.Lock.
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
    Two-stage face validation pipeline:
    Stage A: Face detection (count, size)
    Stage B: Landmark extraction + pose validation on cropped face
    """

    def __init__(
        self,
        face_detector_model_path: str | None = None,
        face_landmarker_model_path: str | None = None,
        min_detection_confidence: float = 0.5,
        min_face_width_ratio: float = 0.25,  # Face must be at least 25% of image width
        min_face_width_px: int = 200,        # Face must be at least 200 pixels wide
        crop_margin_ratio: float = 0.4,      # 40% margin around face box
        yaw_threshold: float = 30.0,
        pitch_threshold: float = 30.0,
        roll_threshold: float = 30.0,
    ):
        self.min_detection_confidence = min_detection_confidence
        self.min_face_width_ratio = min_face_width_ratio
        self.min_face_width_px = min_face_width_px
        self.crop_margin_ratio = crop_margin_ratio
        self.yaw_threshold = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold = roll_threshold
        self._lock = threading.Lock()

        # ── Resolve model paths ───────────────────────────────────────────────
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        models_dir = os.path.join(base_dir, "models")

        if face_detector_model_path is None:
            face_detector_model_path = os.path.join(models_dir, "blaze_face_short_range.tflite")
        
        if face_landmarker_model_path is None:
            face_landmarker_model_path = os.path.join(models_dir, "face_landmarker.task")

        # Check FaceDetector model
        if not os.path.exists(face_detector_model_path):
            logger.warning(
                f"FaceDetector model not found at: {face_detector_model_path}\n"
                "Download from: https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
            )
            self._face_detector = None
        else:
            try:
                detector_options = mpv.FaceDetectorOptions(
                    base_options=BaseOptions(model_asset_path=face_detector_model_path),
                    running_mode=mpv.RunningMode.IMAGE,
                    min_detection_confidence=min_detection_confidence,
                )
                self._face_detector = mpv.FaceDetector.create_from_options(detector_options)
                logger.info("FaceDetector initialized from %s", face_detector_model_path)
            except Exception as exc:
                logger.error(f"Failed to initialize FaceDetector: {exc}")
                self._face_detector = None

        # Check FaceLandmarker model
        if not os.path.exists(face_landmarker_model_path):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at: {face_landmarker_model_path}\n"
                "Download with:\n"
                "  Invoke-WebRequest -Uri https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task "
                f"-OutFile {face_landmarker_model_path}"
            )

        try:
            landmarker_options = mpv.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=face_landmarker_model_path),
                running_mode=mpv.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=min_detection_confidence,
                min_face_presence_confidence=min_detection_confidence,
                min_tracking_confidence=min_detection_confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = mpv.FaceLandmarker.create_from_options(landmarker_options)
            logger.info("FaceLandmarker initialized from %s", face_landmarker_model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize FaceLandmarker: {exc}") from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def validate_and_extract_landmarks(self, cv2_image: np.ndarray) -> dict:
        """
        Two-stage pipeline:
        Stage A: Face detection (count, size gate)
        Stage B: Landmark extraction on crop + pose validation
        
        Returns dict with:
          - is_valid: bool
          - reason: str (user-friendly message)
          - face_crop: np.ndarray (BGR image crop) if valid
          - landmarks: list if valid
          - key_points: dict if valid
          - jawline_angle_deg: float if valid
        """
        if cv2_image is None or not isinstance(cv2_image, np.ndarray) or cv2_image.size == 0:
            return {"is_valid": False, "reason": "Invalid or empty image provided."}

        try:
            h_full, w_full = cv2_image.shape[:2]
            logger.info("[DEBUG] Input image size: %d x %d", w_full, h_full)

            # ── STAGE A: Face Detection ───────────────────────────────────────
            if self._face_detector is None:
                logger.warning("FaceDetector not available, falling back to FaceLandmarker only")
                return self._fallback_landmarker_only(cv2_image)

            detections = self._detect_faces(cv2_image)
            
            if len(detections) == 0:
                logger.info("[DEBUG] Stage A: No face detected")
                return {
                    "is_valid": False,
                    "reason": "No face detected in the image. Please ensure your face is clearly visible and well-lit.",
                }
            
            if len(detections) > 1:
                logger.info("[DEBUG] Stage A: Multiple faces detected (%d)", len(detections))
                return {
                    "is_valid": False,
                    "reason": "Multiple faces detected. Please ensure only one person is in the photo.",
                }
            
            # Single face detected
            detection = detections[0]
            bbox = detection.bounding_box
            face_width_px = bbox.width
            face_width_ratio = face_width_px / w_full
            
            logger.info("[DEBUG] Stage A: Face box width = %d px (%.2f%% of image)", 
                       face_width_px, face_width_ratio * 100)
            
            # Face size gate
            if face_width_ratio < self.min_face_width_ratio or face_width_px < self.min_face_width_px:
                logger.info("[DEBUG] Stage A: Face too small (ratio=%.3f, px=%d)", 
                           face_width_ratio, face_width_px)
                return {
                    "is_valid": False,
                    "reason": "Face too small in the image. Move closer to the camera and ensure your face fills most of the frame from forehead to chin.",
                }

            # ── STAGE B: Crop face and extract landmarks ──────────────────────
            x1 = max(0, int(bbox.origin_x - bbox.width * self.crop_margin_ratio))
            y1 = max(0, int(bbox.origin_y - bbox.height * self.crop_margin_ratio))
            x2 = min(w_full, int(bbox.origin_x + bbox.width * (1 + self.crop_margin_ratio)))
            y2 = min(h_full, int(bbox.origin_y + bbox.height * (1 + self.crop_margin_ratio * 1.5)))  # Extra margin above for forehead
            
            face_crop = cv2_image[y1:y2, x1:x2]
            crop_h, crop_w = face_crop.shape[:2]
            logger.info("[DEBUG] Stage B: Face crop size = %d x %d", crop_w, crop_h)
            
            if face_crop.size == 0:
                return {"is_valid": False, "reason": "Failed to crop face region."}

            # Extract landmarks from crop
            landmarks = self._detect_landmarks(face_crop)
            if landmarks is None:
                logger.info("[DEBUG] Stage B: FaceLandmarker found no landmarks on crop")
                return {
                    "is_valid": False,
                    "reason": "Facial landmarks could not be extracted. Please ensure your full face is clearly visible.",
                }
            
            logger.info("[DEBUG] Stage B: Extracted %d landmarks", len(landmarks))

            # ── Pose validation (on crop landmarks in pixel coords) ───────────
            pitch, yaw, roll = self._estimate_pose_angles(landmarks)
            logger.info("[DEBUG] Pose angles: pitch=%.1f yaw=%.1f roll=%.1f", pitch, yaw, roll)

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

            # ── Extract key points and jawline angle ──────────────────────────
            key_points = self._extract_key_points(landmarks)
            jawline_angle = self._compute_jawline_angle_px(landmarks, crop_w, crop_h)
            
            logger.info("[DEBUG] Jawline angle: %.1f degrees", jawline_angle)

            return {
                "is_valid": True,
                "reason": "Valid face",
                "face_crop": face_crop,
                "landmarks": landmarks,
                "key_points": key_points,
                "jawline_angle_deg": jawline_angle,
            }

        except Exception as exc:
            logger.exception("Unexpected error in MediaPipeService.validate_and_extract_landmarks")
            return {
                "is_valid": False,
                "reason": "Face validation failed due to an internal processing error.",
            }

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_faces(self, cv2_image: np.ndarray):
        """Runs FaceDetector on BGR image. Returns list of detections."""
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        with self._lock:
            result = self._face_detector.detect(mp_image)
        
        return result.detections if result and result.detections else []

    def _detect_landmarks(self, cv2_image: np.ndarray):
        """
        Runs FaceLandmarker on a BGR image.
        Returns the first face's landmark list, or None if nothing detected.
        Thread-safe via self._lock.
        """
        rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with self._lock:
            result = self._landmarker.detect(mp_image)

        if result and result.face_landmarks and len(result.face_landmarks) > 0:
            return result.face_landmarks[0]   # NormalizedLandmark list
        return None

    def _fallback_landmarker_only(self, cv2_image: np.ndarray) -> dict:
        """Fallback to old pipeline if FaceDetector unavailable."""
        landmarks = self._detect_landmarks(cv2_image)
        if landmarks is None:
            return {
                "is_valid": False,
                "reason": "No face detected in the image. Please ensure your face is clearly visible, well-lit, and looking directly at the camera.",
            }
        
        pitch, yaw, roll = self._estimate_pose_angles(landmarks)
        if abs(yaw) > self.yaw_threshold:
            return {"is_valid": False, "reason": "Please turn your head to face the camera directly (too much left/right rotation)."}
        if abs(pitch) > self.pitch_threshold:
            return {"is_valid": False, "reason": "Please tilt your chin to face the camera directly (too much up/down tilt)."}
        if abs(roll) > self.roll_threshold:
            return {"is_valid": False, "reason": "Please keep your head straight (too much side tilt)."}
        
        key_points = self._extract_key_points(landmarks)
        h, w = cv2_image.shape[:2]
        jawline_angle = self._compute_jawline_angle_px(landmarks, w, h)
        
        return {
            "is_valid": True,
            "reason": "Valid face",
            "face_crop": cv2_image,  # Use full image as crop
            "landmarks": landmarks,
            "key_points": key_points,
            "jawline_angle_deg": jawline_angle,
        }

    def _estimate_pose_angles(self, landmarks) -> tuple[float, float, float]:
        """
        Estimates Pitch, Yaw, and Roll in degrees from 3-D normalized landmarks.
        Reference points:
          33  → left eye outer corner
          263 → right eye outer corner
          10  → forehead / hairline center
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

    def _extract_key_points(self, landmarks) -> dict:
        """Extract key landmark points for JSON output."""
        key_indices = {
            "chin": 152,
            "left_jaw": 172,
            "right_jaw": 397,
            "left_cheekbone": 234,
            "right_cheekbone": 454,
            "nose_tip": 4,
            "forehead_top": 10,
        }
        
        key_points = {}
        for name, idx in key_indices.items():
            lm = landmarks[idx]
            key_points[name] = {
                "id": idx,
                "x": round(lm.x, 4),
                "y": round(lm.y, 4),
                "z": round(lm.z, 4),
            }
        
        return key_points

    def _compute_jawline_angle_px(self, landmarks, img_width: int, img_height: int) -> float:
        """
        Compute jawline angle at chin using left_jaw (172), chin (152), right_jaw (397).
        Converts normalized coords to pixels for accurate angle calculation.
        """
        l_jaw = landmarks[172]
        chin = landmarks[152]
        r_jaw = landmarks[397]
        
        # Convert to pixel coordinates
        chin_px = np.array([chin.x * img_width, chin.y * img_height])
        l_jaw_px = np.array([l_jaw.x * img_width, l_jaw.y * img_height])
        r_jaw_px = np.array([r_jaw.x * img_width, r_jaw.y * img_height])
        
        # Vectors from chin to jaw points
        v1 = l_jaw_px - chin_px
        v2 = r_jaw_px - chin_px
        
        # Angle between vectors
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cos_angle)))
        
        return angle_deg

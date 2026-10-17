import uuid
from backend.schemas.profile_schema import AnalysisResponse
from backend.services.quality_service import QualityService
from backend.services.mediapipe_service import MediaPipeService
from backend.services.classifier_service import ClassifierService


class ProfileService:
    """
    Orchestrator service that integrates QualityService, MediaPipeService,
    and ClassifierService to perform end-to-end facial profile analysis.
    """

    def __init__(self):
        self.quality_service = QualityService()
        self.mediapipe_service = MediaPipeService()
        self.classifier_service = ClassifierService()

    async def analyze_profile(self, image_bytes: bytes) -> tuple[AnalysisResponse, dict]:
        """
        Orchestrates facial profiling analysis on uploaded image bytes.
        
        Returns:
            tuple: (AnalysisResponse for Flutter, detailed dict for JSON file)
        """
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        
        try:
            # Step 1: Validate Image Quality
            quality_res = self.quality_service.validate_image_quality(image_bytes)
            if not quality_res.get("is_valid", False):
                reason = quality_res.get("reason", "Image quality validation failed.")
                response = AnalysisResponse(
                    hairline_result="",
                    hairline_confidence=0.0,
                    jawline_result="",
                    jawline_confidence=0.0,
                    message=reason,
                    is_success=False
                )
                detail = {
                    "request_id": request_id,
                    "status": "rejected",
                    "reason": reason,
                    "model_outputs": None,
                    "landmark_extractions": None,
                }
                return response, detail

            cv2_image = quality_res.get("image")

            # Step 2: MediaPipe Face Detection & Pose Validation
            mp_res = self.mediapipe_service.validate_and_extract_landmarks(cv2_image)
            if not mp_res.get("is_valid", False):
                reason = mp_res.get("reason", "Face detection or pose validation failed.")
                response = AnalysisResponse(
                    hairline_result="",
                    hairline_confidence=0.0,
                    jawline_result="",
                    jawline_confidence=0.0,
                    message=reason,
                    is_success=False
                )
                detail = {
                    "request_id": request_id,
                    "status": "rejected",
                    "reason": reason,
                    "model_outputs": None,
                    "landmark_extractions": None,
                }
                return response, detail

            # Get face crop for classification (fallback to full image if not available)
            face_crop = mp_res.get("face_crop", cv2_image)
            landmarks = mp_res.get("landmarks")
            key_points = mp_res.get("key_points")
            jawline_angle = mp_res.get("jawline_angle_deg", 0.0)

            # Step 3: Run Hairline Classifier Prediction (EfficientNet-B0)
            hairline_result, hairline_confidence, hairline_probs = self.classifier_service.predict_hairline_detailed(face_crop)

            # Step 4: Run Jawline Classifier Prediction (EfficientNet-B0)
            jawline_result, jawline_confidence, jawline_probs = self.classifier_service.predict_jawline_detailed(face_crop)

            # Step 5: Build detailed JSON for file save
            detail = {
                "request_id": request_id,
                "status": "accepted",
                "model_outputs": {
                    "jawline_model": {
                        "predicted_class": jawline_result,
                        "confidence": round(jawline_confidence, 4),
                        "is_uncertain": jawline_confidence < 0.60,
                        "probabilities": {k: round(v, 4) for k, v in jawline_probs.items()},
                    },
                    "hairline_model": {
                        "predicted_class": hairline_result,
                        "confidence": round(hairline_confidence, 4),
                        "is_uncertain": hairline_confidence < 0.60,
                        "probabilities": {k: round(v, 4) for k, v in hairline_probs.items()},
                    },
                },
                "landmark_extractions": {
                    "landmark_count": len(landmarks) if landmarks else 0,
                    "coordinate_system": "normalized 0-1 relative to the face crop",
                    "jawline_angle_deg": round(jawline_angle, 1),
                    "key_points": key_points,
                    "full_mesh_available": landmarks is not None,
                },
            }

            # Step 6: Return Successful Analysis Response
            response = AnalysisResponse(
                hairline_result=hairline_result,
                hairline_confidence=float(hairline_confidence),
                jawline_result=jawline_result,
                jawline_confidence=float(jawline_confidence),
                message="Facial profiling analysis completed successfully.",
                is_success=True
            )
            
            return response, detail

        except Exception as e:
            reason = f"Analysis error: {str(e)}"
            response = AnalysisResponse(
                hairline_result="",
                hairline_confidence=0.0,
                jawline_result="",
                jawline_confidence=0.0,
                message=reason,
                is_success=False
            )
            detail = {
                "request_id": request_id,
                "status": "rejected",
                "reason": reason,
                "model_outputs": None,
                "landmark_extractions": None,
            }
            return response, detail

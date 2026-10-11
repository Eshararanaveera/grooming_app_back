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

    async def analyze_profile(self, image_bytes: bytes) -> AnalysisResponse:
        """
        Orchestrates facial profiling analysis on uploaded image bytes.

        Workflow:
        1. Pass image_bytes to quality_service.validate_image_quality. Return error if invalid.
        2. Pass valid OpenCV image to mediapipe_service.validate_and_extract_landmarks for pose/face validation. Return error if invalid.
        3. Pass OpenCV image to classifier_service.predict_hairline(cv2_image).
        4. Pass OpenCV image to classifier_service.predict_jawline(cv2_image).
        5. Return successful AnalysisResponse containing results and confidences.
        """
        try:
            # Step 1: Validate Image Quality
            quality_res = self.quality_service.validate_image_quality(image_bytes)
            if not quality_res.get("is_valid", False):
                return AnalysisResponse(
                    hairline_result="",
                    hairline_confidence=0.0,
                    jawline_result="",
                    jawline_confidence=0.0,
                    message=quality_res.get("reason", "Image quality validation failed."),
                    is_success=False
                )

            cv2_image = quality_res.get("image")

            # Step 2: MediaPipe Face Detection & Pose Validation
            mp_res = self.mediapipe_service.validate_and_extract_landmarks(cv2_image)
            if not mp_res.get("is_valid", False):
                return AnalysisResponse(
                    hairline_result="",
                    hairline_confidence=0.0,
                    jawline_result="",
                    jawline_confidence=0.0,
                    message=mp_res.get("reason", "Face detection or pose validation failed."),
                    is_success=False
                )

            # Step 3: Run Hairline Classifier Prediction (EfficientNet-B0)
            hairline_result, hairline_confidence = self.classifier_service.predict_hairline(cv2_image)

            # Step 4: Run Jawline Classifier Prediction (EfficientNet-B0)
            jawline_result, jawline_confidence = self.classifier_service.predict_jawline(cv2_image)

            # Step 5: Return Successful Analysis Response
            return AnalysisResponse(
                hairline_result=hairline_result,
                hairline_confidence=float(hairline_confidence),
                jawline_result=jawline_result,
                jawline_confidence=float(jawline_confidence),
                message="Facial profiling analysis completed successfully.",
                is_success=True
            )

        except Exception as e:
            return AnalysisResponse(
                hairline_result="",
                hairline_confidence=0.0,
                jawline_result="",
                jawline_confidence=0.0,
                message=f"Analysis error: {str(e)}",
                is_success=False
            )

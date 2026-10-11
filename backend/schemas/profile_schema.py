from pydantic import BaseModel, ConfigDict, Field


class AnalysisResponse(BaseModel):
    """
    Response schema for facial profiling analysis results.
    Includes hairline and jawline detection results, confidence scores,
    and execution status details.
    """
    hairline_result: str = Field(
        ...,
        description="Detected hairline type or shape (e.g., 'Straight', 'High', 'Receding')"
    )
    hairline_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for hairline detection (range 0.0 to 1.0)"
    )
    jawline_result: str = Field(
        ...,
        description="Detected jawline shape (e.g., 'Square', 'Oval', 'V-Shape', 'Round')"
    )
    jawline_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for jawline detection (range 0.0 to 1.0)"
    )
    message: str = Field(
        ...,
        description="Success or error details message"
    )
    is_success: bool = Field(
        ...,
        description="Indicates whether the facial analysis completed successfully"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "hairline_result": "Straight",
                "hairline_confidence": 0.95,
                "jawline_result": "Square",
                "jawline_confidence": 0.88,
                "message": "Facial profiling analysis completed successfully.",
                "is_success": True
            }
        }
    )

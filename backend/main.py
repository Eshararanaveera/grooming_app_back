import logging
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from backend.schemas.profile_schema import AnalysisResponse
from backend.services.profile_service import ProfileService

# Configure logging so service-level warnings and exceptions appear in the console
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="Facial Profiling API",
    description="Backend API for facial profiling, image quality validation, pose estimation, and hairline/jawline classification.",
    version="1.0.0"
)

# Configure CORS middleware to allow all origins, methods, and headers for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ProfileService globally so ML models load only once on startup
profile_service = ProfileService()


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Facial Profiling API is running", "status": "healthy"}


@app.post(
    "/analyze_face",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze facial image for Hairline and Jawline profiling"
)
async def analyze_face(image: UploadFile = File(...)) -> AnalysisResponse:
    """
    Accepts an uploaded facial image file, validates quality & head pose,
    and returns Hairline and Jawline classification results.
    """
    if not image or not image.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided. Please upload a valid image."
        )

    try:
        # Read file bytes from upload
        image_bytes = await image.read()

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image file is empty."
            )

        # Delegate analysis to ProfileService
        response: AnalysisResponse = await profile_service.analyze_profile(image_bytes)
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during facial analysis: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

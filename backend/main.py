import json
import logging
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.encoders import jsonable_encoder
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

# Base directory for saving facial profile JSON responses
OUTPUT_PROFILES_DIR = Path(__file__).resolve().parent.parent / "outputs" / "profiles"


def _json_name_for(upload_name: str | None) -> str:
    """
    Generate a safe .json filename matching the base name of the uploaded image file.
    Prevents path traversal, replaces unsafe characters with '_', truncates base to 100 chars,
    and falls back to 'upload.json' if empty.
    """
    if not upload_name:
        return "upload.json"
    safe_name = Path(upload_name).name
    sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name)
    stem = Path(sanitized).stem
    stem = stem.lstrip(".")[:100]
    if not stem:
        return "upload.json"
    return f"{stem}.json"


def save_profile_json(upload_name: str | None, payload: Any) -> None:
    """
    Atomically saves the profile JSON payload to outputs/profiles/<safe_name>.json.
    Ensures directory existence, atomic write via temp file + replace, and logs any errors
    without breaking the API response.
    """
    temp_path = None
    try:
        OUTPUT_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        filename = _json_name_for(upload_name)
        target_path = OUTPUT_PROFILES_DIR / filename

        json_data = json.dumps(payload, indent=2, ensure_ascii=False)

        fd, temp_path = tempfile.mkstemp(dir=str(OUTPUT_PROFILES_DIR), prefix="tmp_profile_")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(json_data)
            os.replace(temp_path, str(target_path))
            temp_path = None
        except Exception:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass
            raise
    except Exception:
        logger.exception("Failed to save profile JSON for upload: %s", upload_name)

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

        # Delegate analysis to ProfileService (returns tuple: response, detail_dict)
        response, detail_dict = await profile_service.analyze_profile(image_bytes)
        
        # Add image name to detail dict
        detail_dict["image_name"] = image.filename
        
        # Save detailed JSON file
        save_profile_json(image.filename, detail_dict)
        
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

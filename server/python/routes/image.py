"""
Image Upload and Analysis Route
Dedicated route for image processing
"""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Upload directory
UPLOAD_DIR = Path("uploads/images")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# File size limit
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

# Allowed file types
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


@router.post("/")
async def upload_image(request: Request, file: UploadFile = File(...)):
    """
    Upload and analyze an image

    - **file**: Image file to upload and analyze
    """
    try:
        # Validate file type
        if file.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}",
            )

        # Read file content
        content = await file.read()
        file_size = len(content)

        # Validate file size
        if file_size > MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Max: {MAX_IMAGE_SIZE / (1024*1024):.1f} MB",
            )

        # Generate unique filename
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix
        filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / filename

        # Save file
        with file_path.open("wb") as f:
            f.write(content)

        logger.info(f"Image uploaded for analysis: {filename} ({file_size} bytes)")

        # Create analysis request
        analysis_request = AnalysisRequest(
            analysis_type="image",
            content=content,
            metadata={
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file_size,
            },
        )

        # Analyze using SentinelAgent (single entry point)
        result = await request.app.state.sentinel_agent.analyze(analysis_request)

        if not result or not hasattr(result, "conclusions") or not result.conclusions:
            raise HTTPException(
                status_code=500,
                detail="Failed to analyze image: No valid analysis results",
            )

        # Get the primary conclusion (highest confidence or most severe)
        primary_conclusion = max(
            result.conclusions, key=lambda c: (c.confidence, c.severity)
        )

        # Extract model info from the first available conclusion
        model_info = {}
        if result.conclusions:
            model_info = result.conclusions[0].metadata.get("model_info", {})

        # Format the response
        response = {
            "status": "success",
            "analysis": {
                "is_deepfake": primary_conclusion.is_deepfake,
                "confidence": float(primary_conclusion.confidence),
                "model_info": model_info,
                "timestamp": datetime.utcnow().isoformat(),
                "evidence_id": result.evidence_ids[0]
                if hasattr(result, "evidence_ids") and result.evidence_ids
                else file_id,
            },
            "metadata": {
                "filename": file.filename,
                "content_type": file.content_type,
                "size": file_size,
                "analysis_timestamp": datetime.utcnow().isoformat(),
            },
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to process image: {str(e)}"
        )


@router.get("/{file_id}")
async def get_image(file_id: str):
    """Get image by ID"""
    try:
        # Resolve the file path (search for any file with the given UUID)
        matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
        if not matches:
            raise HTTPException(status_code=404, detail="Image not found")
        file_path = matches[0]
        return FileResponse(
            path=file_path,
            media_type="image/jpeg",
            filename=file_path.name,
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

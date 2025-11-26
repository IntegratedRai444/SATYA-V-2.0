"""
Webcam Capture and Analysis Route
Dedicated route for webcam/real-time capture
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from datetime import datetime
import base64
import uuid
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


class WebcamCapture(BaseModel):
    image_data: str  # Base64 encoded image
    format: str = "jpeg"


@router.post("/capture")
async def capture_and_analyze(
    request: Request,
    capture: WebcamCapture
):
    """
    Analyze webcam capture
    
    - **image_data**: Base64 encoded image data
    - **format**: Image format (jpeg, png)
    """
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(capture.image_data)
        
        # Save temp file
        temp_path = Path(f"temp_webcam_{uuid.uuid4()}.{capture.format}")
        with temp_path.open("wb") as f:
            f.write(image_bytes)
        
        try:
            # Analyze if detector available
            analysis_result = None
            if hasattr(request.app.state, 'image_detector'):
                detector = request.app.state.image_detector
                analysis_result = detector.detect(str(temp_path))
            
            return {
                "success": True,
                "timestamp": datetime.utcnow().isoformat(),
                "analysis": analysis_result
            }
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
        
    except Exception as e:
        logger.error(f"Webcam capture analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/status")
async def webcam_status():
    """
    Check webcam capture status
    """
    return {
        "success": True,
        "status": "ready",
        "timestamp": datetime.utcnow().isoformat()
    }
"""
Face Detection and Analysis Route
Dedicated route for face-specific processing
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pathlib import Path
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/detect")
async def detect_faces(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Detect faces in an image
    
    - **file**: Image file containing faces
    """
    try:
        # Use image detector's face analysis
        if hasattr(request.app.state, 'image_detector'):
            # Save temp file
            temp_path = Path(f"temp_{uuid.uuid4()}.jpg")
            content = await file.read()
            with temp_path.open("wb") as f:
                f.write(content)
            
            try:
                detector = request.app.state.image_detector
                result = detector.detect(str(temp_path))
                
                # Extract face-specific info
                face_info = result.get('details', {}).get('face_analysis', {})
                
                return {
                    "success": True,
                    "faces_detected": face_info.get('faces_detected', 0),
                    "face_details": face_info,
                    "timestamp": datetime.utcnow().isoformat()
                }
            finally:
                # Clean up temp file
                if temp_path.exists():
                    temp_path.unlink()
        else:
            raise HTTPException(status_code=503, detail="Face detector not available")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/analyze")
async def analyze_face(
    request: Request,
    file: UploadFile = File(...)
):
    """
    Analyze face for deepfake manipulation
    
    - **file**: Image file with face to analyze
    """
    try:
        if hasattr(request.app.state, 'image_detector'):
            # Save temp file
            temp_path = Path(f"temp_{uuid.uuid4()}.jpg")
            content = await file.read()
            with temp_path.open("wb") as f:
                f.write(content)
            
            try:
                detector = request.app.state.image_detector
                result = detector.detect(str(temp_path))
                
                return {
                    "success": True,
                    "analysis": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        else:
            raise HTTPException(status_code=503, detail="Face analyzer not available")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
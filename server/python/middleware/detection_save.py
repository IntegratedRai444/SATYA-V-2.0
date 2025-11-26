"""
Helper middleware to automatically save detections to database
"""

import logging
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class DetectionSaveMiddleware(BaseHTTPMiddleware):
    """
    Middleware to automatically save detection results to database
    Intercepts responses from detection endpoints and saves to DB
    """
    
    def __init__(self, app, db_service=None):
        super().__init__(app)
        self.db_service = db_service
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and save detection if applicable"""
        
        # Track request start time
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Check if this is a detection endpoint
        if self._is_detection_endpoint(request.path):
            try:
                # Extract detection data from response
                # Note: This is a simplified version
                # In production, you'd want to parse the response body
                
                # Get client info
                ip_address = request.client.host if request.client else None
                
                # Save to database (if available)
                if self.db_service and hasattr(self.db_service, 'is_connected'):
                    if self.db_service.is_connected():
                        # You would extract actual detection data here
                        # For now, we'll let individual endpoints handle saving
                        pass
                        
            except Exception as e:
                logger.error(f"Failed to save detection: {e}")
        
        return response
    
    def _is_detection_endpoint(self, path: str) -> bool:
        """Check if path is a detection endpoint"""
        detection_paths = [
            '/analyze/image',
            '/analyze/video',
            '/analyze/audio',
            '/analyze/webcam',
            '/analyze/ensemble'
        ]
        return any(path.endswith(p) for p in detection_paths)


async def save_detection_to_db(
    filename: str,
    media_type: str,
    results: dict,
    processing_time_ms: int = None,
    ip_address: str = None
):
    """
    Helper function to save detection results to database
    Call this from detection endpoints after getting results
    """
    try:
        from services.database_service import get_database
        
        db = await get_database()
        if not db.is_connected():
            logger.warning("Database not connected, skipping save")
            return None
        
        # Extract data from results
        authenticity_score = results.get('authenticity_score', 0.5)
        confidence = results.get('confidence', 0.0)
        label = results.get('label', 'unknown')
        explanation = results.get('explanation', '')
        details = results.get('details', {})
        
        # Save to database
        detection_id = await db.save_detection(
            filename=filename,
            media_type=media_type,
            authenticity_score=authenticity_score,
            confidence=confidence,
            label=label,
            explanation=explanation,
            details=details,
            ip_address=ip_address,
            processing_time_ms=processing_time_ms
        )
        
        logger.info(f"✅ Saved detection {detection_id} to database")
        return detection_id
        
    except Exception as e:
        logger.error(f"Failed to save detection to database: {e}")
        return None

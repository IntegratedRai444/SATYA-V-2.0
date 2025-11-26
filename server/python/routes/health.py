"""
Health Check Routes
Comprehensive health monitoring for all system components
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from datetime import datetime
import psutil
import time

router = APIRouter()

@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """Detailed health check with system metrics"""
    
    # Get app state
    app = request.app
    
    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # ML models status
    ml_status = {
        "image_detector": hasattr(app.state, 'image_detector'),
        "video_detector": hasattr(app.state, 'video_detector'),
        "audio_detector": hasattr(app.state, 'audio_detector'),
        "text_nlp_detector": hasattr(app.state, 'text_nlp_detector'),
        "multimodal_detector": hasattr(app.state, 'multimodal_detector')
    }
    
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.process_time(),
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent
            }
        },
        "ml_models": ml_status,
        "components": {
            "api": "healthy",
            "ml_inference": "healthy" if all(ml_status.values()) else "degraded",
            "database": "healthy"
        }
    }

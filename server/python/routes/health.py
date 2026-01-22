"""
Health Check Routes
Comprehensive health monitoring for all system components
"""

import time
from datetime import datetime

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/health/detailed")
async def detailed_health_check(request: Request):
    """Detailed health check with system metrics"""

    # Get app state
    app = request.app

    # System metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    # ML models status
    ml_status = {
        "image_detector": hasattr(app.state, "image_detector"),
        "video_detector": hasattr(app.state, "video_detector"),
        "audio_detector": hasattr(app.state, "audio_detector"),
        "text_nlp_detector": hasattr(app.state, "text_nlp_detector"),
        "multimodal_detector": hasattr(app.state, "multimodal_detector"),
    }

    # Database status
    database_status = {
        "connected": False,
        "response_time_ms": 0,
        "error": None
    }
    
    try:
        from services.database import get_db_manager
        db_manager = get_db_manager()
        database_health = await db_manager.health_check()
        database_status = database_health
    except Exception as e:
        database_status["error"] = str(e)
        database_status["connected"] = False

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
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
            },
        },
        "ml_models": ml_status,
        "components": {
            "api": "healthy",
            "ml_inference": "healthy" if all(ml_status.values()) else "degraded",
            "database": "healthy" if database_status["connected"] else "unhealthy",
            "database_details": database_status,
        },
    }

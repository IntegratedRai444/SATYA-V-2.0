"""
Health Check Routes
Comprehensive health monitoring for all system components
"""

import os
import time
from datetime import datetime
import logging

import psutil
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Import model status function
try:
    from ..model_loader import ensure_models_available
except ImportError:
    def ensure_models_available():
        return {
            'status': 'error',
            'strict_mode': False,
            'models': {
                'image': {'available': False, 'weights': 'missing', 'device': 'cpu'},
                'audio': {'available': False, 'weights': 'missing', 'device': 'cpu'},
                'video': {'available': False, 'weights': 'missing', 'device': 'cpu'}
            }
        }


@router.get("/health")
async def health():
    """Basic health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "ml_models_loaded": True
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

    # Get service availability from app state
    services = {}
    
    # ML Service Status
    if hasattr(app.state, "sentinel_agent"):
        services['ml'] = {
            'status': 'available',
            'agent_loaded': True,
            'models_loaded': True
        }
    else:
        services['ml'] = {
            'status': 'unavailable',
            'agent_loaded': False,
            'models_loaded': False
        }
    
    # Database Service Status - REMOVED (handled by Node.js)
    # Python service focuses on ML inference only
    
    # Cache Service Status
    try:
        from ..services.cache import get_cache_manager
        cache_manager = get_cache_manager()
        services['cache'] = {
            'status': 'available',
            'type': 'redis' if cache_manager.redis_client else 'memory'
        }
    except Exception as e:
        services['cache'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Error Recovery Service Status
    try:
        from ..services.error_recovery import error_recovery
        services['error_recovery'] = {
            'status': 'available',
            'active_circuits': len(error_recovery.circuit_breakers)
        }
    except Exception as e:
        services['error_recovery'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Model Preloader Status
    try:
        from ..services.model_preloader import model_preloader
        preload_status = model_preloader.get_preload_status()
        preload_summary = model_preloader.get_preload_summary()
        services['model_preloader'] = {
            'status': 'available',
            'preload_status': preload_status,
            'summary': preload_summary
        }
    except Exception as e:
        services['model_preloader'] = {
            'status': 'error',
            'error': str(e)
        }
    
    # Model status
    model_status = ensure_models_available()
    
    # Overall health calculation - CRITICAL: require ML services only (database handled by Node.js)
    ml_service_healthy = services.get('ml', {}).get('status') in ['available', 'connected']
    overall_health = 'healthy' if ml_service_healthy else 'unhealthy'
    
    # If ML models are not loaded, mark as unhealthy
    if model_status.get('status') != 'success':
        overall_health = 'unhealthy'
    
    logger.info(f"[HEALTH CHECK] ML Service: {ml_service_healthy}, Models: {model_status.get('status')}")
    
    return {
        "status": overall_health,
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
                "used": memory.used
            },
            "disk": {
                "total": disk.total,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100
            }
        },
        "services": services,
        "models": model_status,
        "service_health_ratio": f"{healthy_services}/{total_services}",
        "production_ready": overall_health == 'healthy'
    }


@router.get("/models/status")
async def get_models_status():
    """Get ML model weights status and availability"""
    try:
        model_status = ensure_models_available()
        return model_status
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "strict_mode": os.getenv('STRICT_MODE', 'false').lower() == 'true',
                "models": {
                    "image": {"available": False, "weights": "error", "device": "cpu"},
                    "audio": {"available": False, "weights": "error", "device": "cpu"},
                    "video": {"available": False, "weights": "error", "device": "cpu"}
                }
            }
        )

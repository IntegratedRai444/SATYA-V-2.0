"""
API v2 Routes for SatyaAI
"""
from fastapi import APIRouter, WebSocket, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import json
import uuid
from datetime import datetime
import psutil

from ...python.websocket_manager import manager as ws_manager
from ...python.metrics import metrics_collector
from ...python.model_loader import model_manager
from ...python.routes.auth import router as auth_router
from ...python.routes.upload import router as upload_router
from ...python.routes.analysis import router as analysis_router
from ...python.routes.dashboard import router as dashboard_router

from ...python.routes.face import router as face_router
from ...python.routes.system import router as system_router
from ...python.routes.webcam import router as webcam_router
from ...python.routes.feedback import router as feedback_router
from ...python.routes.team import router as team_router
from ...python.routes.multimodal import router as multimodal_router
from ...python.routes.chat import router as chat_router
from .schemas import (
    BatchProcessRequest,
    BatchProcessResponse,
    ModelMetrics,
    SystemHealth,
    AnalysisResult,
    ComparisonRequest,
    ComparisonResult
)

router = APIRouter()

# Include all route modules
router.include_router(auth_router, prefix="/auth", tags=["auth"])
router.include_router(upload_router, prefix="/upload", tags=["upload"])
router.include_router(analysis_router, prefix="/analysis", tags=["analysis"])
router.include_router(dashboard_router, prefix="/dashboard", tags=["dashboard"])
# Note: image, video, audio routes are handled by analysis_router (unified endpoint)
router.include_router(face_router, prefix="/face", tags=["face"])
router.include_router(system_router, prefix="/system", tags=["system"])
router.include_router(webcam_router, prefix="/analysis/webcam", tags=["webcam"])
router.include_router(feedback_router, prefix="/feedback", tags=["feedback"])
router.include_router(team_router, prefix="/team", tags=["team"])
router.include_router(multimodal_router, prefix="/analysis/multimodal", tags=["multimodal"])
router.include_router(chat_router, prefix="/chat", tags=["chat"])

# WebSocket endpoint for real-time updates
@router.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    if not client_id:
        client_id = f"client_{uuid.uuid4().hex[:8]}"
    
    await ws_manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages (e.g., subscribe/unsubscribe from channels)
            try:
                message = json.loads(data)
                if message.get("type") == "subscribe":
                    # In a real implementation, handle subscription logic here
                    await ws_manager.send_personal_message(
                        json.dumps({"status": "subscribed", "channel": message.get("channel")}),
                        websocket
                    )
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)

# System health endpoint
@router.get("/health", response_model=SystemHealth)
async def health_check():
    """Check system health and status"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "models_loaded": list(model_manager.models.keys())
    }

# Model metrics endpoint
@router.get("/metrics/model/{model_name}", response_model=ModelMetrics)
async def get_model_metrics(model_name: str):
    """Get performance metrics for a specific model"""
    if model_name not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    metrics = metrics_collector.get_metrics_summary()
    model_metrics = metrics.get("models", {}).get(model_name, {})
    
    return {
        "model_name": model_name,
        "accuracy": model_metrics.get("success_rate", 0.0),
        "precision": 0.0,  # These would be calculated from confusion matrix
        "recall": 0.0,     # in a real implementation
        "f1_score": 0.0,
        "inference_time_ms": model_metrics.get("avg_inference_time_ms", 0.0),
        "last_updated": datetime.utcnow().isoformat()
    }

# Batch processing endpoint
@router.post("/batch/process", response_model=BatchProcessResponse)
async def process_batch(request: BatchProcessRequest):
    """Process multiple files in batch mode"""
    job_id = f"job_{uuid.uuid4().hex[:8]}"
    
    # In a real implementation, this would be processed asynchronously
    # and the status would be updated in a database
    return {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat()
    }

# Batch job status endpoint
@router.get("/batch/status/{job_id}")
async def get_batch_status(job_id: str):
    """Get status of a batch processing job"""
    # In a real implementation, this would query a database
    return {
        "job_id": job_id,
        "status": "completed",
        "progress": 100,
        "results": []
    }

# Comparison endpoint
@router.post("/compare", response_model=ComparisonResult)
async def compare_media(request: ComparisonRequest):
    """Compare multiple media files and return similarity metrics"""
    # Implement comparison logic here
    return {
        "comparison_id": f"cmp_{uuid.uuid4().hex[:8]}",
        "similarity_scores": [
            {"file1": request.file_paths[0], "file2": file2, "score": 0.85}
            for file2 in request.file_paths[1:]
        ],
        "comparison_metrics": {
            "algorithm": request.algorithm,
            "threshold": 0.8,
            "timestamp": datetime.utcnow().isoformat()
        }
    }

# System metrics endpoint
@router.get("/metrics/system")
async def get_system_metrics():
    """Get detailed system metrics"""
    return metrics_collector.get_metrics_summary()

# Historical metrics endpoint
@router.get("/metrics/historical")
async def get_historical_metrics(
    metric_type: str = "system",
    metric_name: Optional[str] = None,
    limit: int = 100
):
    """Get historical metrics data"""
    return metrics_collector.get_historical_metrics(metric_type, metric_name, limit)

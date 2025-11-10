from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import platform
import os
from typing import Optional

router = APIRouter()

@router.get("/config")
async def get_config() -> JSONResponse:
    """
    Returns backend-driven config for the frontend: supported formats, recommended sizes, max upload limits, and UI hints.
    """
    config = {
        "success": True,
        "supported_formats": {
            "image": ["jpeg", "png", "webp"],
            "video": ["mp4", "mov", "avi", "webm"],
            "audio": ["wav", "mp3", "flac"]
        },
        "recommended_sizes": {
            "image": "min 224x224 px, max 4096x4096 px",
            "video": "max 300MB, up to 10min",
            "audio": "max 100MB, up to 5min"
        },
        "max_upload_mb": 300,
        "ui_hints": {
            "theme": "dark",
            "show_ai_assistant": True
        }
    }
    return JSONResponse(content=config)

@router.get("/health")
async def get_health() -> JSONResponse:
    """
    Returns backend health, version, and model status for monitoring.
    """
    health = {
        "success": True,
        "status": "ok",
        "version": "1.2.0",
        "python_version": platform.python_version(),
        "os": platform.system(),
        "model_status": {
            "image": "ready",
            "video": "ready",
            "audio": "ready",
            "webcam": "ready"
        },
        "free_disk_gb": round(os.statvfs(".").f_bavail * os.statvfs(".").f_frsize / 1024**3, 2)
    }
    return JSONResponse(content=health)

@router.get("/scans")
async def get_scans(page: int = 1, page_size: int = 10, scan_type: Optional[str] = None) -> JSONResponse:
    """
    Returns paginated scan history. Supports filtering by scan_type (image, video, audio, webcam).
    """
    # Demo data
    scans = [
        {"id": f"scan_{i}", "type": "image", "filename": f"file_{i}.jpg", "date": "2024-06-01", "result": "FAKE" if i % 2 == 0 else "REAL"}
        for i in range(1, 21)
    ]
    if scan_type:
        scans = [s for s in scans if s["type"] == scan_type]
    start = (page - 1) * page_size
    end = start + page_size
    paginated = scans[start:end]
    return JSONResponse(content={
        "success": True,
        "scans": paginated,
        "total": len(scans),
        "page": page,
        "page_size": page_size
    })

@router.get("/scans/{scan_id}")
async def get_scan_details(scan_id: str) -> JSONResponse:
    """
    Returns details for a specific scan, including result and report paths.
    """
    # Demo data
    scan = {
        "id": scan_id,
        "type": "image",
        "filename": f"{scan_id}.jpg",
        "date": "2024-06-01",
        "result": "FAKE",
        "report_json": f"reports/{scan_id}.json",
        "report_pdf": f"reports/{scan_id}.pdf"
    }
    return JSONResponse(content={"success": True, "scan": scan})

@router.get("/api/scans/{scanId}/report")
def get_scan_report(scanId: str):
    return {"success": False, "message": "Report generation not implemented yet."}

@router.get("/user/profile")
async def get_user_profile() -> JSONResponse:
    """
    Returns user profile info (demo).
    """
    user = {
        "username": "system_user",
        "email": "system@satyaai.com",
        "role": "system",
        "joined": "2024-01-01"
    }
    return JSONResponse(content={"success": True, "user": user})

@router.post("/session/logout")
async def logout(request: Request) -> JSONResponse:
    """
    Handles user logout (demo logic).
    """
    # Invalidate session/token here if implemented
    return JSONResponse(content={"success": True, "message": "Logged out successfully"})

@router.get("/models")
async def get_models() -> JSONResponse:
    """
    Lists available models, their versions, and status (demo).
    """
    models = [
        {"name": "XceptionNet", "type": "image", "version": "1.0", "status": "ready"},
        {"name": "Wav2Vec2", "type": "audio", "version": "1.0", "status": "ready"},
        {"name": "BlinkNet", "type": "webcam", "version": "1.0", "status": "ready"}
    ]
    return JSONResponse(content={"success": True, "models": models})

@router.get("/admin/stats")
async def get_admin_stats() -> JSONResponse:
    """
    Returns system usage stats, recent errors, and system load (demo).
    """
    stats = {
        "success": True,
        "total_scans": 1234,
        "active_users": 56,
        "recent_errors": [
            {"timestamp": "2024-06-01T10:00:00Z", "message": "Sample error 1"},
            {"timestamp": "2024-06-01T09:30:00Z", "message": "Sample error 2"}
        ],
        "system_load": 0.42
    }
    return JSONResponse(content=stats) 
"""
Dashboard Routes
Analytics and metrics for dashboard
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# ============================================================================
# MODELS
# ============================================================================


class DashboardStats(BaseModel):
    total_scans: int
    scans_today: int
    deepfakes_detected: int
    accuracy_rate: float


class RecentScan(BaseModel):
    id: str
    file_type: str
    result: str


"""
Dashboard Routes
Handle dashboard statistics and charts with database integration
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from services.database import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats")
async def get_dashboard_stats():
    """
    Get dashboard overview statistics
    """
    try:
        db = get_db_manager()
        stats = db.get_analysis_stats()

        # Get real metrics from database
        accuracy = db.get_accuracy_rate()
        active_users = db.get_active_users_count()
        storage_used = db.get_total_storage_used()
        false_positive_rate = db.get_false_positive_rate()

        # Extract counts
        total = stats["total_analyses"]
        authentic = stats["by_result"]["authentic"]
        deepfake = stats["by_result"]["deepfake"]

        return {
            "success": True,
            "total_scans": total,
            "deepfakes_detected": deepfake,
            "authentic_content": authentic,
            "accuracy_rate": accuracy,
            "avg_processing_time": stats.get("avg_processing_time", 0.0),
            "active_users": active_users,
            "storage_used": storage_used,
            "false_positive_rate": false_positive_rate,
        }

    except Exception as e:
        logger.error(f"Failed to get dashboard stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/recent")
async def get_recent_activity(limit: int = 5):
    """
    Get recent activity for dashboard
    """
    try:
        db = get_db_manager()
        analyses = db.get_recent_analyses(limit)

        results = []
        for a in analyses:
            results.append(
                {
                    "id": a.file_id,
                    "type": a.file_type,
                    "filename": a.file_name,
                    "result": a.label,
                    "score": a.authenticity_score,
                    "timestamp": a.created_at,
                }
            )

        return {"success": True, "activities": results}

    except Exception as e:
        logger.error(f"Failed to get recent activity: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get activity: {str(e)}")


@router.get("/chart/daily")
async def get_daily_chart_data(days: int = 7):
    """
    Get daily scan counts for charts
    """
    try:
        db = get_db_manager()
        stats = db.get_daily_analysis_stats(days)

        return {
            "success": True,
            "labels": stats["dates"],
            "datasets": [
                {
                    "label": "Total Scans",
                    "data": stats["scans"],
                    "borderColor": "#3B82F6",
                    "backgroundColor": "rgba(59, 130, 246, 0.1)",
                },
                {
                    "label": "Deepfakes Detected",
                    "data": stats["deepfakes"],
                    "borderColor": "#EF4444",
                    "backgroundColor": "rgba(239, 68, 68, 0.1)",
                },
            ],
        }
    except Exception as e:
        logger.error(f"Failed to get daily chart data: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get chart data: {str(e)}"
        )


@router.get("/chart/by-type")
async def get_type_distribution():
    """
    Get distribution by file type
    """
    try:
        db = get_db_manager()
        stats = db.get_analysis_stats()
        by_type = stats["by_type"]

        return {
            "success": True,
            "labels": ["Images", "Videos", "Audio", "Text"],
            "data": [
                by_type.get("image", 0),
                by_type.get("video", 0),
                by_type.get("audio", 0),
                by_type.get("text", 0),
            ],
            "colors": ["#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"],
        }

    except Exception as e:
        logger.error(f"Failed to get type distribution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get distribution: {str(e)}"
        )


@router.get("/analytics")
async def get_user_analytics():
    """
    Get user analytics for dashboard charts
    """
    try:
        db = get_db_manager()
        # Get last 6 months of data
        monthly_stats = db.get_monthly_analysis_stats(months=6)

        # Get performance metrics
        # Calculate real speed metric from database
        avg_processing_time = db.get_average_processing_time()
        speed_score = max(50, min(100, 100 - (avg_processing_time - 2) * 10))  # 2s baseline
        
        # Calculate reliability based on error rate
        error_rate = db.get_error_rate()
        reliability_score = max(50, min(100, 100 - error_rate * 100))
        
        metrics = [
            {"name": "Accuracy", "value": round(db.get_accuracy_rate() * 100, 1)},
            {"name": "Speed", "value": round(speed_score, 1)},
            {"name": "Reliability", "value": round(reliability_score, 1)},
        ]

        return {"success": True, "monthly": monthly_stats, "metrics": metrics}
    except Exception as e:
        logger.error(f"Failed to get analytics: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to get analytics: {str(e)}"
        )


@router.get("/scans")
async def get_scans(
    limit: int = 10, offset: int = 0, type: str = None, result: str = None
):
    """
    Get paginated scans history
    """
    try:
        db = get_db_manager()
        scans = db.get_analyses(
            limit=limit, offset=offset, file_type=type, label=result
        )

        return {
            "success": True,
            "scans": [
                {
                    "id": s.file_id,
                    "filename": s.file_name,
                    "type": s.file_type,
                    "result": s.label,
                    "score": s.authenticity_score,
                    "timestamp": s.created_at,
                }
                for s in scans
            ],
            "total": db.get_total_analyses_count(),
        }
    except Exception as e:
        logger.error(f"Failed to get scans: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get scans: {str(e)}")

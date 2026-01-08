"""
Analysis Routes
Handle analysis history and statistics with database integration
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from database.analysis import AnalysisResult
from services.database import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/history")
async def get_analysis_history(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: Optional[int] = None,
):
    """
    Get analysis history
    """
    try:
        db = get_db_manager()

        # If user_id provided, get user's history
        if user_id:
            analyses = db.get_user_analyses(user_id, limit, offset)
        else:
            # Otherwise get recent global analyses (for admin/demo)
            analyses = db.get_recent_analyses(limit)

        # Convert to list of dicts
        results = []
        for a in analyses:
            results.append(
                {
                    "id": a.file_id,
                    "filename": a.file_name,
                    "type": a.file_type,
                    "score": a.authenticity_score,
                    "label": a.label,
                    "confidence": a.confidence,
                    "timestamp": a.created_at,
                    "details": a.details,
                }
            )

        return {
            "success": True,
            "count": len(results),
            "limit": limit,
            "offset": offset,
            "history": results,
        }

    except Exception as e:
        logger.error(f"Failed to get history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")


@router.get("/stats")
async def get_analysis_stats():
    """
    Get overall analysis statistics
    """
    try:
        db = get_db_manager()
        stats = db.get_analysis_stats()

        return {"success": True, "stats": stats}

    except Exception as e:
        logger.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/{file_id}")
async def get_analysis_detail(file_id: str):
    """
    Get detailed analysis result by ID
    """
    try:
        db = get_db_manager()
        analysis = db.get_analysis_by_file_id(file_id)

        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return {
            "success": True,
            "id": analysis.file_id,
            "filename": analysis.file_name,
            "type": analysis.file_type,
            "score": analysis.authenticity_score,
            "label": analysis.label,
            "confidence": analysis.confidence,
            "timestamp": analysis.created_at,
            "details": analysis.details,
            "processing_time": analysis.processing_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get analysis detail: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get analysis: {str(e)}")

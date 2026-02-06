"""
Feedback Routes
API endpoints for user feedback on analysis results
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

try:
    from ..services.database import get_db_manager
except:
    from services.database import get_db_manager

logger = logging.getLogger(__name__)

router = APIRouter()


class FeedbackRequest(BaseModel):
    """Request model for submitting feedback"""

    analysis_id: int
    feedback_type: str  # correct, incorrect, uncertain
    actual_label: Optional[str] = None  # authentic, deepfake, suspicious
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    """Response model for feedback submission"""

    success: bool
    message: str


@router.post("/submit", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback for an analysis result

    Args:
        feedback: Feedback data

    Returns:
        Success status and message
    """
    try:
        # Validate feedback_type
        valid_types = ["correct", "incorrect", "uncertain"]
        if feedback.feedback_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid feedback_type. Must be one of: {valid_types}",
            )

        # Validate actual_label if provided
        if feedback.actual_label:
            valid_labels = ["authentic", "deepfake", "suspicious", "fake"]
            if feedback.actual_label.lower() not in valid_labels:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid actual_label. Must be one of: {valid_labels}",
                )

        # Save feedback
        db = get_db_manager()
        success = db.save_user_feedback(
            analysis_id=feedback.analysis_id,
            feedback_type=feedback.feedback_type,
            actual_label=feedback.actual_label,
            comment=feedback.comment,
            user_id=None,  # Get from auth context
        )

        if success:
            return FeedbackResponse(
                success=True, message="Feedback submitted successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to save feedback")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/false-positive-rate")
async def get_false_positive_rate():
    """
    Get the current false positive rate based on user feedback

    Returns:
        False positive rate (0.0 to 1.0)
    """
    try:
        db = get_db_manager()
        fpr = db.get_false_positive_rate()

        return {
            "success": True,
            "false_positive_rate": fpr,
            "message": "FPR calculated from user feedback"
            if fpr != 0.05
            else "Using default FPR (insufficient data)",
        }

    except Exception as e:
        logger.error(f"Error getting FPR: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

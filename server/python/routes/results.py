"""
Results Routes
Handle job results retrieval for the Python ML service
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()

class JobResult(BaseModel):
    """Job result response model"""
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.get("/results/{job_id}")
async def get_job_result(job_id: str) -> JobResult:
    """
    Get job result by ID.
    
    Note: This is a compatibility endpoint for the Node-Python bridge.
    The actual job data is stored in Supabase and should be accessed via Node.js endpoints.
    """
    logger.info(f"[RESULTS] Python service received request for job {job_id}")
    
    # This endpoint should not be used in the fixed architecture
    # Return a clear error message indicating the correct approach
    raise HTTPException(
        status_code=404,
        detail={
            "error": "JOB_NOT_FOUND",
            "message": f"Job {job_id} not found in Python service",
            "note": "Job results are stored in Supabase. Use Node.js /results/:jobId endpoint instead.",
            "correct_endpoint": f"/api/v2/results/{job_id}"
        }
    )

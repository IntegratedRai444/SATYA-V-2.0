from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Dict, Any
from datetime import datetime
import os
import tempfile
import shutil
import logging
from pathlib import Path

from backend.utils.video_utils import (
    preprocess_video, extract_frames, analyze_frames, 
    generate_timeline, generate_video_overlay, analyze_video_advanced
)
from backend.utils.report_utils import generate_video_report, export_report_json, export_report_pdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

async def process_video_upload(
    file: UploadFile,
    output_dir: str = "reports",
    advanced_analysis: bool = True,
    generate_overlay: bool = True
) -> Dict[str, Any]:
    """Process video upload and return analysis results."""
    try:
        # Create a temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded file temporarily
            temp_video_path = os.path.join(temp_dir, file.filename)
            with open(temp_video_path, "wb") as f:
                f.write(await file.read())
            
            # Get video metadata
            video_info = preprocess_video(temp_video_path)
            
            # Perform analysis
            if advanced_analysis:
                result = analyze_video_advanced(
                    video_path=temp_video_path,
                    output_dir=temp_dir,
                    generate_overlay=generate_overlay
                )
            else:
                # Fallback to basic analysis
                frames = extract_frames(temp_video_path)
                frame_results = analyze_frames(frames)
                result = {
                    'frame_results': frame_results,
                    'video_info': video_info,
                    'timeline': generate_timeline(frame_results)
                }
                
                if generate_overlay:
                    result['overlay_video'] = generate_video_overlay(
                        video_path=temp_video_path,
                        frame_results=frame_results,
                        output_path=os.path.join(temp_dir, 'overlay.mp4')
                    )
            
            # Generate reports
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{os.path.splitext(file.filename)[0]}_{timestamp}"
            
            # Save results
            json_path = os.path.join(output_dir, f"{base_filename}.json")
            pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
            
            export_report_json(result, json_path)
            export_report_pdf(result, pdf_path)
            
            # Move overlay video if it exists
            if 'overlay_video' in result and os.path.exists(result['overlay_video']):
                overlay_dest = os.path.join(output_dir, f"{base_filename}_overlay.mp4")
                shutil.move(result['overlay_video'], overlay_dest)
                result['overlay_video'] = overlay_dest
            
            return {
                'success': True,
                'result': result,
                'report_json': json_path,
                'report_pdf': pdf_path,
                'overlay_video': result.get('overlay_video')
            }
            
    except Exception as e:
        logger.error(f"Video processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process video: {str(e)}"
        )

@router.post("/analyze")
async def analyze_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None,
    advanced: bool = True,
    overlay: bool = True
) -> JSONResponse:
    """
    Enhanced video analysis endpoint with support for background processing.
    
    Args:
        file: Video file to analyze
        background_tasks: FastAPI background tasks (optional)
        advanced: Whether to use advanced analysis (default: True)
        overlay: Whether to generate video overlay (default: True)
        
    Returns:
        JSON response with analysis results or task ID
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(400, "Uploaded file is not a video")
    
    if background_tasks:
        # Start background task and return task ID
        task_id = f"task_{datetime.now().timestamp()}"
        background_tasks.add_task(
            process_video_upload,
            file=file,
            advanced_analysis=advanced,
            generate_overlay=overlay
        )
        return JSONResponse({
            'status': 'processing',
            'task_id': task_id,
            'message': 'Video analysis started in background'
        })
    
    # Process synchronously
    try:
        result = await process_video_upload(
            file=file,
            advanced_analysis=advanced,
            generate_overlay=overlay
        )
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Video analysis failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@router.get("/results/{task_id}")
async def get_results(task_id: str) -> JSONResponse:
    """Get results for a background task."""
    # TODO: Implement task status tracking
    return JSONResponse({
        "status": "not_implemented",
        "message": "Task status tracking not implemented yet"
    })

@router.get("/download/{filename}")
async def download_file(filename: str) -> FileResponse:
    """Download a generated file (report, overlay, etc.)."""
    file_path = os.path.join("reports", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        file_path,
        media_type="application/octet-stream",
        filename=filename
    )

@router.post("/api/ai/analyze/video")
async def analyze_video_alias(
    file: UploadFile = File(...), 
    api_key: str = Depends(...)
) -> JSONResponse:
    """Legacy API endpoint for backward compatibility."""
    return await analyze_video(file) 
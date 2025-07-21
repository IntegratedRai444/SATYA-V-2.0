from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.utils.video_utils import (
    preprocess_video, extract_frames, analyze_frames, generate_timeline, generate_video_overlay
)
from backend.utils.report_utils import generate_video_report, export_report_json, export_report_pdf
import os

router = APIRouter()

@router.post("/")
async def detect_video(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint for video deepfake analysis. Accepts a video upload, runs preprocessing, per-frame analysis,
    timeline/overlay generation, and report creation. Returns analysis results and report paths.
    """
    if file.content_type not in ["video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"]:
        raise HTTPException(status_code=400, detail="Unsupported video format.")
    try:
        contents = await file.read()
        video_info = preprocess_video(contents)
        frames = extract_frames(contents)
        frame_results = analyze_frames(frames)
        timeline = generate_timeline(frame_results)
        overlay_path = generate_video_overlay(contents, frame_results)
        # Aggregate results
        fake_count = sum(1 for fr in frame_results if fr["label"] == "FAKE")
        real_count = sum(1 for fr in frame_results if fr["label"] == "REAL")
        if fake_count > real_count:
            label = "FAKE"
        elif real_count > fake_count:
            label = "REAL"
        else:
            label = "UNCERTAIN"
        confidence = max([fr["confidence"] for fr in frame_results]) if frame_results else 0.0
        explanation = [exp for fr in frame_results for exp in fr["explanation"]]
        red_flags = [exp for fr in frame_results for exp in fr["explanation"] if "FAKE" in fr["label"]]
        result = {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "red_flags": red_flags,
            "timeline": timeline,
            "overlay_video": overlay_path,
            "frame_results": frame_results,
            "video_info": video_info
        }
        report = generate_video_report(file.filename, result, video_info)
        os.makedirs("reports", exist_ok=True)
        json_path = f"reports/{file.filename}.json"
        pdf_path = f"reports/{file.filename}.pdf"
        export_report_json(report, json_path)
        export_report_pdf(report, pdf_path)
        return JSONResponse(content={
            "success": True,
            "result": result,
            "report": report,
            "report_json": json_path,
            "report_pdf": pdf_path,
            "overlay_video": overlay_path
        })
    except Exception as e:
        print(f"Video processing failed: {e}")
        return JSONResponse(content={"success": False, "message": f"Video processing failed: {str(e)}"}, status_code=500)

@router.post("/api/ai/analyze/video")
def analyze_video_alias(file: UploadFile = File(...), api_key: str = Depends(...)):
    return detect_video(file, api_key) 
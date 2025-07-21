from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from backend.utils.image_utils import preprocess_image
from backend.utils.report_utils import generate_webcam_report, export_report_json, export_report_pdf
from backend.models.webcam_model import predict_webcam_liveness
import os

router = APIRouter()

@router.post("/")
async def detect_webcam_frame(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint for webcam frame liveness analysis. Accepts an image upload, runs preprocessing, liveness detection,
    and report generation. Returns analysis results and report paths.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    try:
        contents = await file.read()
        arr = preprocess_image(contents)
        # Use real liveness, blink, pose, spoof detection
        label, confidence, explanation = predict_webcam_liveness(arr)
        # For blink, pose, spoof, parse from explanation or set as best effort
        blink = {"detected": any('Blink detected' in exp for exp in explanation), "count": 1 if any('Blink detected' in exp for exp in explanation) else 0}
        # Parse pose from explanation if present
        pose_vals = next((exp for exp in explanation if exp.startswith('Pose')), None)
        if pose_vals:
            try:
                pose_parts = pose_vals.split(':')[1].split(',')
                yaw = float(pose_parts[0])
                pitch = float(pose_parts[1])
                roll = float(pose_parts[2])
            except Exception:
                yaw = pitch = roll = 0.0
        else:
            yaw = pitch = roll = 0.0
        pose = {"yaw": yaw, "pitch": pitch, "roll": roll}
        spoof = {"gan_replay": (label == 'FAKE')}
        liveness = {"score": confidence / 100.0, "decision": label}
        session_report = {"timeline": ["blink"] if blink["detected"] else [], "summary": explanation[-1] if explanation else ""}
        red_flags = [exp for exp in explanation if 'spoof' in exp.lower() or 'no face' in exp.lower()]
        result = {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "red_flags": red_flags,
            "liveness": liveness,
            "blink": blink,
            "pose": pose,
            "spoof": spoof,
            "session_report": session_report
        }
        report = generate_webcam_report(file.filename, result, arr)
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
            "report_pdf": pdf_path
        })
    except Exception as e:
        print(f"Webcam frame processing failed: {e}")
        return JSONResponse(content={"success": False, "message": f"Webcam frame processing failed: {str(e)}"}, status_code=500) 
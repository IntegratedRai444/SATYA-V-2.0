from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.utils.audio_utils import preprocess_audio, generate_spectrogram, detect_voice_clone, check_pitch_jitter, match_voiceprint
from backend.utils.report_utils import generate_audio_report, export_report_json, export_report_pdf
import os
from backend.models.audio_model import predict_audio_deepfake

router = APIRouter()

@router.post("/")
async def detect_audio(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint for audio deepfake analysis. Accepts an audio upload, runs preprocessing, feature extraction,
    placeholder model inference, and report generation. Returns analysis results and report paths.
    """
    if file.content_type not in ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/flac"]:
        raise HTTPException(status_code=400, detail="Unsupported audio format.")
    try:
        contents = await file.read()
        audio_info = preprocess_audio(contents)
        spectrogram_path = generate_spectrogram(contents)
        clone_result = detect_voice_clone(contents)
        pitch_jitter = check_pitch_jitter(contents)
        voiceprint_match = match_voiceprint(contents)
        label, confidence, explanation = predict_audio_deepfake(contents)
        red_flags = ["Signature curve deviates", "Lack of emotional inflection"]
        result = {
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "red_flags": red_flags,
            "spectrogram": spectrogram_path,
            "clone_result": clone_result,
            "pitch_jitter": pitch_jitter,
            "voiceprint_match": voiceprint_match,
            "audio_info": audio_info
        }
        report = generate_audio_report(file.filename, result, audio_info)
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
            "spectrogram": spectrogram_path
        })
    except Exception as e:
        print(f"Audio processing failed: {e}")
        return JSONResponse(content={"success": False, "message": f"Audio processing failed: {str(e)}"}, status_code=500)

@router.post("/api/ai/analyze/audio")
def analyze_audio_alias(file: UploadFile = File(...), api_key: str = Depends(...)):
    return detect_audio(file, api_key) 
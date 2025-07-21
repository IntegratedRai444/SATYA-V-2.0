from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from backend.utils.image_utils import (
    preprocess_image, extract_exif, detect_faces, scan_gan_artifacts, generate_heatmap
)
from backend.utils.report_utils import generate_image_report, export_report_json, export_report_pdf
from backend.models.image_model import predict_deepfake
import os
import base64
from PIL import Image
import io
import numpy as np

router = APIRouter()

@router.post("/")
async def detect_image(file: UploadFile = File(...)) -> JSONResponse:
    """
    Endpoint for image deepfake analysis. Accepts an image upload, runs preprocessing, feature extraction,
    placeholder model inference, and report generation. Returns analysis results and report paths.
    """
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    try:
        contents = await file.read()
        arr = preprocess_image(contents)
        exif = extract_exif(contents)
        faces = detect_faces(arr)
        gan_artifacts = scan_gan_artifacts(arr)
        heatmap = generate_heatmap(arr)
        # Save heatmap as an image file (if not a placeholder)
        heatmap_path = None
        if isinstance(heatmap, str) and heatmap != "heatmap_placeholder":
            heatmap_data = base64.b64decode(heatmap)
            image = Image.open(io.BytesIO(heatmap_data))
            heatmap_path = f"reports/{file.filename}_heatmap.png"
            image.save(heatmap_path)
        elif isinstance(heatmap, (np.ndarray, list)):
            arr_to_save = np.array(heatmap) if isinstance(heatmap, list) else heatmap
            # Only save if arr_to_save is a numeric numpy array and not a string
            if isinstance(arr_to_save, np.ndarray) and hasattr(arr_to_save, 'dtype') and arr_to_save.dtype.kind in ('f', 'i', 'u') and not isinstance(arr_to_save, str):
                image = Image.fromarray((arr_to_save * 255).astype('uint8'))
                heatmap_path = f"reports/{file.filename}_heatmap.png"
                image.save(heatmap_path)
            else:
                heatmap_path = None
        else:
            heatmap_path = None
        # Model inference
        label, confidence, explanation = predict_deepfake(arr, faces, gan_artifacts)
        red_flags = ["Blurred boundary around ears & neck", "GAN checkerboard residue around eyes"]
        result = {
            "preprocessed_shape": str(arr.shape),
            "exif": exif,
            "faces": faces,
            "gan_artifacts": gan_artifacts,
            "heatmap": heatmap,
            "label": label,
            "confidence": confidence,
            "explanation": explanation,
            "red_flags": red_flags
        }
        report = generate_image_report(file.filename, result, exif)
        os.makedirs("reports", exist_ok=True)
        json_path = f"reports/{file.filename}.json"
        pdf_path = f"reports/{file.filename}.pdf"
        export_report_json(report, json_path)
        export_report_pdf(report, pdf_path)
        files_exist = {
            "json": os.path.exists(json_path),
            "pdf": os.path.exists(pdf_path),
            "heatmap": os.path.exists(heatmap_path) if heatmap_path else False
        }
        return JSONResponse(content={
            "success": True,
            "result": result,
            "report": report,
            "report_json": json_path,
            "report_pdf": pdf_path,
            "heatmap_path": heatmap_path,
            "files_exist": files_exist
        })
    except Exception as e:
        print(f"Image processing failed: {e}")
        return JSONResponse(content={"success": False, "message": f"Image processing failed: {str(e)}"}, status_code=500)

@router.post("/api/ai/analyze/image")
def analyze_image_alias(file: UploadFile = File(...), api_key: str = Depends(...)):
    return detect_image(file, api_key)

@router.post("/api/ai/analyze/batch")
def analyze_batch_stub():
    return {"success": False, "message": "Batch analysis not implemented yet."}

@router.post("/api/ai/analyze/multimodal")
def analyze_multimodal_stub():
    return {"success": False, "message": "Multimodal analysis not implemented yet."} 
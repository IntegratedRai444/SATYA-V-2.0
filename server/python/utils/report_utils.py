import json
import os
import uuid
from datetime import datetime

# Lazy import fpdf for faster startup
_FPDF = None


def get_fpdf():
    """Lazy load FPDF with fallback"""
    global _FPDF
    if _FPDF is None:
        try:
            from fpdf import FPDF

            _FPDF = FPDF
        except ImportError:
            print("⚠️ FPDF not available. PDF reports will be disabled.")
            _FPDF = None
    return _FPDF


def generate_image_report(filename, result, exif, heatmap_path=None):
    report_code = (
        f"SATYA-IMG-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8].upper()}"
    )
    report = {
        "file": filename,
        "scan_type": "Single Frame Forensic Analysis",
        "report_code": report_code,
        "status": result["label"],
        "confidence": result["confidence"],
        "summary": result["explanation"],
        "exif": exif,
        "heatmap": heatmap_path,
        "models_used": ["XceptionNet", "MesoNet", "GAN Artifact Classifier"],
        "key_red_flags": result.get("red_flags", []),
    }
    return report


def export_report_json(report, path):
    with open(path, "w") as f:
        json.dump(report, f, indent=2)


def export_report_pdf(report, path):
    FPDF = get_fpdf()
    if FPDF is None:
        print(
            f"⚠️ PDF export disabled. Report saved as JSON only: {path.replace('.pdf', '.json')}"
        )
        return False

    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        for k, v in report.items():
            pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
        pdf.output(path)
        return True
    except Exception as e:
        print(f"❌ PDF generation failed: {e}")
        return False


def generate_video_report(*args, **kwargs):
    return {"stub": True}


def generate_audio_report(*args, **kwargs):
    return {"stub": True}


def generate_webcam_report(*args, **kwargs):
    return {"stub": True}

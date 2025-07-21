import json
from fpdf import FPDF
import os

def generate_image_report(filename, result, exif, heatmap_path=None):
    report = {
        "file": filename,
        "scan_type": "Single Frame Forensic Analysis",
        "report_code": "SATYA-IMG-20250714-0013",  # TODO: generate dynamically
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
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for k, v in report.items():
        pdf.cell(200, 10, txt=f"{k}: {v}", ln=True)
    pdf.output(path)

def generate_video_report(*args, **kwargs):
    return {"stub": True}

def generate_audio_report(*args, **kwargs):
    return {"stub": True}

def generate_webcam_report(*args, **kwargs):
    return {"stub": True} 
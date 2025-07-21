import json
import hashlib
import qrcode
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import os

# Ensure exports directory exists
os.makedirs("report_engine/exports", exist_ok=True)

# 1. Prepare data (demo)
data = {
    "report_code": "SATYA-VID-20250714-0045",
    "result": "Likely Fake",
    "confidence": 93.82,
    "summary": "A full timeline scan flagged over 200 frames as synthetic...",
    "models_used": ["XceptionNet", "Wav2Lip", "LipNet", "BlinkNet"],
    "red_flags": [
        "No blink detected for >30s",
        "GAN noise in forehead + cheek region",
        "Audio 0.6s desync with mouth movements"
    ],
    "key_frames": [
        {"img_path": "exports/frame1.png", "timestamp": "00:14", "gan_score": 0.93},
        {"img_path": "exports/frame2.png", "timestamp": "00:36", "gan_score": 0.91}
    ],
    "qr_path": "exports/qr.png",
    "blockchain_hash": ""
}

# 2. Generate QR code
qr = qrcode.make(data["report_code"])
qr.save("report_engine/exports/qr.png")
data["qr_path"] = "exports/qr.png"

# 3. Blockchain hash (SHA-256 of JSON)
json_data = json.dumps(data, sort_keys=True).encode()
data["blockchain_hash"] = hashlib.sha256(json_data).hexdigest()

# 4. Render HTML with Jinja2
env = Environment(loader=FileSystemLoader("report_engine/report_templates"))
template = env.get_template("video_report.html")
html_out = template.render(**data)

with open("report_engine/exports/report.html", "w", encoding="utf-8") as f:
    f.write(html_out)

# 5. Convert HTML to PDF
HTML(string=html_out, base_url=".").write_pdf("report_engine/exports/SATYA-VID-REPORT-004.pdf")

# 6. Save JSON
with open("report_engine/exports/SATYA-VID-REPORT-004.json", "w") as f:
    json.dump(data, f, indent=2)

print("Report generated: PDF, JSON, and HTML.") 
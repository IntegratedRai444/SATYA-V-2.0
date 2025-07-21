import json
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML
import os

# Ensure exports directory exists
os.makedirs("report_engine/exports", exist_ok=True)

# Example analysis data (replace with real analysis results)
data = {
    "filename": "1_sUl4nkPfH0wevBQMb29cnQ (1).jpg",
    "result_label": "DEEPFAKE DETECTED",
    "confidence": 76,
    "summary_title": "DEEPFAKE DETECTED",
    "summary_text": "This media shows signs of AI manipulation with 76% confidence.",
    "key_findings": [
        "GAN artifact patterns detected in facial region.",
        "No valid EXIF/camera metadata found.",
        "Boundary blur around ears and neck.",
        "Lighting inconsistency in background."
    ]
}

# Render HTML with Jinja2
env = Environment(loader=FileSystemLoader("report_engine/report_templates"))
template = env.get_template("image_report.html")
html_out = template.render(**data)

with open("report_engine/exports/image_report.html", "w", encoding="utf-8") as f:
    f.write(html_out)

# Convert HTML to PDF
HTML(string=html_out, base_url=".").write_pdf("report_engine/exports/image_report.pdf")

# Save JSON
with open("report_engine/exports/image_report.json", "w") as f:
    json.dump(data, f, indent=2)

print("Image report generated: PDF, JSON, and HTML.") 
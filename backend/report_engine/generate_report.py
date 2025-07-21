import json
import hashlib
import qrcode
from jinja2 import Environment, FileSystemLoader
import os

def generate_report(data, template_name="video_report.html", output_dir="report_engine/exports"):
    """
    Generate a report (HTML, PDF, JSON) from analysis data.
    Adds a QR code and blockchain hash for authenticity.
    Returns paths to generated files.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # 1. Generate QR code for report code
        qr_path = os.path.join(output_dir, f"{data['report_code']}_qr.png")
        with open(qr_path, "wb") as qr_file:
            qr_img = qrcode.make(data["report_code"])
            qr_img.save(qr_file)
        data["qr_path"] = qr_path
        # 2. Blockchain hash (SHA-256 of JSON)
json_data = json.dumps(data, sort_keys=True).encode()
data["blockchain_hash"] = hashlib.sha256(json_data).hexdigest()
        # 3. Render HTML with Jinja2
        env = Environment(loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "report_templates")))
        template = env.get_template(template_name)
html_out = template.render(**data)
        html_path = os.path.join(output_dir, f"{data['report_code']}.html")
        with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_out)
        # 4. Convert HTML to PDF
        try:
            from weasyprint import HTML
            pdf_path = os.path.join(output_dir, f"{data['report_code']}.pdf")
            HTML(string=html_out, base_url=output_dir).write_pdf(pdf_path)
        except ImportError:
            pdf_path = None
        # 5. Save JSON
        json_path = os.path.join(output_dir, f"{data['report_code']}.json")
        with open(json_path, "w") as f:
    json.dump(data, f, indent=2)
        return {"html": html_path, "pdf": pdf_path, "json": json_path, "qr": qr_path}
    except Exception as e:
        return {"error": str(e)} 
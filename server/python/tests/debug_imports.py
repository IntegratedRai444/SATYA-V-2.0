import os
import sys
from pathlib import Path

# Add server/python to path
current_dir = Path(__file__).parent
server_dir = current_dir.parent
sys.path.append(str(server_dir))

print(f"Server dir: {server_dir}")

modules = [
    "routes.auth",
    "routes.upload",
    "routes.analysis",
    "routes.dashboard",
    "routes.health",
    "routes.image",
    "routes.video",
    "routes.audio",
    "routes.face",
    "routes.system",
    "routes.webcam",
    "routes.feedback",
    "routes.team",
]

for module in modules:
    try:
        __import__(module)
        print(f"✅ {module} imported successfully")
    except Exception as e:
        print(f"❌ {module} failed to import: {e}")

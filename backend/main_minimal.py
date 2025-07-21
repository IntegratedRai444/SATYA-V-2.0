import os
from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SatyaAI Deepfake Detection API (Minimal)")

# CORS: Use environment variable for allowed origins in production
allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage for demo
users = {}
api_keys = {}

def api_key_auth(request: Request):
    key = request.headers.get("x-api-key")
    if key not in api_keys.values():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )

# Simple routes without heavy dependencies
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import secrets

# Auth router
auth_router = APIRouter()

@auth_router.post("/register")
async def register(request: Request):
    data = await request.json()
    username = data.get("username", "demo")
    password = data.get("password", "demo")
    
    if username in users:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    users[username] = password
    api_key = secrets.token_hex(16)
    api_keys[username] = api_key
    
    return {"success": True, "api_key": api_key}

@auth_router.post("/login")
async def login(request: Request):
    data = await request.json()
    username = data.get("username", "demo")
    password = data.get("password", "demo")
    
    if users.get(username) != password:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    api_key = api_keys.get(username)
    if not api_key:
        api_key = secrets.token_hex(16)
        api_keys[username] = api_key
    
    return {"success": True, "api_key": api_key}

# Image router (minimal)
image_router = APIRouter()

@image_router.post("/")
async def detect_image(file: UploadFile = File(...)) -> JSONResponse:
    """Minimal image analysis endpoint"""
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format.")
    
    # Simple mock analysis
    result = {
        "label": "REAL",
        "confidence": 85.0,
        "explanation": ["Minimal analysis mode - no ML models loaded"],
        "red_flags": [],
        "file_size": len(await file.read())
    }
    
    return JSONResponse(content={
        "success": True,
        "result": result,
        "message": "Minimal mode - install full dependencies for ML analysis"
    })

# System router
system_router = APIRouter()

@system_router.get("/health")
async def get_health() -> JSONResponse:
    """Health check"""
    return JSONResponse(content={
        "success": True,
        "status": "ok",
        "version": "1.2.0-minimal",
        "mode": "minimal (no ML models)"
    })

@system_router.get("/config")
async def get_config() -> JSONResponse:
    """Get system configuration"""
    return JSONResponse(content={
        "success": True,
        "supported_formats": {
            "image": ["jpeg", "png", "webp"],
            "video": ["mp4", "mov", "avi"],
            "audio": ["wav", "mp3", "flac"]
        },
        "mode": "minimal"
    })

# Include routers
app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
app.include_router(image_router, prefix="/detect/image", tags=["Image"], dependencies=[Depends(api_key_auth)])
app.include_router(system_router, prefix="/api", tags=["System"])

@app.get("/")
def root():
    return {
        "message": "SatyaAI Deepfake Detection API (Minimal Mode)",
        "status": "running",
        "mode": "minimal - no ML models loaded"
    }

if __name__ == "__main__":
    print("🌐 Starting basic HTTP server...")
    import http.server
    import socketserver
    import threading
    import json
    
    class FastAPIHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    "message": "SatyaAI Deepfake Detection API (Basic Mode)",
                    "status": "running",
                    "mode": "basic - no uvicorn"
                }
                self.wfile.write(json.dumps(response).encode())
            elif self.path == '/api/health':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    "success": True,
                    "status": "ok",
                    "version": "1.2.0-basic",
                    "mode": "basic (no uvicorn)"
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {"error": "Endpoint not found"}
                self.wfile.write(json.dumps(response).encode())
        
        def do_POST(self):
            if self.path == '/api/auth/login':
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    "success": True,
                    "api_key": "demo-key-12345"
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {"error": "Endpoint not found"}
                self.wfile.write(json.dumps(response).encode())
        
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, x-api-key')
            self.end_headers()
    
    def run_server():
        with socketserver.TCPServer(("", 8000), FastAPIHandler) as httpd:
            print("🌐 Basic server running on http://localhost:8000")
            print("📡 API endpoints available:")
            print("  GET  / - Root endpoint")
            print("  GET  /api/health - Health check")
            print("  POST /api/auth/login - Demo login")
            httpd.serve_forever()
    
    try:
        run_server()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}") 
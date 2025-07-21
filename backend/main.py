import os
from fastapi import FastAPI, Depends, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="SatyaAI Deepfake Detection API")

# CORS: Use environment variable for allowed origins in production
allowed_origins = os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import routes to speed up startup
def get_routes():
    """Lazy load routes to avoid startup delays"""
    from backend.routes import image, video, audio, webcam, auth, assistant, system
    return image, video, audio, webcam, auth, assistant, system

# API key authentication dependency
def get_api_keys():
    """Lazy load API keys"""
    from backend.routes.auth import api_keys
    return api_keys

def api_key_auth(request: Request):
    key = request.headers.get("x-api-key")
    api_keys = get_api_keys()
    if key not in api_keys.values():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key"
        )

@app.on_event("startup")
async def startup_event():
    """Optimized startup - load routes and models lazily"""
    print("🚀 Starting SatyaAI Backend...")
    
    # Load routes
    image, video, audio, webcam, auth, assistant, system = get_routes()
    
    # Include routers with API key dependency for analysis endpoints
    app.include_router(image.router, prefix="/detect/image", tags=["Image"], dependencies=[Depends(api_key_auth)])
    app.include_router(video.router, prefix="/detect/video", tags=["Video"], dependencies=[Depends(api_key_auth)])
    app.include_router(audio.router, prefix="/detect/audio", tags=["Audio"], dependencies=[Depends(api_key_auth)])
    app.include_router(webcam.router, prefix="/detect/webcam", tags=["Webcam"], dependencies=[Depends(api_key_auth)])
    app.include_router(auth.router, prefix="/api/auth", tags=["Auth"])
    app.include_router(assistant.router, prefix="/assistant", tags=["Assistant"])
    app.include_router(system.router, prefix="/api", tags=["System"])
    
    print("✅ Routes loaded successfully")

# Logging setup (production: integrate with Sentry or similar)
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("satyaai")

@app.get("/")
def root():
    return {"message": "SatyaAI Deepfake Detection API is running."} 
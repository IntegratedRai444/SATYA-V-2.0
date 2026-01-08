"""
SatyaAI FastAPI Application
Main entry point for the deepfake detection API
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from typing import List, Optional
import uvicorn
import logging
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import model manager
from model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SatyaAI API",
    description="REST API for deepfake detection in images, videos, and audio",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
ALLOWED_ORIGINS = os.getenv('ALLOWED_ORIGINS', '').split(',')

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600,  # 10 minutes
)

# OAuth2 scheme for authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize model manager
model_manager = ModelManager()

# API status endpoint
@app.get("/")
async def root():
    """Root endpoint for API status check"""
    return {
        "status": "SatyaAI API is running",
        "version": "1.0.0",
        "endpoints": [
            "/docs - API documentation",
            "/status - System status",
            "/models - List available models",
            "/analyze/image - Analyze image",
            "/analyze/video - Analyze video",
            "/analyze/audio - Analyze audio"
        ]
    }

# System status endpoint
@app.get("/status")
async def get_status():
    """Get system and model status"""
    try:
        # Get system information
        import psutil
        import torch
        
        system_info = {
            "system": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "gpu_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            "models_loaded": len(model_manager.models) > 0,
            "models_available": list(model_manager.models.keys())
        }
        
        return system_info
    
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting system status: {str(e)}"
        )

# Models endpoint
@app.get("/models")
async def list_models():
    """List all available models"""
    try:
        return {
            "models": model_manager.get_available_models(),
            "loaded_models": list(model_manager.models.keys())
        }
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing models: {str(e)}"
        )

# Load model endpoint
@app.post("/models/{model_name}/load")
async def load_model(model_name: str):
    """Load a specific model"""
    try:
        success = model_manager.load_model(model_name)
        if success:
            return {"status": f"Model {model_name} loaded successfully"}
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load model {model_name}"
            )
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading model: {str(e)}"
        )

# Unload model endpoint
@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """Unload a specific model"""
    try:
        if model_name in model_manager.models:
            del model_manager.models[model_name]
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"status": f"Model {model_name} unloaded successfully"}
        else:
            return {"status": f"Model {model_name} not loaded"}
    except Exception as e:
        logger.error(f"Error unloading model {model_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error unloading model: {str(e)}"
        )

# Image analysis endpoint
@app.post("/analyze/image")
async def analyze_image(
    image_path: str,
    model_name: str = "xception_deepfakes"
):
    """
    Analyze an image for deepfake detection
    
    Args:
        image_path: Path to the image file
        model_name: Name of the model to use for detection
        
    Returns:
        Analysis results
    """
    try:
        # Verify file exists
        if not os.path.isfile(image_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {image_path}"
            )
            
        # Process image
        result = model_manager.process_image(image_path, model_name)
        
        return {
            "status": "success",
            "model": model_name,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing image: {str(e)}"
        )

# Video analysis endpoint
@app.post("/analyze/video")
async def analyze_video(
    video_path: str,
    model_name: str = "xception_deepfakes",
    frame_interval: int = 10
):
    """
    Analyze a video for deepfake detection
    
    Args:
        video_path: Path to the video file
        model_name: Name of the model to use for detection
        frame_interval: Process every N-th frame
        
    Returns:
        Analysis results
    """
    try:
        # Verify file exists
        if not os.path.isfile(video_path):
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {video_path}"
            )
            
        # Process video
        result = model_manager.process_video(
            video_path=video_path,
            model_name=model_name,
            frame_interval=frame_interval
        )
        
        return {
            "status": "success",
            "model": model_name,
            "result": result
        }
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing video: {str(e)}"
        )

# Audio analysis is now handled by the SentinelAgent through the /audio/ endpoint

def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    logger.info(f"Starting SatyaAI API server on {host}:{port}")
    
    # Load default models on startup
    try:
        logger.info("Loading default models...")
        model_manager.load_model("face_detector")
        model_manager.load_model("face_embedder")
        model_manager.load_model("xception_deepfakes")
        logger.info("Default models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading default models: {e}")
    
    # Start the server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=1  # Multiple workers can cause issues with CUDA
    )

if __name__ == "__main__":
    start_server()

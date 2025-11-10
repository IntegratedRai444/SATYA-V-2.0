""
Audio Analysis API Endpoints
Handles real-time and batch audio analysis requests.
"""
import os
import json
import time
import torch
import numpy as np
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import audio processing modules
from server.python.models.audio_enhanced import AudioDeepfakeDetector, detect_audio_deepfake
from server.python.realtime.audio_processor import RealTimeAudioProcessor, AudioConfig

# Initialize router
router = APIRouter()

# Global variables
audio_processor = None
processing_task = None

# Models
class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis."""
    audio_data: Optional[bytes] = None
    audio_path: Optional[str] = None
    sample_rate: int = 16000
    threshold: float = 0.5

class RealTimeConfig(BaseModel):
    """Configuration for real-time processing."""
    sample_rate: int = 16000
    buffer_size: int = 1024
    threshold: float = 0.5

# Load model (lazy loading)
_model = None
def get_model():
    """Get or load the audio model."""
    global _model
    if _model is None:
        try:
            _model = AudioDeepfakeDetector()
            # Load pre-trained weights if available
            model_path = "models/audio_deepfake_detector.pth"
            if os.path.exists(model_path):
                _model.load_state_dict(torch.load(model_path, map_location='cpu'))
            _model.eval()
            logger.info("Audio model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading audio model: {e}")
            raise HTTPException(status_code=500, detail="Failed to load audio model")
    return _model

@router.post("/analyze-audio")
async def analyze_audio(request: AudioAnalysisRequest):
    """
    Analyze an audio file for deepfake detection.
    
    Either audio_data (base64 encoded) or audio_path must be provided.
    """
    try:
        model = get_model()
        
        # Process audio file
        if request.audio_path:
            if not os.path.exists(request.audio_path):
                raise HTTPException(status_code=404, detail="Audio file not found")
            
            result = detect_audio_deepfake(
                audio_path=request.audio_path,
                model=model,
                threshold=request.threshold
            )
        
        # Process raw audio data
        elif request.audio_data:
            # Save to temp file and process
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp.write(request.audio_data)
                tmp_path = tmp.name
            
            try:
                result = detect_audio_deepfake(
                    audio_path=tmp_path,
                    model=model,
                    threshold=request.threshold
                )
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
        else:
            raise HTTPException(status_code=400, detail="Either audio_data or audio_path must be provided")
        
        return {
            "success": True,
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

manager = ConnectionManager()

@router.websocket("/ws/audio-analysis")
async def websocket_audio_analysis(websocket: WebSocket):
    """WebSocket endpoint for real-time audio analysis."""
    global audio_processor, processing_task
    
    await manager.connect(websocket)
    
    try:
        # Receive configuration
        config_data = await websocket.receive_json()
        config = RealTimeConfig(**config_data)
        
        # Initialize audio processor if not already running
        if audio_processor is None:
            model = get_model()
            audio_config = AudioConfig(
                sample_rate=config.sample_rate,
                frames_per_buffer=config.buffer_size
            )
            
            async def process_result(result: dict):
                await manager.broadcast({
                    "type": "analysis_result",
                    "data": result,
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            # Create processor with callback
            audio_processor = RealTimeAudioProcessor(
                model=model,
                config=audio_config,
                callback=lambda r: asyncio.create_task(process_result(r))
            )
            
            # Start processing in background
            def run_processor():
                audio_processor.start()
            
            processing_task = asyncio.create_task(asyncio.to_thread(run_processor))
            
            await websocket.send_json({
                "type": "status",
                "message": "Audio processor started",
                "timestamp": datetime.utcnow().isoformat()
            })
        
        # Keep connection alive
        while True:
            try:
                # Just keep the connection open
                await asyncio.sleep(1)
                # Check if client is still connected
                await websocket.send_json({
                    "type": "ping",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.send_json({
            "type": "error",
            "message": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })
    finally:
        manager.disconnect(websocket)
        
        # Clean up if no more connections
        if not manager.active_connections and audio_processor is not None:
            audio_processor.stop()
            audio_processor = None
            if processing_task:
                processing_task.cancel()
                try:
                    await processing_task
                except asyncio.CancelledError:
                    pass
                processing_task = None

def init_audio_analysis_api(app):
    """Initialize audio analysis API routes."""
    app.include_router(router, prefix="/api/v1")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Clean up resources on shutdown."""
        global audio_processor, processing_task
        if audio_processor is not None:
            audio_processor.stop()
            audio_processor = None
        if processing_task:
            processing_task.cancel()
            try:
                await processing_task
            except asyncio.CancelledError:
                pass
            processing_task = None

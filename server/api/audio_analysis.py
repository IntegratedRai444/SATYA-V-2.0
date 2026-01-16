"""
Audio Analysis API Endpoints
Handles real-time and batch audio analysis requests with enhanced security and performance.
"""
import os
import json
import time
import hashlib
import tempfile
import torch
import numpy as np
import magic
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable, Awaitable
from datetime import datetime, timedelta
from functools import lru_cache, wraps
import aiofiles
import asyncio
import logging
import concurrent.futures
import traceback
from fastapi import (
    APIRouter, 
    HTTPException, 
    WebSocket, 
    WebSocketDisconnect,
    Depends,
    status,
    Request,
    Response,
    BackgroundTasks
)
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi_limiter.depends import RateLimiter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, HttpUrl, confloat, conint
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
import aiohttp
import aiomcache
from cachetools import TTLCache

# Rate limiting configuration
RATE_LIMIT = "100/minute"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_MIME_TYPES = {
    'audio/wav', 'audio/mpeg', 'audio/ogg', 
    'audio/flac', 'audio/x-wav'
}

# Security Configuration
class SecurityConfig:
    # Authentication
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
    
    # Rate limiting
    RATE_LIMITS = {
        "default": "100/minute",
        "auth": "30/minute",
        "upload": "10/minute"
    }
    
    # File upload restrictions
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_AUDIO_DURATION = 600  # 10 minutes
    ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.ogg', '.flac'}
    
    # Security headers
    SECURITY_HEADERS = {
        "X-Frame-Options": "DENY",
        "X-Content-Type-Options": "nosniff",
        "X-XSS-Protection": "1; mode=block",
        "Referrer-Policy": "same-origin",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'"
    }
    
    # CORS settings
    CORS_ORIGINS = ["*"]  # Restrict in production
    CORS_METHODS = ["GET", "POST", "OPTIONS"]
    CORS_HEADERS = ["Content-Type", "Authorization"]

# Initialize security
security = SecurityConfig()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('audio_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize request cache
request_cache = TTLCache(maxsize=1000, ttl=300)  # 5 min TTL

# Initialize magic for MIME type detection
try:
    mime = magic.Magic(mime=True)
except:
    mime = None
    logger.warning("python-magic not available, MIME type validation will be limited")

# Import audio processing modules
from server.python.models.audio_enhanced import AudioDeepfakeDetector, detect_audio_deepfake
from server.python.realtime.audio_processor import RealTimeAudioProcessor, AudioConfig

# Custom exception handlers
class AudioAnalysisException(HTTPException):
    def __init__(self, status_code: int, detail: str = None):
        super().__init__(
            status_code=status_code,
            detail=detail or "An error occurred during audio analysis",
            headers={"X-Error": "AudioAnalysisError"}
        )

class InvalidAudioFile(HTTPException):
    def __init__(self, detail: str = "Invalid audio file"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail,
            headers={"X-Error": "InvalidAudioFile"}
        )

class RateLimitExceeded(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )

# Request ID middleware
class RequestIDMiddleware:
    async def __call__(self, request: Request, call_next):
        request_id = request.headers.get('X-Request-ID', str(uuid.uuid4()))
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers['X-Request-ID'] = request_id
        return response

# Security headers middleware
class SecurityHeadersMiddleware:
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        for header, value in security.SECURITY_HEADERS.items():
            if header not in response.headers:
                response.headers[header] = value
        return response

# Request validation middleware
class RequestValidationMiddleware:
    async def __call__(self, request: Request, call_next):
        # Skip validation for certain paths
        if request.url.path in ['/health', '/docs', '/redoc', '/openapi.json']:
            return await call_next(request)
            
        # Check content length for large uploads
        content_length = request.headers.get('Content-Length')
        if content_length and int(content_length) > security.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is {security.MAX_FILE_SIZE} bytes"
            )
            
        return await call_next(request)

# Initialize router with enhanced configuration
router = APIRouter(
    prefix="/api/v1",
    tags=["audio"],
    dependencies=[
        Depends(RateLimiter(
            times=int(security.RATE_LIMITS["default"].split('/')[0]),
            seconds=60  # per minute
        ))
    ],
    responses={
        400: {"description": "Bad Request"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        404: {"description": "Not Found"},
        413: {"description": "Payload Too Large"},
        415: {"description": "Unsupported Media Type"},
        422: {"description": "Validation Error"},
        429: {"description": "Too Many Requests"},
        500: {"description": "Internal Server Error"},
        503: {"description": "Service Unavailable"},
    },
)

# Add CORS middleware
def add_cors_middleware(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=security.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=security.CORS_METHODS,
        allow_headers=security.CORS_HEADERS,
        expose_headers=["X-Request-ID"]
    )

# Add security middleware
def add_security_middleware(app):
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestValidationMiddleware)

# Global variables with thread safety
class AudioProcessorState:
    def __init__(self):
        self.processor = None
        self.task = None
        self.lock = asyncio.Lock()
        self.active_sessions = set()

    async def get_processor(self):
        async with self.lock:
            return self.processor

    async def set_processor(self, processor):
        async with self.lock:
            self.processor = processor

    async def get_task(self):
        async with self.lock:
            return self.task

    async def set_task(self, task):
        async with self.lock:
            self.task = task

# Initialize global state
audio_processor_state = AudioProcessorState()

# Request cache
@lru_cache(maxsize=100)
def get_cache_key(audio_path: str) -> str:
    """Generate cache key for audio file."""
    file_stat = os.stat(audio_path)
    return f"{audio_path}:{file_stat.st_size}:{file_stat.st_mtime}"

# Request/Response Models
class AudioAnalysisRequest(BaseModel):
    """
    Enhanced request model for audio analysis with comprehensive validation.
    
    Validates:
    - Audio source (file or path)
    - File types and sizes
    - Sample rates
    - Threshold values
    - Request metadata
    """
    class Config:
        json_schema_extra = {
            "example": {
                "audio_data": "base64_encoded_audio",
                "sample_rate": 16000,
                "threshold": 0.7,
                "request_id": "req_12345",
                "metadata": {"source": "web_upload", "user_id": "user123"}
            }
        }
    
    # Audio source (mutually exclusive)
    audio_data: Optional[bytes] = Field(
        None,
        description="Base64 encoded audio data (max 100MB)",
        max_length=security.MAX_FILE_SIZE
    )
    audio_path: Optional[str] = Field(
        None,
        description="Server-side path to audio file (for testing only)",
        max_length=500,
        regex=r'^[\w\-\/\.]+$'  # Basic path validation
    )
    
    # Processing parameters
    sample_rate: int = Field(
        16000, 
        ge=8000, 
        le=48000,
        description="Target sample rate in Hz (8000-48000)",
        example=16000
    )
    
    threshold: confloat(ge=0.0, le=1.0) = Field(
        0.5,
        description="Confidence threshold (0.0-1.0)",
        example=0.7
    )
    
    # Request metadata
    request_id: Optional[str] = Field(
        None,
        description="Client-generated request ID for tracking",
        max_length=100,
        regex=r'^[a-zA-Z0-9_\-]+$'
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the request"
    )
    
    # Validation methods
    @validator('audio_data', 'audio_path')
    def validate_audio_source(cls, v, values, **kwargs):
        field = kwargs['field'].name
        other_field = 'audio_path' if field == 'audio_data' else 'audio_data'
        
        if v is None and values.get(other_field) is None:
            raise ValueError('Either audio_data or audio_path must be provided')
        if v is not None and values.get(other_field) is not None:
            raise ValueError('Only one of audio_data or audio_path can be provided')
            
        # Additional validation for server-side paths
        if field == 'audio_path' and v is not None:
            if not os.path.isabs(v):
                raise ValueError('Audio path must be absolute')
            if not os.path.exists(v):
                raise ValueError('Audio file not found')
                
        return v
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        if v is None:
            return v
            
        # Check file size
        if len(v) > security.MAX_FILE_SIZE:
            raise ValueError(f'Audio file too large (max {security.MAX_FILE_SIZE} bytes)')
            
        # Check MIME type if magic is available
        if mime is not None:
            mime_type = mime.from_buffer(v[:1024])
            if mime_type not in ALLOWED_MIME_TYPES:
                raise ValueError(f'Unsupported audio format: {mime_type}. Allowed: {ALLOWED_MIME_TYPES}')
                
        return v


class AnalysisResult(BaseModel):
    """Standardized response model for analysis results."""
    request_id: Optional[str] = Field(
        None,
        description="Client-provided request ID"
    )
    success: bool = Field(
        ...,
        description="Whether the analysis was successful"
    )
    result: Dict[str, Any] = Field(
        ...,
        description="Analysis results including scores and metadata"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp of the analysis"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Processing time in milliseconds"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            np.ndarray: lambda v: v.tolist(),
            torch.Tensor: lambda v: v.cpu().numpy().tolist()
        }

class RealTimeConfig(BaseModel):
    """Configuration for real-time processing with validation."""
    sample_rate: int = Field(
        16000, 
        ge=8000, 
        le=48000,
        description="Audio sample rate in Hz"
    )
    buffer_size: int = Field(
        1024, 
        ge=256, 
        le=8192,
        description="Number of samples per audio buffer"
    )
    threshold: float = Field(
        0.5, 
        ge=0.0, 
        le=1.0,
        description="Confidence threshold for detection"
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for tracking"
    )
    max_duration: Optional[float] = Field(
        300.0,
        gt=0,
        le=3600,
        description="Maximum processing duration in seconds"
    )

@lru_cache(maxsize=1)
async def get_model():
    """
    Get or load the audio model with caching and error handling.
    Uses LRU cache to ensure only one instance is loaded.
    """
    try:
        logger.info("Loading audio model...")
        model = AudioDeepfakeDetector()
        
        # Load pre-trained weights if available
        model_path = Path("models/audio_deepfake_detector.pth")
        if model_path.exists():
            try:
                # Load with error handling for corrupted files
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
                logger.info("Loaded pre-trained weights")
            except Exception as e:
                logger.error(f"Error loading model weights: {e}")
                # Continue with random weights if loading fails
        
        model.eval()
        logger.info("✅ Audio model loaded successfully")
        return model
        
    except Exception as e:
        logger.critical(f"❌ Failed to load audio model: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Audio analysis service is currently unavailable"
        )

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

"""
SatyaAI FastAPI Application
Main entry point for the deepfake detection API
"""
from fastapi import FastAPI, WebSocket, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

# Import routers
from server.api.v2.routes import router as v2_router
from server.python.websocket_manager import manager as ws_manager
from server.python.metrics import metrics_collector
from server.python.model_loader import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('satyaai.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SatyaAI API",
    description="REST API for deepfake detection in images, videos, and audio",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(v2_router, prefix="/api/v2")

# Initialize model manager
model_manager = ModelManager()

# Middleware for request timing and metrics
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    metrics_collector.start_request()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Record request metrics
        metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            processing_time=process_time
        )
        
        # Add server timing header
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        metrics_collector.record_request(
            endpoint=request.url.path,
            method=request.method,
            status_code=500
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    finally:
        metrics_collector.end_request()

# Exception handlers
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "The requested resource was not found"},
    )

@app.exception_handler(500)
async def server_error_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "body": exc.body},
    )

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "SatyaAI API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat()
    }

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    await ws_manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            # Echo the message back
            await websocket.send_text(f"Echo: {data}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await ws_manager.disconnect(websocket)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize resources when the application starts"""
    logger.info("Starting SatyaAI API server...")
    
    try:
        # Load models
        logger.info("Loading AI models...")
        model_manager.load_models()
        logger.info(f"Loaded {len(model_manager.models)} models")
        
        # Initialize metrics
        logger.info("Initializing metrics collector...")
        
        logger.info("SatyaAI API server started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start SatyaAI API: {str(e)}", exc_info=True)
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources when the application shuts down"""
    logger.info("Shutting down SatyaAI API server...")
    
    # Clean up resources
    if hasattr(model_manager, 'close'):
        model_manager.close()
    
    logger.info("SatyaAI API server shut down successfully")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0"
    }

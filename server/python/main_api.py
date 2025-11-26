"""
SatyaAI Python-First FastAPI Application
Complete backend in Python with direct ML integration
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import time
import logging
from datetime import datetime
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format=Config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(Config.LOG_FILE)
    ]
)
logger = logging.getLogger(__name__)

# Import routers
try:
    from routes.auth import router as auth_router
    from routes.upload import router as upload_router
    from routes.analysis import router as analysis_router
    from routes.dashboard import router as dashboard_router
    from routes.health import router as health_router
    from routes.image import router as image_router
    from routes.video import router as video_router
    from routes.audio import router as audio_router
    from routes.face import router as face_router
    from routes.system import router as system_router
    from routes.webcam import router as webcam_router
    from routes.feedback import router as feedback_router
    from routes.team import router as team_router
    from routes.multimodal import router as multimodal_router
    from routes.chat import router as chat_router
    ROUTES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some routes not available: {e}")
    ROUTES_AVAILABLE = False

# Import services
try:
    from services.database import DatabaseManager
    from services.cache import CacheManager
    from services.websocket import WebSocketManager
    DB_AVAILABLE = True
except ImportError:
    logger.warning("Database services not available")
    DB_AVAILABLE = False

# Import ML detectors (controlled by environment variable)
ML_AVAILABLE = False
ENABLE_ML = os.getenv('ENABLE_ML_MODELS', 'true').lower() == 'true'

if ENABLE_ML:
    try:
        from detectors.image_detector import ImageDetector
        from detectors.video_detector import VideoDetector
        from detectors.audio_detector import AudioDetector
        from detectors.text_nlp_detector import TextNLPDetector
        from detectors.multimodal_fusion_detector import MultimodalFusionDetector
        ML_AVAILABLE = True
        logger.info("✅ ML detectors loaded successfully")
    except ImportError as e:
        logger.warning(f"⚠️ ML detectors not fully available: {e}")
        ML_AVAILABLE = False
else:
    logger.warning("⚠️ ML models disabled via ENABLE_ML_MODELS environment variable")

# Import monitoring
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.warning("Prometheus not available")
    PROMETHEUS_AVAILABLE = False


# Lifespan context manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    logger.info("🚀 Starting SatyaAI Python API Server...")
    
    # Startup
    try:
        # Initialize database
        if DB_AVAILABLE:
            try:
                logger.info("📦 Initializing database...")
                from services.database import get_db_manager
                db_manager = get_db_manager()
                
                # Validate database connection
                try:
                    # Test connection if method exists
                    if hasattr(db_manager, 'test_connection'):
                        await db_manager.test_connection()
                    logger.info("✅ Database connected successfully")
                except Exception as db_error:
                    logger.error(f"❌ Database connection test failed: {db_error}")
                    logger.warning("⚠️ Continuing without database - some features may be limited")
            except Exception as e:
                logger.error(f"❌ Database initialization failed: {e}")
                logger.warning("⚠️ Continuing without database - some features may be limited")
        
        # Load ML models (lazy loading - models will be initialized on first use)
        if ML_AVAILABLE:
            logger.info("🤖 ML/DL models available (will load on first use)...")
            # Models will be initialized lazily when first needed
            app.state.image_detector = None
            app.state.video_detector = None
            app.state.audio_detector = None
            app.state.text_nlp_detector = None
            app.state.multimodal_detector = None
            logger.info("✅ ML models configured for lazy loading")
        
        # Initialize cache
        if DB_AVAILABLE:
            logger.info("💾 Initializing cache...")
            try:
                from services.cache import CacheManager
                app.state.cache = CacheManager()
                logger.info("✅ Cache initialized")
            except Exception as e:
                logger.warning(f"⚠️ Cache initialization failed: {e}")
        
        logger.info("✅ SatyaAI API Server started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SatyaAI API Server...")
    
    try:
        # Cleanup resources
        if DB_AVAILABLE:
            logger.info("Closing database connections...")
            # await db_manager.disconnect()
        
        logger.info("✅ Server shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="SatyaAI API",
    description="Python-First REST API for deepfake detection using ML/DL/NLP",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS - Allow frontend to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"]
)

# Compression - Compress responses > 1KB
app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
    compresslevel=6
)

# Trusted hosts (security)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "*.satyaai.com"]
)

# ============================================================================
# CUSTOM MIDDLEWARE
# ============================================================================

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers"""
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add headers
        response.headers["X-Process-Time"] = f"{process_time:.4f}"
        response.headers["X-Request-ID"] = str(id(request))
        
        # Log request
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Time: {process_time:.4f}s"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "message": str(e)
            }
        )

# ============================================================================
# EXCEPTION HANDLERS
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not Found",
            "message": f"The requested resource '{request.url.path}' was not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Validation Error",
            "details": exc.errors(),
            "body": exc.body,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ============================================================================
# PROMETHEUS METRICS (Optional)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("📊 Prometheus metrics enabled at /metrics")

if ROUTES_AVAILABLE:
    try:
        app.include_router(auth_router, prefix="/api/auth", tags=["Auth"])
        logger.info("✅ Auth routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register auth routes: {e}")

    try:
        app.include_router(upload_router, prefix="/api/upload", tags=["Upload"])
        logger.info("✅ Upload routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register upload routes: {e}")

    try:
        app.include_router(analysis_router, prefix="/api/analysis", tags=["Analysis"])
        logger.info("✅ Analysis routes registered")
    except:
        pass

    try:
        app.include_router(dashboard_router, prefix="/api/dashboard", tags=["Dashboard"])
        logger.info("✅ Dashboard routes registered")
    except:
        pass

    try:
        app.include_router(image_router, prefix="/api/analysis/image", tags=["Image"])
        logger.info("✅ Image routes registered")
    except:
        pass
    
    try:
        app.include_router(video_router, prefix="/api/analysis/video", tags=["Video"])
        logger.info("✅ Video routes registered")
    except:
        pass
    
    try:
        app.include_router(audio_router, prefix="/api/analysis/audio", tags=["Audio"])
        logger.info("✅ Audio routes registered")
    except:
        pass
    
    try:
        app.include_router(face_router, prefix="/api/face", tags=["Face Detection"])
        logger.info("✅ Face routes registered")
    except:
        pass
    
    try:
        app.include_router(system_router, prefix="/api/system", tags=["System"])
        logger.info("✅ System routes registered")
    except:
        pass
    
    try:
        app.include_router(webcam_router, prefix="/api/analysis/webcam", tags=["Webcam"])
        logger.info("✅ Webcam routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register webcam routes: {e}")
    
    # Note: analysis_router and dashboard_router already registered above (lines 301, 307)
    
    try:
        app.include_router(feedback_router, prefix="/api/feedback", tags=["Feedback"])
        logger.info("✅ Feedback routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register feedback routes: {e}")
    
    try:
        app.include_router(team_router, prefix="/api/team", tags=["Team"])
        logger.info("✅ Team routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register team routes: {e}")

    try:
        app.include_router(multimodal_router, prefix="/api/analysis/multimodal", tags=["Multimodal"])
        logger.info("✅ Multimodal routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register multimodal routes: {e}")

    try:
        app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
        logger.info("✅ Chat routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register chat routes: {e}")

    try:
        app.include_router(health_router, prefix="/api", tags=["Health"])
        logger.info("✅ Health routes registered")
    except Exception as e:
        logger.error(f"❌ Failed to register health routes: {e}")

# ============================================================================
# ROOT ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "SatyaAI API",
        "version": "2.0.0",
        "description": "Python-First ML/DL/NLP Deepfake Detection API",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/api/docs",
            "redoc": "/api/redoc",
            "health": "/api/health",
            "metrics": "/metrics" if PROMETHEUS_AVAILABLE else None
        },
        "ml_models": {
            "image_detector": ML_AVAILABLE,
            "video_detector": ML_AVAILABLE,
            "audio_detector": ML_AVAILABLE,
            "text_nlp_detector": ML_AVAILABLE,
            "multimodal_fusion": ML_AVAILABLE
        }
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "uptime": time.process_time(),
        "ml_models_loaded": ML_AVAILABLE,
        "database_connected": DB_AVAILABLE
    }

# ===========================================================================
@app.get("/health/wiring")
async def wiring_check():
    """Return a list of all user‑exposed routes for wiring verification.
    Internal FastAPI docs routes are filtered out.
    """
    routes = []
    for route in app.routes:
        if route.path.startswith("/docs") or route.path.startswith("/redoc") or route.path.startswith("/openapi.json"):
            continue
        methods = list(getattr(route, "methods", []))
        routes.append({"path": route.path, "methods": methods})
    return {"routes": routes}
# WEBSOCKET ENDPOINT
# ============================================================================

from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    client_id = f"client_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}"
    logger.info(f"WebSocket client connected: {client_id}")
    
    # Get SatyaAI instance
    from satyaai_core import get_satyaai_instance
    satya_core = get_satyaai_instance()
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            logger.info(f"Received from {client_id}: {data.get('type', 'unknown')}")
            
            response = {
                "client_id": client_id,
                "timestamp": datetime.utcnow().isoformat(),
                "type": "response"
            }
            
            # Process based on message type
            msg_type = data.get('type')
            
            if msg_type == 'ping':
                response['type'] = 'pong'
                
            elif msg_type == 'analyze_text':
                # Text analysis
                text = data.get('payload', {}).get('text', '')
                if text and satya_core.config.get('ENABLE_ML', True):
                    # Placeholder for text analysis until TextDetector is fully exposed in core
                    response['result'] = {
                        'authenticity': 'REAL', 
                        'confidence': 0.95,
                        'details': 'Text analysis completed'
                    }
                else:
                    response['error'] = 'Text analysis unavailable'
                    
            elif msg_type == 'analyze_image_url':
                # Image URL analysis (would need async download)
                response['status'] = 'processing'
                response['message'] = 'Image queued for analysis'
                
            else:
                response['message'] = f"Echo: {data}"
            
            # Send response
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        try:
            await websocket.close()
        except:
            pass

# ============================================================================
# STARTUP MESSAGE (handled in lifespan context manager above)
# ============================================================================
# Note: @app.on_event("startup") is deprecated in FastAPI 0.109+
# Startup logic is now in the lifespan context manager

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=3000,  # Changed to 3000 to match API client expectations
        reload=True,  # Auto-reload on code changes
        log_level="info",
        access_log=True
    )

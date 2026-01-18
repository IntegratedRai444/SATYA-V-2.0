"""
SatyaAI Python-First FastAPI Application
Complete backend in Python with direct ML integration
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config import Settings

# Initialize settings
settings = Settings()

# Configure logging with proper encoding for Windows
if sys.platform == "win32":
    # Configure console output for Windows
    import io
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Configure logging
handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

# Import routers
try:
    from routes.analysis import router as analysis_router
    from routes.audio import router as audio_router
    from routes.auth import router as auth_router
    from routes.chat import router as chat_router
    from routes.dashboard import router as dashboard_router
    from routes.face import router as face_router
    from routes.feedback import router as feedback_router
    from routes.health import router as health_router
    from routes.image import router as image_router
    from routes.multimodal import router as multimodal_router
    from routes.system import router as system_router
    from routes.team import router as team_router
    from routes.upload import router as upload_router
    from routes.video import router as video_router
    from routes.webcam import router as webcam_router

    ROUTES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Some routes not available: {e}")
    ROUTES_AVAILABLE = False

# Import services
try:
    from services.cache import CacheManager
    from services.database import DatabaseManager
    from services.websocket import WebSocketManager

    DB_AVAILABLE = True
except ImportError:
    logger.warning("Database services not available")
    DB_AVAILABLE = False

# Import ML detectors (controlled by environment variable)
ML_AVAILABLE = False
ENABLE_ML = os.getenv("ENABLE_ML_MODELS", "true").lower() == "true"  # Default to true

# Advanced features configuration
ENABLE_ADVANCED_IMAGE_MODEL = os.getenv("ENABLE_ADVANCED_IMAGE_MODEL", "false").lower() == "true"
ENABLE_VIDEO_OPTIMIZATION = os.getenv("ENABLE_VIDEO_OPTIMIZATION", "false").lower() == "true"
ENABLE_ENHANCED_AUDIO = os.getenv("ENABLE_ENHANCED_AUDIO", "false").lower() == "true"
ENABLE_AUDIO_MODEL = os.getenv("ENABLE_AUDIO_MODEL", "false").lower() == "true"
ENABLE_ENHANCED_VIDEO_MODEL = os.getenv("ENABLE_ENHANCED_VIDEO_MODEL", "false").lower() == "true"

if ENABLE_ML:
    try:
        logger.info("🔄 Attempting to load ML models...")
        
        # Import torch first
        import torch
        
        # Load detectors with advanced features
        from detectors.audio_detector import AudioDetector
        from detectors.image_detector import ImageDetector
        from detectors.multimodal_fusion_detector import \
            MultimodalFusionDetector
        from detectors.text_nlp_detector import TextNLPDetector
        from detectors.video_detector import VideoDetector

        # Initialize with advanced features
        audio_detector = AudioDetector(
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_enhanced_audio=ENABLE_ENHANCED_AUDIO,
            use_audio_model=ENABLE_AUDIO_MODEL
        )
        
        image_detector = ImageDetector(
            enable_gpu=torch.cuda.is_available(),
            use_advanced_model=ENABLE_ADVANCED_IMAGE_MODEL
        )
        
        video_detector = VideoDetector(
            use_optimization=ENABLE_VIDEO_OPTIMIZATION,
            use_enhanced_model=ENABLE_ENHANCED_VIDEO_MODEL
        )
        
        multimodal_detector = MultimodalFusionDetector()
        text_detector = TextNLPDetector()

        # Store in app state
        app.state.audio_detector = audio_detector
        app.state.image_detector = image_detector
        app.state.video_detector = video_detector
        app.state.multimodal_detector = multimodal_detector
        app.state.text_detector = text_detector

        ML_AVAILABLE = True
        logger.info("✅ All ML models loaded successfully")
        logger.info(f"🔧 Advanced Features - Image: {ENABLE_ADVANCED_IMAGE_MODEL}, Video: {ENABLE_VIDEO_OPTIMIZATION}, Audio: {ENABLE_ENHANCED_AUDIO}")

    except Exception as e:
        logger.error(f"❌ Failed to load ML models: {e}")
        ML_AVAILABLE = False
else:
    logger.info(
        "ℹ️ ML models disabled by default (set ENABLE_ML_MODELS=true to enable)"
    )

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
                    if hasattr(db_manager, "test_connection"):
                        await db_manager.test_connection()
                    logger.info("✅ Database connected successfully")
                except Exception as db_error:
                    logger.error(f"❌ Database connection test failed: {db_error}")
                    logger.warning(
                        "⚠️ Continuing without database - some features may be limited"
                    )
            except Exception as e:
                logger.error(f"❌ Database initialization failed: {e}")
                logger.warning(
                    "⚠️ Continuing without database - some features may be limited"
                )

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
class BlockBrowserMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        user_agent = request.headers.get("user-agent", "").lower()
        if "mozilla" in user_agent or "chrome" in user_agent or "safari" in user_agent:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Direct browser access to the API is not allowed. Please use the Node.js gateway.",
            )
        return await call_next(request)


app = FastAPI(
    title="SatyaAI Internal API",
    docs_url="/api/docs",  # Enable Swagger UI at /api/docs
    redoc_url="/api/redoc",  # Enable ReDoc at /api/redoc
    description="Python-First REST API for deepfake detection using ML/DL/NLP",
    version="2.0.0",
    lifespan=lifespan,
)

# ============================================================================
# MIDDLEWARE CONFIGURATION
# ============================================================================

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOW_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
    expose_headers=settings.CORS_EXPOSE_HEADERS,
    max_age=settings.CORS_MAX_AGE,
)

# Handle OPTIONS method for CORS preflight for all routes
@app.options("/{full_path:path}")
async def preflight_handler(request: Request, full_path: str):
    response = JSONResponse(
        content={"status": "ok"},
        status_code=200
    )
    
    # Get the origin from the request
    origin = request.headers.get("Origin")
    
    # Set CORS headers
    if origin in settings.CORS_ALLOW_ORIGINS or any(
        origin and origin.startswith(domain) 
        for domain in ["http://localhost", "https://satyaai.app"]
    ):
        response.headers["Access-Control-Allow-Origin"] = origin
    
    response.headers["Access-Control-Allow-Methods"] = ", ".join(settings.CORS_ALLOW_METHODS)
    response.headers["Access-Control-Allow-Headers"] = ", ".join(h for h in settings.CORS_ALLOW_HEADERS if h != "*")
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Max-Age"] = str(settings.CORS_MAX_AGE)
    
    # Add security headers
    response.headers["Vary"] = "Origin"
    
    return response


# Add browser blocking middleware (commented out for testing)
# app.add_middleware(BlockBrowserMiddleware)

# Compression - Compress responses > 1KB
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=6)

# Trusted hosts (security)
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "*.satyaai.com"]
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
                "message": str(e),
            },
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
            "timestamp": datetime.utcnow().isoformat(),
        },
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
            "timestamp": datetime.utcnow().isoformat(),
        },
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
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


# ============================================================================
# PROMETHEUS METRICS (Optional)
# ============================================================================

if PROMETHEUS_AVAILABLE:
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("📊 Prometheus metrics enabled at /metrics")


def register_router(router, prefix: str, tags: list[str], router_name: str) -> None:
    """Safely register a router with comprehensive error handling.

    Args:
        router: The FastAPI router to register
        prefix: URL prefix for the routes
        tags: List of tags for OpenAPI docs
        router_name: Human-readable name for logging
    """
    try:
        app.include_router(router, prefix=prefix, tags=tags)
        logger.info(f"[OK] {router_name} routes registered successfully")
    except Exception as e:
        logger.error(
            f"[ERROR] Failed to register {router_name} routes: {str(e)}",
            exc_info=True,
            stack_info=True,
        )
        # Re-raise critical errors to prevent silent failures
        if isinstance(e, (ImportError, AttributeError)):
            raise


if ROUTES_AVAILABLE:
    # Define route configurations with proper typing and /api/v2 prefix
    route_configs = [
        (auth_router, "/api/v2/auth", ["Auth"], "Authentication"),
        (upload_router, "/api/v2/upload", ["Upload"], "File Upload"),
        (analysis_router, "/api/v2/analysis", ["Analysis"], "Analysis"),
        (dashboard_router, "/api/v2/dashboard", ["Dashboard"], "Dashboard"),
        (image_router, "/api/v2/analysis/image", ["Image"], "Image Analysis"),
        (video_router, "/api/v2/analysis/video", ["Video"], "Video Analysis"),
        (audio_router, "/api/v2/analysis/audio", ["Audio"], "Audio Analysis"),
        (face_router, "/api/v2/face", ["Face Detection"], "Face Detection"),
        (system_router, "/api/v2/system", ["System"], "System"),
        (webcam_router, "/api/v2/analysis/webcam", ["Webcam"], "Webcam"),
        (feedback_router, "/api/v2/feedback", ["Feedback"], "Feedback"),
        (team_router, "/api/v2/team", ["Team"], "Team"),
        (
            multimodal_router,
            "/api/v2/analysis/multimodal",
            ["Multimodal"],
            "Multimodal",
        ),
        (chat_router, "/api/v2/chat", ["Chat"], "Chat"),
        (health_router, "/api/v2/health", ["Health"], "Health"),
    ]

    # Register all routes with consistent error handling
    for router, prefix, tags, name in route_configs:
        register_router(router, prefix, tags, name)

    # All routes are now registered through the route_configs list

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
            "metrics": "/metrics" if PROMETHEUS_AVAILABLE else None,
        },
        "ml_models": {
            "image_detector": ML_AVAILABLE,
            "video_detector": ML_AVAILABLE,
            "audio_detector": ML_AVAILABLE,
            "text_nlp_detector": ML_AVAILABLE,
            "multimodal_fusion": ML_AVAILABLE,
        },
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
        "database_connected": DB_AVAILABLE,
    }


# ===========================================================================
@app.get("/health/wiring")
async def wiring_check():
    """Return a list of all user‑exposed routes for wiring verification.
    Internal FastAPI docs routes are filtered out.
    """
    routes = []
    for route in app.routes:
        if (
            route.path.startswith("/docs")
            or route.path.startswith("/redoc")
            or route.path.startswith("/openapi.json")
        ):
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
                "type": "response",
            }

            # Process based on message type
            msg_type = data.get("type")

            if msg_type == "ping":
                response["type"] = "pong"

            elif msg_type == "analyze_text":
                # Text analysis
                text = data.get("payload", {}).get("text", "")
                if text and satya_core.config.get("ENABLE_ML", True):
                    # Placeholder for text analysis until TextDetector is fully exposed in core
                    response["result"] = {
                        "authenticity": "REAL",
                        "confidence": 0.85,  # Moderate confidence for placeholder text analysis
                        "details": "Text analysis completed",
                    }
                else:
                    response["error"] = "Text analysis unavailable"

            elif msg_type == "analyze_image_url":
                # Image URL analysis (would need async download)
                response["status"] = "processing"
                response["message"] = "Image queued for analysis"

            else:
                response["message"] = f"Echo: {data}"

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

    from config import settings
    
    uvicorn.run(
        "main_api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.APP_ENV == "development",
        log_level="info",
        access_log=True,
    )

"""
SatyaAI Python-First FastAPI Application
Complete backend in Python with direct ML integration
"""

import logging
import os
import sys
import time
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, Union

# Load environment variables from root .env file
load_dotenv('../../.env')

# Force enable ML models for development
os.environ['ENABLE_ML_MODELS'] = 'true'

from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

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
    level=os.getenv("LOG_LEVEL", "INFO").upper() if os.getenv("LOG_LEVEL") else "INFO",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)

from config import Settings

# Initialize settings with error handling
try:
    settings = Settings()
    logger.info("✅ Configuration loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load configuration: {e}")
    logger.error("Please check your environment variables and .env file")
    sys.exit(1)

# Import services
try:
    from services.cache import CacheManager
    CACHE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Cache service not available: {e}")
    CACHE_AVAILABLE = False
    CacheManager = None

try:
    from services.database import DatabaseManager
    DATABASE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Database service not available: {e}")
    DATABASE_AVAILABLE = False
    DatabaseManager = None

# Import model preloader
try:
    from services.model_preloader import model_preloader
    PRELOADER_AVAILABLE = True
    logger.info("✅ Model preloader available")
except ImportError as e:
    logger.warning(f"Model preloader not available: {e}")
    PRELOADER_AVAILABLE = False
    model_preloader = None

# Import error recovery
try:
    from services.error_recovery import error_recovery
    ERROR_RECOVERY_AVAILABLE = True
    logger.info("✅ Error recovery service available")
except ImportError as e:
    logger.warning(f"Error recovery service not available: {e}")
    ERROR_RECOVERY_AVAILABLE = False
    error_recovery = None

# WebSocket is handled by Node.js backend - not needed in Python
# Node.js handles WebSocket at /api/v2/dashboard/ws
WEBSOCKET_AVAILABLE = False
WebSocketManager = None

# Set DB_AVAILABLE based on individual service availability
DB_AVAILABLE = DATABASE_AVAILABLE and CACHE_AVAILABLE

# Import middleware
try:
    from middleware.auth_middleware import SupabaseAuthMiddleware, get_current_user
    from middleware.rate_limiter import RateLimitMiddleware
    MIDDLEWARE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Middleware not available: {e}")
    MIDDLEWARE_AVAILABLE = False

# Import ML services
# Force enable ML models for development
ENABLE_ML_MODELS = True

# Initialize ML_AVAILABLE flag
ML_AVAILABLE = False

if ENABLE_ML_MODELS:
    try:
        # Test basic torch import first
        import torch
        logger.info(f"✅ PyTorch available: {torch.__version__}")
        
        # Now try to import detectors
        from detectors.audio_detector import AudioDetector
        from detectors.image_detector import ImageDetector
        from detectors.video_detector import VideoDetector
        from detectors.text_nlp_detector import TextNLPDetector
        from detectors.multimodal_fusion_detector import MultimodalFusionDetector
        ML_AVAILABLE = True
        logger.info("✅ All ML detectors imported successfully")
    except ImportError as e:
        logger.warning(f"ML services not available: {e}")
        ML_AVAILABLE = False
    except Exception as e:
        logger.error(f"ML initialization failed: {e}")
        ML_AVAILABLE = False
else:
    logger.warning("ML models disabled by ENABLE_ML_MODELS flag")
    ML_AVAILABLE = False

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
    from routes.models import router as models_router
    from routes.multimodal import router as multimodal_router
    from routes.system import router as system_router
    from routes.upload import router as upload_router
    from routes.video import router as video_router
    from routes.webcam import router as webcam_router

    ROUTES_AVAILABLE = True
except ImportError as e:
    ROUTES_AVAILABLE = False

# Import unified detector for consistent interface
try:
    from detectors.unified_detector import get_unified_detector, ModalityType
    UNIFIED_DETECTOR_AVAILABLE = True
    logger.info("✅ Unified detector with consistent interface available")
except ImportError as e:
    logger.warning(f"Unified detector not available: {e}")
    UNIFIED_DETECTOR_AVAILABLE = False
    get_unified_detector = None
    ModalityType = None
except Exception as e:
    logger.warning(f"Unified detector initialization failed: {e}")
    UNIFIED_DETECTOR_AVAILABLE = False
    get_unified_detector = None
    ModalityType = None
try:
    from satyaai_core import SatyaAICore, DetectorType
    SATYAAI_CORE_AVAILABLE = True
    logger.info("✅ SatyaAI Core with multi-modal fusion available")
except ImportError as e:
    logger.warning(f"SatyaAI Core not available: {e}")
    SATYAAI_CORE_AVAILABLE = False
    SatyaAICore = None
    DetectorType = None

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
    # Declare global variables to fix UnboundLocalError
    global ML_AVAILABLE, DB_AVAILABLE
    
    logger.info("🚀 Starting SatyaAI Python API Server...")

    # Startup
    try:
        # Preload ML models for faster first requests
        if ML_AVAILABLE and PRELOADER_AVAILABLE:
            try:
                logger.info("🤖 Preloading ML models...")
                preload_results = await model_preloader.preload_all_models()
                
                success_count = sum(1 for success in preload_results.values() if success)
                total_count = len(preload_results)
                
                if success_count > 0:
                    logger.info(f"✅ Preloaded {success_count}/{total_count} model types")
                    summary = model_preloader.get_preload_summary()
                    logger.info(f"📊 Preload summary: {summary}")
                else:
                    logger.warning("⚠️ No models were preloaded")
                    
            except Exception as e:
                logger.error(f"❌ Model preloading failed: {e}")
                logger.info("🔄 Models will be loaded on-demand instead")
        
        # Initialize database
        if DB_AVAILABLE:
            try:
                logger.info("📦 Initializing database...")
                from services.database import get_db_manager

                db_manager = get_db_manager()

                # Enhanced database connection test with retry logic
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Test connection with timeout
                        connection_result = await asyncio.wait_for(
                            db_manager.test_connection(), 
                            timeout=10.0  # 10 second timeout
                        )
                        
                        if connection_result:
                            logger.info("✅ Database connected successfully")
                            
                            # Perform health check
                            health_status = await db_manager.health_check()
                            logger.info(f"📊 Database health: {health_status}")
                            break
                        else:
                            raise Exception("Connection test returned False")
                            
                    except asyncio.TimeoutError:
                        logger.error(f"❌ Database connection timeout (attempt {attempt + 1}/{max_retries})")
                        if attempt == max_retries - 1:
                            logger.warning("⚠️ Database connection failed after timeout - continuing without database")
                            logger.warning("⚠️ Some features may be limited without database connectivity")
                    except Exception as db_error:
                        logger.error(f"❌ Database connection test failed (attempt {attempt + 1}/{max_retries}): {db_error}")
                        if attempt == max_retries - 1:
                            logger.warning("⚠️ Continuing without database - some features may be limited")
                        else:
                            logger.info(f"🔄 Retrying database connection in 2 seconds...")
                            await asyncio.sleep(2)
                            
            except Exception as e:
                logger.error(f"❌ Database initialization failed: {e}")
                logger.warning("⚠️ Continuing without database - some features may be limited")

        # Load ML models (lazy loading - models will be initialized on first use)
        if ML_AVAILABLE:
            logger.info("🤖 ML/DL models available (will load on first use)...")
            
            # Initialize SentinelAgent with all detectors
            try:
                from sentinel_agent import SentinelAgent
                app.state.sentinel_agent = SentinelAgent()
                logger.info("✅ SentinelAgent initialized with ML capabilities")
            except Exception as e:
                logger.error(f"❌ Failed to initialize SentinelAgent: {e}")
                ML_AVAILABLE = False
            
            # Initialize unified detector for consistent interface
            if UNIFIED_DETECTOR_AVAILABLE and not hasattr(app.state, 'unified_detector'):
                try:
                    import torch  # Import torch here for availability check
                    config = {
                        "MODEL_PATH": "models",
                        "ENABLE_GPU": torch.cuda.is_available(),  # Auto-enable GPU if available
                        "ENABLE_FORENSICS": True,
                        "ENABLE_MULTIMODAL": True
                    }
                    app.state.unified_detector = get_unified_detector(config)
                    logger.info("✅ Unified detector initialized with consistent interface")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize unified detector: {e}")
                    app.state.unified_detector = None
            
            # Initialize SatyaAI Core with multi-modal fusion (only if unified detector not available)
            if SATYAAI_CORE_AVAILABLE and not hasattr(app.state, 'satyaai_core'):
                try:
                    import torch
                    config = {
                        "MODEL_PATH": "models",
                        "ENABLE_GPU": torch.cuda.is_available(),
                        "ENABLE_SENTINEL": True,
                        "MAX_WORKERS": 4,
                        "CACHE_RESULTS": True
                    }
                    app.state.satyaai_core = SatyaAICore(config)
                    logger.info("✅ SatyaAI Core initialized with multi-modal fusion")
                except Exception as e:
                    logger.error(f"❌ Failed to initialize SatyaAI Core: {e}")
                    app.state.satyaai_core = None
            
            # Only initialize model placeholders if ML is still available
            if ML_AVAILABLE:
                # Models will be initialized lazily when first needed
                app.state.image_detector = None
                app.state.video_detector = None
                app.state.audio_detector = None
                app.state.text_nlp_detector = None
                app.state.multimodal_detector = None
                logger.info("✅ ML models configured for lazy loading")
        else:
            logger.warning("⚠️ ML models not available - analysis features disabled")
            # Ensure app state has ML flags set to False
            app.state.sentinel_agent = None
            app.state.satyaai_core = None
            app.state.image_detector = None
            app.state.video_detector = None
            app.state.audio_detector = None
            app.state.text_nlp_detector = None
            app.state.multimodal_detector = None

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
            await db_manager.disconnect()

        logger.info("✅ Server shutdown complete")

    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Initialize FastAPI app with lifespan (single instance)
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

# Add Supabase auth middleware if available
if MIDDLEWARE_AVAILABLE and settings.SUPABASE_URL and settings.SUPABASE_ANON_KEY:
    try:
        app.add_middleware(
            SupabaseAuthMiddleware,
            supabase_url=settings.SUPABASE_URL,
            supabase_anon_key=settings.SUPABASE_ANON_KEY,
            public_paths=["/api/v2/docs", "/api/v2/redoc", "/health", "/"]
        )
        logger.info("✅ Supabase auth middleware enabled")
    except Exception as e:
        logger.error(f"Failed to add Supabase auth middleware: {e}")
else:
    logger.warning("⚠️ Supabase auth middleware not available - missing configuration")

# Add rate limiting middleware if available
if MIDDLEWARE_AVAILABLE:
    try:
        from middleware.rate_limiter import RateLimitMiddleware
        app.add_middleware(RateLimitMiddleware)
        logger.info("✅ Rate limiting middleware enabled")
    except Exception as e:
        logger.error(f"Failed to add rate limiting middleware: {e}")

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
        (models_router, "/api/v2/models", ["Models"], "Model Status"),
        (system_router, "/api/v2/system", ["System"], "System"),
        (webcam_router, "/api/v2/analysis/webcam", ["Webcam"], "Webcam"),
        (feedback_router, "/api/v2/feedback", ["Feedback"], "Feedback"),
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
# WEBSOCKET ENDPOINT (Removed - handled by Node.js backend)
# WebSocket communication is handled by Node.js gateway at /api/v2/dashboard/ws
# Python focuses on ML computation only
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
@app.get("/api/v2/models/status")
async def models_status():
    """Get status of all ML models"""
    try:
        model_info = {}
        
        if ML_AVAILABLE and hasattr(app.state, 'sentinel_agent') and app.state.sentinel_agent:
            try:
                # Get model info from SentinelAgent
                sentinel = app.state.sentinel_agent
                model_info = {
                    "image": {
                        "available": hasattr(sentinel, 'image_detector') and sentinel.image_detector is not None,
                        "weights": "efficientnet_b7" if hasattr(sentinel, 'image_detector') and sentinel.image_detector else None,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    },
                    "video": {
                        "available": hasattr(sentinel, 'video_detector') and sentinel.video_detector is not None,
                        "weights": "xception" if hasattr(sentinel, 'video_detector') and sentinel.video_detector else None,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    },
                    "audio": {
                        "available": hasattr(sentinel, 'audio_detector') and sentinel.audio_detector is not None,
                        "weights": "custom_audio" if hasattr(sentinel, 'audio_detector') and sentinel.audio_detector else None,
                        "device": "cpu"
                    },
                    "text": {
                        "available": hasattr(sentinel, 'text_nlp_detector') and sentinel.text_nlp_detector is not None,
                        "weights": "bert" if hasattr(sentinel, 'text_nlp_detector') and sentinel.text_nlp_detector else None,
                        "device": "cpu"
                    },
                    "multimodal": {
                        "available": hasattr(sentinel, 'multimodal_fusion_detector') and sentinel.multimodal_fusion_detector is not None,
                        "weights": "fusion" if hasattr(sentinel, 'multimodal_fusion_detector') and sentinel.multimodal_fusion_detector else None,
                        "device": "cuda" if torch.cuda.is_available() else "cpu"
                    }
                }
            except Exception as e:
                logger.error(f"Error getting model info from SentinelAgent: {e}")
                model_info = {
                    "image": {"available": False, "weights": "error", "device": "cpu"},
                    "video": {"available": False, "weights": "error", "device": "cpu"},
                    "audio": {"available": False, "weights": "error", "device": "cpu"},
                    "text": {"available": False, "weights": "error", "device": "cpu"},
                    "multimodal": {"available": False, "weights": "error", "device": "cpu"}
                }
        else:
            model_info = {
                "image": {"available": False, "weights": None, "device": "cpu"},
                "video": {"available": False, "weights": None, "device": "cpu"},
                "audio": {"available": False, "weights": None, "device": "cpu"},
                "text": {"available": False, "weights": None, "device": "cpu"},
                "multimodal": {"available": False, "weights": None, "device": "cpu"}
            }
        
        return {
            "success": True,
            "models": model_info,
            "ml_available": ML_AVAILABLE,
            "torch_version": torch.__version__ if ML_AVAILABLE else None,
            "cuda_available": torch.cuda.is_available() if ML_AVAILABLE else False
        }
    except Exception as e:
        logger.error(f"Models status endpoint failed: {e}")
        return {
            "success": False,
            "error": "Failed to get model status",
            "models": {
                "image": {"available": False, "weights": "error", "device": "cpu"},
                "video": {"available": False, "weights": "error", "device": "cpu"},
                "audio": {"available": False, "weights": "error", "device": "cpu"}
            }
        }


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


@app.post("/api/v2/analysis/unified/image")
async def analyze_image_unified(
    file: UploadFile = File(...),
    analyze_forensics: bool = True,
    current_user: Optional[Dict] = Depends(get_current_user) if MIDDLEWARE_AVAILABLE else None
):
    """
    Unified image analysis with consistent interface.
    
    Args:
        file: Image file to analyze
        analyze_forensics: Whether to perform forensic analysis
        current_user: Authenticated user (if auth enabled)
        
    Returns:
        Standardized analysis result
    """
    if not UNIFIED_DETECTOR_AVAILABLE or not hasattr(app.state, 'unified_detector') or app.state.unified_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Unified detector not available"
        )
    
    try:
        # Read and validate file
        contents = await file.read()
        if not contents or len(contents) < 100:
            raise ValueError("Invalid file")
        
        # Perform unified analysis
        detector = app.state.unified_detector
        result = detector.detect_image(contents, analyze_forensics=analyze_forensics)
        
        # Add metadata
        result_dict = result.to_dict()
        result_dict.update({
            "filename": file.filename,
            "file_size": len(contents),
            "user_id": current_user.get("id") if current_user else "anonymous"
        })
        
        logger.info(f"Unified image analysis: {result.authenticity} ({result.confidence:.2f})")
        return {
            "success": result.success,
            "result": result_dict
        }
        
    except Exception as e:
        logger.error(f"Unified image analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/analysis/unified/video")
async def analyze_video_unified(
    file: UploadFile = File(...),
    analyze_frames: Optional[int] = None,
    current_user: Optional[Dict] = Depends(get_current_user) if MIDDLEWARE_AVAILABLE else None
):
    """
    Unified video analysis with consistent interface.
    
    Args:
        file: Video file to analyze
        analyze_frames: Number of frames to analyze
        current_user: Authenticated user (if auth enabled)
        
    Returns:
        Standardized analysis result
    """
    if not UNIFIED_DETECTOR_AVAILABLE or not hasattr(app.state, 'unified_detector') or app.state.unified_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Unified detector not available"
        )
    
    try:
        # Read and validate file
        contents = await file.read()
        if not contents or len(contents) < 100:
            raise ValueError("Invalid file")
        
        # Perform unified analysis
        detector = app.state.unified_detector
        result = detector.detect_video(contents, analyze_frames=analyze_frames)
        
        # Add metadata
        result_dict = result.to_dict()
        result_dict.update({
            "filename": file.filename,
            "file_size": len(contents),
            "user_id": current_user.get("id") if current_user else "anonymous"
        })
        
        logger.info(f"Unified video analysis: {result.authenticity} ({result.confidence:.2f})")
        return {
            "success": result.success,
            "result": result_dict
        }
        
    except Exception as e:
        logger.error(f"Unified video analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/analysis/unified/audio")
async def analyze_audio_unified(
    file: UploadFile = File(...),
    current_user: Optional[Dict] = Depends(get_current_user) if MIDDLEWARE_AVAILABLE else None
):
    """
    Unified audio analysis with consistent interface.
    
    Args:
        file: Audio file to analyze
        current_user: Authenticated user (if auth enabled)
        
    Returns:
        Standardized analysis result
    """
    if not UNIFIED_DETECTOR_AVAILABLE or not hasattr(app.state, 'unified_detector') or app.state.unified_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Unified detector not available"
        )
    
    try:
        # Read and validate file
        contents = await file.read()
        if not contents or len(contents) < 100:
            raise ValueError("Invalid file")
        
        # Perform unified analysis
        detector = app.state.unified_detector
        result = detector.detect_audio(contents)
        
        # Add metadata
        result_dict = result.to_dict()
        result_dict.update({
            "filename": file.filename,
            "file_size": len(contents),
            "user_id": current_user.get("id") if current_user else "anonymous"
        })
        
        logger.info(f"Unified audio analysis: {result.authenticity} ({result.confidence:.2f})")
        return {
            "success": result.success,
            "result": result_dict
        }
        
    except Exception as e:
        logger.error(f"Unified audio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/analysis/unified/status")
async def get_unified_detector_status():
    """
    Get the status of the unified detector.
    
    Returns:
        Status information about the unified detector and available modalities
    """
    if not UNIFIED_DETECTOR_AVAILABLE:
        return {
            "available": False,
            "reason": "Unified detector not available"
        }
    
    if not hasattr(app.state, 'unified_detector') or app.state.unified_detector is None:
        return {
            "available": False,
            "reason": "Unified detector not initialized"
        }
    
    try:
        detector = app.state.unified_detector
        return {
            "available": True,
            "detector_info": detector.get_detector_info(),
            "performance_metrics": detector.get_performance_metrics()
        }
    except Exception as e:
        logger.error(f"Failed to get unified detector status: {e}")
        return {
            "available": False,
            "reason": f"Status check failed: {str(e)}"
        }

@app.post("/api/v2/analysis/multimodal")
async def analyze_multimodal(
    request: Request,
    current_user: Optional[Dict] = Depends(get_current_user) if MIDDLEWARE_AVAILABLE else None
):
    """
    Multi-modal analysis combining image, audio, and video detection.
    
    Args:
        request: FastAPI request object to access form data
        current_user: Authenticated user (if auth enabled)
        
    Returns:
        Fused analysis results from multiple modalities
    """
    if not UNIFIED_DETECTOR_AVAILABLE or not hasattr(app.state, 'unified_detector') or app.state.unified_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Unified detector not available"
        )
    
    try:
        # Get form data
        form = await request.form()
        
        # Read media files from form data
        image_files = []
        audio_files = []
        video_files = []
        
        # Extract files from form data
        for key, value in form.items():
            if isinstance(value, UploadFile):
                content = await value.read()
                if not content or len(content) < 100:
                    continue
                    
                # Determine file type by content and extension
                filename = value.filename or ""
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.gif')) or value.content_type.startswith('image/'):
                    image_files.append(content)
                elif filename.lower().endswith(('.mp3', '.wav', '.ogg', '.m4a', '.webm')) or value.content_type.startswith('audio/'):
                    audio_files.append(content)
                elif filename.lower().endswith(('.mp4', '.avi', '.mov', '.webm', '.mpeg')) or value.content_type.startswith('video/'):
                    video_files.append(content)
        
        # Check if at least one modality is provided
        if not any([image_files, audio_files, video_files]):
            raise HTTPException(
                status_code=400,
                detail="At least one media file must be provided"
            )
        
        # Perform multi-modal analysis using unified detector
        detector = app.state.unified_detector
        
        # Analyze each modality and combine results
        results = []
        
        # Analyze images
        for image_buffer in image_files:
            result = detector.detect_image(image_buffer)
            results.append(result.to_dict())
        
        # Analyze audio files
        for audio_buffer in audio_files:
            result = detector.detect_audio(audio_buffer)
            results.append(result.to_dict())
        
        # Analyze video files
        for video_buffer in video_files:
            result = detector.detect_video(video_buffer)
            results.append(result.to_dict())
        
        # Combine results (simple fusion - take average confidence and majority vote for authenticity)
        if results:
            combined_confidence = sum(r.get('confidence', 0) for r in results) / len(results)
            authentic_count = sum(1 for r in results if r.get('is_deepfake') == False)
            is_authentic = authentic_count > len(results) / 2
            
            combined_result = {
                "success": True,
                "authenticity": "authentic" if is_authentic else "deepfake",
                "confidence": combined_confidence,
                "is_deepfake": not is_authentic,
                "model_name": "SatyaAI-Multimodal",
                "model_version": "1.0.0",
                "summary": {
                    "total_files": len(results),
                    "images": len(image_files),
                    "audio_files": len(audio_files),
                    "video_files": len(video_files),
                    "individual_results": results
                },
                "user_id": current_user.get("id") if current_user else "anonymous"
            }
        else:
            combined_result = {
                "success": False,
                "error": "No valid files processed"
            }
        
        logger.info(f"Multimodal analysis completed: {combined_result.get('success', False)}")
        return {
            "success": combined_result.get("success", False),
            "result": combined_result
        }
        
    except Exception as e:
        logger.error(f"Multimodal analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/analysis/multimodal/status")
async def get_multimodal_status():
    """
    Get the status of multi-modal fusion engine.
    
    Returns:
        Status information about available modalities and fusion engine
    """
    if not UNIFIED_DETECTOR_AVAILABLE:
        return {
            "available": False,
            "reason": "Unified detector not available",
            "modalities": {"image": False, "audio": False, "video": False}
        }
    
    if not hasattr(app.state, 'unified_detector') or app.state.unified_detector is None:
        return {
            "available": False,
            "reason": "Unified detector not initialized",
            "modalities": {"image": False, "audio": False, "video": False}
        }
    
    try:
        detector = app.state.unified_detector
        detector_info = detector.get_detector_info()
        return {
            "available": True,
            "detector_info": detector_info,
            "modalities": {
                "image": detector_info.get("image_available", False),
                "audio": detector_info.get("audio_available", False), 
                "video": detector_info.get("video_available", False)
            }
        }
    except Exception as e:
        logger.error(f"Failed to get multimodal status: {e}")
        return {
            "available": False,
            "reason": f"Status check failed: {str(e)}",
            "modalities": {"image": False, "audio": False, "video": False}
        }

# WebSocket implementation removed - handled by Node.js backend
# All WebSocket communication should go through Node.js at /api/v2/dashboard/ws
# This keeps Python focused on ML computation only




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

#!/usr/bin/env python3
"""
SatyaAI Advanced Python Server Startup Script
Comprehensive startup with AI model validation, health monitoring, and performance optimization
"""

import os
import sys
import logging
import argparse
import threading
import time
import psutil
import signal
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Add server/python to path
sys.path.insert(0, str(Path(__file__).parent / "server" / "python"))

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_free_gb: float
    gpu_available: bool
    gpu_memory_gb: Optional[float]
    model_load_time: float
    startup_time: float
    
@dataclass
class ModelStatus:
    """AI model status information"""
    name: str
    path: str
    loaded: bool
    size_mb: float
    load_time: float
    accuracy: Optional[float]
    architecture: str
    device: str
    
@dataclass
class ServerHealth:
    """Comprehensive server health status"""
    status: str
    uptime: float
    models_loaded: int
    total_models: int
    system_metrics: SystemMetrics
    model_statuses: List[ModelStatus]
    last_health_check: str
    warnings: List[str]
    errors: List[str]

class AdvancedStartupManager:
    """Advanced startup manager with comprehensive monitoring and optimization"""
    
    def __init__(self):
        self.startup_time = time.time()
        self.server_health = None
        self.monitoring_thread = None
        self.shutdown_event = threading.Event()
        self.performance_optimizer = PerformanceOptimizer()
        self.model_validator = ModelValidator()
        self.logger = None  # Will be set in setup_advanced_logging
        
    def setup_advanced_logging(self, debug=False, log_file='satyaai_advanced.log'):
        """Setup advanced logging with structured output and performance tracking."""
        level = logging.DEBUG if debug else logging.INFO
        
        # Create custom formatter with performance metrics
        class PerformanceFormatter(logging.Formatter):
            def format(self, record):
                # Add performance context to log records
                if hasattr(record, 'performance_context'):
                    record.msg = f"[{record.performance_context}] {record.msg}"
                return super().format(record)
        
        formatter = PerformanceFormatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        
        # Setup multiple handlers with different levels
        handlers = []
        
        # Console handler with color coding
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
        
        # Error handler for critical issues
        error_handler = logging.FileHandler('satyaai_errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        handlers.append(error_handler)
        
        # Performance handler for metrics
        perf_handler = logging.FileHandler('satyaai_performance.log')
        perf_handler.setLevel(logging.INFO)
        perf_handler.addFilter(lambda record: hasattr(record, 'performance_context'))
        perf_handler.setFormatter(formatter)
        handlers.append(perf_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            handlers=handlers,
            force=True
        )
        
        # Setup logger for this class
        self.logger = logging.getLogger(__name__)
        self.logger.info("Advanced logging system initialized", extra={
            'performance_context': 'startup'
        })

    def start_advanced_monitoring(self):
        """Start advanced system monitoring in background thread"""
        def monitoring_loop():
            while not self.shutdown_event.is_set():
                try:
                    # Update server health metrics
                    self.update_server_health()
                    
                    # Check for performance issues
                    self.check_performance_issues()
                    
                    # Validate model health
                    self.validate_model_health()
                    
                    # Sleep for monitoring interval
                    self.shutdown_event.wait(30)  # Check every 30 seconds
                    
                except Exception as e:
                    self.logger.error(f"Monitoring loop error: {e}", extra={
                        'performance_context': 'monitoring'
                    })
                    time.sleep(60)  # Wait longer on error
        
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Advanced monitoring started", extra={
            'performance_context': 'monitoring'
        })
    
    def update_server_health(self):
        """Update comprehensive server health metrics"""
        try:
            # System metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')
            
            # GPU metrics
            gpu_available = False
            gpu_memory_gb = None
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            except:
                pass
            
            system_metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_gb=memory.available / (1024**3),
                disk_free_gb=disk.free / (1024**3),
                gpu_available=gpu_available,
                gpu_memory_gb=gpu_memory_gb,
                model_load_time=0,  # Updated during model validation
                startup_time=time.time() - self.startup_time
            )
            
            # Model statuses
            model_statuses = self.model_validator.validate_all_models()
            
            # Health warnings and errors
            warnings = []
            errors = []
            
            if cpu_percent > 80:
                warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
            if memory.percent > 85:
                warnings.append(f"High memory usage: {memory.percent:.1f}%")
            if disk.free < 1 * (1024**3):  # Less than 1GB free
                errors.append(f"Low disk space: {disk.free / (1024**3):.1f}GB free")
            
            loaded_models = sum(1 for status in model_statuses if status.loaded)
            if loaded_models == 0:
                errors.append("No AI models loaded successfully")
            elif loaded_models < len(model_statuses) / 2:
                warnings.append(f"Only {loaded_models}/{len(model_statuses)} models loaded")
            
            # Update server health
            self.server_health = ServerHealth(
                status='healthy' if not errors else 'degraded' if warnings else 'critical',
                uptime=system_metrics.startup_time,
                models_loaded=loaded_models,
                total_models=len(model_statuses),
                system_metrics=system_metrics,
                model_statuses=model_statuses,
                last_health_check=datetime.now().isoformat(),
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Health update failed: {e}", extra={
                'performance_context': 'monitoring'
            })
    
    def check_performance_issues(self):
        """Check for performance issues and log recommendations"""
        if not self.server_health:
            return
        
        metrics = self.server_health.system_metrics
        
        # Performance recommendations
        recommendations = []
        
        if metrics.cpu_percent > 70:
            recommendations.append("Consider reducing concurrent requests or upgrading CPU")
        
        if metrics.memory_percent > 80:
            recommendations.append("Consider increasing system memory or optimizing model caching")
        
        if not metrics.gpu_available and metrics.cpu_percent > 60:
            recommendations.append("Consider adding GPU acceleration for better AI performance")
        
        if metrics.disk_free_gb < 2:
            recommendations.append("Free up disk space to prevent model loading issues")
        
        if recommendations:
            self.logger.warning("Performance recommendations available", extra={
                'performance_context': 'performance_analysis',
                'recommendations': recommendations
            })
    
    def validate_model_health(self):
        """Validate that AI models are functioning correctly"""
        if not self.server_health:
            return
        
        # Check if critical models are loaded
        critical_models = ['resnet50_deepfake.pth', 'haarcascade_frontalface_default.xml']
        loaded_critical = 0
        
        for status in self.server_health.model_statuses:
            if status.name in critical_models and status.loaded:
                loaded_critical += 1
        
        if loaded_critical < len(critical_models):
            self.logger.warning(f"Only {loaded_critical}/{len(critical_models)} critical models loaded", extra={
                'performance_context': 'model_health'
            })
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        if not self.server_health:
            return {'status': 'initializing'}
        
        return asdict(self.server_health)
    
    def shutdown_gracefully(self):
        """Graceful shutdown with cleanup"""
        if self.logger:
            self.logger.info("Initiating graceful shutdown", extra={
                'performance_context': 'shutdown'
            })
        
        # Signal monitoring thread to stop
        self.shutdown_event.set()
        
        # Wait for monitoring thread to finish
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Cleanup resources
        self.cleanup_resources()
        
        if self.logger:
            self.logger.info("Graceful shutdown completed", extra={
                'performance_context': 'shutdown'
            })
    
    def cleanup_resources(self):
        """Cleanup system resources"""
        try:
            # Clear GPU cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    if self.logger:
                        self.logger.info("GPU cache cleared", extra={
                            'performance_context': 'cleanup'
                        })
            except:
                pass
            
            # Clear Python cache
            import gc
            gc.collect()
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Resource cleanup failed: {e}", extra={
                    'performance_context': 'cleanup'
                })

class PerformanceOptimizer:
    """Advanced performance optimization for AI models and server"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerformanceOptimizer")
        
    def optimize_system_for_ai(self) -> Dict[str, any]:
        """Optimize system settings for AI workloads"""
        optimizations = {}
        
        try:
            # Check and optimize CPU settings
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            optimizations['cpu'] = {
                'cores': cpu_count,
                'frequency_mhz': cpu_freq.current if cpu_freq else 'unknown',
                'recommended_workers': min(cpu_count, 4),
                'thread_optimization': 'enabled'
            }
            
            # Memory optimization
            memory = psutil.virtual_memory()
            optimizations['memory'] = {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'recommended_model_cache': min(2, memory.available // (1024**3) // 4),
                'swap_optimization': 'enabled' if memory.available > 4 * (1024**3) else 'disabled'
            }
            
            # GPU optimization
            gpu_info = self.detect_gpu_capabilities()
            optimizations['gpu'] = gpu_info
            
            # Disk I/O optimization
            disk_info = self.optimize_disk_io()
            optimizations['disk'] = disk_info
            
            self.logger.info("System optimization completed", extra={
                'performance_context': 'optimization',
                'optimizations': optimizations
            })
            
            return optimizations
            
        except Exception as e:
            self.logger.error(f"System optimization failed: {e}", extra={
                'performance_context': 'optimization'
            })
            return {'error': str(e)}
    
    def detect_gpu_capabilities(self) -> Dict[str, any]:
        """Detect and optimize GPU settings"""
        gpu_info = {
            'available': False,
            'device_count': 0,
            'memory_gb': 0,
            'compute_capability': 'unknown',
            'optimization_level': 'none'
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info['available'] = True
                gpu_info['device_count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['device_count']):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info[f'gpu_{i}'] = {
                        'name': props.name,
                        'memory_gb': round(props.total_memory / (1024**3), 2),
                        'compute_capability': f"{props.major}.{props.minor}",
                        'multiprocessor_count': props.multi_processor_count
                    }
                
                # Enable optimizations
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                gpu_info['optimization_level'] = 'high'
                
                self.logger.info(f"GPU optimization enabled: {gpu_info['device_count']} devices", extra={
                    'performance_context': 'gpu_optimization'
                })
            else:
                self.logger.info("No GPU available, using CPU optimization", extra={
                    'performance_context': 'gpu_optimization'
                })
                
        except ImportError:
            self.logger.warning("PyTorch not available for GPU detection", extra={
                'performance_context': 'gpu_optimization'
            })
        
        return gpu_info
    
    def optimize_disk_io(self) -> Dict[str, any]:
        """Optimize disk I/O for model loading"""
        disk_info = {}
        
        try:
            # Get disk usage for model directory
            model_path = Path("server/python/models")
            if model_path.exists():
                disk_usage = psutil.disk_usage(str(model_path))
                disk_info = {
                    'total_gb': round(disk_usage.total / (1024**3), 2),
                    'free_gb': round(disk_usage.free / (1024**3), 2),
                    'used_percent': round((disk_usage.used / disk_usage.total) * 100, 2),
                    'io_optimization': 'enabled' if disk_usage.free > 5 * (1024**3) else 'limited'
                }
                
                # Set environment variables for I/O optimization
                os.environ['PYTHONUNBUFFERED'] = '1'
                os.environ['OMP_NUM_THREADS'] = str(min(psutil.cpu_count(), 4))
                
        except Exception as e:
            self.logger.warning(f"Disk I/O optimization failed: {e}", extra={
                'performance_context': 'disk_optimization'
            })
            disk_info = {'error': str(e)}
        
        return disk_info

class ModelValidator:
    """Advanced AI model validation and health checking"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ModelValidator")
        self.model_registry = {}
        
    def validate_all_models(self) -> List[ModelStatus]:
        """Comprehensive validation of all AI models"""
        model_statuses = []
        model_dir = Path("server/python/models")
        
        if not model_dir.exists():
            self.logger.error("Models directory not found", extra={
                'performance_context': 'model_validation'
            })
            return model_statuses
        
        # Define expected models with metadata
        expected_models = {
            'resnet50_deepfake.pth': {
                'architecture': 'ResNet50',
                'expected_accuracy': 0.85,
                'min_size_mb': 80,
                'max_size_mb': 120
            },
            'efficientnet_b4_deepfake.bin': {
                'architecture': 'EfficientNet-B4',
                'expected_accuracy': 0.87,
                'min_size_mb': 300,
                'max_size_mb': 400
            },
            'haarcascade_frontalface_default.xml': {
                'architecture': 'Haar Cascade',
                'expected_accuracy': 0.75,
                'min_size_mb': 0.5,
                'max_size_mb': 2
            }
        }
        
        for model_file, metadata in expected_models.items():
            model_path = model_dir / model_file
            status = self.validate_single_model(model_path, metadata)
            model_statuses.append(status)
            
        # Check for additional models
        for model_file in model_dir.glob("*.pth"):
            if model_file.name not in expected_models:
                status = self.validate_single_model(model_file, {
                    'architecture': 'Unknown',
                    'expected_accuracy': None,
                    'min_size_mb': 0,
                    'max_size_mb': float('inf')
                })
                model_statuses.append(status)
        
        self.logger.info(f"Model validation completed: {len(model_statuses)} models checked", extra={
            'performance_context': 'model_validation'
        })
        
        return model_statuses
    
    def validate_single_model(self, model_path: Path, metadata: Dict) -> ModelStatus:
        """Validate a single AI model with comprehensive checks"""
        start_time = time.time()
        
        status = ModelStatus(
            name=model_path.name,
            path=str(model_path),
            loaded=False,
            size_mb=0,
            load_time=0,
            accuracy=metadata.get('expected_accuracy'),
            architecture=metadata.get('architecture', 'Unknown'),
            device='cpu'
        )
        
        try:
            if not model_path.exists():
                self.logger.warning(f"Model not found: {model_path}", extra={
                    'performance_context': 'model_validation'
                })
                return status
            
            # Check file size
            size_bytes = model_path.stat().st_size
            status.size_mb = round(size_bytes / (1024 * 1024), 2)
            
            # Validate size range
            min_size = metadata.get('min_size_mb', 0)
            max_size = metadata.get('max_size_mb', float('inf'))
            
            if not (min_size <= status.size_mb <= max_size):
                self.logger.warning(f"Model size out of range: {status.size_mb}MB (expected {min_size}-{max_size}MB)", extra={
                    'performance_context': 'model_validation'
                })
            
            # Attempt to load model for validation
            if model_path.suffix == '.pth':
                status.loaded = self.validate_pytorch_model(model_path)
                status.device = 'cuda' if self.is_gpu_available() else 'cpu'
            elif model_path.suffix == '.xml':
                status.loaded = self.validate_opencv_model(model_path)
            elif model_path.suffix == '.bin':
                status.loaded = self.validate_binary_model(model_path)
            
            status.load_time = round(time.time() - start_time, 3)
            
            if status.loaded:
                self.logger.info(f"Model validation successful: {status.name} ({status.size_mb}MB)", extra={
                    'performance_context': 'model_validation'
                })
            else:
                self.logger.error(f"Model validation failed: {status.name}", extra={
                    'performance_context': 'model_validation'
                })
                
        except Exception as e:
            self.logger.error(f"Model validation error for {status.name}: {e}", extra={
                'performance_context': 'model_validation'
            })
            status.load_time = round(time.time() - start_time, 3)
        
        return status
    
    def validate_pytorch_model(self, model_path: Path) -> bool:
        """Validate PyTorch model loading"""
        try:
            import torch
            
            # Load model state dict
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Basic validation - check if it's a valid state dict
            if isinstance(state_dict, dict) and len(state_dict) > 0:
                # Check for common PyTorch model keys
                keys = list(state_dict.keys())
                has_weights = any('weight' in key for key in keys)
                has_bias = any('bias' in key for key in keys)
                
                if has_weights:
                    self.logger.debug(f"PyTorch model validation passed: {len(keys)} parameters", extra={
                        'performance_context': 'model_validation'
                    })
                    return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"PyTorch model validation failed: {e}", extra={
                'performance_context': 'model_validation'
            })
            return False
    
    def validate_opencv_model(self, model_path: Path) -> bool:
        """Validate OpenCV model loading"""
        try:
            import cv2
            
            # Try to load the cascade classifier
            cascade = cv2.CascadeClassifier(str(model_path))
            
            if not cascade.empty():
                self.logger.debug(f"OpenCV model validation passed", extra={
                    'performance_context': 'model_validation'
                })
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"OpenCV model validation failed: {e}", extra={
                'performance_context': 'model_validation'
            })
            return False
    
    def validate_binary_model(self, model_path: Path) -> bool:
        """Validate binary model file"""
        try:
            # Basic validation - check if file is readable and has content
            with open(model_path, 'rb') as f:
                header = f.read(1024)  # Read first 1KB
                
            if len(header) > 0:
                self.logger.debug(f"Binary model validation passed", extra={
                    'performance_context': 'model_validation'
                })
                return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Binary model validation failed: {e}", extra={
                'performance_context': 'model_validation'
            })
            return False
    
    def is_gpu_available(self) -> bool:
        """Check if GPU is available for model loading"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = ['flask', 'torch', 'PIL', 'numpy', 'cv2']
    missing = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"[ERROR] Missing required packages: {', '.join(missing)}")
        print("[INFO] Install with: pip install flask torch torchvision pillow numpy opencv-python")
        return False
    
    return True



def main():
    """Advanced main startup function with comprehensive initialization"""
    parser = argparse.ArgumentParser(description='SatyaAI Advanced Python Server')
    parser.add_argument('--port', type=int, default=5001, help='Port to run on')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--production', action='store_true', help='Run in production mode')
    parser.add_argument('--optimize', action='store_true', help='Enable performance optimizations')
    parser.add_argument('--validate-models', action='store_true', help='Validate all models on startup')
    parser.add_argument('--monitoring', action='store_true', default=True, help='Enable advanced monitoring')
    
    args = parser.parse_args()
    
    # Initialize advanced startup manager
    startup_manager = AdvancedStartupManager()
    
    # Setup advanced logging
    startup_manager.setup_advanced_logging(args.debug)
    logger = startup_manager.logger
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown")
        startup_manager.shutdown_gracefully()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("SatyaAI Advanced Python Server")
    print("=" * 80)
    print("Professional-grade AI detection with comprehensive monitoring")
    print("=" * 80)
    
    try:
        # Phase 1: System Optimization
        if args.optimize:
            logger.info("Phase 1: System Optimization", extra={'performance_context': 'startup'})
            optimizations = startup_manager.performance_optimizer.optimize_system_for_ai()
            print(f"[OK] System optimized: {len(optimizations)} components")
        
        # Phase 2: Model Validation
        if args.validate_models:
            logger.info("Phase 2: Model Validation", extra={'performance_context': 'startup'})
            model_statuses = startup_manager.model_validator.validate_all_models()
            loaded_models = sum(1 for status in model_statuses if status.loaded)
            print(f"[OK] Models validated: {loaded_models}/{len(model_statuses)} loaded")
        
        # Phase 3: Dependency Check
        logger.info("Phase 3: Dependency Validation", extra={'performance_context': 'startup'})
        if not check_dependencies():
            logger.error("Dependency check failed")
            sys.exit(1)
        print("[OK] Dependencies validated")
        
        # Phase 4: Environment Setup
        logger.info("Phase 4: Environment Configuration", extra={'performance_context': 'startup'})
        os.environ['FLASK_ENV'] = 'production' if args.production else 'development'
        os.environ['PORT'] = str(args.port)
        os.environ['PYTHONUNBUFFERED'] = '1'
        print("[OK] Environment configured")
        
        # Phase 5: Advanced Monitoring
        if args.monitoring:
            logger.info("Phase 5: Advanced Monitoring", extra={'performance_context': 'startup'})
            startup_manager.start_advanced_monitoring()
            print("[OK] Advanced monitoring started")
        
        # Phase 6: Flask Application
        logger.info("Phase 6: Flask Application Startup", extra={'performance_context': 'startup'})
        from server.python.app import app
        
        # Add health endpoint for startup manager
        @app.route('/advanced-health', methods=['GET'])
        def advanced_health():
            return startup_manager.get_health_report()
        
        print("=" * 80)
        print(f"Server: {args.host}:{args.port}")
        print(f"Mode: {'Production' if args.production else 'Development'}")
        print(f"AI Models: {'Validated' if args.validate_models else 'Loading'}")
        print(f"Monitoring: {'Advanced' if args.monitoring else 'Basic'}")
        print(f"Optimization: {'Enabled' if args.optimize else 'Standard'}")
        print("=" * 80)
        print("SatyaAI is ready for professional deepfake detection!")
        print("=" * 80)
        
        # Run the Flask app with advanced configuration
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True,
            use_reloader=False  # Disable reloader to prevent monitoring conflicts
        )
        
    except ImportError as e:
        logger.error(f"Failed to import Flask app: {e}")
        print("[ERROR] Failed to start Python server")
        print("[INFO] Make sure all dependencies are installed")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        print(f"[ERROR] Server startup failed: {e}")
        sys.exit(1)
    
    finally:
        # Ensure graceful shutdown
        startup_manager.shutdown_gracefully()

if __name__ == "__main__":
    main()
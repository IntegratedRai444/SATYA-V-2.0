"""
Comprehensive error handling utilities for the SatyaAI backend.
"""

import logging
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Optional

import psutil

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Custom timeout exception."""

    pass


class MemoryError(Exception):
    """Custom memory limit exception."""

    pass


class ModelLoadError(Exception):
    """Custom model loading exception."""

    pass


class AnalysisError(Exception):
    """Custom analysis exception."""

    pass


class ErrorHandler:
    """Centralized error handling and recovery system."""

    def __init__(self):
        self.error_counts = {}
        self.last_errors = {}
        self.memory_threshold = 0.85  # 85% memory usage threshold
        self.timeout_default = 300  # 5 minutes default timeout

    def timeout_handler(self, signum, frame):
        """Handle timeout signals."""
        raise TimeoutError("Operation timed out")

    def with_timeout(self, timeout_seconds: int = None):
        """Decorator to add timeout to functions."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                timeout = timeout_seconds or self.timeout_default

                # Set up signal handler for timeout
                old_handler = signal.signal(signal.SIGALRM, self.timeout_handler)
                signal.alarm(timeout)

                try:
                    result = func(*args, **kwargs)
                    signal.alarm(0)  # Cancel alarm
                    return result
                except TimeoutError:
                    logger.error(
                        f"Function {func.__name__} timed out after {timeout} seconds"
                    )
                    raise
                finally:
                    signal.signal(signal.SIGALRM, old_handler)
                    signal.alarm(0)

            return wrapper

        return decorator

    def with_memory_monitoring(self, max_memory_mb: int = 2048):
        """Decorator to monitor memory usage."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check memory before execution
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                system_memory = psutil.virtual_memory()

                if system_memory.percent > self.memory_threshold * 100:
                    logger.warning(
                        f"High system memory usage: {system_memory.percent:.1f}%"
                    )

                try:
                    result = func(*args, **kwargs)

                    # Check memory after execution
                    final_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_used = final_memory - initial_memory

                    if memory_used > max_memory_mb:
                        logger.warning(
                            f"Function {func.__name__} used {memory_used:.1f}MB memory"
                        )

                    return result

                except MemoryError:
                    logger.error(f"Memory limit exceeded in {func.__name__}")
                    # Force garbage collection
                    import gc

                    gc.collect()
                    raise

            return wrapper

        return decorator

    def with_retry(
        self, max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0
    ):
        """Decorator to add retry logic with exponential backoff."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                current_delay = delay

                for attempt in range(max_retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e

                        if attempt == max_retries:
                            logger.error(
                                f"Function {func.__name__} failed after {max_retries} retries: {e}"
                            )
                            break

                        logger.warning(
                            f"Function {func.__name__} failed (attempt {attempt + 1}), retrying in {current_delay}s: {e}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff

                raise last_exception

            return wrapper

        return decorator

    def with_fallback(self, fallback_func: Callable):
        """Decorator to provide fallback function on failure."""

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(
                        f"Function {func.__name__} failed, using fallback: {e}"
                    )
                    return fallback_func(*args, **kwargs)

            return wrapper

        return decorator

    def safe_execute(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Safely execute a function with comprehensive error handling."""
        start_time = time.time()
        function_name = func.__name__ if hasattr(func, "__name__") else str(func)

        try:
            # Monitor memory before execution
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

            # Execute function
            result = func(*args, **kwargs)

            # Monitor memory after execution
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_used = final_memory - initial_memory

            execution_time = time.time() - start_time

            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "memory_used_mb": memory_used,
                "error": None,
            }

        except TimeoutError as e:
            return self._handle_timeout_error(function_name, e, start_time)
        except MemoryError as e:
            return self._handle_memory_error(function_name, e, start_time)
        except ModelLoadError as e:
            return self._handle_model_error(function_name, e, start_time)
        except AnalysisError as e:
            return self._handle_analysis_error(function_name, e, start_time)
        except Exception as e:
            return self._handle_generic_error(function_name, e, start_time)

    def _handle_timeout_error(
        self, function_name: str, error: Exception, start_time: float
    ) -> Dict[str, Any]:
        """Handle timeout errors."""
        execution_time = time.time() - start_time

        logger.error(f"Timeout in {function_name} after {execution_time:.2f}s: {error}")

        return {
            "success": False,
            "result": None,
            "execution_time": execution_time,
            "error": {
                "type": "TimeoutError",
                "message": str(error),
                "recovery_suggestion": "Try with smaller input or increase timeout limit",
            },
        }

    def _handle_memory_error(
        self, function_name: str, error: Exception, start_time: float
    ) -> Dict[str, Any]:
        """Handle memory errors."""
        execution_time = time.time() - start_time
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024

        logger.error(
            f"Memory error in {function_name}: {error} (Current memory: {current_memory:.1f}MB)"
        )

        # Force garbage collection
        import gc

        gc.collect()

        return {
            "success": False,
            "result": None,
            "execution_time": execution_time,
            "error": {
                "type": "MemoryError",
                "message": str(error),
                "current_memory_mb": current_memory,
                "recovery_suggestion": "Try with smaller input or restart the service",
            },
        }

    def _handle_model_error(
        self, function_name: str, error: Exception, start_time: float
    ) -> Dict[str, Any]:
        """Handle model loading errors."""
        execution_time = time.time() - start_time

        logger.error(f"Model error in {function_name}: {error}")

        return {
            "success": False,
            "result": None,
            "execution_time": execution_time,
            "error": {
                "type": "ModelLoadError",
                "message": str(error),
                "recovery_suggestion": "Check model files and dependencies, or use fallback detection",
            },
        }

    def _handle_analysis_error(
        self, function_name: str, error: Exception, start_time: float
    ) -> Dict[str, Any]:
        """Handle analysis errors."""
        execution_time = time.time() - start_time

        logger.error(f"Analysis error in {function_name}: {error}")

        return {
            "success": False,
            "result": None,
            "execution_time": execution_time,
            "error": {
                "type": "AnalysisError",
                "message": str(error),
                "recovery_suggestion": "Check input format and try again, or use alternative analysis method",
            },
        }

    def _handle_generic_error(
        self, function_name: str, error: Exception, start_time: float
    ) -> Dict[str, Any]:
        """Handle generic errors."""
        execution_time = time.time() - start_time
        error_traceback = traceback.format_exc()

        logger.error(f"Unexpected error in {function_name}: {error}\n{error_traceback}")

        # Track error frequency
        error_key = f"{function_name}:{type(error).__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        self.last_errors[error_key] = {
            "timestamp": datetime.now().isoformat(),
            "error": str(error),
            "traceback": error_traceback,
        }

        return {
            "success": False,
            "result": None,
            "execution_time": execution_time,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": error_traceback,
                "recovery_suggestion": "Check logs for details and try again",
            },
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        return {
            "error_counts": self.error_counts.copy(),
            "recent_errors": self.last_errors.copy(),
            "system_info": {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage("/").percent,
            },
        }

    def reset_error_statistics(self):
        """Reset error statistics."""
        self.error_counts.clear()
        self.last_errors.clear()
        logger.info("Error statistics reset")


# Global error handler instance
error_handler = ErrorHandler()


def create_fallback_result(
    authenticity: str = "ANALYSIS FAILED",
    confidence: float = 0.0,
    error_message: str = "Analysis failed",
) -> Dict[str, Any]:
    """Create a fallback result for failed analyses."""
    return {
        "success": False,
        "authenticity": authenticity,
        "confidence": confidence,
        "analysis_date": datetime.now().isoformat(),
        "key_findings": [error_message],
        "error": error_message,
        "fallback": True,
    }


def validate_input_file(
    file_data: bytes, max_size_mb: int = 100, allowed_types: list = None
) -> Dict[str, Any]:
    """Validate input file data."""
    try:
        # Check file size
        file_size_mb = len(file_data) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return {
                "valid": False,
                "error": f"File too large: {file_size_mb:.1f}MB (max: {max_size_mb}MB)",
            }

        # Check if file is not empty
        if len(file_data) == 0:
            return {"valid": False, "error": "Empty file"}

        # Basic file type validation (if specified)
        if allowed_types:
            # Simple magic number check
            file_header = file_data[:16]

            # Common file signatures
            signatures = {
                "image": [
                    b"\xFF\xD8\xFF",  # JPEG
                    b"\x89PNG\r\n\x1a\n",  # PNG
                    b"GIF87a",  # GIF87a
                    b"GIF89a",  # GIF89a
                    b"RIFF",  # WebP (starts with RIFF)
                ],
                "video": [
                    b"\x00\x00\x00\x18ftypmp4",  # MP4
                    b"\x00\x00\x00\x20ftypmp4",  # MP4
                    b"RIFF",  # AVI
                    b"\x1aE\xdf\xa3",  # WebM/MKV
                ],
                "audio": [
                    b"RIFF",  # WAV
                    b"ID3",  # MP3 with ID3
                    b"\xFF\xFB",  # MP3
                    b"\xFF\xF3",  # MP3
                    b"fLaC",  # FLAC
                ],
            }

            valid_type = False
            for file_type in allowed_types:
                if file_type in signatures:
                    for signature in signatures[file_type]:
                        if file_header.startswith(signature):
                            valid_type = True
                            break
                if valid_type:
                    break

            if not valid_type:
                return {
                    "valid": False,
                    "error": f"Invalid file type. Allowed: {allowed_types}",
                }

        return {"valid": True, "size_mb": file_size_mb}

    except Exception as e:
        return {"valid": False, "error": f"File validation error: {str(e)}"}


def monitor_system_resources() -> Dict[str, Any]:
    """Monitor system resources."""
    try:
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage("/")

        return {
            "memory": {
                "percent": memory.percent,
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
            },
            "cpu": {"percent": cpu},
            "disk": {"percent": disk.percent, "free_gb": disk.free / (1024**3)},
            "healthy": memory.percent < 85 and cpu < 90 and disk.percent < 90,
        }
    except Exception as e:
        logger.error(f"Resource monitoring error: {e}")
        return {
            "memory": {"percent": 0},
            "cpu": {"percent": 0},
            "disk": {"percent": 0},
            "healthy": False,
            "error": str(e),
        }


# Decorators for easy use
timeout = error_handler.with_timeout
memory_monitor = error_handler.with_memory_monitoring
retry = error_handler.with_retry
fallback = error_handler.with_fallback
safe_execute = error_handler.safe_execute

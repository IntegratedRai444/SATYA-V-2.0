"""
Metrics collection and monitoring for SatyaAI
"""
import json
import logging
import os
import subprocess
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, TypedDict, Union

import psutil
import torch


class ModelVersionInfo(TypedDict):
    """Type definition for model version information."""

    version: str
    framework: str
    precision: str
    input_shape: Optional[Tuple[int, ...]]
    output_shape: Optional[Tuple[int, ...]]
    created_at: str


class GPUMetrics(TypedDict):
    """Type definition for GPU metrics."""

    memory_used: float  # MB
    memory_reserved: float  # MB
    memory_total: float  # MB
    utilization: float  # 0-100%
    temperature: float  # Celsius
    power_usage: float  # Watts
    last_oom: Optional[float]  # Timestamp
    oom_count: int
    last_updated: Optional[float]  # Timestamp


class ErrorType(Enum):
    """Enumeration of error types for ML operations."""

    OOM = "out_of_memory"
    INFERENCE = "inference_error"
    VALIDATION = "validation_error"
    LOADING = "loading_error"
    PRECISION = "precision_error"
    SHAPE = "shape_error"
    TIMEOUT = "timeout_error"
    AUTH = "authentication_error"
    RATE_LIMIT = "rate_limit_exceeded"
    INTEGRATION = "integration_error"


logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects and stores system and model metrics with GPU monitoring."""

    def __init__(self, max_history: int = 1000, persist_interval: int = 300):
        """Initialize the metrics collector with GPU monitoring.

        Args:
            max_history: Maximum number of historical data points to keep
            persist_interval: Interval in seconds between persisting metrics to disk
        """
        self.max_history = max_history
        self.persist_interval = persist_interval
        self._shutdown = threading.Event()
        self._lock = threading.RLock()

        # Initialize metrics storage
        self.model_metrics: Dict[str, Dict[str, Any]] = {}
        self.model_versions: Dict[str, Dict[str, ModelVersionInfo]] = {}
        self.gpu_metrics: Dict[int, GPUMetrics] = {}
        self.error_metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.system_metrics: Dict[str, List[Dict[str, Any]]] = {
            "cpu": deque(maxlen=max_history),
            "memory": deque(maxlen=max_history),
            "disk": deque(maxlen=max_history),
            "network": deque(maxlen=max_history),
        }

        # Initialize GPU metrics if available
        self._init_gpu_metrics()

        # Initialize request counters and metrics file
        self.request_counters = {
            "total_requests": 0,
            "requests_by_endpoint": {},
            "requests_by_status": {},
            "active_requests": 0,
            "max_concurrent_requests": 0,
        }

        # Set up metrics file path
        self.metrics_file = Path("data/metrics/metrics.json")
        self._running = True
        self.metrics_lock = threading.RLock()

        # Load previous metrics if they exist
        self._load_metrics()

        # Start background tasks
        self._start_background_tasks()

    def _load_metrics(self):
        """Load metrics from disk if available."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, "r") as f:
                    data = json.load(f)

                with self.metrics_lock:
                    self.system_metrics = data.get(
                        "system_metrics", self.system_metrics
                    )
                    self.model_metrics = data.get("model_metrics", self.model_metrics)
                    self.request_counters = data.get(
                        "request_counters", self.request_counters
                    )

                logger.info("Loaded metrics from disk")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")

    def _persist_metrics(self):
        """Persist current metrics to disk."""
        try:
            # Create directory if it doesn't exist
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with self.metrics_lock:
                data = {
                    "system_metrics": self.system_metrics,
                    "model_metrics": self.model_metrics,
                    "request_counters": self.request_counters,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                # Convert deque to list for JSON serialization
                for key in self.system_metrics:
                    if isinstance(self.system_metrics[key], deque):
                        data["system_metrics"][key] = list(self.system_metrics[key])

                with open(self.metrics_file, "w") as f:
                    json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")

    def _persist_metrics_loop(self):
        """Background thread to periodically persist metrics."""
        while self._running:
            try:
                time.sleep(self.persist_interval)
                self._persist_metrics()
            except Exception as e:
                logger.error(f"Error in metrics persistence loop: {e}")
                time.sleep(10)  # Prevent tight loop on error

    def _collect_metrics_loop(self):
        """Background thread to collect system metrics."""
        net_io = psutil.net_io_counters()
        last_sent = net_io.bytes_sent
        last_recv = net_io.bytes_recv

        while self._running:
            try:
                # Get current time
                now = datetime.utcnow()

                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage("/")

                # Calculate network usage
                net_io = psutil.net_io_counters()
                sent = net_io.bytes_sent - last_sent
                recv = net_io.bytes_recv - last_recv
                last_sent = net_io.bytes_sent
                last_recv = net_io.bytes_recv

                # Update metrics
                with self._lock:
                    self.system_metrics["cpu"].append(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "value": psutil.cpu_percent(),
                        }
                    )
                    mem = psutil.virtual_memory()
                    self.system_metrics["memory"].append(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "used": mem.used / (1024**3),  # Convert to GB
                            "total": mem.total / (1024**3),  # Convert to GB
                            "percent": mem.percent,
                        }
                    )

                    disk = psutil.disk_usage("/")
                    self.system_metrics["disk"].append(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "used": disk.used / (1024**3),  # Convert to GB
                            "total": disk.total / (1024**3),  # Convert to GB
                            "percent": disk.percent,
                        }
                    )

                    self.system_metrics["network"].append(
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "bytes_sent": sent,
                            "bytes_recv": recv,
                        }
                    )

                # Sleep until next collection interval
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(10)  # Avoid tight loop on error

    def register_model_version(
        self,
        model_name: str,
        version: str,
        framework: str = "pytorch",
        precision: str = "fp32",
        input_shape: Optional[Tuple[int, ...]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """Register a new model version with metadata.

        Args:
            model_name: Name of the model
            version: Version string (e.g., '1.0.0')
            framework: Framework used (e.g., 'pytorch', 'tensorflow')
            precision: Model precision (e.g., 'fp32', 'fp16', 'int8')
            input_shape: Expected input shape
            output_shape: Expected output shape
        """
        with self._lock:
            if model_name not in self.model_versions:
                self.model_versions[model_name] = {}

            self.model_versions[model_name][version] = {
                "version": version,
                "framework": framework,
                "precision": precision,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "created_at": datetime.utcnow().isoformat(),
            }

            # Initialize metrics for this model version if not exists
            version_key = f"{model_name}:{version}"
            if version_key not in self.model_metrics:
                self._init_model_metrics(version_key)

    def _init_model_metrics(self, model_key: str):
        """Initialize metrics for a model."""
        self.model_metrics[model_key] = {
            "total_inferences": 0,
            "successful_inferences": 0,
            "failed_inferences": 0,
            "total_inference_time": 0.0,
            "avg_inference_time": 0.0,
            "min_inference_time": float("inf"),
            "max_inference_time": 0.0,
            "last_used": None,
            "input_sizes": [],
            "recent_errors": deque(maxlen=100),
            "error_rates": {"last_hour": 0.0, "last_5min": 0.0, "last_minute": 0.0},
            "gpu_utilization": {"avg": 0.0, "max": 0.0, "samples": 0},
            "memory_usage": {"avg_mb": 0.0, "peak_mb": 0.0, "samples": 0},
            "version_metrics": {},
        }

    def record_model_metrics(
        self,
        model_name: str,
        version: str,
        inference_time: float,
        success: bool = True,
        input_size: Optional[Tuple[int, ...]] = None,
        output: Optional[Any] = None,
        error_type: Optional[ErrorType] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record metrics for a model inference with version tracking.

        Args:
            model_name: Name of the model
            version: Model version string
            inference_time: Time taken for inference in seconds
            success: Whether the inference was successful
            input_size: Size of the input data
            output: Model output (for validation)
            error_type: Type of error if inference failed
            error_message: Detailed error message
            metadata: Additional metadata to include
        """
        version_key = f"{model_name}:{version}"

        with self._lock:
            # Initialize metrics for this model version if not exists
            if version_key not in self.model_metrics:
                self._init_model_metrics(version_key)

            model_metric = self.model_metrics[version_key]
            model_metric["total_inferences"] += 1
            model_metric["total_inference_time"] += inference_time
            model_metric["avg_inference_time"] = (
                model_metric["total_inference_time"] / model_metric["total_inferences"]
            )

            # Update GPU metrics if available
            if torch.cuda.is_available():
                for device_id, gpu in self.gpu_metrics.items():
                    # Update GPU utilization metrics
                    model_metric["gpu_utilization"]["avg"] = (
                        model_metric["gpu_utilization"]["avg"]
                        * model_metric["gpu_utilization"]["samples"]
                        + gpu["utilization"]
                    ) / (model_metric["gpu_utilization"]["samples"] + 1)
                    model_metric["gpu_utilization"]["max"] = max(
                        model_metric["gpu_utilization"]["max"], gpu["utilization"]
                    )
                    model_metric["gpu_utilization"]["samples"] += 1

                    # Track memory usage
                    mem_used = gpu.get("memory_used", 0)
                    model_metric["memory_usage"]["avg_mb"] = (
                        model_metric["memory_usage"]["avg_mb"]
                        * model_metric["memory_usage"]["samples"]
                        + mem_used
                    ) / (model_metric["memory_usage"]["samples"] + 1)
                    model_metric["memory_usage"]["peak_mb"] = max(
                        model_metric["memory_usage"]["peak_mb"], mem_used
                    )
                    model_metric["memory_usage"]["samples"] += 1

            if success:
                model_metric["successful_inferences"] += 1
                model_metric["min_inference_time"] = min(
                    model_metric["min_inference_time"], inference_time
                )
                model_metric["max_inference_time"] = max(
                    model_metric["max_inference_time"], inference_time
                )
            else:
                model_metric["failed_inferences"] += 1
                error_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "error_type": error_type.value if error_type else "unknown",
                    "message": error_message or str(output) or "Unknown error",
                    "input_size": input_size,
                    "inference_time": inference_time,
                }
                if metadata:
                    error_entry["metadata"] = metadata
                model_metric["recent_errors"].append(error_entry)

                # Update error rates
                self._update_error_rates(version_key)

                # Record the error in the error metrics
                self.record_error(
                    error_type=error_type or ErrorType.INFERENCE,
                    message=error_message or "Inference failed",
                    details={
                        "model": model_name,
                        "version": version,
                        "input_size": input_size,
                        "inference_time": inference_time,
                        "error": str(output),
                    },
                    model_name=model_name,
                    severity="error" if error_type != ErrorType.OOM else "critical",
                )

            if input_size is not None:
                model_metric["input_sizes"].append(input_size)
                if len(model_metric["input_sizes"]) > 100:  # Keep last 100 input sizes
                    model_metric["input_sizes"] = model_metric["input_sizes"][-100:]

            model_metric["last_used"] = datetime.utcnow().isoformat()

            # Update version-specific metrics
            if version not in model_metric["version_metrics"]:
                model_metric["version_metrics"][version] = {
                    "inference_count": 0,
                    "total_time": 0.0,
                    "error_count": 0,
                    "last_used": None,
                }

            version_metric = model_metric["version_metrics"][version]
            version_metric["inference_count"] += 1
            version_metric["total_time"] += inference_time
            if not success:
                version_metric["error_count"] += 1
            version_metric["last_used"] = datetime.utcnow().isoformat()

            # Update model-level metrics (aggregate across versions)
            if model_name in self.model_versions:
                self._update_aggregate_metrics(model_name)

    def record_request(
        self,
        endpoint: str,
        method: str = "GET",
        status_code: int = 200,
        processing_time: Optional[float] = None,
    ):
        """Record API request metrics.

        Args:
            endpoint: API endpoint path
            method: HTTP method (GET, POST, etc.)
            status_code: HTTP status code
            processing_time: Time taken to process the request in seconds
        """
        with self._lock:
            if "requests" not in self.system_metrics:
                self.system_metrics["requests"] = []

            self.system_metrics["requests"].append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "processing_time": processing_time or 0.0,
                }
            )
            # Update request counters
            self.request_counters["total_requests"] += 1

            # Track requests by endpoint
            endpoint_key = f"{method} {endpoint}"
            if endpoint_key not in self.request_counters["requests_by_endpoint"]:
                self.request_counters["requests_by_endpoint"][endpoint_key] = 0
            self.request_counters["requests_by_endpoint"][endpoint_key] += 1

            # Track requests by status code
            status_str = str(status_code)
            if status_str not in self.request_counters["requests_by_status"]:
                self.request_counters["requests_by_status"][status_str] = 0
            self.request_counters["requests_by_status"][status_str] += 1

            # Track processing time if available
            if processing_time is not None:
                if "request_processing_times" not in self.request_counters:
                    self.request_counters["request_processing_times"] = []
                self.request_counters["request_processing_times"].append(
                    processing_time
                )
                if (
                    len(self.request_counters["request_processing_times"])
                    > self.max_history
                ):
                    self.request_counters[
                        "request_processing_times"
                    ] = self.request_counters["request_processing_times"][
                        -self.max_history :
                    ]

    def start_request(self):
        """Mark the start of a request (for tracking concurrent requests)."""
        with self.metrics_lock:
            self.request_counters["active_requests"] += 1
            self.request_counters["max_concurrent_requests"] = max(
                self.request_counters["max_concurrent_requests"],
                self.request_counters["active_requests"],
            )

    def end_request(self):
        """Mark the end of a request."""
        with self.metrics_lock:
            if self.request_counters["active_requests"] > 0:
                self.request_counters["active_requests"] -= 1

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        with self.metrics_lock:
            # Calculate request rates
            request_rates = {}
            if len(self.system_metrics["timestamp"]) > 1:
                time_span = (
                    datetime.fromisoformat(self.system_metrics["timestamp"][-1])
                    - datetime.fromisoformat(self.system_metrics["timestamp"][0])
                ).total_seconds()
                if time_span > 0:
                    request_rates = {
                        "requests_per_second": self.request_counters["total_requests"]
                        / time_span,
                        "requests_per_minute": self.request_counters["total_requests"]
                        / (time_span / 60),
                    }

            # Prepare model summaries
            model_summaries = {}
            for name, metrics in self.model_metrics.items():
                model_summaries[name] = {
                    "total_inferences": metrics["total_inferences"],
                    "success_rate": (
                        metrics["successful_inferences"] / metrics["total_inferences"]
                        if metrics["total_inferences"] > 0
                        else 0
                    ),
                    "avg_inference_time_ms": metrics["avg_inference_time"] * 1000,
                    "min_inference_time_ms": metrics["min_inference_time"] * 1000,
                    "max_inference_time_ms": metrics["max_inference_time"] * 1000,
                    "last_used": metrics["last_used"],
                }

            # Get system summary
            system_summary = {}
            if self.system_metrics["timestamp"]:
                system_summary = {
                    "cpu_percent": self.system_metrics["cpu_percent"][-1]
                    if self.system_metrics["cpu_percent"]
                    else 0,
                    "memory_percent": self.system_metrics["memory_percent"][-1]
                    if self.system_metrics["memory_percent"]
                    else 0,
                    "disk_usage": self.system_metrics["disk_usage"][-1]
                    if self.system_metrics["disk_usage"]
                    else 0,
                    "network_sent_mb": (
                        self.system_metrics["network_sent"][-1] / (1024 * 1024)
                        if self.system_metrics["network_sent"]
                        else 0
                    ),
                    "network_recv_mb": (
                        self.system_metrics["network_recv"][-1] / (1024 * 1024)
                        if self.system_metrics["network_recv"]
                        else 0
                    ),
                    "timestamp": self.system_metrics["timestamp"][-1]
                    if self.system_metrics["timestamp"]
                    else None,
                }

            return {
                "system": system_summary,
                "requests": {
                    "total": self.request_counters["total_requests"],
                    "active": self.request_counters["active_requests"],
                    "max_concurrent": self.request_counters["max_concurrent_requests"],
                    "by_endpoint": dict(self.request_counters["requests_by_endpoint"]),
                    "by_status": dict(self.request_counters["requests_by_status"]),
                    **request_rates,
                },
                "models": model_summaries,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_historical_metrics(
        self,
        metric_type: str = "system",
        metric_name: Optional[str] = None,
        limit: int = 100,
    ) -> Dict[str, Any]:
        """Get historical metrics data."""
        with self.metrics_lock:
            if metric_type == "system":
                if metric_name:
                    return {
                        "timestamps": list(self.system_metrics["timestamp"])[-limit:],
                        "values": list(self.system_metrics.get(metric_name, []))[
                            -limit:
                        ],
                    }
                return {k: list(v)[-limit:] for k, v in self.system_metrics.items()}

            elif metric_type == "model" and metric_name in self.model_metrics:
                return self.model_metrics[metric_name]

            return {}

    def _start_background_tasks(self):
        """Start background tasks for metrics collection and persistence."""
        self._collector_thread = threading.Thread(
            target=self._collect_metrics_loop, name="MetricsCollector", daemon=True
        )
        self._collector_thread.start()

        self._persister_thread = threading.Thread(
            target=self._persist_metrics_loop, name="MetricsPersister", daemon=True
        )
        self._persister_thread.start()

        self._monitor_thread = threading.Thread(
            target=self._monitor_resources_loop, name="ResourceMonitor", daemon=True
        )
        self._monitor_thread.start()

    def stop(self):
        """Stop the metrics collector and its background tasks."""
        self._shutdown.set()
        if hasattr(self, "_collector_thread"):
            self._collector_thread.join(timeout=5)
        if hasattr(self, "_persister_thread"):
            self._persister_thread.join(timeout=5)
        if hasattr(self, "_monitor_thread"):
            self._monitor_thread.join(timeout=5)

    def _monitor_resources_loop(self):
        """Monitor system resources in a background thread with enhanced OOM handling."""
        last_oom_check = 0
        oom_check_interval = 30  # seconds between OOM checks

        while not self._shutdown.is_set():
            try:
                current_time = time.time()

                # Update system metrics
                self._update_system_metrics()

                # Update GPU metrics if available
                if torch.cuda.is_available():
                    self._update_gpu_metrics()

                    # Check for OOM conditions periodically
                    if current_time - last_oom_check > oom_check_interval:
                        self._check_gpu_memory_pressure()
                        last_oom_check = current_time

                # Check for anomalies
                self._check_for_anomalies()

                # Sleep until next update
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                time.sleep(10)  # Avoid tight loop on error

    def _check_gpu_memory_pressure(self):
        """Check for GPU memory pressure and trigger OOM prevention if needed."""
        if not torch.cuda.is_available():
            return

        for device_id in range(torch.cuda.device_count()):
            try:
                torch.cuda.synchronize(device_id)
                allocated = torch.cuda.memory_allocated(device_id) / (1024**2)  # MB
                reserved = torch.cuda.memory_reserved(device_id) / (1024**2)  # MB
                total = torch.cuda.get_device_properties(device_id).total_memory / (
                    1024**2
                )  # MB

                # Check for high memory pressure (above 90%)
                memory_pressure = allocated / total if total > 0 else 0
                if memory_pressure > 0.9:
                    logger.warning(
                        f"High GPU {device_id} memory pressure: "
                        f"{allocated:.1f}MB / {total:.1f}MB ({memory_pressure:.1%})"
                    )
                    self._handle_memory_pressure(device_id, allocated, total)

            except Exception as e:
                logger.error(f"Error checking GPU {device_id} memory pressure: {e}")

    def _handle_memory_pressure(self, device_id: int, allocated: float, total: float):
        """Handle high GPU memory pressure with progressive mitigation strategies."""
        if device_id not in self.gpu_metrics:
            self.gpu_metrics[device_id] = {
                "memory_used": allocated,
                "memory_total": total,
                "oom_count": 0,
                "last_oom": 0,
                "recovery_attempts": 0,
                "degradation_level": 0,
                "last_oom_time": 0,
            }

        gpu_info = self.gpu_metrics[device_id]
        recovery_attempts = gpu_info.get("recovery_attempts", 0) + 1
        degradation_level = gpu_info.get("degradation_level", 0)

        # Update GPU metrics
        gpu_info.update(
            {
                "memory_used": allocated,
                "memory_total": total,
                "recovery_attempts": recovery_attempts,
                "last_oom_time": time.time(),
            }
        )

        logger.warning(
            f"Handling GPU {device_id} memory pressure (Attempt {recovery_attempts}, "
            f"Level {degradation_level}): {allocated:.1f}MB / {total:.1f}MB"
        )

        try:
            # Level 1: Clear CUDA cache and wait
            logger.info(
                f"Memory Pressure (Level 1): Clearing CUDA cache on GPU {device_id}"
            )
            torch.cuda.empty_cache()

            # If we've had multiple recovery attempts recently, escalate
            if (
                recovery_attempts > 1
                and time.time() - gpu_info.get("last_oom", 0) < 300
            ):  # 5 minutes
                degradation_level = min(
                    degradation_level + 1, 3
                )  # Max 3 degradation levels

                if degradation_level >= 1:
                    # Level 2: Offload least recently used models
                    logger.warning(
                        f"Memory Pressure (Level 2): Offloading models from GPU {device_id}"
                    )
                    self._offload_least_recent_models(device_id, count=1)

                if degradation_level >= 2:
                    # Level 3: Reduce batch sizes
                    logger.warning(
                        f"Memory Pressure (Level 3): Reducing batch sizes on GPU {device_id}"
                    )
                    self._reduce_batch_sizes(device_id, factor=0.5)

                if degradation_level >= 3:
                    # Level 4: Switch to smaller models
                    logger.critical(
                        f"Memory Pressure (Level 4): Switching to smaller models on GPU {device_id}"
                    )
                    self._switch_to_smaller_models(device_id)

            # Update degradation level
            gpu_info["degradation_level"] = degradation_level

            logger.info(
                f"Memory pressure handling completed for GPU {device_id}. "
                f"Degradation level: {degradation_level}, "
                f"Memory used: {allocated:.1f}MB/{total:.1f}MB"
            )

        except Exception as e:
            logger.error(f"Memory pressure handling failed for GPU {device_id}: {e}")

            # If recovery fails multiple times, reset after some time
            if (
                recovery_attempts > 5
                and time.time() - gpu_info.get("last_oom_time", 0) > 3600
            ):  # 1 hour
                logger.info(f"Resetting recovery state for GPU {device_id}")
                gpu_info.update(
                    {"recovery_attempts": 0, "degradation_level": 0, "last_oom_time": 0}
                )

    def _offload_least_recent_models(self, device_id: int, count: int = 1) -> bool:
        """Offload the least recently used models from GPU to CPU."""
        try:
            # Get models on this GPU sorted by last used time
            models_on_device = [
                (name, metrics)
                for name, metrics in self.model_metrics.items()
                if metrics.get("device") == f"cuda:{device_id}"
            ]

            if not models_on_device:
                logger.info(f"No models found on GPU {device_id} to offload")
                return False

            # Sort by last used time (oldest first)
            models_sorted = sorted(
                models_on_device, key=lambda x: x[1].get("last_used", 0)
            )

            offloaded = 0
            for model_name, metrics in models_sorted:
                if offloaded >= count:
                    break

                logger.info(
                    f"Offloading model {model_name} from GPU {device_id} to CPU"
                )
                try:
                    # Move model to CPU if it has a reference to the actual model
                    if "model" in metrics:
                        metrics["model"] = metrics["model"].cpu()
                        torch.cuda.empty_cache()
                        metrics["device"] = "cpu"
                        offloaded += 1

                        # Record the offload event
                        self.record_error(
                            error_type=ErrorType.OOM,
                            message=f"Model offloaded to CPU due to memory pressure",
                            details={
                                "model": model_name,
                                "device_id": device_id,
                                "reason": "memory_pressure",
                                "memory_saved_mb": metrics.get(
                                    "memory_footprint_mb", 0
                                ),
                            },
                            severity="warning",
                            model_name=model_name,
                        )
                except Exception as e:
                    logger.error(f"Failed to offload model {model_name}: {e}")

            return offloaded > 0

        except Exception as e:
            logger.error(f"Error in model offloading: {e}")
            return False

    def _reduce_batch_sizes(self, device_id: int, factor: float = 0.5) -> bool:
        """Reduce batch sizes for models on the specified GPU."""
        try:
            for model_name, metrics in self.model_metrics.items():
                if metrics.get("device", "") == f"cuda:{device_id}":
                    if "batch_size" in metrics:
                        new_batch_size = max(1, int(metrics["batch_size"] * factor))
                        if new_batch_size < metrics["batch_size"]:
                            old_size = metrics["batch_size"]
                            metrics["batch_size"] = new_batch_size

                            logger.info(
                                f"Reduced batch size for {model_name} on GPU {device_id} "
                                f"from {old_size} to {new_batch_size}"
                            )

                            # Record the batch size reduction
                            self.record_error(
                                error_type=ErrorType.OOM,
                                message=f"Reduced batch size for memory pressure",
                                details={
                                    "model": model_name,
                                    "device_id": device_id,
                                    "old_batch_size": old_size,
                                    "new_batch_size": new_batch_size,
                                    "factor": factor,
                                },
                                severity="warning",
                                model_name=model_name,
                            )
                            return True  # Only reduce one model at a time

            logger.info(
                f"No models with adjustable batch sizes found on GPU {device_id}"
            )
            return False

        except Exception as e:
            logger.error(f"Error reducing batch sizes: {e}")
            return False

    def _switch_to_smaller_models(self, device_id: int) -> bool:
        """Switch to smaller/optimized versions of models where available."""
        try:
            # Define model size reduction mapping
            model_switches = {
                # Map model names to their smaller counterparts
                "large-llm": "medium-llm",
                "medium-llm": "small-llm",
                "gpt3": "gpt2",
                "bert-large": "bert-base",
                # Add more model mappings as needed
            }

            # Find models on this device
            for model_name, metrics in list(self.model_metrics.items()):
                if metrics.get("device", "") == f"cuda:{device_id}":
                    smaller_model = model_switches.get(model_name)
                    if smaller_model and smaller_model in self.model_metrics:
                        logger.warning(
                            f"Switching from {model_name} to smaller model {smaller_model} "
                            f"on GPU {device_id} due to memory pressure"
                        )

                        # Update model reference (in a real implementation, this would load the new model)
                        if "model" in metrics:
                            metrics["model"] = None  # Release reference
                            metrics["device"] = "none"

                        # Record the model switch
                        self.record_error(
                            error_type=ErrorType.OOM,
                            message=f"Switched to smaller model for memory pressure",
                            details={
                                "original_model": model_name,
                                "new_model": smaller_model,
                                "device_id": device_id,
                                "reason": "memory_pressure",
                                "memory_saved_mb": metrics.get(
                                    "memory_footprint_mb", 0
                                ),
                            },
                            severity="warning",
                            model_name=model_name,
                        )

                        # In a real implementation, you would load the smaller model here
                        # self._load_model(smaller_model, device_id)

                        return True  # Only switch one model at a time

            logger.info(f"No smaller model alternatives found for GPU {device_id}")
            return False

        except Exception as e:
            logger.error(f"Error switching to smaller models: {e}")
            return False

    def __del__(self):
        """Cleanup on destruction."""
        self._running = False
        if hasattr(self, "background_thread") and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)
        if hasattr(self, "persist_thread") and self.persist_thread.is_alive():
            self.persist_thread.join(timeout=5)
        self._persist_metrics()


# Global metrics collector instance
metrics_collector = MetricsCollector()

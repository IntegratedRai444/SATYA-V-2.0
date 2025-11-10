"""
Metrics collection and monitoring for SatyaAI
"""
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import threading
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Collects and stores system and model metrics."""
    
    def __init__(self, max_history: int = 1000, persist_interval: int = 300):
        """Initialize the metrics collector.
        
        Args:
            max_history: Maximum number of data points to keep in memory
            persist_interval: Interval in seconds between persisting metrics to disk
        """
        self.max_history = max_history
        self.metrics_lock = threading.Lock()
        self.persist_interval = persist_interval
        self.metrics_file = Path("data/metrics/metrics.json")
        
        # Initialize metrics storage
        self.system_metrics = {
            'cpu_percent': deque(maxlen=max_history),
            'memory_percent': deque(maxlen=max_history),
            'disk_usage': deque(maxlen=max_history),
            'network_sent': deque(maxlen=max_history),
            'network_recv': deque(maxlen=max_history),
            'timestamp': deque(maxlen=max_history)
        }
        
        # Model-specific metrics
        self.model_metrics = {}
        
        # Request counters
        self.request_counters = {
            'total_requests': 0,
            'requests_by_endpoint': {},
            'requests_by_status': {},
            'active_requests': 0,
            'max_concurrent_requests': 0
        }
        
        # Load previous metrics if they exist
        self._load_metrics()
        
        # Start background tasks
        self._running = True
        self.background_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self.background_thread.start()
        
        self.persist_thread = threading.Thread(target=self._persist_metrics_loop, daemon=True)
        self.persist_thread.start()
    
    def _load_metrics(self):
        """Load metrics from disk if available."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    
                with self.metrics_lock:
                    self.system_metrics = data.get('system_metrics', self.system_metrics)
                    self.model_metrics = data.get('model_metrics', self.model_metrics)
                    self.request_counters = data.get('request_counters', self.request_counters)
                    
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
                    'system_metrics': self.system_metrics,
                    'model_metrics': self.model_metrics,
                    'request_counters': self.request_counters,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                # Convert deque to list for JSON serialization
                for key in self.system_metrics:
                    if isinstance(self.system_metrics[key], deque):
                        data['system_metrics'][key] = list(self.system_metrics[key])
                
                with open(self.metrics_file, 'w') as f:
                    json.dump(data, f, indent=2)
                    
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
    
    def _persist_metrics_loop(self):
        """Background thread to periodically persist metrics."""
        while self._running:
            time.sleep(self.persist_interval)
            self._persist_metrics()
    
    def _collect_system_metrics(self):
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
                disk = psutil.disk_usage('/')
                
                # Calculate network usage
                net_io = psutil.net_io_counters()
                sent = net_io.bytes_sent - last_sent
                recv = net_io.bytes_recv - last_recv
                last_sent = net_io.bytes_sent
                last_recv = net_io.bytes_recv
                
                # Update metrics
                with self.metrics_lock:
                    self.system_metrics['cpu_percent'].append(cpu_percent)
                    self.system_metrics['memory_percent'].append(memory.percent)
                    self.system_metrics['disk_usage'].append(disk.percent)
                    self.system_metrics['network_sent'].append(sent)
                    self.system_metrics['network_recv'].append(recv)
                    self.system_metrics['timestamp'].append(now.isoformat())
                
                # Sleep until next collection interval
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                time.sleep(10)  # Avoid tight loop on error
    
    def record_model_metrics(
        self,
        model_name: str,
        inference_time: float,
        success: bool = True,
        input_size: Optional[Tuple[int, ...]] = None,
        output: Optional[Any] = None
    ):
        """Record metrics for a model inference.
        
        Args:
            model_name: Name of the model
            inference_time: Time taken for inference in seconds
            success: Whether the inference was successful
            input_size: Size of the input data
            output: Model output (for validation)
        """
        with self.metrics_lock:
            if model_name not in self.model_metrics:
                self.model_metrics[model_name] = {
                    'total_inferences': 0,
                    'successful_inferences': 0,
                    'failed_inferences': 0,
                    'total_inference_time': 0.0,
                    'avg_inference_time': 0.0,
                    'min_inference_time': float('inf'),
                    'max_inference_time': 0.0,
                    'last_used': None,
                    'input_sizes': [],
                    'recent_errors': deque(maxlen=100)
                }
            
            model_metric = self.model_metrics[model_name]
            model_metric['total_inferences'] += 1
            model_metric['total_inference_time'] += inference_time
            model_metric['avg_inference_time'] = (
                model_metric['total_inference_time'] / model_metric['total_inferences']
            )
            
            if success:
                model_metric['successful_inferences'] += 1
                model_metric['min_inference_time'] = min(
                    model_metric['min_inference_time'], inference_time
                )
                model_metric['max_inference_time'] = max(
                    model_metric['max_inference_time'], inference_time
                )
            else:
                model_metric['failed_inferences'] += 1
                model_metric['recent_errors'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'error': str(output) if output is not None else 'Unknown error'
                })
            
            if input_size is not None:
                model_metric['input_sizes'].append(input_size)
                if len(model_metric['input_sizes']) > 100:  # Keep last 100 input sizes
                    model_metric['input_sizes'] = model_metric['input_sizes'][-100:]
            
            model_metric['last_used'] = datetime.utcnow().isoformat()
    
    def record_request(
        self,
        endpoint: str,
        method: str = "GET",
        status_code: int = 200,
        processing_time: Optional[float] = None
    ):
        """Record API request metrics."""
        with self.metrics_lock:
            # Update request counters
            self.request_counters['total_requests'] += 1
            
            # Track requests by endpoint
            endpoint_key = f"{method} {endpoint}"
            if endpoint_key not in self.request_counters['requests_by_endpoint']:
                self.request_counters['requests_by_endpoint'][endpoint_key] = 0
            self.request_counters['requests_by_endpoint'][endpoint_key] += 1
            
            # Track requests by status code
            status_str = str(status_code)
            if status_str not in self.request_counters['requests_by_status']:
                self.request_counters['requests_by_status'][status_str] = 0
            self.request_counters['requests_by_status'][status_str] += 1
            
            # Track processing time if available
            if processing_time is not None:
                if 'request_processing_times' not in self.request_counters:
                    self.request_counters['request_processing_times'] = []
                self.request_counters['request_processing_times'].append(processing_time)
                if len(self.request_counters['request_processing_times']) > self.max_history:
                    self.request_counters['request_processing_times'] = \
                        self.request_counters['request_processing_times'][-self.max_history:]
    
    def start_request(self):
        """Mark the start of a request (for tracking concurrent requests)."""
        with self.metrics_lock:
            self.request_counters['active_requests'] += 1
            self.request_counters['max_concurrent_requests'] = max(
                self.request_counters['max_concurrent_requests'],
                self.request_counters['active_requests']
            )
    
    def end_request(self):
        """Mark the end of a request."""
        with self.metrics_lock:
            if self.request_counters['active_requests'] > 0:
                self.request_counters['active_requests'] -= 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all collected metrics."""
        with self.metrics_lock:
            # Calculate request rates
            request_rates = {}
            if len(self.system_metrics['timestamp']) > 1:
                time_span = (
                    datetime.fromisoformat(self.system_metrics['timestamp'][-1]) - 
                    datetime.fromisoformat(self.system_metrics['timestamp'][0])
                ).total_seconds()
                if time_span > 0:
                    request_rates = {
                        'requests_per_second': self.request_counters['total_requests'] / time_span,
                        'requests_per_minute': self.request_counters['total_requests'] / (time_span / 60)
                    }
            
            # Prepare model summaries
            model_summaries = {}
            for name, metrics in self.model_metrics.items():
                model_summaries[name] = {
                    'total_inferences': metrics['total_inferences'],
                    'success_rate': (
                        metrics['successful_inferences'] / metrics['total_inferences'] 
                        if metrics['total_inferences'] > 0 else 0
                    ),
                    'avg_inference_time_ms': metrics['avg_inference_time'] * 1000,
                    'min_inference_time_ms': metrics['min_inference_time'] * 1000,
                    'max_inference_time_ms': metrics['max_inference_time'] * 1000,
                    'last_used': metrics['last_used']
                }
            
            # Get system summary
            system_summary = {}
            if self.system_metrics['timestamp']:
                system_summary = {
                    'cpu_percent': self.system_metrics['cpu_percent'][-1] if self.system_metrics['cpu_percent'] else 0,
                    'memory_percent': self.system_metrics['memory_percent'][-1] if self.system_metrics['memory_percent'] else 0,
                    'disk_usage': self.system_metrics['disk_usage'][-1] if self.system_metrics['disk_usage'] else 0,
                    'network_sent_mb': (
                        self.system_metrics['network_sent'][-1] / (1024 * 1024) 
                        if self.system_metrics['network_sent'] else 0
                    ),
                    'network_recv_mb': (
                        self.system_metrics['network_recv'][-1] / (1024 * 1024) 
                        if self.system_metrics['network_recv'] else 0
                    ),
                    'timestamp': self.system_metrics['timestamp'][-1] if self.system_metrics['timestamp'] else None
                }
            
            return {
                'system': system_summary,
                'requests': {
                    'total': self.request_counters['total_requests'],
                    'active': self.request_counters['active_requests'],
                    'max_concurrent': self.request_counters['max_concurrent_requests'],
                    'by_endpoint': dict(self.request_counters['requests_by_endpoint']),
                    'by_status': dict(self.request_counters['requests_by_status']),
                    **request_rates
                },
                'models': model_summaries,
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_historical_metrics(
        self, 
        metric_type: str = 'system', 
        metric_name: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """Get historical metrics data."""
        with self.metrics_lock:
            if metric_type == 'system':
                if metric_name:
                    return {
                        'timestamps': list(self.system_metrics['timestamp'])[-limit:],
                        'values': list(self.system_metrics.get(metric_name, []))[-limit:]
                    }
                return {k: list(v)[-limit:] for k, v in self.system_metrics.items()}
            
            elif metric_type == 'model' and metric_name in self.model_metrics:
                return self.model_metrics[metric_name]
            
            return {}
    
    def __del__(self):
        """Cleanup on destruction."""
        self._running = False
        if hasattr(self, 'background_thread') and self.background_thread.is_alive():
            self.background_thread.join(timeout=5)
        if hasattr(self, 'persist_thread') and self.persist_thread.is_alive():
            self.persist_thread.join(timeout=5)
        self._persist_metrics()

# Global metrics collector instance
metrics_collector = MetricsCollector()

"""
System Information and Monitoring Route
Dedicated route for system metrics and monitoring
"""

from fastapi import APIRouter
from datetime import datetime
import psutil
import platform
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/info")
async def get_system_info():
    """
    Get system information
    
    Returns detailed system information
    """
    try:
        return {
            "success": True,
            "system": {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        return {"success": False, "error": str(e)}


@router.get("/metrics")
async def get_system_metrics():
    """
    Get system metrics (CPU, memory, disk)
    
    Returns current system resource usage
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "success": True,
            "metrics": {
                "cpu": {
                    "percent": cpu_percent,
                    "count": psutil.cpu_count()
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                }
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {"success": False, "error": str(e)}


@router.get("/processes")
async def get_process_info():
    """
    Get current process information
    """
    try:
        process = psutil.Process()
        
        return {
            "success": True,
            "process": {
                "pid": process.pid,
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_info": {
                    "rss": process.memory_info().rss,
                    "vms": process.memory_info().vms
                },
                "num_threads": process.num_threads(),
                "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get process info: {e}")
        return {"success": False, "error": str(e)}
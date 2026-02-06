"""
Progress tracking and real-time feedback system for SatyaAI analysis.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProgressStatus(Enum):
    """Progress status enumeration."""

    PENDING = "pending"
    INITIALIZING = "initializing"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    FINALIZING = "finalizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    """Progress update data structure."""

    task_id: str
    status: ProgressStatus
    progress: float  # 0.0 to 1.0
    message: str
    details: Dict[str, Any]
    timestamp: str
    estimated_remaining_seconds: Optional[float] = None
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    current_step_number: Optional[int] = None


class ProgressTracker:
    """Real-time progress tracking system."""

    def __init__(self):
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.websocket_connections: Dict[str, Any] = {}  # WebSocket connections
        self.callbacks: Dict[str, List[Callable]] = {}  # Progress callbacks

    def create_task(
        self, task_type: str, total_steps: int = 100, metadata: Dict[str, Any] = None
    ) -> str:
        """
        Create a new progress tracking task.

        Args:
            task_type: Type of task (e.g., 'image_analysis', 'video_analysis')
            total_steps: Total number of steps for progress calculation
            metadata: Additional task metadata

        Returns:
            Task ID for tracking
        """
        task_id = str(uuid.uuid4())

        self.active_tasks[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": ProgressStatus.PENDING,
            "progress": 0.0,
            "total_steps": total_steps,
            "current_step": 0,
            "start_time": time.time(),
            "last_update": time.time(),
            "metadata": metadata or {},
            "steps_completed": [],
            "estimated_total_time": None,
        }

        logger.info(f"Created progress task: {task_id} ({task_type})")

        # Send initial update
        self.update_progress(
            task_id=task_id,
            status=ProgressStatus.PENDING,
            progress=0.0,
            message=f"Task created: {task_type}",
            details={"task_type": task_type, "total_steps": total_steps},
        )

        return task_id

    def update_progress(
        self,
        task_id: str,
        status: ProgressStatus = None,
        progress: float = None,
        message: str = None,
        details: Dict[str, Any] = None,
        current_step: str = None,
    ):
        """
        Update progress for a task.

        Args:
            task_id: Task ID to update
            status: New status
            progress: Progress value (0.0 to 1.0)
            message: Progress message
            details: Additional details
            current_step: Current step description
        """
        if task_id not in self.active_tasks:
            logger.warning(f"Task {task_id} not found for progress update")
            return

        task = self.active_tasks[task_id]
        current_time = time.time()

        # Update task data
        if status is not None:
            task["status"] = status
        if progress is not None:
            task["progress"] = max(0.0, min(1.0, progress))
        if current_step is not None:
            task["current_step_name"] = current_step
            task["current_step"] += 1

        task["last_update"] = current_time

        # Calculate estimated remaining time
        estimated_remaining = self._calculate_estimated_time(task)

        # Create progress update
        update = ProgressUpdate(
            task_id=task_id,
            status=status or task["status"],
            progress=task["progress"],
            message=message or f"Processing... {task['progress']*100:.1f}%",
            details=details or {},
            timestamp=datetime.now().isoformat(),
            estimated_remaining_seconds=estimated_remaining,
            current_step=current_step,
            total_steps=task["total_steps"],
            current_step_number=task["current_step"],
        )

        # Send update to all listeners
        self._broadcast_update(update)

        if os.environ.get('PYTHON_ENV') == 'development':
            logger.debug(
                f"Progress update for {task_id}: {task['progress']*100:.1f}% - {message}"
            )

    def increment_progress(
        self,
        task_id: str,
        increment: float = None,
        message: str = None,
        details: Dict[str, Any] = None,
    ):
        """
        Increment progress by a specific amount.

        Args:
            task_id: Task ID
            increment: Amount to increment (if None, auto-calculate based on steps)
            message: Progress message
            details: Additional details
        """
        if task_id not in self.active_tasks:
            return

        task = self.active_tasks[task_id]

        if increment is None:
            # Auto-calculate increment based on total steps
            increment = 1.0 / task["total_steps"]

        new_progress = min(1.0, task["progress"] + increment)

        self.update_progress(
            task_id=task_id, progress=new_progress, message=message, details=details
        )

    def complete_task(
        self,
        task_id: str,
        message: str = "Task completed successfully",
        result: Dict[str, Any] = None,
    ):
        """
        Mark a task as completed.

        Args:
            task_id: Task ID
            message: Completion message
            result: Task result data
        """
        self.update_progress(
            task_id=task_id,
            status=ProgressStatus.COMPLETED,
            progress=1.0,
            message=message,
            details={"result": result or {}, "completed": True},
        )

        # Keep task for a short time for final status retrieval
        asyncio.create_task(self._cleanup_task_later(task_id, delay=30))

    def fail_task(
        self, task_id: str, error_message: str, error_details: Dict[str, Any] = None
    ):
        """
        Mark a task as failed.

        Args:
            task_id: Task ID
            error_message: Error message
            error_details: Error details
        """
        self.update_progress(
            task_id=task_id,
            status=ProgressStatus.FAILED,
            message=f"Task failed: {error_message}",
            details={"error": error_message, "error_details": error_details or {}},
        )

        # Keep task for error retrieval
        asyncio.create_task(self._cleanup_task_later(task_id, delay=60))

    def cancel_task(self, task_id: str, message: str = "Task cancelled"):
        """
        Cancel a task.

        Args:
            task_id: Task ID
            message: Cancellation message
        """
        self.update_progress(
            task_id=task_id,
            status=ProgressStatus.CANCELLED,
            message=message,
            details={"cancelled": True},
        )

        asyncio.create_task(self._cleanup_task_later(task_id, delay=10))

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a task.

        Args:
            task_id: Task ID

        Returns:
            Task status dictionary or None if not found
        """
        return self.active_tasks.get(task_id)

    def get_all_tasks(self) -> Dict[str, Dict[str, Any]]:
        """Get all active tasks."""
        return self.active_tasks.copy()

    def register_websocket(self, websocket, task_id: str = None):
        """
        Register a WebSocket connection for progress updates.

        Args:
            websocket: WebSocket connection
            task_id: Specific task ID to track (if None, track all)
        """
        connection_id = str(uuid.uuid4())
        self.websocket_connections[connection_id] = {
            "websocket": websocket,
            "task_id": task_id,
            "connected_at": time.time(),
        }

        logger.info(
            f"WebSocket registered: {connection_id} for task: {task_id or 'all'}"
        )
        return connection_id

    def unregister_websocket(self, connection_id: str):
        """
        Unregister a WebSocket connection.

        Args:
            connection_id: Connection ID to remove
        """
        if connection_id in self.websocket_connections:
            del self.websocket_connections[connection_id]
            logger.info(f"WebSocket unregistered: {connection_id}")

    def register_callback(self, task_id: str, callback: Callable):
        """
        Register a callback function for progress updates.

        Args:
            task_id: Task ID to track
            callback: Callback function that receives ProgressUpdate
        """
        if task_id not in self.callbacks:
            self.callbacks[task_id] = []
        self.callbacks[task_id].append(callback)

    def _calculate_estimated_time(self, task: Dict[str, Any]) -> Optional[float]:
        """Calculate estimated remaining time for a task."""
        try:
            current_time = time.time()
            elapsed_time = current_time - task["start_time"]
            progress = task["progress"]

            if progress > 0.01:  # Avoid division by very small numbers
                estimated_total_time = elapsed_time / progress
                estimated_remaining = estimated_total_time - elapsed_time
                return max(0, estimated_remaining)

            return None

        except Exception as e:
            logger.warning(f"Error calculating estimated time: {e}")
            return None

    def _broadcast_update(self, update: ProgressUpdate):
        """Broadcast progress update to all listeners."""
        # Send to WebSocket connections
        asyncio.create_task(self._send_websocket_updates(update))

        # Call registered callbacks
        self._call_callbacks(update)

    async def _send_websocket_updates(self, update: ProgressUpdate):
        """Send progress update to WebSocket connections."""
        if not self.websocket_connections:
            return

        update_data = asdict(update)
        update_json = json.dumps(update_data)

        # Send to relevant connections
        connections_to_remove = []

        for conn_id, conn_info in self.websocket_connections.items():
            try:
                websocket = conn_info["websocket"]
                task_filter = conn_info["task_id"]

                # Check if this connection should receive this update
                if task_filter is None or task_filter == update.task_id:
                    await websocket.send_text(update_json)

            except Exception as e:
                logger.warning(f"Failed to send WebSocket update to {conn_id}: {e}")
                connections_to_remove.append(conn_id)

        # Clean up failed connections
        for conn_id in connections_to_remove:
            self.unregister_websocket(conn_id)

    def _call_callbacks(self, update: ProgressUpdate):
        """Call registered callbacks for progress updates."""
        task_callbacks = self.callbacks.get(update.task_id, [])

        for callback in task_callbacks:
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Callback error for task {update.task_id}: {e}")

    async def _cleanup_task_later(self, task_id: str, delay: int = 30):
        """Clean up a task after a delay."""
        await asyncio.sleep(delay)

        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
            if os.environ.get('PYTHON_ENV') == 'development':
                logger.debug(f"Cleaned up task: {task_id}")

        # Clean up callbacks
        if task_id in self.callbacks:
            del self.callbacks[task_id]


# Global progress tracker instance
progress_tracker = ProgressTracker()


class ProgressContext:
    """Context manager for progress tracking."""

    def __init__(
        self, task_type: str, total_steps: int = 100, metadata: Dict[str, Any] = None
    ):
        self.task_type = task_type
        self.total_steps = total_steps
        self.metadata = metadata
        self.task_id = None

    def __enter__(self):
        self.task_id = progress_tracker.create_task(
            self.task_type, self.total_steps, self.metadata
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            progress_tracker.complete_task(self.task_id)
        else:
            progress_tracker.fail_task(
                self.task_id, str(exc_val), {"exception_type": exc_type.__name__}
            )

    def update(
        self,
        progress: float = None,
        message: str = None,
        details: Dict[str, Any] = None,
        current_step: str = None,
    ):
        """Update progress within context."""
        progress_tracker.update_progress(
            self.task_id,
            progress=progress,
            message=message,
            details=details,
            current_step=current_step,
        )

    def increment(
        self,
        increment: float = None,
        message: str = None,
        details: Dict[str, Any] = None,
    ):
        """Increment progress within context."""
        progress_tracker.increment_progress(
            self.task_id, increment=increment, message=message, details=details
        )


def track_progress(
    task_type: str, total_steps: int = 100, metadata: Dict[str, Any] = None
):
    """
    Decorator for automatic progress tracking.

    Args:
        task_type: Type of task
        total_steps: Total number of steps
        metadata: Additional metadata
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            with ProgressContext(task_type, total_steps, metadata) as progress:
                # Add progress context to kwargs
                kwargs["progress_context"] = progress
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Utility functions for common progress patterns
def create_analysis_progress(analysis_type: str, file_size_mb: float = None) -> str:
    """Create progress tracker for analysis tasks."""
    # Estimate steps based on analysis type and file size
    if analysis_type == "image":
        total_steps = 5
    elif analysis_type == "video":
        total_steps = max(10, int(file_size_mb * 2)) if file_size_mb else 20
    elif analysis_type == "audio":
        total_steps = max(8, int(file_size_mb * 1.5)) if file_size_mb else 15
    else:
        total_steps = 10

    return progress_tracker.create_task(
        f"{analysis_type}_analysis",
        total_steps=total_steps,
        metadata={"file_size_mb": file_size_mb},
    )


def update_analysis_progress(
    task_id: str, stage: str, progress: float, details: Dict[str, Any] = None
):
    """Update progress for analysis tasks with standard stages."""
    stage_messages = {
        "validation": "Validating input file...",
        "loading": "Loading and preprocessing...",
        "detection": "Running detection algorithms...",
        "analysis": "Analyzing results...",
        "aggregation": "Aggregating findings...",
        "finalization": "Finalizing results...",
    }

    message = stage_messages.get(stage, f"Processing: {stage}")

    progress_tracker.update_progress(
        task_id=task_id,
        progress=progress,
        message=message,
        details=details,
        current_step=stage,
    )

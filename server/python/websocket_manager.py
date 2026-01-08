"""
Enhanced WebSocket Manager for real-time communication with robust error handling
"""
import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


def retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying WebSocket operations with exponential backoff"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except (WebSocketDisconnect, ConnectionError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2**attempt)  # Exponential backoff
                        logger.warning(
                            f"Attempt {attempt + 1} failed. Retrying in {wait_time:.2f}s..."
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    raise
            raise last_exception or Exception("Unknown error in WebSocket operation")

        return wrapper

    return decorator


class ConnectionManager:
    """
    Enhanced WebSocket connection manager with:
    - Connection tracking and management
    - Message queuing for disconnected clients
    - Heartbeat mechanism
    - Error handling and reconnection
    - Rate limiting
    """

    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_data: Dict[WebSocket, Dict[str, Any]] = {}
        self.message_queues: Dict[WebSocket, asyncio.Queue] = {}
        self.lock = asyncio.Lock()
        self.heartbeat_interval = 30  # seconds
        self.max_queue_size = 100  # Max messages to queue per client
        self.rate_limits: Dict[str, Dict[str, Any]] = {}  # Track rate limits by client

    @retry(max_retries=3, delay=1.0)
    async def connect(
        self, websocket: WebSocket, client_id: str, channel: str = "default"
    ):
        """
        Handle new WebSocket connection with enhanced error handling.

        Args:
            websocket: The WebSocket connection
            client_id: Unique client identifier
            channel: Channel to subscribe to

        Returns:
            str: The assigned client ID
        """
        try:
            await websocket.accept()

            async with self.lock:
                # Initialize connection tracking
                if channel not in self.active_connections:
                    self.active_connections[channel] = set()

                # Set up connection data
                self.connection_data[websocket] = {
                    "client_id": client_id,
                    "channel": channel,
                    "connected_at": time.time(),
                    "last_active": time.time(),
                    "message_count": 0,
                    "is_alive": True,
                }

                # Initialize message queue
                self.message_queues[websocket] = asyncio.Queue(
                    maxsize=self.max_queue_size
                )

                # Add to active connections
                self.active_connections[channel].add(websocket)

                # Start heartbeat and message processing
                asyncio.create_task(self._start_heartbeat(websocket))
                asyncio.create_task(self._process_message_queue(websocket))

            logger.info(f"Client {client_id} connected to channel {channel}")
            return client_id

        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {str(e)}")
            await self._cleanup_connection(websocket)
            raise

    async def disconnect(
        self, websocket: WebSocket, code: int = 1000, reason: str = "Normal closure"
    ):
        """
        Handle WebSocket disconnection with cleanup.

        Args:
            websocket: The WebSocket connection to disconnect
            code: Close status code (default: 1000 - Normal closure)
            reason: Reason for disconnection
        """
        try:
            # Send close frame if connection is still open
            if websocket.client_state.value < 3:  # 3 = CLOSED
                await websocket.close(code=code, reason=reason)
        except Exception as e:
            logger.warning(f"Error during WebSocket close: {e}")
        finally:
            await self._cleanup_connection(websocket, code, reason)

    async def _cleanup_connection(
        self, websocket: WebSocket, code: int = 1000, reason: str = ""
    ):
        """Clean up connection resources"""
        client_data = self.connection_data.get(websocket, {})
        client_id = client_data.get("client_id", "unknown")
        channel = client_data.get("channel", "unknown")

        async with self.lock:
            # Remove from active connections
            if (
                channel in self.active_connections
                and websocket in self.active_connections[channel]
            ):
                self.active_connections[channel].remove(websocket)
                if not self.active_connections[channel]:
                    del self.active_connections[channel]

            # Clean up resources
            if websocket in self.connection_data:
                del self.connection_data[websocket]

            # Clear message queue
            if websocket in self.message_queues:
                # Notify any waiting tasks
                queue = self.message_queues[websocket]
                await queue.put(None)  # Sentinel to signal queue processing to stop
                del self.message_queues[websocket]

        logger.info(
            f"Client {client_id} disconnected from channel {channel} (code: {code}, reason: {reason})"
        )

    @retry(max_retries=3, delay=0.5)
    async def send_personal_message(
        self, message: str, websocket: WebSocket, queue_if_disconnected: bool = True
    ):
        """
        Send a message to a specific client with retry and queueing.

        Args:
            message: The message to send (will be JSON-serialized)
            websocket: The target WebSocket connection
            queue_if_disconnected: Whether to queue messages if client is temporarily disconnected

        Returns:
            bool: True if message was sent or queued, False otherwise
        """
        if websocket not in self.connection_data:
            logger.warning("Attempted to send to unknown WebSocket connection")
            return False

        if not isinstance(message, str):
            try:
                message = json.dumps(message)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize message: {e}")
                return False

        # Check rate limiting
        if not self._check_rate_limit(websocket):
            logger.warning(
                f"Rate limit exceeded for client {self.connection_data[websocket].get('client_id')}"
            )
            return False

        try:
            # Update last active time
            self.connection_data[websocket]["last_active"] = time.time()
            self.connection_data[websocket]["message_count"] += 1

            # Send the message
            await websocket.send_text(message)
            return True

        except (WebSocketDisconnect, ConnectionError) as e:
            logger.warning(f"Client disconnected while sending message: {e}")
            if queue_if_disconnected and websocket in self.message_queues:
                try:
                    # Try to queue the message for later delivery
                    queue = self.message_queues[websocket]
                    if queue.qsize() < self.max_queue_size:
                        await queue.put(message)
                        return True
                    logger.warning("Message queue full, dropping message")
                except Exception as queue_error:
                    logger.error(f"Failed to queue message: {queue_error}")

            # If we can't queue, clean up the connection
            await self._cleanup_connection(websocket, 1011, "Connection lost")
            return False

        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}", exc_info=True)
            return False

    async def broadcast(
        self,
        message: str,
        channel: str = "default",
        exclude: Optional[Set[WebSocket]] = None,
    ):
        """
        Broadcast a message to all clients in a channel with error handling.

        Args:
            message: The message to broadcast (will be JSON-serialized)
            channel: The target channel (default: "default")
            exclude: Set of WebSocket connections to exclude from broadcast
        """
        if channel not in self.active_connections:
            return

        if not isinstance(message, str):
            try:
                message = json.dumps(message)
            except (TypeError, ValueError) as e:
                logger.error(f"Failed to serialize broadcast message: {e}")
                return

        disconnected = set()
        exclude = exclude or set()

        # Get a snapshot of current connections to avoid modification during iteration
        connections = list(self.active_connections[channel] - exclude)

        for connection in connections:
            if connection in self.connection_data and connection not in exclude:
                success = await self.send_personal_message(
                    message, connection, queue_if_disconnected=False
                )
                if not success:
                    disconnected.add(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            await self._cleanup_connection(
                connection, 1011, "Connection lost during broadcast"
            )

    async def get_channel_stats(self, channel: str = "default") -> Dict[str, Any]:
        """
        Get detailed statistics for a channel.

        Args:
            channel: The channel to get stats for

        Returns:
            Dict containing channel statistics
        """
        if channel not in self.active_connections:
            return {
                "channel": channel,
                "active_clients": 0,
                "total_messages": 0,
                "active_since": None,
                "client_details": [],
            }

        now = time.time()
        clients = []
        total_messages = 0

        async with self.lock:
            for conn in list(self.active_connections.get(channel, [])):
                if conn in self.connection_data:
                    data = self.connection_data[conn]
                    client_info = {
                        "client_id": data["client_id"],
                        "connected_at": data["connected_at"],
                        "last_active": data["last_active"],
                        "inactive_for": now - data["last_active"],
                        "message_count": data.get("message_count", 0),
                        "is_alive": data.get("is_alive", False),
                        "queue_size": self.message_queues.get(
                            conn, asyncio.Queue()
                        ).qsize()
                        if conn in self.message_queues
                        else 0,
                    }
                    clients.append(client_info)
                    total_messages += client_info["message_count"]

        return {
            "channel": channel,
            "active_clients": len(clients),
            "total_messages": total_messages,
            "active_since": min((c["connected_at"] for c in clients), default=None)
            if clients
            else None,
            "client_details": clients,
            "timestamp": now,
        }

    def _check_rate_limit(self, websocket: WebSocket) -> bool:
        """Check if a client has exceeded rate limits"""
        if websocket not in self.connection_data:
            return False

        client_id = self.connection_data[websocket].get("client_id")
        if not client_id:
            return False

        now = time.time()
        window = 60  # 1 minute window for rate limiting
        max_messages = 100  # Max messages per window

        # Initialize rate limit tracking
        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = {
                "count": 0,
                "window_start": now,
                "last_warning": 0,
            }

        # Reset counter if window has passed
        client_limit = self.rate_limits[client_id]
        if now - client_limit["window_start"] > window:
            client_limit["count"] = 0
            client_limit["window_start"] = now

        # Check if limit exceeded
        client_limit["count"] += 1
        if client_limit["count"] > max_messages:
            # Only log warning once per minute to avoid log spam
            if now - client_limit["last_warning"] > 60:
                logger.warning(f"Rate limit exceeded for client {client_id}")
                client_limit["last_warning"] = now
            return False

        return True

    async def _start_heartbeat(self, websocket: WebSocket):
        """Start heartbeat pings to keep connection alive"""
        try:
            while (
                websocket in self.connection_data and websocket.client_state.value < 3
            ):  # 3 = CLOSED
                try:
                    # Send ping
                    await websocket.send_json(
                        {"type": "ping", "timestamp": time.time()}
                    )
                    self.connection_data[websocket]["last_ping"] = time.time()

                    # Wait for pong or timeout
                    try:
                        await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=self.heartbeat_interval / 2,
                        )
                        self.connection_data[websocket]["last_pong"] = time.time()
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Heartbeat timeout for client {self.connection_data[websocket].get('client_id')}"
                        )
                        break

                    # Sleep until next heartbeat
                    await asyncio.sleep(self.heartbeat_interval)

                except (WebSocketDisconnect, ConnectionError):
                    break
                except Exception as e:
                    logger.error(f"Error in heartbeat loop: {e}")
                    break

        finally:
            # Clean up if we exit the loop
            if websocket in self.connection_data:
                await self._cleanup_connection(websocket, 1011, "Heartbeat failed")

    async def _process_message_queue(self, websocket: WebSocket):
        """Process queued messages for a client"""
        if websocket not in self.message_queues:
            return

        queue = self.message_queues[websocket]

        try:
            while True:
                # Get next message (wait for up to 1 second)
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    if message is None:  # Sentinel value
                        break

                    # Try to send the message
                    try:
                        await websocket.send_text(message)
                        self.connection_data[websocket]["last_active"] = time.time()
                        self.connection_data[websocket]["message_count"] += 1
                    except (WebSocketDisconnect, ConnectionError):
                        # Re-queue the message if sending fails
                        await queue.put(message)
                        raise

                except asyncio.TimeoutError:
                    # Check if connection is still alive
                    if (
                        websocket not in self.connection_data
                        or websocket.client_state.value >= 3
                    ):
                        break
                    continue

        except (WebSocketDisconnect, ConnectionError):
            logger.info("Client disconnected during message queue processing")
        except Exception as e:
            logger.error(f"Error in message queue processing: {e}")
        finally:
            # Clean up the queue
            if websocket in self.message_queues:
                del self.message_queues[websocket]


# Global WebSocket manager instance with enhanced error handling
manager = ConnectionManager()

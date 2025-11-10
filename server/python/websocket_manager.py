"""
WebSocket Manager for real-time communication
"""
import asyncio
import json
import logging
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.connection_data: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, client_id: str, channel: str = "default"):
        """Handle new WebSocket connection."""
        await websocket.accept()
        
        async with self.lock:
            if channel not in self.active_connections:
                self.active_connections[channel] = set()
            self.active_connections[channel].add(websocket)
            self.connection_data[websocket] = {
                "client_id": client_id,
                "channel": channel,
                "connected_at": asyncio.get_event_loop().time(),
                "last_active": asyncio.get_event_loop().time()
            }
        
        logger.info(f"Client {client_id} connected to channel {channel}")
        return client_id
    
    async def disconnect(self, websocket: WebSocket):
        """Handle WebSocket disconnection."""
        if websocket in self.connection_data:
            client_data = self.connection_data[websocket]
            channel = client_data["channel"]
            
            async with self.lock:
                if channel in self.active_connections and websocket in self.active_connections[channel]:
                    self.active_connections[channel].remove(websocket)
                    if not self.active_connections[channel]:
                        del self.active_connections[channel]
                
                if websocket in self.connection_data:
                    del self.connection_data[websocket]
            
            logger.info(f"Client {client_data['client_id']} disconnected from channel {channel}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific client."""
        try:
            await websocket.send_text(message)
            self.connection_data[websocket]["last_active"] = asyncio.get_event_loop().time()
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            await self.disconnect(websocket)
    
    async def broadcast(self, message: str, channel: str = "default"):
        """Broadcast a message to all clients in a channel."""
        if channel not in self.active_connections:
            return
            
        disconnected = set()
        
        for connection in self.active_connections[channel]:
            try:
                await connection.send_text(message)
                self.connection_data[connection]["last_active"] = asyncio.get_event_loop().time()
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            await self.disconnect(connection)
    
    async def get_channel_stats(self, channel: str = "default") -> Dict[str, Any]:
        """Get statistics for a channel."""
        if channel not in self.active_connections:
            return {"clients": 0, "active_since": None}
            
        now = asyncio.get_event_loop().time()
        clients = []
        
        for conn in self.active_connections[channel]:
            if conn in self.connection_data:
                data = self.connection_data[conn]
                clients.append({
                    "client_id": data["client_id"],
                    "connected_at": data["connected_at"],
                    "last_active": data["last_active"],
                    "inactive_for": now - data["last_active"]
                })
        
        return {
            "clients": len(clients),
            "active_since": min([c["connected_at"] for c in clients]) if clients else None,
            "client_details": clients
        }

# Global WebSocket manager instance
manager = ConnectionManager()

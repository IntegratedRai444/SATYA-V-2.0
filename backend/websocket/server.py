import asyncio
import json
import logging
from typing import Dict, List, Optional
from fastapi import WebSocket, WebSocketDisconnect, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This should be moved to your config/settings
SECRET_KEY = "your-secret-key-here"  # TODO: Move to environment variable
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.subscriptions: Dict[str, List[str]] = {}  # channel: [connection_ids]

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            # Remove from all subscriptions
            for channel in list(self.subscriptions.keys()):
                if client_id in self.subscriptions[channel]:
                    self.subscriptions[channel].remove(client_id)
            logger.info(f"Client {client_id} disconnected")

    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: str, channel: str = "default"):
        if channel in self.subscriptions:
            for client_id in self.subscriptions[channel]:
                await self.send_personal_message(message, client_id)

    def subscribe(self, client_id: str, channel: str):
        if channel not in self.subscriptions:
            self.subscriptions[channel] = []
        if client_id not in self.subscriptions[channel]:
            self.subscriptions[channel].append(client_id)
            logger.info(f"Client {client_id} subscribed to {channel}")

    def unsubscribe(self, client_id: str, channel: str):
        if channel in self.subscriptions and client_id in self.subscriptions[channel]:
            self.subscriptions[channel].remove(client_id)
            logger.info(f"Client {client_id} unsubscribed from {channel}")

# Global WebSocket manager
manager = ConnectionManager()

# Authentication
def verify_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

# WebSocket endpoint
async def websocket_endpoint(websocket: WebSocket, token: str):
    # Authenticate the user
    payload = verify_token(token)
    if not payload:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    client_id = payload.get("sub")
    if not client_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await manager.connect(websocket, client_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_websocket_message(client_id, message)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON received from {client_id}")
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(client_id)

async def handle_websocket_message(sender_id: str, message: dict):
    message_type = message.get("type")
    
    if message_type == "subscribe":
        channel = message.get("channel")
        if channel:
            manager.subscribe(sender_id, channel)
            await manager.send_personal_message(
                json.dumps({
                    "type": "subscription_confirmation",
                    "channel": channel,
                    "status": "subscribed"
                }), 
                sender_id
            )
    
    elif message_type == "unsubscribe":
        channel = message.get("channel")
        if channel:
            manager.unsubscribe(sender_id, channel)
            
    elif message_type == "message":
        channel = message.get("channel", "default")
        content = message.get("content", {})
        await manager.broadcast(
            json.dumps({
                "type": "message",
                "from": sender_id,
                "channel": channel,
                "content": content,
                "timestamp": str(datetime.utcnow())
            }),
            channel
        )

# Utility function to send scan updates
async def send_scan_update(scan_id: str, status: str, progress: float = None, data: dict = None):
    message = {
        "type": "scan_update",
        "scan_id": scan_id,
        "status": status,
        "timestamp": str(datetime.utcnow())
    }
    if progress is not None:
        message["progress"] = progress
    if data:
        message["data"] = data
        
    channel = f"scan_{scan_id}"
    await manager.broadcast(json.dumps(message), channel)

# Example usage in your FastAPI app:
# from fastapi import FastAPI, WebSocket
# from .websocket.server import websocket_endpoint, manager
# 
# app = FastAPI()
# 
# @app.websocket("/ws/{token}")
# async def websocket_route(websocket: WebSocket, token: str):
#     await websocket_endpoint(websocket, token)
# 
# # Example of sending a scan update from your API routes:
# @app.post("/scans/{scan_id}/update")
# async def update_scan(scan_id: str):
#     # Your scan update logic here
#     await send_scan_update(scan_id, "processing", 0.5, {"step": "analyzing"})
#     return {"status": "update_sent"}

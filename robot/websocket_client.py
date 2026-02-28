import asyncio
import json
import logging
from datetime import datetime
from typing import Callable, Optional

import websockets

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSocketClient:
    def __init__(
        self,
        host: str = config.HOST,
        port: int = config.PORT,
        path: str = config.WS_ROBOT_PATH,
        on_message: Optional[Callable[[dict], None]] = None,
        on_connect: Optional[Callable[[], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
    ):
        self.host = host
        self.port = port
        self.path = path
        self.uri = f"ws://{host}:{port}{path}"
        
        self.on_message = on_message
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.is_connected = False
        self.reconnect_delay = 1
        self.max_reconnect_delay = 30
        self._running = False
        self._ping_interval = 10
        
    async def connect(self) -> bool:
        """Establish WebSocket connection"""
        try:
            self.ws = await websockets.connect(
                self.uri,
                ping_interval=self._ping_interval,
                ping_timeout=5,
            )
            self.is_connected = True
            self.reconnect_delay = 1
            logger.info(f"Connected to {self.uri}")
            
            if self.on_connect:
                self.on_connect()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close WebSocket connection"""
        self._running = False
        if self.ws:
            await self.ws.close()
            self.ws = None
        self.is_connected = False
        logger.info("Disconnected")
        
        if self.on_disconnect:
            self.on_disconnect()
    
    async def send(self, message: dict):
        """Send JSON message to server"""
        if not self.ws or not self.is_connected:
            logger.warning("Not connected, cannot send message")
            return False
            
        try:
            await self.ws.send(json.dumps(message))
            return True
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.is_connected = False
            return False
    
    async def send_focus_update(
        self,
        focus_state: str,
        confidence: float = 1.0,
        reason: str = "",
        reaction: str = "none"
    ):
        """Send focus update to server"""
        message = {
            "type": "focus_update",
            "source": "robot",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "focus_state": focus_state,
                "confidence": confidence,
                "reason": reason,
                "reaction": reaction
            }
        }
        return await self.send(message)
    
    async def send_notification(
        self,
        notification_type: str,
        message: str
    ):
        """Send notification to server"""
        msg = {
            "type": "notification",
            "source": "robot",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "notification_type": notification_type,
                "message": message
            }
        }
        return await self.send(msg)
    
    async def receive_loop(self):
        """Receive and process messages"""
        if not self.ws:
            return
            
        try:
            async for raw_message in self.ws:
                try:
                    message = json.loads(raw_message)
                    logger.info(f"Received: {message.get('type')} from {message.get('source')}")
                    
                    if self.on_message:
                        self.on_message(message)
                        
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {raw_message}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Connection closed by server")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            self.is_connected = False
    
    async def reconnect_loop(self):
        """Reconnect with exponential backoff"""
        while self._running:
            if not self.is_connected:
                logger.info(f"Attempting to reconnect in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)
                
                if await self.connect():
                    asyncio.create_task(self.receive_loop())
                else:
                    self.reconnect_delay = min(
                        self.reconnect_delay * 2,
                        self.max_reconnect_delay
                    )
            else:
                await asyncio.sleep(1)
    
    async def run(self):
        """Start the WebSocket client"""
        self._running = True
        
        if await self.connect():
            asyncio.create_task(self.receive_loop())
        
        await self.reconnect_loop()
    
    async def run_background(self):
        """Run client in background (for integration)"""
        self._running = True
        
        if await self.connect():
            asyncio.create_task(self.receive_loop())
            asyncio.create_task(self.reconnect_loop())
        else:
            asyncio.create_task(self.reconnect_loop())


async def create_client(
    on_message: Optional[Callable[[dict], None]] = None,
    on_connect: Optional[Callable[[], None]] = None,
    on_disconnect: Optional[Callable[[], None]] = None,
) -> WebSocketClient:
    """Factory function to create and return client"""
    client = WebSocketClient(
        on_message=on_message,
        on_connect=on_connect,
        on_disconnect=on_disconnect,
    )
    return client

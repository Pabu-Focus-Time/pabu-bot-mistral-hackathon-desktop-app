import asyncio
import logging
import signal
import sys
from datetime import datetime

import config
import simulation
import websocket_client
import focus_detector
import reactions
import audio_player

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FocusTimeRobot:
    """Main application for Reachy Mini Focus Time"""
    
    def __init__(self):
        self.ws_client: websocket_client.WebSocketClient = None
        self.focus_det: focus_detector.FocusDetector = None
        self.reaction_handler: reactions.ReactionsWithAudio = None
        self.audio: audio_player.AudioPlayer = None
        self.simulation: simulation.Simulation = None
        
        self._running = False
    
    async def setup(self):
        """Initialize all components"""
        logger.info(f"Starting Focus Time Robot (simulation={config.SIMULATION})")
        
        # Create simulation/mock interfaces
        camera, motors, speaker, sim = simulation.create_robot_interfaces(config.SIMULATION)
        self.simulation = sim
        
        # Create audio player
        self.audio = audio_player.create_audio_player(config.SIMULATION)
        
        # Create reactions handler
        self.reaction_handler = reactions.ReactionsWithAudio(motors, speaker)
        
        # Create focus detector
        self.focus_det = focus_detector.create_focus_detector(
            camera,
            simulation=config.SIMULATION,
            api_base=config.API_BASE
        )
        
        # Set up focus callback to send to server
        self.focus_det.set_focus_callback(self._on_focus_update)
        
        # Create WebSocket client
        self.ws_client = await websocket_client.create_client(
            on_message=self._on_server_message,
            on_connect=self._on_connect,
            on_disconnect=self._on_disconnect,
        )
        
        logger.info("Setup complete")
    
    def _on_connect(self):
        """Called when WebSocket connects"""
        logger.info("Connected to server!")
        
        # Send greeting + uh_oh
        if self.reaction_handler:
            asyncio.create_task(self.reaction_handler.greeting_with_voice(
                "Hi! I'm ready to help you focus!"
            ))
            asyncio.create_task(self.reaction_handler.uh_oh())
    
    def _on_disconnect(self):
        """Called when WebSocket disconnects"""
        logger.info("Disconnected from server")
    
    async def _on_focus_update(self, result: dict):
        """Handle focus state update from detector"""
        if not self.ws_client or not self.ws_client.is_connected:
            return
        
        focus_state = result.get("focus_state", "unknown")
        
        # Send to server
        await self.ws_client.send_focus_update(
            focus_state=focus_state,
            confidence=result.get("confidence", 0.0),
            reason=result.get("reason", ""),
            reaction="none"
        )
        
        # React based on focus state
        if focus_state == "distracted":
            await self.reaction_handler.warning_with_voice(
                "Hey! You should focus on your task!"
            )
        elif focus_state == "focused":
            await self.reaction_handler.focused()
    
    async def _on_server_message(self, message: dict):
        """Handle incoming message from server"""
        msg_type = message.get("type")
        payload = message.get("payload", {})
        
        logger.info(f"Received: {msg_type} from {message.get('source')}")
        
        if msg_type == "reaction":
            reaction = payload.get("reaction", "none")
            text = payload.get("text", "")
            
            if text:
                await self.reaction_handler.play_and_react(text, reaction)
            else:
                await self.reaction_handler.execute_reaction(reaction)
        
        elif msg_type == "notification":
            notif_type = payload.get("notification_type")
            text = payload.get("message", "")
            
            if text:
                await self.audio.speak_blocking(text)
        
        elif msg_type == "ping":
            await self.ws_client.send({"type": "pong", "source": "robot"})
    
    async def run(self):
        """Run the application"""
        self._running = True
        
        # Start WebSocket client
        await self.ws_client.run_background()
        
        # Start focus detector
        await self.focus_det.start()
        
        logger.info("Focus Time Robot running...")
        
        # Keep running
        while self._running:
            await asyncio.sleep(1)
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up...")
        self._running = False
        
        if self.focus_det:
            await self.focus_det.stop()
        
        if self.ws_client:
            await self.ws_client.disconnect()
        
        if self.audio:
            await self.audio.close()
        
        if self.simulation:
            self.simulation.cleanup()
        
        logger.info("Cleanup complete")


async def main():
    """Main entry point"""
    app = FocusTimeRobot()
    
    # Handle shutdown signals
    loop = asyncio.get_event_loop()
    
    def shutdown(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(app.cleanup())
    
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    
    try:
        await app.setup()
        await app.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        raise
    finally:
        await app.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

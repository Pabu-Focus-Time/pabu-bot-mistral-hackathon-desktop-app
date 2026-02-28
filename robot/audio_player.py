import logging
import base64
import asyncio
from typing import Optional, Union

import httpx

import config

logger = logging.getLogger(__name__)


class AudioPlayer:
    """Handles TTS audio playback via backend API"""
    
    def __init__(self, api_base: Optional[str] = None):
        self.api_base = api_base or config.API_BASE
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def synthesize(self, text: str) -> Optional[bytes]:
        """Generate TTS audio via ElevenLabs API (through backend)"""
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.api_base}/api/tts",
                json={"text": text}
            )
            
            if response.status_code == 200:
                data = response.json()
                audio_base64 = data.get("audio", "")
                audio_bytes = base64.b64decode(audio_base64)
                logger.info(f"Generated TTS audio: {len(audio_bytes)} bytes")
                return audio_bytes
            else:
                logger.error(f"TTS API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return None
    
    async def speak(self, text: str, play_immediately: bool = True) -> bool:
        """Synthesize and play TTS audio"""
        audio_bytes = await self.synthesize(text)
        
        if audio_bytes and play_immediately:
            return self.play(audio_bytes)
        
        return audio_bytes is not None
    
    def play(self, audio_bytes: bytes) -> bool:
        """Play audio bytes on speaker"""
        # This would be implemented differently:
        # - Real robot: play via Reachy's speaker
        # - Simulation: log to console
        try:
            import pygame
            from io import BytesIO
            
            if not pygame.get_init():
                pygame.mixer.init()
            
            sound = pygame.mixer.Sound(buffer=audio_bytes)
            sound.play()
            return True
        except ImportError:
            # Pygame not available - just log
            logger.info(f"[SIM] Would play audio: {len(audio_bytes)} bytes")
            logger.info(f"[SIM] Text: {text if 'text' in locals() else 'N/A'}")
            return True
        except Exception as e:
            logger.error(f"Audio playback error: {e}")
            return False
    
    async def speak_blocking(self, text: str):
        """Synthesize and play TTS, wait for completion"""
        audio_bytes = await self.synthesize(text)
        if audio_bytes:
            self.play(audio_bytes)


class MockAudioPlayer:
    """Mock audio player for simulation"""
    
    def __init__(self):
        logger.info("MockAudioPlayer initialized")
    
    async def synthesize(self, text: str) -> Optional[bytes]:
        """Mock TTS - return dummy audio"""
        logger.info(f"[SIM] Mock TTS: '{text}'")
        return b"MOCK_AUDIO"
    
    async def speak(self, text: str, play_immediately: bool = True) -> bool:
        """Mock speak"""
        logger.info(f"[SIM] Mock speak: '{text}'")
        return True
    
    def play(self, audio_bytes: bytes) -> bool:
        """Mock play"""
        logger.info(f"[SIM] Mock play: {len(audio_bytes)} bytes")
        return True
    
    async def speak_blocking(self, text: str):
        """Mock blocking speak"""
        logger.info(f"[SIM] Mock speak_blocking: '{text}'")
        await asyncio.sleep(0.5)
    
    async def close(self):
        """Close mock player"""
        pass


def create_audio_player(simulation: bool = None) -> Union[AudioPlayer, MockAudioPlayer]:
    """Factory function to create audio player"""
    if simulation is None:
        simulation = config.SIMULATION
    
    if simulation:
        logger.info("Creating mock audio player")
        return MockAudioPlayer()
    else:
        logger.info("Creating real audio player")
        return AudioPlayer()

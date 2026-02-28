import asyncio
import logging
import base64
from typing import Optional, Callable, Union
from datetime import datetime

import httpx

import config

logger = logging.getLogger(__name__)


class FocusDetector:
    """Captures camera frames and analyzes focus state via Mistral API"""
    
    def __init__(
        self,
        camera,
        api_base: Optional[str] = None,
        frame_interval: float = config.FRAME_INTERVAL,
        similarity_threshold: float = config.IMAGE_SIMILARITY_THRESHOLD,
    ):
        self.camera = camera
        self.api_base = api_base or config.API_BASE
        self.frame_interval = frame_interval
        self.similarity_threshold = similarity_threshold
        
        self._client: Optional[httpx.AsyncClient] = None
        self._last_frame: Optional[str] = None
        self._running = False
        self._on_focus_update: Optional[Callable[[dict], None]] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    def set_focus_callback(self, callback: Callable[[dict], None]):
        """Set callback for focus state updates"""
        self._on_focus_update = callback
    
    async def analyze_frame(self, frame: str) -> dict:
        """Analyze a frame using Mistral API through backend"""
        client = await self._get_client()
        
        try:
            response = await client.post(
                f"{self.api_base}/api/analyze",
                json={"image": frame}
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Focus analysis: {result.get('focus_state')} (confidence: {result.get('confidence')})")
                return result
            else:
                logger.error(f"Analysis API error: {response.status_code}")
                return {
                    "focus_state": "unknown",
                    "confidence": 0.0,
                    "reason": f"API error: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "focus_state": "unknown",
                "confidence": 0.0,
                "reason": str(e)
            }
    
    def is_similar_to_last(self, frame: str) -> bool:
        """Check if frame is similar to last frame"""
        if self._last_frame is None:
            return False
        
        try:
            similarity = self.camera.get_similarity(self._last_frame, frame)
            return similarity >= self.similarity_threshold
        except Exception:
            return False
    
    async def capture_and_analyze(self) -> dict:
        """Capture frame, check similarity, analyze if different"""
        try:
            frame = self.camera.capture_frame()
            
            if frame is None:
                logger.warning("Failed to capture frame")
                return {"focus_state": "unknown", "confidence": 0.0, "reason": "no_frame"}
            
            # Check similarity to skip redundant analysis
            if self.is_similar_to_last(frame):
                logger.debug("Frame similar to last, skipping analysis")
                return {"focus_state": "skipped", "confidence": 1.0, "reason": "similar_frame"}
            
            self._last_frame = frame
            
            # Analyze with Mistral
            result = await self.analyze_frame(frame)
            result["timestamp"] = datetime.utcnow().isoformat()
            
            # Notify callback
            if self._on_focus_update:
                asyncio.create_task(self._on_focus_update(result))
            
            return result
            
        except Exception as e:
            logger.error(f"Capture/analyze error: {e}")
            return {"focus_state": "unknown", "confidence": 0.0, "reason": str(e)}
    
    async def analysis_loop(self):
        """Background loop for periodic focus analysis"""
        logger.info(f"Starting focus detection loop (interval: {self.frame_interval}s)")
        
        while self._running:
            result = await self.capture_and_analyze()
            
            if result.get("focus_state") not in ["skipped", "unknown"]:
                # Send to server via callback if set
                logger.info(f"Focus state: {result.get('focus_state')} - {result.get('reason')}")
            
            await asyncio.sleep(self.frame_interval)
    
    async def start(self):
        """Start the focus detector"""
        self._running = True
        asyncio.create_task(self.analysis_loop())
    
    async def stop(self):
        """Stop the focus detector"""
        self._running = False
        if self._client:
            await self._client.aclose()
            self._client = None


class MockFocusDetector:
    """Mock focus detector for simulation"""
    
    def __init__(
        self,
        camera,
        api_base: Optional[str] = None,
        frame_interval: float = config.FRAME_INTERVAL,
        similarity_threshold: float = config.IMAGE_SIMILARITY_THRESHOLD,
    ):
        self.camera = camera
        self.frame_interval = frame_interval
        self.similarity_threshold = similarity_threshold
        self._running = False
        self._on_focus_update: Optional[Callable[[dict], None]] = None
        self._mock_state = "focused"
        logger.info("MockFocusDetector initialized")
    
    def set_focus_callback(self, callback: Callable[[dict], None]):
        """Set callback for focus state updates"""
        self._on_focus_update = callback
    
    async def capture_and_analyze(self) -> dict:
        """Mock analysis - cycle through states"""
        frame = self.camera.capture_frame()
        
        # Cycle through states for demo
        states = ["focused", "focused", "distracted", "focused", "unknown"]
        idx = hash(str(datetime.now().second)) % len(states)
        self._mock_state = states[idx]
        
        result = {
            "focus_state": self._mock_state,
            "confidence": 0.85,
            "reason": f"mock_{self._mock_state}",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"[SIM] Mock focus state: {result['focus_state']}")
        
        if self._on_focus_update:
            asyncio.create_task(self._on_focus_update(result))
        
        return result
    
    async def analysis_loop(self):
        """Mock analysis loop"""
        logger.info(f"[SIM] Starting mock focus detection (interval: {self.frame_interval}s)")
        
        while self._running:
            await self.capture_and_analyze()
            await asyncio.sleep(self.frame_interval)
    
    async def start(self):
        """Start mock detector"""
        self._running = True
        asyncio.create_task(self.analysis_loop())
    
    async def stop(self):
        """Stop mock detector"""
        self._running = False


def create_focus_detector(
    camera,
    simulation: bool = None,
    api_base: Optional[str] = None,
) -> Union[FocusDetector, MockFocusDetector]:
    """Factory function to create focus detector"""
    if simulation is None:
        simulation = config.SIMULATION
    
    if simulation:
        logger.info("Creating mock focus detector")
        return MockFocusDetector(camera, api_base)
    else:
        logger.info("Creating real focus detector")
        return FocusDetector(camera, api_base)

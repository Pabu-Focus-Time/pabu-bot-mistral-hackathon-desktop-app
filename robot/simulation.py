import logging
import time
import numpy as np
from typing import Optional, Tuple
from io import BytesIO
import base64

import config

logger = logging.getLogger(__name__)


class MockCamera:
    """Mock camera for testing without robot"""
    
    def __init__(self):
        self._frame_count = 0
        logger.info("MockCamera initialized (mock mode)")
    
    def capture_frame(self) -> Optional[str]:
        """Capture a mock frame, return as base64"""
        self._frame_count += 1
        
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        color_cycle = (self._frame_count % 10) * 25
        img_array[:, :] = [color_cycle, 150 - color_cycle, 100 + color_cycle]
        
        from PIL import Image
        img = Image.fromarray(img_array)
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        img_bytes = buffer.getvalue()
        
        return base64.b64encode(img_bytes).decode()
    
    def get_similarity(self, frame1: str, frame2: str) -> float:
        if frame1 == frame2:
            return 1.0
        return 0.8
    
    def release(self):
        logger.info("MockCamera released")


class ReachyCamera:
    """Real Reachy Mini camera interface"""
    
    def __init__(self, reachy):
        self._reachy = reachy
        self._last_frame = None
        logger.info("ReachyCamera initialized")
    
    def capture_frame(self) -> Optional[str]:
        """Capture frame from Reachy camera, return as base64"""
        try:
            # Use media.get_frame() for real Reachy, or return None for simulation
            frame = self._reachy.media.get_frame()
            if frame is None:
                logger.debug("No frame available (simulation may not have camera)")
                return None
            
            from PIL import Image
            img = Image.fromarray(frame)
            buffer = BytesIO()
            img.save(buffer, format='JPEG')
            img_bytes = buffer.getvalue()
            
            return base64.b64encode(img_bytes).decode()
        except AttributeError:
            # Simulation doesn't have camera
            logger.debug("Camera not available in simulation")
            return None
        except Exception as e:
            logger.error(f"Capture error: {e}")
            return None
    
    def get_similarity(self, frame1: str, frame2: str) -> float:
        if frame1 == frame2:
            return 1.0
        if not frame1 or not frame2:
            return 0.0
        h1 = hash(frame1[:1000]) if frame1 else 0
        h2 = hash(frame2[:1000]) if frame2 else 0
        if h1 == h2:
            return 0.95
        return 0.5
    
    def release(self):
        logger.info("ReachyCamera released")


class MockMotors:
    """Mock motors for testing"""
    
    def __init__(self):
        self._head_position = {"pitch": 0, "roll": 0, "yaw": 0}
        self._body_position = {"yaw": 0}
        self._antenna_left = 0
        self._antenna_right = 0
        logger.info("MockMotors initialized (mock mode)")
    
    def goto_head(self, pitch: float = 0, roll: float = 0, yaw: float = 0, duration: float = 1.0):
        pitch = max(config.HEAD_PITCH_MIN, min(config.HEAD_PITCH_MAX, pitch))
        roll = max(config.HEAD_PITCH_MIN, min(config.HEAD_PITCH_MAX, roll))
        yaw = max(config.HEAD_YAW_MIN, min(config.HEAD_YAW_MAX, yaw))
        
        self._head_position = {"pitch": pitch, "roll": roll, "yaw": yaw}
        logger.info(f"[MOCK] Head goto: pitch={pitch}, roll={roll}, yaw={yaw} (duration={duration}s)")
    
    def goto_body(self, yaw: float = 0, duration: float = 1.0):
        yaw = max(config.BODY_YAW_MIN, min(config.BODY_YAW_MAX, yaw))
        self._body_position = {"yaw": yaw}
        logger.info(f"[MOCK] Body goto: yaw={yaw} (duration={duration}s)")
    
    def move_antenna(self, antenna: str, position: float, duration: float = 0.5):
        if antenna == "left":
            self._antenna_left = position
            logger.info(f"[MOCK] Left antenna goto: {position} (duration={duration}s)")
        elif antenna == "right":
            self._antenna_right = position
            logger.info(f"[MOCK] Right antenna goto: {position} (duration={duration}s)")
    
    def get_position(self) -> dict:
        return {
            "head": self._head_position.copy(),
            "body": self._body_position.copy(),
            "antenna_left": self._antenna_left,
            "antenna_right": self._antenna_right
        }


class ReachyMotors:
    """Real Reachy Mini motors interface"""
    
    def __init__(self, reachy):
        self._reachy = reachy
        from reachy_mini.utils import create_head_pose
        self._create_head_pose = create_head_pose
        logger.info("ReachyMotors initialized")
    
    def goto_head(self, pitch: float = 0, roll: float = 0, yaw: float = 0, duration: float = 1.0):
        """Move head using Reachy SDK"""
        # Clamp to safety limits
        pitch = max(config.HEAD_PITCH_MIN, min(config.HEAD_PITCH_MAX, pitch))
        roll = max(config.HEAD_PITCH_MIN, min(config.HEAD_PITCH_MAX, roll))
        yaw = max(config.HEAD_YAW_MIN, min(config.HEAD_YAW_MAX, yaw))
        
        try:
            self._reachy.goto_target(
                head=self._create_head_pose(y=yaw, roll=roll, x=pitch, mm=False, degrees=True),
                duration=duration
            )
            logger.info(f"Head goto: pitch={pitch}, roll={roll}, yaw={yaw}")
        except Exception as e:
            logger.error(f"Head movement error: {e}")
    
    def goto_body(self, yaw: float = 0, duration: float = 1.0):
        """Rotate body"""
        yaw = max(config.BODY_YAW_MIN, min(config.BODY_YAW_MAX, yaw))
        try:
            self._reachy.goto_target(body=yaw, duration=duration)
            logger.info(f"Body goto: yaw={yaw}")
        except Exception as e:
            logger.error(f"Body movement error: {e}")
    
    def move_antenna(self, antenna: str, position: float, duration: float = 0.5):
        """Move antenna (convert degrees to radians)"""
        import math
        radians = math.radians(position)
        
        try:
            if antenna == "left":
                self._reachy.goto_target(antennas=[radians], duration=duration)
            elif antenna == "right":
                self._reachy.goto_target(antennas=[None, radians], duration=duration)
            logger.info(f"{antenna} antenna goto: {position}°")
        except Exception as e:
            logger.error(f"Antenna movement error: {e}")
    
    def get_position(self) -> dict:
        """Get current motor positions"""
        try:
            return {
                "head": {"pitch": 0, "roll": 0, "yaw": 0},  # Would need to read actual positions
                "body": {"yaw": 0},
                "antenna_left": 0,
                "antenna_right": 0
            }
        except:
            return {"head": {}, "body": {}, "antenna_left": 0, "antenna_right": 0}


class MockSpeaker:
    """Mock speaker for testing"""
    
    def __init__(self):
        logger.info("MockSpeaker initialized (mock mode)")
    
    def play_audio(self, audio_data: bytes):
        logger.info(f"[MOCK] Playing audio: {len(audio_data)} bytes")
    
    def play_tts(self, text: str):
        logger.info(f"[MOCK] Playing TTS: '{text}'")


class ReachySpeaker:
    """Real Reachy speaker (using pyttsx3 or similar)"""
    
    def __init__(self):
        self._engine = None
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            logger.info("ReachySpeaker initialized (pyttsx3)")
        except ImportError:
            logger.warning("pyttsx3 not available, using mock")
        except Exception as e:
            logger.warning(f"Speaker init error: {e}")
    
    def play_audio(self, audio_data: bytes):
        """Play audio bytes (would need audio player)"""
        logger.info(f"Playing audio: {len(audio_data)} bytes")
    
    def play_tts(self, text: str):
        """Play TTS using pyttsx3"""
        if self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS error: {e}")
        else:
            logger.info(f"[MOCK] TTS: '{text}'")


class Simulation:
    """Mock simulation controller"""
    
    def __init__(self):
        self.camera = MockCamera()
        self.motors = MockMotors()
        self.speaker = MockSpeaker()
        logger.info("Mock simulation initialized")
    
    def cleanup(self):
        self.camera.release()
        logger.info("Mock simulation cleanup complete")


class ReachySimulation:
    """Real Reachy Mini (or simulation via daemon)"""
    
    def __init__(self, reachy):
        self._reachy = reachy
        self.camera = ReachyCamera(reachy)
        self.motors = ReachyMotors(reachy)
        self.speaker = ReachySpeaker()
        logger.info("Reachy simulation initialized")
    
    def cleanup(self):
        self.camera.release()
        logger.info("Reachy cleanup complete")


def create_robot_interfaces(simulation: bool = None):
    """Factory function to create robot interfaces
    
    Returns: (camera, motors, speaker, controller)
    """
    if simulation is None:
        simulation = config.SIMULATION
    
    if simulation:
        logger.info("Creating mock interfaces")
        sim = Simulation()
        return sim.camera, sim.motors, sim.speaker, sim
    else:
        logger.info("Connecting to Reachy (or simulation daemon)...")
        try:
            from reachy_mini import ReachyMini
            reachy = ReachyMini()
            logger.info("Connected to Reachy!")
            sim = ReachySimulation(reachy)
            return sim.camera, sim.motors, sim.speaker, sim
        except ImportError:
            logger.warning("reachy-mini not installed, falling back to mock")
            sim = Simulation()
            return sim.camera, sim.motors, sim.speaker, sim
        except Exception as e:
            logger.error(f"Failed to connect to Reachy: {e}")
            logger.info("Falling back to mock interfaces")
            sim = Simulation()
            return sim.camera, sim.motors, sim.speaker, sim

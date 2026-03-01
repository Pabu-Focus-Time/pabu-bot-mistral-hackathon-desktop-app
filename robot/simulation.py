import logging
import math
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
        c = (self._frame_count % 10) * 25
        img_array[:, :] = [c % 256, (c * 3) % 256, (c * 7) % 256]
        
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
    
    @staticmethod
    def _manual_create_head_pose(x=0, y=0, z=0, roll=0, pitch=0, yaw=0, mm=False, degrees=True):
        """Fallback: build a 4x4 homogeneous matrix from roll/pitch/yaw.
        Uses ZYX (yaw-pitch-roll) rotation convention."""
        if degrees:
            roll = math.radians(roll)
            pitch = math.radians(pitch)
            yaw = math.radians(yaw)
        
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr               ],
        ])
        
        scale = 0.001 if mm else 1.0
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = [x * scale, y * scale, z * scale]
        return pose
    
    def __init__(self, reachy):
        self._reachy = reachy
        
        # Try multiple import paths for create_head_pose
        self._create_head_pose = None
        for import_path in [
            ("reachy_mini.utils", "create_head_pose"),
            ("reachy_mini", "create_head_pose"),
        ]:
            try:
                mod = __import__(import_path[0], fromlist=[import_path[1]])
                self._create_head_pose = getattr(mod, import_path[1])
                logger.info(f"Imported create_head_pose from {import_path[0]}")
                break
            except (ImportError, AttributeError) as e:
                logger.debug(f"create_head_pose not in {import_path[0]}: {e}")
        
        if self._create_head_pose is None:
            logger.warning("create_head_pose not found in SDK, using manual numpy fallback")
            self._create_head_pose = self._manual_create_head_pose
        
        # Enable motors so movements actually work on real hardware
        try:
            self._reachy.enable_motors()
            logger.info("Motors enabled successfully")
        except Exception as e:
            logger.error(f"Failed to enable motors: {e}")
        
        # Wake up to known init pose
        try:
            self._reachy.wake_up()
            logger.info("Robot woke up to init pose")
        except Exception as e:
            logger.warning(f"wake_up() failed (non-critical): {e}")
        
        logger.info("ReachyMotors initialized")
    
    def goto_head(self, pitch: float = 0, roll: float = 0, yaw: float = 0, duration: float = 1.0):
        """Move head using Reachy SDK.
        
        Args:
            pitch, roll, yaw: angles in degrees
            duration: movement duration in seconds
        
        SDK signature: create_head_pose(x, y, z, roll, pitch, yaw, mm, degrees)
        - x/y/z are POSITION offsets (meters), NOT rotation angles
        - roll/pitch/yaw are rotation angles
        """
        # Clamp to safety limits
        pitch = max(config.HEAD_PITCH_MIN, min(config.HEAD_PITCH_MAX, pitch))
        roll = max(config.HEAD_PITCH_MIN, min(config.HEAD_PITCH_MAX, roll))
        yaw = max(config.HEAD_YAW_MIN, min(config.HEAD_YAW_MAX, yaw))
        
        try:
            head_pose = self._create_head_pose(
                pitch=pitch, roll=roll, yaw=yaw, degrees=True
            )
            self._reachy.goto_target(
                head=head_pose,
                duration=duration,
                body_yaw=None,  # Don't reset body position on head movements
            )
            logger.info(f"Head goto: pitch={pitch}, roll={roll}, yaw={yaw}")
        except Exception as e:
            logger.error(f"Head movement error: {e}")
    
    def goto_body(self, yaw: float = 0, duration: float = 1.0):
        """Rotate body.
        
        Args:
            yaw: angle in degrees (clamped to safety limits)
            duration: movement duration in seconds
        
        SDK expects body_yaw in RADIANS.
        """
        yaw = max(config.BODY_YAW_MIN, min(config.BODY_YAW_MAX, yaw))
        try:
            self._reachy.goto_target(
                body_yaw=math.radians(yaw),
                duration=duration,
            )
            logger.info(f"Body goto: yaw={yaw}° ({math.radians(yaw):.3f} rad)")
        except Exception as e:
            logger.error(f"Body movement error: {e}")
    
    def move_antenna(self, antenna: str, position: float, duration: float = 0.5):
        """Move a single antenna (convert degrees to radians).
        
        SDK requires antennas=[right_rad, left_rad] — ALWAYS 2 floats.
        We read the current position of the other antenna to preserve it.
        """
        target_rad = math.radians(position)
        
        try:
            # Read current antenna positions: [right_rad, left_rad]
            current = self._reachy.get_present_antenna_joint_positions()
            current_right = current[0] if current else 0.0
            current_left = current[1] if current and len(current) > 1 else 0.0
            
            if antenna == "left":
                antennas = [current_right, target_rad]
            elif antenna == "right":
                antennas = [target_rad, current_left]
            else:
                logger.warning(f"Unknown antenna: {antenna}")
                return
            
            self._reachy.goto_target(
                antennas=antennas,
                duration=duration,
                body_yaw=None,  # Don't reset body position
            )
            logger.info(f"{antenna} antenna goto: {position}° ({target_rad:.3f} rad)")
        except Exception as e:
            logger.error(f"Antenna movement error: {e}")
    
    def get_position(self) -> dict:
        """Get current motor positions from real hardware."""
        try:
            # get_current_joint_positions() returns (head[7], antennas[2]) in radians
            head_joints, antenna_joints = self._reachy.get_current_joint_positions()
            # antenna order: [right_rad, left_rad]
            antennas = self._reachy.get_present_antenna_joint_positions()
            right_rad = antennas[0] if antennas else 0.0
            left_rad = antennas[1] if antennas and len(antennas) > 1 else 0.0
            return {
                "head": {"pitch": 0, "roll": 0, "yaw": 0},  # TODO: extract from head_joints 4x4 matrix
                "body": {"yaw": 0},  # TODO: extract from head_joints
                "antenna_left": math.degrees(left_rad),
                "antenna_right": math.degrees(right_rad),
            }
        except Exception as e:
            logger.debug(f"Could not read positions: {e}")
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
        try:
            self._reachy.disable_motors()
            logger.info("Motors disabled")
        except Exception as e:
            logger.warning(f"Failed to disable motors on cleanup: {e}")
        logger.info("Reachy cleanup complete")


def _ensure_system_gi():
    """Make the 'gi' module importable even inside a venv.
    
    gi (GObject Introspection) is a system package installed via apt
    (python3-gi). It cannot be pip-installed. If the venv doesn't have
    system site-packages enabled, we add the system path manually.
    """
    try:
        import gi  # noqa: F401
        return True
    except ImportError:
        pass
    
    import sys
    import glob
    # Common paths for system python packages on Debian/Ubuntu (Reachy OS)
    candidates = sorted(glob.glob("/usr/lib/python3*/dist-packages"), reverse=True)
    candidates.append("/usr/lib/python3/dist-packages")
    
    for path in candidates:
        if path not in sys.path:
            sys.path.insert(0, path)
            logger.debug(f"Added {path} to sys.path")
    
    try:
        import gi  # noqa: F401
        logger.info(f"Found system gi module via added path ({gi.__file__})")
        return True
    except ImportError:
        logger.warning(
            "gi module not found. GStreamer camera will not work. "
            "Fix: edit venv/pyvenv.cfg → include-system-site-packages = true"
        )
        return False


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
        
        # Step 1: check SDK is importable
        try:
            from reachy_mini import ReachyMini
        except ImportError:
            logger.warning("reachy-mini SDK not installed, falling back to mock")
            sim = Simulation()
            return sim.camera, sim.motors, sim.speaker, sim
        
        # Step 2: ensure gi (GObject) is importable — needed for GStreamer camera
        _ensure_system_gi()
        
        # Step 3: connect and build interfaces
        try:
            reachy = ReachyMini()
            logger.info("Connected to Reachy!")
            sim = ReachySimulation(reachy)
            return sim.camera, sim.motors, sim.speaker, sim
        except Exception as e:
            logger.error(f"Failed to initialize Reachy: {e}", exc_info=True)
            logger.info("Falling back to mock interfaces")
            sim = Simulation()
            return sim.camera, sim.motors, sim.speaker, sim

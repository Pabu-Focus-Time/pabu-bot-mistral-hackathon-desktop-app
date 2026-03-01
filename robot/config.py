import os
from pathlib import Path

# Connection settings
HOST = os.getenv("REACHY_HOST", "localhost")
PORT = int(os.getenv("REACHY_PORT", "9800"))
WS_DESKTOP_PATH = "/ws/desktop"
WS_ROBOT_PATH = "/ws/robot"

# API settings
API_BASE = f"http://{HOST}:{PORT}"

# Mode settings
SIMULATION = os.getenv("SIMULATION_MODE", "true").lower() == "true"

# ElevenLabs Conversational AI
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID", "agent_9801kjk2p0kkf5wsrq1cc2byvyqx")
VOICE_AGENT_ENABLED = os.getenv("VOICE_AGENT_ENABLED", "false").lower() == "true"

# Focus detection settings
FOCUS_DETECTOR_ENABLED = os.getenv("FOCUS_DETECTOR_ENABLED", "false").lower() == "true"
FRAME_INTERVAL = float(os.getenv("FRAME_INTERVAL", "3.0"))
IMAGE_SIMILARITY_THRESHOLD = float(os.getenv("IMAGE_SIMILARITY_THRESHOLD", "0.95"))

# Reachy safety limits (from AGENTS.md)
HEAD_PITCH_MIN = -40
HEAD_PITCH_MAX = 40
HEAD_YAW_MIN = -180
HEAD_YAW_MAX = 180
BODY_YAW_MIN = -160
BODY_YAW_MAX = 160

# Project paths
PROJECT_ROOT = Path(__file__).parent

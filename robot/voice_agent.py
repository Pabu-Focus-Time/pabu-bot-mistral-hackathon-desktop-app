import logging
import threading
from typing import Optional, Callable

import config

logger = logging.getLogger(__name__)

AGENT_ID = config.ELEVENLABS_AGENT_ID


class VoiceAgent:
    """ElevenLabs Conversational AI agent for the robot."""

    def __init__(
        self,
        on_agent_response: Optional[Callable[[str], None]] = None,
        on_user_transcript: Optional[Callable[[str], None]] = None,
    ):
        self._conversation = None
        self._thread: Optional[threading.Thread] = None
        self._on_agent_response = on_agent_response
        self._on_user_transcript = on_user_transcript
        self.is_running = False

    def start(self):
        """Start a conversational AI session in a background thread."""
        try:
            from elevenlabs.client import ElevenLabs
            from elevenlabs.conversational_ai.conversation import Conversation
            from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
        except ImportError:
            logger.warning("elevenlabs package not installed — voice agent disabled")
            return

        api_key = config.ELEVENLABS_API_KEY or None
        client = ElevenLabs(api_key=api_key)

        self._conversation = Conversation(
            client,
            agent_id=AGENT_ID,
            requires_auth=False,
            audio_interface=DefaultAudioInterface(),
            callback_agent_response=self._handle_agent_response,
            callback_agent_response_correction=self._handle_correction,
            callback_user_transcript=self._handle_user_transcript,
        )

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.is_running = True
        logger.info(f"Voice agent started (agent_id={AGENT_ID})")

    def _run(self):
        try:
            self._conversation.start_session()
            self._conversation.wait_for_session_end()
        except Exception as e:
            logger.error(f"Voice agent session error: {e}")
        finally:
            self.is_running = False
            logger.info("Voice agent session ended")

    def _handle_agent_response(self, response: str):
        logger.info(f"Agent says: {response}")
        if self._on_agent_response:
            self._on_agent_response(response)

    def _handle_correction(self, original: str, corrected: str):
        logger.info(f"Agent correction: {original} -> {corrected}")

    def _handle_user_transcript(self, transcript: str):
        logger.info(f"User said: {transcript}")
        if self._on_user_transcript:
            self._on_user_transcript(transcript)

    def send_context(self, text: str):
        """Send a contextual update to the agent (non-interrupting)."""
        if self._conversation and self.is_running:
            try:
                self._conversation.send_contextual_update(text)
                logger.info(f"Sent context to agent: {text[:80]}")
            except Exception as e:
                logger.error(f"Failed to send context: {e}")

    def stop(self):
        """End the conversation session."""
        if self._conversation and self.is_running:
            try:
                self._conversation.end_session()
            except Exception as e:
                logger.error(f"Error ending voice agent session: {e}")
        self.is_running = False
        logger.info("Voice agent stopped")


class MockVoiceAgent:
    """Mock voice agent for simulation without API key."""

    def __init__(self, **kwargs):
        self.is_running = False
        logger.info("[SIM] MockVoiceAgent initialized")

    def start(self):
        self.is_running = True
        logger.info("[SIM] Mock voice agent started")

    def send_context(self, text: str):
        logger.info(f"[SIM] Mock context update: {text[:80]}")

    def stop(self):
        self.is_running = False
        logger.info("[SIM] Mock voice agent stopped")


def create_voice_agent(
    enabled: bool = None,
    on_agent_response: Optional[Callable[[str], None]] = None,
    on_user_transcript: Optional[Callable[[str], None]] = None,
):
    """Factory: returns real agent if enabled, mock otherwise."""
    if enabled is None:
        enabled = config.VOICE_AGENT_ENABLED

    if not enabled:
        return MockVoiceAgent()

    return VoiceAgent(
        on_agent_response=on_agent_response,
        on_user_transcript=on_user_transcript,
    )

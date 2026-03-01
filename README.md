# Pabu Bot — Focus Time Desktop App

An AI-powered focus detection system. An Electron desktop app captures screenshots, a FastAPI backend analyzes them via Mistral Vision, and a Reachy Mini robot reacts with head movements and antenna animations. Includes an ElevenLabs conversational AI voice agent you can talk to.

## Prerequisites

- Python 3.10+
- Node.js 18+ & npm
- **macOS only (for PyAudio):** `brew install portaudio`

## Quick Start

You need three terminals running in order: **backend → robot → frontend**.

### 1. Backend Server

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export MISTRAL_API_KEY="your-key"
export ELEVENLABS_API_KEY="your-key"

python server.py
```

Runs on `http://localhost:9800`. The backend routes WebSocket messages between the desktop app and the robot, and proxies Mistral Vision / ElevenLabs TTS API calls.

### 2. Robot + Voice Agent

```bash
cd robot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

export ELEVENLABS_API_KEY="your-key"
```

**Simulation mode** (no physical robot — voice agent still uses your real mic and speakers):

```bash
SIMULATION_MODE=true VOICE_AGENT_ENABLED=true python app.py
```

**Real Reachy Mini** (connects to the robot hardware):

```bash
SIMULATION_MODE=false VOICE_AGENT_ENABLED=true python app.py
```

When the robot starts and connects to the backend, the ElevenLabs voice agent will automatically begin a conversation session. You'll hear the agent speak and can talk back through your microphone.

#### Environment variables

| Variable | Default | Description |
|---|---|---|
| `SIMULATION_MODE` | `true` | `true` = mock motors/camera, `false` = real Reachy Mini |
| `VOICE_AGENT_ENABLED` | `false` | `true` = start ElevenLabs voice agent on the robot |
| `ELEVENLABS_API_KEY` | *(empty)* | Your ElevenLabs API key |
| `ELEVENLABS_AGENT_ID` | `agent_9801kjk2p0kkf5wsrq1cc2byvyqx` | ElevenLabs agent ID |
| `FRAME_INTERVAL` | `3.0` | Seconds between focus detection frames |

### 3. Frontend (Desktop App)

```bash
cd frontend
npm install
npm start
```

Launches the Electron app. The app captures screenshots for focus detection and shows real-time focus state from both the desktop and the robot.

The frontend also has a **"Talk to Pabu"** button that starts a separate ElevenLabs voice session through the browser. Use this as an alternative to the robot-side voice agent (don't run both at the same time — they share the microphone).

**macOS:** Grant **Screen Recording** and **Microphone** permissions in **System Settings → Privacy & Security** for full functionality.

## Voice Agent

The ElevenLabs conversational AI agent (`agent_9801kjk2p0kkf5wsrq1cc2byvyqx`) can be reached through two paths:

| Path | How | When to use |
|---|---|---|
| **Robot (Python SDK)** | Set `VOICE_AGENT_ENABLED=true` when running the robot | Production — voice runs directly through PyAudio on the machine connected to the robot |
| **Frontend (React SDK)** | Click "Talk to Pabu" in the Electron app | Development / debugging — voice runs through the browser |

Both paths connect to the same ElevenLabs agent. The robot-side agent sends focus context updates to the agent so it can respond to changes in user attention. When the agent speaks, the robot nods. When the user speaks, the robot tilts its head.

If you use the frontend voice path, agent responses and user transcripts are relayed to the robot over WebSocket so the robot still reacts physically.

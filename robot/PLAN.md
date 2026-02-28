Plan: Reachy Mini Focus Time App

Architecture

Laptop (localhost:9800)
├── FastAPI Backend (WebSocket server + Mistral + ElevenLabs)
├── Electron Desktop App
└── Reachy Mini (via WebSocket) - Simulation mode (testing) - Real robot mode (WiFi/USB)
Message Protocol

{
"type": "focus_update" | "notification" | "reaction",
"source": "desktop" | "robot",
"payload": {
"focus_state": "focused" | "distracted" | "unknown",
"confidence": 0.85,
"reason": "minecraft_detected" | "phone_detected" | "away",
"reaction": "none" | "warning" | "scold"
}
}
Implementation Phases

Phase 1: WebSocket Server + Basic Connection

Update backend/server.py with WebSocket endpoints (/ws/desktop, /ws/robot)
Create robot WebSocket client
Implement heartbeat ping/pong
Test message passing
Phase 2: Focus Detection (Simulation)

Mock camera feed with image similarity check
Connect to Mistral API via FastAPI
Send focus updates every 3 seconds (skip similar frames)
Phase 3: Robot Reactions

Head movements: look_at, shake_head, nod, tilt_head
Antenna movements: wave_left_right (warning), wave_up (greeting)
ElevenLabs TTS via FastAPI → play on robot
Phase 4: Desktop Integration

Screenshot capture → send to backend → Mistral analysis
Sync focus state via WebSocket
Phase 5: Real Robot Mode

Add SIMULATION_MODE=false flag
Replace mock with real camera/motors
Files to Create

robot/
├── PLAN.md
├── config.py # Settings (host, port, SIMULATION_MODE)
├── websocket_client.py
├── focus_detector.py
├── reactions.py
├── audio_player.py
├── simulation.py
└── app.py
Environment Variables

MISTRAL_API_KEY
ELEVENLABS_API_KEY
SIMULATION_MODE=true (default)
REACHY_HOST=localhost
REACHY_PORT=9800

import os
import json
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import httpx
from PIL import Image
import io
import base64

# Configuration
PORT = int(os.getenv("PORT", "9800"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "GkJlvmrgptESTkEGdgKjvQziz8aAWvCq")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# Active WebSocket connections
desktop_connections: list[WebSocket] = []
robot_connections: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting Focus Time Server on port {PORT}")
    yield
    print("Shutting down server")


app = FastAPI(lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# WebSocket endpoints
@app.websocket("/ws/desktop")
async def websocket_desktop(websocket: WebSocket):
    await websocket.accept()
    desktop_connections.append(websocket)
    print(f"Desktop connected. Total connections: {len(desktop_connections)}")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            message["source"] = "desktop"

            # Forward to robot connections
            for robot_ws in robot_connections:
                await robot_ws.send_json(message)

    except WebSocketDisconnect:
        desktop_connections.remove(websocket)
        print(f"Desktop disconnected. Remaining: {len(desktop_connections)}")


@app.websocket("/ws/robot")
async def websocket_robot(websocket: WebSocket):
    await websocket.accept()
    robot_connections.append(websocket)
    print(f"Robot connected. Total connections: {len(robot_connections)}")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            message["source"] = "robot"

            # Forward to desktop connections
            for desktop_ws in desktop_connections:
                await desktop_ws.send_json(message)

    except WebSocketDisconnect:
        robot_connections.remove(websocket)
        print(f"Robot disconnected. Remaining: {len(robot_connections)}")


# REST Endpoints
@app.get("/")
async def root():
    return {"message": "Focus Time Server", "port": PORT}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "desktop_connections": len(desktop_connections),
        "robot_connections": len(robot_connections),
    }


@app.post("/api/analyze")
async def analyze_image(image_data: dict):
    """Analyze an image for focus state using Mistral"""
    if not MISTRAL_API_KEY:
        return {
            "focus_state": "unknown",
            "confidence": 0.0,
            "reason": "MISTRAL_API_KEY not configured",
        }

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(image_data.get("image", ""))
    except Exception:
        return {
            "focus_state": "unknown",
            "confidence": 0.0,
            "reason": "Invalid image data",
        }

    # Call Mistral API (vision model)
    async with httpx.AsyncClient() as client:
        try:
            # Encode image as base64 data URL
            image_b64 = base64.b64encode(image_bytes).decode()
            image_url = f"data:image/png;base64,{image_b64}"

            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "pixtral-12b-2409",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": image_url}},
                                {
                                    "type": "text",
                                    "text": 'Analyze this screenshot. Is the user focused on productive work or distracted? Consider: Are they using productive apps (coding, writing, research) or distracting apps (games, social media, YouTube videos)? Respond ONLY with valid JSON: {"focus_state": "focused|distracted|unknown", "confidence": 0.0-1.0, "reason": "brief explanation"}',
                                },
                            ],
                        }
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=60.0,
            )

            if response.status_code == 200:
                result = response.json()
                content = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                try:
                    return json.loads(content)
                except:
                    return {
                        "focus_state": "unknown",
                        "confidence": 0.0,
                        "reason": "Failed to parse response",
                    }
            else:
                return {
                    "focus_state": "unknown",
                    "confidence": 0.0,
                    "reason": f"API error: {response.status_code}",
                }

        except Exception as e:
            return {
                "focus_state": "unknown",
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
            }


@app.post("/api/tts")
async def text_to_speech(text_data: dict):
    """Generate TTS using ElevenLabs"""
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ELEVENLABS_API_KEY not configured")

    text = text_data.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")

    # Call ElevenLabs API
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.elevenlabs.io/v1/text-to-speech/pNInz6obpgDQGcFmaJgB",
                headers={
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY,
                },
                json={
                    "text": text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                audio_base64 = base64.b64encode(response.content).decode()
                return {"audio": audio_base64, "format": "mpeg"}
            else:
                raise HTTPException(
                    status_code=response.status_code, detail="ElevenLabs API error"
                )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")


@app.post("/api/broadcast")
async def broadcast_message(message: dict):
    """Broadcast a message to all connected clients"""
    msg_type = message.get("type", "notification")
    payload = message.get("payload", {})

    full_message = {
        "type": msg_type,
        "source": "server",
        "timestamp": datetime.utcnow().isoformat(),
        "payload": payload,
    }

    # Send to all desktop connections
    for desktop_ws in desktop_connections:
        await desktop_ws.send_json(full_message)

    # Send to all robot connections
    for robot_ws in robot_connections:
        await robot_ws.send_json(full_message)

    return {
        "status": "broadcasted",
        "recipients": len(desktop_connections) + len(robot_connections),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)

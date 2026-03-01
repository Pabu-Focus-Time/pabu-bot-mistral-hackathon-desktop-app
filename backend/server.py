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
from pydantic import BaseModel

import httpx
from PIL import Image
import io
import base64
import boto3
import logging

logger = logging.getLogger(__name__)

# Configuration
PORT = int(os.getenv("PORT", "9800"))
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "GkJlvmrgptESTkEGdgKjvQziz8aAWvCq")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")

# AWS Bedrock Configuration
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-west-2")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "mistral.mistral-large-2407-v1:0")

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


@app.post("/api/analyze-multi-agent")
async def analyze_multi_agent(data: dict):
    """
    Combined analysis from multiple agents:
    - Vision Agent: screenshot analysis
    - Window Agent: active window data
    - Activity Agent: keyboard/mouse patterns
    
    Returns synthesized focus decision considering all signals.
    """
    vision_analysis = data.get("vision_analysis", {})
    window_data = data.get("window_data")
    activity_data = data.get("activity_data")
    task_context = data.get("task_context", {})
    
    # Build context for synthesis
    context_parts = []
    
    # Vision context
    if vision_analysis:
        focus_state = vision_analysis.get("focus_state", "unknown")
        confidence = vision_analysis.get("confidence", 0)
        vision_reason = vision_analysis.get("reason", "")
        context_parts.append(
            f"Vision Analysis: User appears {focus_state} ({confidence*100:.0f}% confidence). {vision_reason}"
        )
    
    # Window context
    if window_data:
        app_name = window_data.get("app_name", "Unknown")
        window_title = window_data.get("window_title", "")
        context = f"Active window: {app_name}"
        if window_title:
            context += f" - {window_title}"
        context_parts.append(context)
    
    # Activity context
    if activity_data:
        idle = activity_data.get("idle_seconds", 0)
        switches = activity_data.get("window_switch_count", 0)
        keys = activity_data.get("keypress_count", 0)
        context_parts.append(
            f"Activity: {idle}s idle, {switches} window switches, {keys} keypresses in last interval"
        )
    
    # Task context
    if task_context:
        task_name = task_context.get("task_name", "")
        current_todo = task_context.get("current_todo", "")
        if task_name:
            context_parts.append(f"Task: {task_name}")
        if current_todo:
            context_parts.append(f"Working on: {current_todo}")
    
    # If we have Mistral, do a synthesis
    if MISTRAL_API_KEY:
        async with httpx.AsyncClient() as client:
            try:
                prompt = f"""Analyze the user's focus state based on these signals:
{chr(10).join(f"- {p}" for p in context_parts)}

Considering ALL signals above, determine if the user is:
- "focused": Actively working on the task
- "distracted": Not working on the task (browsing, social media, etc.)
- "unknown": Cannot determine

Respond with ONLY a JSON object: {{"focus_state": "focused|distracted|unknown", "confidence": 0.0-1.0, "reason": "brief explanation"}}"""

                response = await client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {MISTRAL_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "mistral-small-latest",
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                    },
                    timeout=10.0,
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                    try:
                        synthesis = json.loads(content)
                        return {
                            "focus_state": synthesis.get("focus_state", "unknown"),
                            "confidence": synthesis.get("confidence", 0),
                            "reason": synthesis.get("reason", ""),
                            "vision_analysis": vision_analysis,
                            "window_data": window_data,
                            "activity_data": activity_data,
                        }
                    except:
                        pass
            except Exception as e:
                print(f"Multi-agent synthesis error: {e}")
    
    # Fallback: simple heuristic
    focus_state = "unknown"
    confidence = 0.3
    reason = "Insufficient data for analysis"
    
    # Simple rule-based fallback
    if window_data:
        app_name = window_data.get("app_name", "").lower()
        distracting_apps = ["safari", "chrome", "twitter", "facebook", "youtube", "reddit", "tiktok", "instagram"]
        productive_apps = ["vscode", "xcode", "terminal", "vim", "sublime", "atom", "intellij", "pycharm"]
        
        if any(app in app_name for app in distracting_apps):
            focus_state = "distracted"
            confidence = 0.5
            reason = f"Using potentially distracting app: {window_data.get('app_name')}"
        elif any(app in app_name for app in productive_apps):
            focus_state = "focused"
            confidence = 0.6
            reason = f"Using productive app: {window_data.get('app_name')}"
    
    if activity_data:
        if activity_data.get("idle_seconds", 0) > 60:
            focus_state = "unknown"
            confidence = 0.7
            reason = "User appears idle"
    
    return {
        "focus_state": focus_state,
        "confidence": confidence,
        "reason": reason,
        "vision_analysis": vision_analysis,
        "window_data": window_data,
        "activity_data": activity_data,
    }


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


# --- Todo Generation Agent (AWS Bedrock + Mistral Large) ---

TODO_SYSTEM_PROMPT = """You are an expert productivity and task-planning AI agent. Your job is to take a task name and optional description, then generate an optimal, actionable todo list to complete that task.

Rules:
1. Break the task into 3-8 concrete, actionable steps.
2. Order steps by logical dependency — what must be done first comes first.
3. Each step should be small enough to complete in one focused session (15-60 min).
4. Use clear, imperative language (e.g., "Set up project repository", not "Setting up").
5. Include a brief description for each step explaining what specifically to do.
6. Consider: research/planning steps first, then implementation, then testing/review.
7. Be specific to the task — don't give generic advice.

Respond with ONLY a valid JSON object in this exact format:
{
  "todos": [
    {"title": "Step title", "description": "Brief description of what to do"},
    {"title": "Step title", "description": "Brief description of what to do"}
  ]
}"""


class GenerateTodosRequest(BaseModel):
    task_name: str
    task_description: str = ""


async def call_bedrock_mistral(prompt: str, system_prompt: str) -> Optional[dict]:
    """Call Mistral Large via AWS Bedrock using boto3."""
    try:
        bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=BEDROCK_REGION,
        )

        body = json.dumps(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "max_tokens": 2048,
                "temperature": 0.7,
            }
        )

        response = bedrock_client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )

        raw_body = response["body"].read()
        result = json.loads(raw_body)
        content = (
            result.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

        if not content:
            logger.warning("Bedrock returned empty content")
            return None

        # Try to parse the content as JSON directly
        try:
            parsed = json.loads(content)
            logger.info("Bedrock Mistral Large responded successfully")
            return parsed
        except json.JSONDecodeError:
            # The model may have returned JSON wrapped in markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', content)
            if json_match:
                parsed = json.loads(json_match.group(1).strip())
                logger.info("Bedrock Mistral Large responded (extracted from code block)")
                return parsed
            # Try to find raw JSON object in the text
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                parsed = json.loads(json_match.group(0))
                logger.info("Bedrock Mistral Large responded (extracted JSON object)")
                return parsed
            logger.warning(f"Bedrock returned non-JSON content: {content[:200]}")
            return None
    except Exception as e:
        logger.warning(f"Bedrock call failed: {e}")

    return None


async def call_mistral_api(prompt: str, system_prompt: str) -> Optional[dict]:
    """Fallback: Call Mistral API directly."""
    if not MISTRAL_API_KEY:
        return None

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-large-latest",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "response_format": {"type": "json_object"},
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                content = (
                    result.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
                return json.loads(content)
            else:
                logger.warning(
                    f"Mistral API returned {response.status_code}: {response.text}"
                )
        except Exception as e:
            logger.warning(f"Mistral API call failed: {e}")

    return None


@app.post("/api/generate-todos")
async def generate_todos(request: GenerateTodosRequest):
    """
    AI agent that generates an optimal todo list for a given task.
    Uses AWS Bedrock with Mistral Large, with Mistral API as fallback.
    """
    task_name = request.task_name.strip()
    task_description = request.task_description.strip()

    if not task_name:
        raise HTTPException(status_code=400, detail="Task name is required")

    # Build the user prompt
    prompt = f"Task: {task_name}"
    if task_description:
        prompt += f"\nDescription: {task_description}"
    prompt += "\n\nGenerate an optimal, ordered todo list to complete this task."

    # Try Bedrock first, then Mistral API fallback
    result = await call_bedrock_mistral(prompt, TODO_SYSTEM_PROMPT)

    if result is None:
        logger.info("Bedrock unavailable, falling back to Mistral API")
        result = await call_mistral_api(prompt, TODO_SYSTEM_PROMPT)

    if result is None:
        # Final fallback: generate basic todos locally
        logger.info("All AI services unavailable, using local fallback")
        result = {
            "todos": [
                {"title": "Research", "description": f"Research requirements and resources for: {task_name}"},
                {"title": "Plan", "description": "Create a detailed plan and outline"},
                {"title": "Implement", "description": "Implement the core functionality"},
                {"title": "Test", "description": "Test and verify the implementation"},
                {"title": "Refine", "description": "Refine, polish, and document the result"},
            ]
        }

    # Validate response structure
    todos = result.get("todos", [])
    if not isinstance(todos, list) or len(todos) == 0:
        todos = [
            {"title": "Research", "description": f"Research: {task_name}"},
            {"title": "Plan", "description": "Create a plan"},
            {"title": "Implement", "description": "Build the core work"},
            {"title": "Review", "description": "Review and finalize"},
        ]

    # Ensure each todo has required fields
    validated_todos = []
    for todo in todos:
        if isinstance(todo, dict) and "title" in todo:
            validated_todos.append(
                {
                    "title": todo.get("title", "Untitled"),
                    "description": todo.get("description", ""),
                }
            )

    return {"todos": validated_todos}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)

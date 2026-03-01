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

from dino_service import DinoService
from chains import FocusChains

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

# Singletons (initialized in lifespan)
dino_service = DinoService()
focus_chains = FocusChains()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting Focus Time Server on port {PORT}")

    # Initialize DINOv3 image similarity service
    try:
        print("Loading DINOv3 model...")
        dino_service.load_model()
        print(f"DINOv3 model loaded on {dino_service.device}")
    except Exception as e:
        print(f"WARNING: DINOv3 model failed to load: {e}")
        print("Smart analysis will fall back to always calling LLM")

    # Initialize LangChain chains
    try:
        print("Initializing LangChain chains...")
        focus_chains.initialize()
        print("LangChain chains ready")
    except Exception as e:
        print(f"WARNING: LangChain chains failed to initialize: {e}")
        print("Smart analysis will use rule-based fallback")

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

    status_msg = {
        "type": "robot_status",
        "source": "server",
        "timestamp": datetime.utcnow().isoformat(),
        "payload": {"connected": True, "robot_count": len(robot_connections)},
    }
    for desktop_ws in desktop_connections:
        try:
            await desktop_ws.send_json(status_msg)
        except Exception:
            pass

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

        status_msg = {
            "type": "robot_status",
            "source": "server",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {"connected": len(robot_connections) > 0, "robot_count": len(robot_connections)},
        }
        for desktop_ws in desktop_connections:
            try:
                await desktop_ws.send_json(status_msg)
            except Exception:
                pass


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
                    "model": "pixtral-large-latest",
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

TODO_SYSTEM_PROMPT = """You are an expert productivity and task-planning AI agent. Your job is to take a task name and optional description, then generate an optimal, actionable todo list with realistic time estimates.

Rules:
1. Break the task into 3-8 concrete, actionable steps.
2. Order steps by logical dependency — what must be done first comes first.
3. Each step should be small enough to complete in one focused session (15-60 min).
4. Use clear, imperative language (e.g., "Set up project repository", not "Setting up").
5. Include a brief description for each step explaining what specifically to do.
6. Estimate realistic time in minutes for each step. Consider: research/reading = longer, simple edits/config = shorter, implementation = varies by complexity. Be honest — don't underestimate.
7. Consider: research/planning steps first, then implementation, then testing/review.
8. Be specific to the task — don't give generic advice.

Respond with ONLY a valid JSON object in this exact format:
{
  "todos": [
    {"title": "Step title", "description": "Brief description of what to do", "estimated_minutes": 30},
    {"title": "Step title", "description": "Brief description of what to do", "estimated_minutes": 20}
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
                {"title": "Research", "description": f"Research requirements and resources for: {task_name}", "estimated_minutes": 20},
                {"title": "Plan", "description": "Create a detailed plan and outline", "estimated_minutes": 15},
                {"title": "Implement", "description": "Implement the core functionality", "estimated_minutes": 45},
                {"title": "Test", "description": "Test and verify the implementation", "estimated_minutes": 20},
                {"title": "Refine", "description": "Refine, polish, and document the result", "estimated_minutes": 15},
            ]
        }

    # Validate response structure
    todos = result.get("todos", [])
    if not isinstance(todos, list) or len(todos) == 0:
        todos = [
            {"title": "Research", "description": f"Research: {task_name}", "estimated_minutes": 20},
            {"title": "Plan", "description": "Create a plan", "estimated_minutes": 15},
            {"title": "Implement", "description": "Build the core work", "estimated_minutes": 45},
            {"title": "Review", "description": "Review and finalize", "estimated_minutes": 15},
        ]

    # Ensure each todo has required fields
    validated_todos = []
    for todo in todos:
        if isinstance(todo, dict) and "title" in todo:
            # Parse estimated_minutes, default to 30 if missing or invalid
            est = todo.get("estimated_minutes", 30)
            try:
                est = int(est)
                if est <= 0 or est > 480:
                    est = 30
            except (ValueError, TypeError):
                est = 30

            validated_todos.append(
                {
                    "title": todo.get("title", "Untitled"),
                    "description": todo.get("description", ""),
                    "estimated_minutes": est,
                }
            )

    return {"todos": validated_todos}


# --- Resource Suggestion Agent ---

RESOURCE_SYSTEM_PROMPT = """You are a focus coach AI. The user is working on a specific task but has been detected as distracted. Your job is to give them 2-3 specific, actionable suggestions to help them refocus on their task.

Rules:
1. Be specific to their actual task and current todo item — not generic advice.
2. Each suggestion should be something they can do RIGHT NOW (not "plan for later").
3. Include concrete actions: what app to open, what to search for, what to read/write.
4. Keep suggestions brief and encouraging — not preachy.
5. If they are behind schedule on a todo, acknowledge it constructively.

Respond with ONLY a valid JSON object:
{
  "resources": [
    {"title": "Brief action title", "action": "Specific thing to do right now"},
    {"title": "Brief action title", "action": "Specific thing to do right now"}
  ]
}"""


@app.post("/api/suggest-resources")
async def suggest_resources(data: dict):
    """
    AI agent that suggests specific resources/actions to help a distracted user refocus.
    Uses Bedrock -> Mistral API fallback, with hardcoded fallback if both fail.
    """
    task_name = data.get("task_name", "").strip()
    task_description = data.get("task_description", "").strip()
    current_todo = data.get("current_todo", "").strip()
    todos = data.get("todos", [])
    elapsed_seconds = data.get("elapsed_seconds", 0)
    estimated_minutes = data.get("estimated_minutes", 0)

    # Build context prompt
    parts = []
    if task_name:
        parts.append(f'Task: "{task_name}"')
    if task_description:
        parts.append(f"Description: {task_description}")
    if current_todo:
        parts.append(f'Currently working on: "{current_todo}"')
    if estimated_minutes > 0:
        elapsed_min = round(elapsed_seconds / 60, 1)
        parts.append(f"Time spent on current todo: {elapsed_min} min (estimated: {estimated_minutes} min)")
        if elapsed_seconds > estimated_minutes * 60:
            parts.append("NOTE: The user is BEHIND SCHEDULE on this todo.")
    if todos:
        completed = sum(1 for t in todos if t.get("status") == "completed")
        total = len(todos)
        parts.append(f"Progress: {completed}/{total} todos completed")

    prompt = "\n".join(parts) if parts else "The user is working but got distracted. Suggest ways to refocus."
    prompt += "\n\nSuggest 2-3 specific actions to help them refocus right now."

    # Try Bedrock first
    result = await call_bedrock_mistral(prompt, RESOURCE_SYSTEM_PROMPT)

    if result is None:
        result = await call_mistral_api(prompt, RESOURCE_SYSTEM_PROMPT)

    if result is None:
        # Hardcoded fallback
        resources = []
        if current_todo:
            resources.append({
                "title": f"Continue: {current_todo[:40]}",
                "action": f"Open the relevant app and pick up where you left off on this step."
            })
        resources.append({
            "title": "Close distracting tabs",
            "action": "Close social media, news, and other non-work tabs. Keep only task-relevant ones."
        })
        resources.append({
            "title": "Quick re-read your plan",
            "action": f"Review your task description and todo list to rebuild context on '{task_name or 'your task'}'."
        })
        return {"resources": resources}

    # Validate
    resources = result.get("resources", [])
    if not isinstance(resources, list) or len(resources) == 0:
        return {"resources": [
            {"title": "Refocus", "action": "Take a deep breath and return to your current todo."},
            {"title": "Close distractions", "action": "Close non-work tabs and apps."},
        ]}

    validated = []
    for r in resources[:3]:
        if isinstance(r, dict) and "title" in r:
            validated.append({
                "title": r.get("title", ""),
                "action": r.get("action", ""),
            })

    return {"resources": validated}


# --- Smart Analysis Endpoint (DINOv3 pre-filter + LangChain synthesis) ---


@app.post("/api/analyze-smart")
async def analyze_smart(data: dict):
    """
    Smart focus analysis with DINOv3 pre-filter.

    Flow:
    1. Run DINOv3 similarity check on the screenshot
    2. If screen UNCHANGED (similarity < threshold): skip LLM, return cached focus state
    3. If screen CHANGED: run vision analysis + synthesis chain, cache result

    Receives:
    {
        "image": "base64_string",
        "task_context": {"task_name": str, "current_todo": str},  # optional
        "window_data": {"app_name": str, "window_title": str},    # optional
        "activity_data": {"idle_seconds": int, ...}               # optional
    }

    Returns:
    {
        "focus_state": "focused|distracted|unknown",
        "confidence": 0.0-1.0,
        "reason": str,
        "content_changed": bool,
        "similarity_score": float,
        "analysis_source": "llm|cached|rule_based",
        "dino_available": bool
    }
    """
    image_b64 = data.get("image", "")
    task_context = data.get("task_context")
    window_data = data.get("window_data")
    activity_data = data.get("activity_data")

    # Log received context for debugging
    if task_context:
        task_name = task_context.get("task_name", "?")
        current_todo = task_context.get("current_todo", "none")
        todos = task_context.get("todos", [])
        completed = sum(1 for t in todos if t.get("status") == "completed")
        print(
            f"[CONTEXT] Task: \"{task_name}\" | "
            f"Current todo: \"{current_todo}\" | "
            f"Todos: {completed}/{len(todos)} done",
            flush=True,
        )
    else:
        print("[CONTEXT] No task context received", flush=True)
    if window_data:
        print(
            f"[CONTEXT] Window: {window_data.get('app_name', '?')} "
            f"— {window_data.get('window_title', '?')[:60]}",
            flush=True,
        )
    if activity_data:
        print(
            f"[CONTEXT] Activity: "
            f"idle={activity_data.get('idle_seconds', 0)}s "
            f"switches={activity_data.get('window_switch_count', 0)} "
            f"keys={activity_data.get('keypress_count', 0)}",
            flush=True,
        )

    if not image_b64:
        return {
            "focus_state": "unknown",
            "confidence": 0.0,
            "reason": "No image provided",
            "content_changed": True,
            "similarity_score": 1.0,
            "analysis_source": "error",
            "dino_available": dino_service.is_loaded,
        }

    # Self-app detection: skip LLM when user is looking at Pabu Focus itself
    SELF_APP_NAMES = {"electron", "pabu focus", "pabu"}
    if window_data:
        active_app = window_data.get("app_name", "").strip().lower()
        if active_app in SELF_APP_NAMES:
            print(
                f"[SMART] Active window is self (\"{active_app}\") — skipping analysis (paused)",
                flush=True,
            )
            return {
                "focus_state": "paused",
                "confidence": 1.0,
                "reason": f"User is viewing Pabu Focus app",
                "content_changed": False,
                "similarity_score": 0.0,
                "analysis_source": "self_app",
                "dino_available": dino_service.is_loaded,
            }

    # Step 1: DINOv3 similarity check (offloaded to thread to avoid blocking)
    dino_result = {"changed": True, "similarity_score": 1.0, "is_first_frame": True}

    if dino_service.is_loaded:
        try:
            dino_result = await asyncio.to_thread(dino_service.compare, image_b64, "desktop")
            score = dino_result.get("similarity_score", -1)
            changed = dino_result.get("changed", True)
            first = dino_result.get("is_first_frame", False)
            thresh = dino_result.get("threshold", 0.15)
            if first:
                print(f"[DINO] First frame — no comparison yet (score=N/A)", flush=True)
            else:
                status = "CHANGED" if changed else "UNCHANGED"
                print(
                    f"[DINO] Inference OK — score={score:.4f}  "
                    f"threshold={thresh}  status={status}",
                    flush=True,
                )
        except Exception as e:
            print(f"[DINO] ERROR — comparison failed: {e}", flush=True)
            dino_result = {"changed": True, "similarity_score": 1.0, "is_first_frame": False}
    else:
        print("[DINO] Model not loaded — skipping similarity check", flush=True)

    content_changed = dino_result.get("changed", True)
    similarity_score = dino_result.get("similarity_score", 1.0)
    is_first_frame = dino_result.get("is_first_frame", False)

    # Step 2: If screen unchanged, return cached result (respects TTL)
    if not content_changed and not is_first_frame:
        cached = dino_service.get_cached_focus("desktop")
        if cached:
            print(
                f"[SMART] Screen unchanged — returning CACHED result: "
                f"{cached.get('focus_state')} ({cached.get('confidence', 0)*100:.0f}%)",
                flush=True,
            )
            return {
                **cached,
                "content_changed": False,
                "similarity_score": similarity_score,
                "analysis_source": "cached",
                "dino_available": True,
            }
        else:
            print(
                f"[SMART] Screen unchanged but cache expired — forcing fresh LLM analysis...",
                flush=True,
            )

    # Step 3: Screen changed (or first frame / no cache / cache expired) — run full analysis
    if content_changed or is_first_frame:
        reason_tag = "first frame" if is_first_frame else f"score={similarity_score:.4f}"
        print(f"[SMART] Screen changed ({reason_tag}) — running LLM analysis...", flush=True)

    # 3a: Vision analysis via LangChain (task-aware)
    vision_result = None
    if focus_chains.is_ready and focus_chains.vision_llm:
        try:
            vision_result = await focus_chains.analyze_vision(image_b64, task_context=task_context)
            print(
                f"[VISION] LangChain Pixtral result: "
                f"{vision_result.get('focus_state')} "
                f"({vision_result.get('confidence', 0)*100:.0f}%) "
                f"— {vision_result.get('reason', '')[:60]}",
                flush=True,
            )
        except Exception as e:
            print(f"[VISION] LangChain failed: {e}", flush=True)

    # Fallback to original /api/analyze logic if LangChain vision failed
    if vision_result is None or vision_result.get("focus_state") == "unknown":
        if MISTRAL_API_KEY:
            try:
                print("[VISION] Falling back to direct Mistral API...", flush=True)

                # Build task-aware vision prompt for fallback too
                from chains import _build_vision_prompt
                fallback_vision_prompt = _build_vision_prompt(task_context)

                async with httpx.AsyncClient() as client:
                    image_url = f"data:image/png;base64,{image_b64}"
                    response = await client.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {MISTRAL_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "pixtral-large-latest",
                            "messages": [{
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": image_url}},
                                    {
                                        "type": "text",
                                        "text": fallback_vision_prompt,
                                    },
                                ],
                            }],
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
                        vision_result = json.loads(content)
                        print(
                            f"[VISION] Direct Mistral result: "
                            f"{vision_result.get('focus_state')} "
                            f"({vision_result.get('confidence', 0)*100:.0f}%) "
                            f"— {vision_result.get('reason', '')[:60]}",
                            flush=True,
                        )
                    else:
                        print(f"[VISION] Direct Mistral HTTP error: {response.status_code}", flush=True)
            except Exception as e:
                print(f"[VISION] Direct Mistral fallback failed: {e}", flush=True)

    # 3b: Synthesis — combine all signals
    analysis_source = "llm"

    if focus_chains.is_ready:
        try:
            synthesis = await focus_chains.synthesize_focus(
                vision_analysis=vision_result,
                window_data=window_data,
                activity_data=activity_data,
                task_context=task_context,
                content_change=dino_result,
            )
            print(
                f"[SYNTHESIS] LangChain result: "
                f"{synthesis.get('focus_state')} "
                f"({synthesis.get('confidence', 0)*100:.0f}%) "
                f"— {synthesis.get('reason', '')[:80]}",
                flush=True,
            )
        except Exception as e:
            print(f"[SYNTHESIS] LangChain failed, using fallback: {e}", flush=True)
            synthesis = vision_result or {
                "focus_state": "unknown",
                "confidence": 0.0,
                "reason": "Analysis failed",
            }
            analysis_source = "rule_based"
    else:
        # No LangChain — use vision result directly or rule-based
        if vision_result and vision_result.get("focus_state") != "unknown":
            synthesis = vision_result
            print("[SYNTHESIS] Using vision result directly (no LangChain)", flush=True)
        else:
            synthesis = FocusChains._rule_based_synthesis(
                vision_result, window_data, activity_data
            )
            analysis_source = "rule_based"
            print(f"[SYNTHESIS] Rule-based fallback: {synthesis.get('focus_state')}", flush=True)

    # Build final result
    final_result = {
        "focus_state": synthesis.get("focus_state", "unknown"),
        "confidence": synthesis.get("confidence", 0.0),
        "reason": synthesis.get("reason", ""),
    }

    # Cache the result for next time (in case screen doesn't change)
    dino_service.set_cached_focus(final_result, "desktop")

    print(
        f"[SMART] === RESULT: {final_result['focus_state']} "
        f"({final_result['confidence']*100:.0f}%) | "
        f"changed={content_changed} | source={analysis_source} ===",
        flush=True,
    )

    return {
        **final_result,
        "content_changed": content_changed,
        "similarity_score": similarity_score,
        "analysis_source": analysis_source,
        "dino_available": dino_service.is_loaded,
    }


@app.post("/api/analyze-smart/reset")
async def reset_smart_analysis():
    """Reset DINOv3 state for desktop channel (e.g., when starting a new session)."""
    dino_service.reset("desktop")
    return {"status": "reset", "message": "DINOv3 desktop state cleared"}


# --- Robot Camera Analysis Endpoint ---


@app.post("/api/analyze-robot")
async def analyze_robot(data: dict):
    """
    Analyze a robot camera frame for physical distraction.

    The robot's camera watches the user at their desk. This endpoint detects:
    - Phone usage (holding, scrolling, looking at phone)
    - Looking away from the laptop
    - Not engaged with work (sleeping, eating, talking, daydreaming)
    - General body language indicating distraction

    Uses DINOv3 pre-filter (robot channel) + Pixtral Large vision analysis.
    Results are also broadcast to desktop clients via WebSocket.

    Receives:
    {
        "image": "base64_string"
    }

    Returns:
    {
        "focus_state": "focused|distracted|unknown",
        "confidence": 0.0-1.0,
        "reason": str,
        "content_changed": bool,
        "similarity_score": float,
        "analysis_source": "ai|cached|fallback",
        "dino_available": bool
    }
    """
    image_b64 = data.get("image", "")

    if not image_b64:
        return {
            "focus_state": "unknown",
            "confidence": 0.0,
            "reason": "No image provided",
            "content_changed": True,
            "similarity_score": 1.0,
            "analysis_source": "error",
            "dino_available": dino_service.is_loaded,
        }

    # Step 1: DINOv3 similarity check (robot channel)
    dino_result = {"changed": True, "similarity_score": 1.0, "is_first_frame": True}

    if dino_service.is_loaded:
        try:
            dino_result = await asyncio.to_thread(
                dino_service.compare, image_b64, "robot"
            )
            score = dino_result.get("similarity_score", -1)
            changed = dino_result.get("changed", True)
            first = dino_result.get("is_first_frame", False)
            thresh = dino_result.get("threshold", 0.10)
            if first:
                print(
                    f"[ROBOT-DINO] First frame — no comparison yet",
                    flush=True,
                )
            else:
                status = "CHANGED" if changed else "UNCHANGED"
                print(
                    f"[ROBOT-DINO] score={score:.4f}  "
                    f"threshold={thresh}  status={status}",
                    flush=True,
                )
        except Exception as e:
            print(f"[ROBOT-DINO] ERROR: {e}", flush=True)
            dino_result = {
                "changed": True,
                "similarity_score": 1.0,
                "is_first_frame": False,
            }
    else:
        print("[ROBOT-DINO] Model not loaded — skipping", flush=True)

    content_changed = dino_result.get("changed", True)
    similarity_score = dino_result.get("similarity_score", 1.0)
    is_first_frame = dino_result.get("is_first_frame", False)

    # Step 2: If scene unchanged, return cached result
    if not content_changed and not is_first_frame:
        cached = dino_service.get_cached_focus("robot")
        if cached:
            print(
                f"[ROBOT] Scene unchanged — returning CACHED: "
                f"{cached.get('focus_state')} ({cached.get('confidence', 0)*100:.0f}%)",
                flush=True,
            )
            return {
                **cached,
                "content_changed": False,
                "similarity_score": similarity_score,
                "analysis_source": "cached",
                "dino_available": True,
            }

    # Step 3: Scene changed — run Pixtral Large robot vision analysis
    reason_tag = "first frame" if is_first_frame else f"score={similarity_score:.4f}"
    print(
        f"[ROBOT] Scene changed ({reason_tag}) — running vision analysis...",
        flush=True,
    )

    vision_result = None
    analysis_source = "ai"

    # 3a: Try LangChain robot vision (Pixtral Large)
    if focus_chains.is_ready and focus_chains.vision_llm:
        try:
            vision_result = await focus_chains.analyze_robot_vision(image_b64)
            print(
                f"[ROBOT-VISION] Pixtral result: "
                f"{vision_result.get('focus_state')} "
                f"({vision_result.get('confidence', 0)*100:.0f}%) "
                f"— {vision_result.get('reason', '')[:80]}",
                flush=True,
            )
        except Exception as e:
            print(f"[ROBOT-VISION] LangChain failed: {e}", flush=True)

    # 3b: Fallback to direct Mistral API if LangChain failed
    if vision_result is None or vision_result.get("focus_state") == "unknown":
        if MISTRAL_API_KEY:
            try:
                print(
                    "[ROBOT-VISION] Falling back to direct Mistral API...",
                    flush=True,
                )
                from chains import ROBOT_VISION_PROMPT

                async with httpx.AsyncClient() as client:
                    image_url = f"data:image/jpeg;base64,{image_b64}"
                    response = await client.post(
                        "https://api.mistral.ai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {MISTRAL_API_KEY}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "pixtral-large-latest",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image_url",
                                            "image_url": {"url": image_url},
                                        },
                                        {
                                            "type": "text",
                                            "text": ROBOT_VISION_PROMPT,
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
                        vision_result = json.loads(content)
                        print(
                            f"[ROBOT-VISION] Direct Mistral result: "
                            f"{vision_result.get('focus_state')} "
                            f"({vision_result.get('confidence', 0)*100:.0f}%) "
                            f"— {vision_result.get('reason', '')[:80]}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[ROBOT-VISION] Direct Mistral HTTP error: "
                            f"{response.status_code}",
                            flush=True,
                        )
            except Exception as e:
                print(
                    f"[ROBOT-VISION] Direct Mistral fallback failed: {e}",
                    flush=True,
                )

    # Step 4: Build final result
    if vision_result and vision_result.get("focus_state") != "unknown":
        final_result = {
            "focus_state": vision_result.get("focus_state", "unknown"),
            "confidence": vision_result.get("confidence", 0.0),
            "reason": vision_result.get("reason", ""),
        }
    else:
        final_result = {
            "focus_state": "unknown",
            "confidence": 0.0,
            "reason": "Could not analyze robot camera image",
        }
        analysis_source = "fallback"

    # Cache the result for next time
    dino_service.set_cached_focus(final_result, "robot")

    print(
        f"[ROBOT] === RESULT: {final_result['focus_state']} "
        f"({final_result['confidence']*100:.0f}%) | "
        f"changed={content_changed} | source={analysis_source} ===",
        flush=True,
    )

    # Step 5: Broadcast result to desktop clients via WebSocket
    if final_result["focus_state"] != "unknown":
        robot_message = {
            "type": "robot_focus_update",
            "source": "server",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "focus_state": final_result["focus_state"],
                "confidence": final_result["confidence"],
                "reason": final_result["reason"],
                "content_changed": content_changed,
                "similarity_score": similarity_score,
                "analysis_source": analysis_source,
            },
        }
        disconnected = []
        for desktop_ws in desktop_connections:
            try:
                await desktop_ws.send_json(robot_message)
            except Exception:
                disconnected.append(desktop_ws)
        for ws in disconnected:
            desktop_connections.remove(ws)

        if desktop_connections:
            print(
                f"[ROBOT] Broadcast to {len(desktop_connections)} desktop client(s)",
                flush=True,
            )

    return {
        **final_result,
        "content_changed": content_changed,
        "similarity_score": similarity_score,
        "analysis_source": analysis_source,
        "dino_available": dino_service.is_loaded,
    }


@app.post("/api/analyze-robot/reset")
async def reset_robot_analysis():
    """Reset DINOv3 state for robot channel."""
    dino_service.reset("robot")
    return {"status": "reset", "message": "DINOv3 robot state cleared"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)

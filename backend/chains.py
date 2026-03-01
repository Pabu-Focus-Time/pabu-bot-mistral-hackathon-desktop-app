"""
LangChain Chains for Focus Analysis

Task-aware focus analysis pipeline:
- Vision analysis: Pixtral vision model analyzes screenshots WITH task context
- Synthesis chain: Combines vision + window + activity + task context into a focus decision

The key insight: "focused" means doing something RELEVANT TO THE TASK, not just
using a "productive" app. Coding in VSCode while your task is "Learn Multivariate
Calculus" is distracted, not focused.
"""

import os
import json
import logging
from typing import Optional, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

# -- Configuration ---

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "GkJlvmrgptESTkEGdgKjvQziz8aAWvCq")
BEDROCK_REGION = os.getenv("BEDROCK_REGION", "us-west-2")


# -- Prompts ---
# NOTE: Double curly braces {{}} are used to escape literal braces in
# LangChain ChatPromptTemplate strings. In non-template strings (like
# the vision prompt built per-request), use single braces {}.

SYNTHESIS_SYSTEM_PROMPT = """You are a focus-state analyzer for a productivity app. Your job is to determine whether the user is ACTUALLY WORKING ON THEIR STATED TASK — not just whether they are using a "productive" app.

CRITICAL RULE: "Focused" means the user's screen content and activity are DIRECTLY RELEVANT to their specific task and current todo item. Using a code editor when the task is "Learn Multivariate Calculus" is DISTRACTED, not focused. Watching a calculus lecture on YouTube when the task is "Learn Multivariate Calculus" IS focused.

You will receive:
- The user's task name, description, and todo list with completion status
- The currently active todo item they should be working on
- Vision analysis of what's on their screen
- Which app/window is active
- Keyboard/mouse activity metrics
- Whether the screen content recently changed

Decision rules (in priority order):
1. TASK RELEVANCE is the #1 factor. Ask: "Does the screen content relate to the stated task?"
2. If the active app and window title clearly relate to the task topic, the user is likely focused.
3. If the active app is unrelated to the task (e.g., code editor for a math task, social media for any task), the user is likely distracted.
4. Browser usage depends on content: researching the task topic = focused, social media = distracted.
5. High idle time (>30s) with no activity suggests the user stepped away.
6. The vision analysis provides what's literally on screen — cross-reference it with the task.

Respond with ONLY valid JSON: {{"focus_state": "focused|distracted|unknown", "confidence": 0.0-1.0, "reason": "brief explanation referencing the task"}}"""


ROBOT_VISION_PROMPT = """You are analyzing a camera image from a small desk robot that is watching a person who should be working on their laptop.

Your job is to determine whether the person is FOCUSED on their laptop work or DISTRACTED by something else.

Signs of DISTRACTION (return "distracted"):
- Holding or looking at a phone/smartphone
- Scrolling on a phone or tablet
- Looking away from the laptop screen for an extended period
- Turned away from the desk entirely
- Sleeping, dozing off, or head down on desk
- Eating a meal (snacking briefly is OK)
- Talking to someone else / on a phone call
- Playing with objects, fidgeting extensively
- Staring blankly / clearly daydreaming (eyes unfocused, not looking at screen)

Signs of FOCUS (return "focused"):
- Looking at the laptop screen
- Typing on the keyboard
- Reading something on screen (even if still)
- Writing notes while referencing the screen
- Briefly looking away to think (< a few seconds is normal)
- Using a mouse/trackpad

Signs of UNKNOWN (return "unknown"):
- Person is not visible in frame
- Image is too dark/blurry to determine
- Cannot clearly see what the person is doing

IMPORTANT: A person can be sitting at their desk but still be distracted if they are on their phone. Phone usage while at the desk is the #1 distraction signal.

Respond ONLY with valid JSON: {"focus_state": "focused|distracted|unknown", "confidence": 0.0-1.0, "reason": "brief description of what you see the person doing"}"""


def _build_vision_prompt(task_context: Optional[dict] = None) -> str:
    """Build the vision analysis prompt, optionally with task context."""
    base = "Analyze this screenshot and determine if the user is focused on their task or distracted."

    if task_context:
        task_name = task_context.get("task_name", "")
        task_desc = task_context.get("task_description", "")
        current_todo = task_context.get("current_todo", "")
        todos = task_context.get("todos", [])

        parts = [base, ""]
        parts.append(f'The user\'s current task is: "{task_name}"')
        if task_desc:
            parts.append(f"Task description: {task_desc}")
        if current_todo:
            parts.append(f'They should currently be working on: "{current_todo}"')
        if todos:
            todo_summary = ", ".join(
                t.get("title", "") for t in todos[:5]
            )
            parts.append(f"Their todo list includes: {todo_summary}")

        parts.append("")
        parts.append(
            "IMPORTANT: Determine if the screen content is RELEVANT to this specific task. "
            "A code editor is NOT relevant if the task is about math/science/writing. "
            "A YouTube video IS relevant if it's a lecture on the task topic."
        )
    else:
        parts = [
            base,
            "",
            "No specific task context available. Judge based on whether the user "
            "appears to be doing productive work vs. entertainment/social media.",
        ]

    parts.append("")
    parts.append(
        'Respond ONLY with valid JSON: '
        '{"focus_state": "focused|distracted|unknown", '
        '"confidence": 0.0-1.0, '
        '"reason": "brief explanation"}'
    )

    return "\n".join(parts)


def _format_task_context(task_context: Optional[dict]) -> List[str]:
    """Format rich task context into human-readable lines for the synthesis prompt."""
    if not task_context:
        return []

    lines = []

    task_name = task_context.get("task_name", "")
    if task_name:
        lines.append(f'TASK: "{task_name}"')

    task_desc = task_context.get("task_description", "")
    if task_desc:
        lines.append(f"Task description: {task_desc}")

    current_todo = task_context.get("current_todo", "")
    if current_todo:
        lines.append(f'Currently working on: "{current_todo}"')

    todos = task_context.get("todos", [])
    if todos:
        completed = sum(1 for t in todos if t.get("status") == "completed")
        total = len(todos)
        lines.append(f"Progress: {completed}/{total} todos completed")
        lines.append("Todo list:")
        for t in todos:
            status_icon = "x" if t.get("status") == "completed" else " "
            lines.append(f"  [{status_icon}] {t.get('title', 'Untitled')}")

    return lines


# -- Chain Builder ---


class FocusChains:
    """
    Manages LangChain chains for focus analysis.

    Initialize once at server startup, use throughout the app lifecycle.
    """

    def __init__(self):
        self.synthesis_chain = None
        self.synthesis_chain_fallback = None
        self.vision_llm = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all chains. Call once at server startup."""
        if self._initialized:
            return

        try:
            self._init_synthesis_chain()
            self._init_vision_llm()
            self._initialized = True
            logger.info("LangChain chains initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize LangChain chains: {e}")
            self._initialized = False

    def _init_synthesis_chain(self) -> None:
        """Build the synthesis chain with Bedrock primary + Mistral fallback."""
        parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_messages([
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        # Primary: AWS Bedrock with Mistral Large
        try:
            from langchain_aws import ChatBedrockConverse

            bedrock_llm = ChatBedrockConverse(
                model="mistral.mistral-large-2407-v1:0",
                region_name=BEDROCK_REGION,
                temperature=0.3,
                max_tokens=512,
            )
            self.synthesis_chain = prompt | bedrock_llm | parser
            logger.info("Bedrock synthesis chain ready")
        except Exception as e:
            logger.warning(f"Bedrock synthesis chain unavailable: {e}")
            self.synthesis_chain = None

        # Fallback: Mistral API direct
        try:
            if MISTRAL_API_KEY:
                from langchain_mistralai import ChatMistralAI

                mistral_llm = ChatMistralAI(
                    model="mistral-small-latest",
                    mistral_api_key=MISTRAL_API_KEY,
                    temperature=0.3,
                    max_tokens=512,
                )
                self.synthesis_chain_fallback = prompt | mistral_llm | parser
                logger.info("Mistral synthesis chain fallback ready")
        except Exception as e:
            logger.warning(f"Mistral synthesis chain unavailable: {e}")
            self.synthesis_chain_fallback = None

        # Wire up with_fallbacks if both are available
        if self.synthesis_chain and self.synthesis_chain_fallback:
            self.synthesis_chain = self.synthesis_chain.with_fallbacks(
                [self.synthesis_chain_fallback]
            )

    def _init_vision_llm(self) -> None:
        """Initialize the Pixtral vision model via LangChain."""
        try:
            if MISTRAL_API_KEY:
                from langchain_mistralai import ChatMistralAI

                self.vision_llm = ChatMistralAI(
                    model="pixtral-large-latest",
                    mistral_api_key=MISTRAL_API_KEY,
                    temperature=0.3,
                    max_tokens=512,
                )
                logger.info("Vision LLM (Pixtral Large) ready")
        except Exception as e:
            logger.warning(f"Vision LLM unavailable: {e}")
            self.vision_llm = None

    @property
    def is_ready(self) -> bool:
        return self._initialized

    async def synthesize_focus(
        self,
        vision_analysis: Optional[dict] = None,
        window_data: Optional[dict] = None,
        activity_data: Optional[dict] = None,
        task_context: Optional[dict] = None,
        content_change: Optional[dict] = None,
    ) -> dict:
        """
        Synthesize focus state from multiple signals using LangChain.

        Returns: {"focus_state": str, "confidence": float, "reason": str}
        """
        # Build the context string with rich task info
        context_parts = []

        # Task context FIRST — most important signal
        task_lines = _format_task_context(task_context)
        if task_lines:
            context_parts.extend(task_lines)
        else:
            context_parts.append("TASK: No task specified (general productivity check)")

        context_parts.append("")  # blank separator

        # Vision context
        if vision_analysis:
            state = vision_analysis.get("focus_state", "unknown")
            conf = vision_analysis.get("confidence", 0)
            reason = vision_analysis.get("reason", "")
            context_parts.append(
                f"Vision Analysis: Screen shows user appears {state} "
                f"({conf * 100:.0f}% confidence). {reason}"
            )

        # Window context
        if window_data:
            app_name = window_data.get("app_name", "Unknown")
            title = window_data.get("window_title", "")
            line = f"Active application: {app_name}"
            if title:
                line += f" — window title: \"{title}\""
            context_parts.append(line)

        # Activity context
        if activity_data:
            idle = activity_data.get("idle_seconds", 0)
            switches = activity_data.get("window_switch_count", 0)
            keys = activity_data.get("keypress_count", 0)
            context_parts.append(
                f"Activity: {idle}s idle, {switches} window switches, "
                f"{keys} keypresses in last interval"
            )

        # Content change context
        if content_change:
            changed = content_change.get("changed", True)
            score = content_change.get("similarity_score", 0)
            context_parts.append(
                f"Screen content: {'Changed' if changed else 'Unchanged'} "
                f"(change score: {score:.4f})"
            )

        input_text = "\n".join(context_parts)

        # Try LangChain synthesis
        if self.synthesis_chain:
            try:
                result = await self.synthesis_chain.ainvoke({"input": input_text})
                if isinstance(result, dict) and "focus_state" in result:
                    return result
            except Exception as e:
                logger.warning(f"LangChain synthesis failed: {e}")

        # If LangChain failed entirely, use rule-based fallback
        return self._rule_based_synthesis(
            vision_analysis, window_data, activity_data, task_context
        )

    async def analyze_vision(
        self,
        image_b64: str,
        task_context: Optional[dict] = None,
    ) -> dict:
        """
        Analyze a screenshot using Pixtral vision model via LangChain.
        Now task-aware: tells the model what the user should be working on.

        Returns: {"focus_state": str, "confidence": float, "reason": str}
        """
        if not self.vision_llm:
            return {
                "focus_state": "unknown",
                "confidence": 0.0,
                "reason": "Vision LLM not available",
            }

        try:
            vision_prompt = _build_vision_prompt(task_context)
            image_url = f"data:image/png;base64,{image_b64}"
            message = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": vision_prompt},
            ])

            response = await self.vision_llm.ainvoke([message])
            content = response.content

            # Parse JSON from response
            if isinstance(content, str):
                result = json.loads(content)
                return {
                    "focus_state": result.get("focus_state", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "reason": result.get("reason", ""),
                }
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            if isinstance(content, str):
                match = re.search(r"\{[\s\S]*?\}", content)
                if match:
                    try:
                        result = json.loads(match.group(0))
                        return {
                            "focus_state": result.get("focus_state", "unknown"),
                            "confidence": result.get("confidence", 0.0),
                            "reason": result.get("reason", ""),
                        }
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")

        return {
            "focus_state": "unknown",
            "confidence": 0.0,
            "reason": "Failed to analyze screenshot",
        }

    async def analyze_robot_vision(self, image_b64: str) -> dict:
        """
        Analyze a robot camera image for physical distraction (phone usage,
        looking away, not engaged with laptop, etc.).

        Uses Pixtral Large with a prompt tailored for physical-world observation,
        not screen content analysis.

        Returns: {"focus_state": str, "confidence": float, "reason": str}
        """
        if not self.vision_llm:
            return {
                "focus_state": "unknown",
                "confidence": 0.0,
                "reason": "Vision LLM not available",
            }

        try:
            image_url = f"data:image/jpeg;base64,{image_b64}"
            message = HumanMessage(content=[
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": ROBOT_VISION_PROMPT},
            ])

            response = await self.vision_llm.ainvoke([message])
            content = response.content

            # Parse JSON from response
            if isinstance(content, str):
                result = json.loads(content)
                return {
                    "focus_state": result.get("focus_state", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "reason": result.get("reason", ""),
                }
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re

            if isinstance(content, str):
                match = re.search(r"\{[\s\S]*?\}", content)
                if match:
                    try:
                        result = json.loads(match.group(0))
                        return {
                            "focus_state": result.get("focus_state", "unknown"),
                            "confidence": result.get("confidence", 0.0),
                            "reason": result.get("reason", ""),
                        }
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.error(f"Robot vision analysis error: {e}")

        return {
            "focus_state": "unknown",
            "confidence": 0.0,
            "reason": "Failed to analyze robot camera image",
        }

    @staticmethod
    def _rule_based_synthesis(
        vision_analysis: Optional[dict],
        window_data: Optional[dict],
        activity_data: Optional[dict],
        task_context: Optional[dict] = None,
    ) -> dict:
        """Simple heuristic fallback when LLM is unavailable."""
        focus_state = "unknown"
        confidence = 0.3
        reason = "Insufficient data for analysis"

        if window_data:
            app_name = window_data.get("app_name", "").lower()
            distracting = [
                "safari", "chrome", "twitter", "facebook",
                "youtube", "reddit", "tiktok", "instagram",
            ]
            productive = [
                "vscode", "code", "xcode", "terminal", "iterm",
                "vim", "neovim", "sublime", "atom", "intellij",
                "pycharm", "webstorm", "cursor",
            ]

            if any(app in app_name for app in distracting):
                focus_state = "distracted"
                confidence = 0.5
                reason = f"Using potentially distracting app: {window_data.get('app_name')}"
            elif any(app in app_name for app in productive):
                focus_state = "focused"
                confidence = 0.6
                reason = f"Using productive app: {window_data.get('app_name')}"

        if vision_analysis and vision_analysis.get("focus_state") != "unknown":
            focus_state = vision_analysis.get("focus_state", focus_state)
            confidence = max(confidence, vision_analysis.get("confidence", 0))
            reason = vision_analysis.get("reason", reason)

        if activity_data and activity_data.get("idle_seconds", 0) > 60:
            focus_state = "unknown"
            confidence = 0.7
            reason = "User appears idle"

        return {
            "focus_state": focus_state,
            "confidence": confidence,
            "reason": reason,
        }

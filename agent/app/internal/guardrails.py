"""LangChain guardrails for the Clinical Companion agent.

The stack follows the LangChain guardrails guide: a deterministic before-agent
content filter, PII middleware, a tool-call filter for Tavily, and a model-based
after-agent safety check.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    PIIMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from .bmi import BmiTool
from .web_search import format_search_results, format_source_links

REFUSAL = (
    "I can't help with diagnosis or treatment plans. "
    "Please contact a clinician or specialist for this question. "
    "This is informational guidance only."
)

HABIT_BOUNDARY = (
    "I can't recommend medication or a treatment plan. "
    "Please contact a clinician or specialist for this question. "
    "I can talk about general lifestyle habits that support day-to-day wellbeing, "
    "such as sleep, movement, and eating patterns. "
    "This is informational guidance only."
)

NO_WEB_SOURCES = "No usable web sources were found for this question."
TAVILY_PAYLOAD = "TAVILY_JSON\n"

INTRODUCTION = (
    "Hello! I'm Clinical Companion, your assistant for diet, exercise, "
    "and healthy habits. How can I help?"
)

OUT_OF_SCOPE = (
    "I can only help with diet, exercise, healthy habits, and related wellbeing. "
    "This is informational guidance only."
)

NOT_COVERED = (
    "This topic is not covered by the book or the allowed health sources. "
    "This is informational guidance only."
)

_DANGEROUS = (
    "diagnos",
    "prescribe",
    "treatment plan",
    "what medication",
    "which medication",
    "do i have",
)

_TREATMENT = (
    "what should i take",
    "what can i take",
    "what do i take",
    "what to take",
    "should i take",
    "what medicine",
    "which medicine",
    "what drug",
    "which drug",
    "medicine for",
    "medication for",
    "drug for",
    "que debo tomar",
    "qué debo tomar",
    "que puedo tomar",
    "qué puedo tomar",
    "medicamento",
    "medicina para",
)

_SNIPPET_MEDICATION = _TREATMENT + (
    "medication",
    "medicine",
    "medicina",
    "prescription",
    "dosage",
    "tablet",
    "capsule",
)

_INJECTION = re.compile(
    r"ignore previous instructions|disregard your|you are now|system prompt",
    re.IGNORECASE,
)
_DOSE = re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:mg|mcg|ml|iu)\b", re.IGNORECASE)
_DOSE_QUESTION = re.compile(r"\b(?:dose|doses|dosis|dosage|dosages)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_PHONE = r"(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{3}\)[\s.-]?|\d{3}[\s.-])\d{3}[\s.-]\d{4}|\b\d{10}\b"

SAFETY_PROMPT = (
    "Evaluate if this response is safe and appropriate for Clinical Companion. "
    "Clinical Companion may discuss sleep, movement, and eating patterns. "
    "Respond with only SAFE or UNSAFE. "
    "UNSAFE means the response diagnoses, prescribes, names a medication, "
    "states a dose, or gives a treatment regimen. "
    "SAFE means informational lifestyle guidance with no diagnosis, medication, "
    "dose, or treatment regimen.\n\n"
    "Response: {content}"
)


def _text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(str(part.get("text") or ""))
        return "".join(parts)
    return "" if content is None else str(content)


def latest_human(state: AgentState) -> str:
    """Return the most recent human message in this thread."""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") == "human":
            return _text(message.content)
    return ""


def _blocked_reply(user_text: str) -> str | None:
    text = user_text.lower()
    if any(phrase in text for phrase in _DANGEROUS):
        return REFUSAL
    if any(phrase in text for phrase in _TREATMENT):
        return HABIT_BOUNDARY
    if _DOSE_QUESTION.search(user_text) or _DOSE.search(user_text):
        return HABIT_BOUNDARY
    reading = BmiTool().from_text(user_text)
    if reading:
        return reading
    return None


def _medication_sentence(sentence: str) -> bool:
    lowered = sentence.lower()
    if _DOSE.search(sentence):
        return True
    return any(phrase in lowered for phrase in _SNIPPET_MEDICATION)


def clean_web_text(text: str) -> str:
    """Drop injection lines and sentences that name a dose or a medication."""
    kept_lines: list[str] = []
    for line in (text or "").splitlines():
        if _INJECTION.search(line):
            continue
        safe = [
            sentence.strip()
            for sentence in _SENTENCE.split(line.strip())
            if sentence.strip() and not _medication_sentence(sentence)
        ]
        if safe:
            kept_lines.append(" ".join(safe))
    return "\n".join(kept_lines).strip()


def kept_tavily_results(results: list[dict]) -> list[dict]:
    """Drop injection lines and dose or medication sentences, then keep survivors."""
    kept: list[dict] = []
    for result in results:
        content = clean_web_text(str(result.get("content") or ""))
        if not content:
            continue
        title = clean_web_text(str(result.get("title") or ""))
        kept.append({**result, "title": title, "content": content})
    return kept


def guard_tavily_results(results: list[dict]) -> str:
    """Return model-facing evidence, citing only sources that still have text."""
    kept = kept_tavily_results(results)
    if not kept:
        return NO_WEB_SOURCES
    body = format_search_results(kept)
    links = format_source_links(kept)
    if not links:
        return body
    return f"{body}\n\n{links}"


def present_tavily_payload(content: str) -> str:
    """Turn a Tavily tool payload into filtered evidence."""
    if not content.startswith(TAVILY_PAYLOAD):
        cleaned = clean_web_text(content)
        return cleaned or NO_WEB_SOURCES
    try:
        results = json.loads(content[len(TAVILY_PAYLOAD) :])
    except json.JSONDecodeError:
        return NO_WEB_SOURCES
    if not isinstance(results, list):
        return NO_WEB_SOURCES
    return guard_tavily_results(results)


def _rewrite_tool_result(result: ToolMessage | Command) -> ToolMessage | Command:
    if not isinstance(result, ToolMessage):
        return result
    filtered = present_tavily_payload(_text(result.content))
    return ToolMessage(
        content=filtered,
        tool_call_id=result.tool_call_id,
        name=result.name,
    )


class ContentFilterMiddleware(AgentMiddleware):
    """Block diagnosis, treatment, doses, and complete BMI readings before the model."""

    def _filter(self, state: AgentState) -> dict[str, Any] | None:
        reply = _blocked_reply(latest_human(state))
        if reply is None:
            return None
        return {
            "messages": [AIMessage(content=reply)],
            "citations": "",
            "jump_to": "end",
        }

    @hook_config(can_jump_to=["end"])
    def before_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return self._filter(state)

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return self._filter(state)


def pii_middleware() -> list[PIIMiddleware]:
    """Redact or mask PII on input, output, and tool results.

    URL detection stays off so Tavily source links remain in the reply.
    """
    shared = {
        "apply_to_input": True,
        "apply_to_output": True,
        "apply_to_tool_results": True,
    }
    return [
        PIIMiddleware("email", strategy="redact", **shared),
        PIIMiddleware("credit_card", strategy="mask", **shared),
        PIIMiddleware("phone", detector=_PHONE, strategy="redact", **shared),
    ]


class TavilyResultMiddleware(AgentMiddleware):
    """Filter search_web results before they enter the model context."""

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "search_web":
            return handler(request)
        return _rewrite_tool_result(handler(request))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler,
    ) -> ToolMessage | Command:
        if request.tool_call.get("name") != "search_web":
            return await handler(request)
        return _rewrite_tool_result(await handler(request))


def _skip_safety(text: str) -> bool:
    stripped = text.strip()
    return stripped in {
        REFUSAL,
        HABIT_BOUNDARY,
        INTRODUCTION,
        OUT_OF_SCOPE,
        NOT_COVERED,
    } or stripped.startswith("BMI ")


class SafetyGuardrailMiddleware(AgentMiddleware):
    """Ask the chat model whether the final reply is SAFE or UNSAFE."""

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def _review(self, state: AgentState) -> None:
        messages = state.get("messages") or []
        if not messages:
            return
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return
        content = _text(last_message.content).strip()
        if not content or _skip_safety(content):
            return
        result = self.model.invoke(
            [{"role": "user", "content": SAFETY_PROMPT.format(content=content)}]
        )
        verdict = _text(getattr(result, "content", result))
        if "UNSAFE" in verdict:
            last_message.content = HABIT_BOUNDARY

    @hook_config(can_jump_to=["end"])
    def after_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        self._review(state)
        return None

    @hook_config(can_jump_to=["end"])
    async def aafter_agent(
        self, state: AgentState, runtime: Runtime
    ) -> dict[str, Any] | None:
        self._review(state)
        return None

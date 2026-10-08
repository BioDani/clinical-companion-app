"""Clinical Companion: a LangChain agent with guardrail middleware."""

from __future__ import annotations

import re
from typing import Any, NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.runtime import Runtime

from .guardrails import (
    HABIT_BOUNDARY,
    INTRODUCTION,
    NOT_COVERED,
    NO_WEB_SOURCES,
    OUT_OF_SCOPE,
    REFUSAL,
    ContentFilterMiddleware,
    SafetyGuardrailMiddleware,
    guard_tavily_results,
    kept_tavily_results,
    latest_human,
    pii_middleware,
)
from .llm import HuggingFaceChat
from .web_search import HEALTH_DOMAINS, format_source_links

BOOK_TITLE = "Dr. Carlos Jaramillo, Pilares"

SYSTEM_PROMPT = (
    "You are Clinical Companion, a helpful assistant for diet, exercise, "
    "and healthy habits. "
    "You are not a diagnostic device and you do not replace a physician. "
    "Keep answers concise and in English. "
    "Answer only from the evidence message in this turn. "
    "Do not add facts from memory. "
    "Do not say the answer is general knowledge. "
    "Do not diagnose or prescribe. "
    "Do not name a medication, a dose, or a treatment regimen. "
    "If a source mentions drugs, leave them out and keep only lifestyle habits. "
    "Remind the user this is informational guidance only."
)

SCOPE_PROMPT = (
    "Classify the user message for Clinical Companion. "
    "Reply with only one of these labels: GREETING, IN_SCOPE, OUT_OF_SCOPE. "
    "GREETING is a hello, thanks, or a short social opener with no health question. "
    "IN_SCOPE is diet, exercise, sleep, nutrition, healthy habits, or related wellbeing. "
    "OUT_OF_SCOPE is anything else, including the assistant's training, dates, "
    "coding, news, or trivia."
)

_PAGE = re.compile(r"\bpage (\d+)\b")
_GENERAL_KNOWLEDGE = re.compile(r"^.*general knowledge.*$", re.IGNORECASE | re.MULTILINE)


class CompanionState(AgentState):
    """Agent state plus the citation block built from kept sources."""

    citations: NotRequired[str]


def format_book_reference(retrieved: str) -> str:
    """Cite the ebook pages already tagged on the retrieved passages."""
    pages: list[str] = []
    for page in _PAGE.findall(retrieved or ""):
        if page not in pages:
            pages.append(page)
    if not pages:
        return ""
    listed = ", ".join(f"page {page}" for page in pages)
    return f"### Reference\n\n- {BOOK_TITLE}, {listed}"


def _scope_label(verdict: str) -> str:
    text = (verdict or "").upper()
    if "OUT_OF_SCOPE" in text:
        return "OUT_OF_SCOPE"
    if "GREETING" in text:
        return "GREETING"
    if "IN_SCOPE" in text:
        return "IN_SCOPE"
    return "OUT_OF_SCOPE"


def _book_text(hits: list) -> str:
    return "\n\n".join(hit.text for hit in hits if getattr(hit, "text", ""))


def _citation_block(book: str, web_results: list[dict]) -> str:
    parts: list[str] = []
    reference = format_book_reference(book)
    if reference:
        parts.append(reference)
    links = format_source_links(web_results)
    if links:
        parts.append(links)
    return "\n\n".join(parts)


def _evidence_message(book: str, web_text: str) -> str:
    parts = [
        "Use only the evidence below. Do not add facts from memory. "
        "Do not say this is general knowledge."
    ]
    if book.strip():
        parts.append(f"Book passages:\n{book.strip()}")
    if web_text.strip():
        parts.append(f"Web evidence:\n{web_text.strip()}")
    return "\n\n".join(parts)


def _end(reply: str) -> dict[str, Any]:
    return {
        "messages": [AIMessage(content=reply)],
        "citations": "",
        "jump_to": "end",
    }


class EvidenceMiddleware(AgentMiddleware):
    """Classify scope, retrieve evidence, and store the citation block."""

    state_schema = CompanionState

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def _prepare(self, state: CompanionState) -> dict[str, Any] | None:
        question = latest_human(state)
        verdict = self.model.invoke(
            [
                {"role": "system", "content": SCOPE_PROMPT},
                {"role": "user", "content": question},
            ]
        )
        label = _scope_label(getattr(verdict, "content", verdict))
        if label == "GREETING":
            return _end(INTRODUCTION)
        if label == "OUT_OF_SCOPE":
            return _end(OUT_OF_SCOPE)

        from .vector_store import grade_coverage, search_hits

        hits = search_hits(question)
        grade = grade_coverage(question, hits)
        book = "" if grade == "miss" else _book_text(hits)
        web_results: list[dict] = []
        web_text = ""
        if grade != "explained":
            from .web_search import search_web

            web_results = kept_tavily_results(
                search_web(question, include_domains=list(HEALTH_DOMAINS))
            )
            web_text = guard_tavily_results(web_results)
            if web_text == NO_WEB_SOURCES:
                web_text = ""
                web_results = []
        if grade == "miss" and not web_results:
            return _end(NOT_COVERED)
        return {
            "citations": _citation_block(book, web_results),
            "messages": [HumanMessage(content=_evidence_message(book, web_text))],
        }

    @hook_config(can_jump_to=["end"])
    def before_model(
        self, state: CompanionState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return self._prepare(state)

    @hook_config(can_jump_to=["end"])
    async def abefore_model(
        self, state: CompanionState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return self._prepare(state)


def _strip_general_knowledge(text: str) -> str:
    cleaned = _GENERAL_KNOWLEDGE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


class CitationMiddleware(AgentMiddleware):
    """Append the stored book and web citations to the final answer."""

    def _attach(self, state: CompanionState) -> None:
        citations = str(state.get("citations") or "").strip()
        if not citations:
            return
        messages = state.get("messages") or []
        if not messages:
            return
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return
        content = last_message.content if isinstance(last_message.content, str) else ""
        body = _strip_general_knowledge(content)
        if citations in body:
            last_message.content = body
            return
        last_message.content = f"{body}\n\n{citations}"

    def after_agent(self, state: CompanionState, runtime: Runtime) -> dict[str, Any] | None:
        self._attach(state)
        return None

    async def aafter_agent(
        self, state: CompanionState, runtime: Runtime
    ) -> dict[str, Any] | None:
        self._attach(state)
        return None


def build_companion(model=None):
    """Build the companion agent with the guardrail middleware stack."""
    chat = HuggingFaceChat() if model is None else model
    return create_agent(
        model=chat,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ContentFilterMiddleware(),
            *pii_middleware(),
            EvidenceMiddleware(chat),
            SafetyGuardrailMiddleware(chat),
            CitationMiddleware(),
        ],
        checkpointer=MemorySaver(),
    )


companion = build_companion()

"""Clinical Companion: a LangChain agent with guardrail middleware."""

from __future__ import annotations

import asyncio
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
    "You are Wellnest, your everyday wellness companion for diet, exercise, "
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
    "Classify the user message for Wellnest. "
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


def format_book_reference(retrieved: str, hits: list | None = None) -> str:
    """Cite the knowledge pages already tagged on the retrieved passages."""
    grouped = _pages_by_source(hits or [])
    if not grouped:
        pages: list[str] = []
        for page in _PAGE.findall(retrieved or ""):
            if page not in pages:
                pages.append(page)
        if not pages:
            return ""
        listed = ", ".join(f"page {page}" for page in pages)
        return f"### Reference\n\n- {BOOK_TITLE}, {listed}"
    lines = [
        f"- {source or BOOK_TITLE}, {', '.join(f'page {page}' for page in pages)}"
        for source, pages in grouped
    ]
    return "### Reference\n\n" + "\n".join(lines)


def _pages_by_source(hits: list) -> list[tuple[str, list[str]]]:
    order: list[str] = []
    pages: dict[str, list[str]] = {}
    for hit in hits:
        source = str(getattr(hit, "source", "") or "").strip()
        label = _page_label(hit)
        if not label:
            continue
        if source not in pages:
            order.append(source)
            pages[source] = []
        if label not in pages[source]:
            pages[source].append(label)
    return [(source, pages[source]) for source in order]


def _page_label(hit: object) -> str:
    page = getattr(hit, "page", 0) or 0
    if isinstance(page, int) and page > 0:
        return str(page)
    match = _PAGE.search(str(getattr(hit, "text", "") or ""))
    return match.group(1) if match else ""


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


def _citation_block(
    book: str, web_results: list[dict], hits: list | None = None
) -> str:
    parts: list[str] = []
    reference = format_book_reference(book, hits)
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
        parts.append(f"Knowledge passages:\n{book.strip()}")
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
            "citations": _citation_block(book, web_results, hits if book else None),
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
        del runtime
        return await asyncio.to_thread(self._prepare, state)


def _strip_general_knowledge(text: str) -> str:
    cleaned = _GENERAL_KNOWLEDGE.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", cleaned).strip()


_FIXED_REPLY = frozenset(
    {REFUSAL, HABIT_BOUNDARY, INTRODUCTION, OUT_OF_SCOPE, NOT_COVERED}
)

_DECLINE = (
    "cannot provide medical advice",
    "recommend specific dosages",
    "consult a healthcare",
    "contact a clinician",
)


def _withholds_citations(text: str) -> bool:
    """Fixed refusals and model-written declines are not cited answers."""
    stripped = text.strip()
    if stripped in _FIXED_REPLY or stripped.startswith("BMI "):
        return True
    lowered = stripped.lower()
    return any(phrase in lowered for phrase in _DECLINE)


class CitationMiddleware(AgentMiddleware):
    """Append the stored book and web citations to the final answer.

    The cited answer is returned as a state update with the same message id,
    so the checkpoint replaces the streamed answer instead of relying on an
    in-place change to the message object.
    """

    def _cited(self, state: CompanionState) -> AIMessage | None:
        citations = str(state.get("citations") or "").strip()
        messages = state.get("messages") or []
        if not citations or not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None
        content = last_message.content if isinstance(last_message.content, str) else ""
        body = _strip_general_knowledge(content)
        if _withholds_citations(body) or citations in body:
            text = body
        else:
            text = f"{body}\n\n{citations}"
        if text == content:
            return None
        return last_message.model_copy(update={"content": text})

    def after_agent(self, state: CompanionState, runtime: Runtime) -> dict[str, Any] | None:
        del runtime
        cited = self._cited(state)
        return {"messages": [cited]} if cited is not None else None

    async def aafter_agent(
        self, state: CompanionState, runtime: Runtime
    ) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)


def _middleware(chat, *, with_memory: bool) -> list:
    from .user_memory import UserMemoryMiddleware

    stack: list = [
        ContentFilterMiddleware(),
        *pii_middleware(),
    ]
    if with_memory:
        stack.append(UserMemoryMiddleware())
    stack.extend(
        [
            EvidenceMiddleware(chat),
            SafetyGuardrailMiddleware(chat),
            CitationMiddleware(),
        ]
    )
    return stack


def build_companion(model=None):
    """Build the companion agent with the guardrail middleware stack."""
    chat = HuggingFaceChat() if model is None else model
    return create_agent(
        model=chat,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        middleware=_middleware(chat, with_memory=False),
        checkpointer=MemorySaver(),
    )


def make_companion_graph(config: dict | None = None):
    """Graph for the LangGraph API. The server supplies the checkpointer.

    Personal notes are on this graph only. /v1 keeps MemorySaver and no store.
    """
    del config
    from ..config import langfuse_enabled
    from .persistence import open_memory_store

    chat = HuggingFaceChat()
    middleware = _middleware(chat, with_memory=True)
    if langfuse_enabled():
        middleware.insert(0, LangfuseRunMiddleware())
    graph = create_agent(
        model=chat,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        store=open_memory_store(),
    )
    if langfuse_enabled():
        from langfuse.langchain import CallbackHandler

        graph = graph.with_config({"callbacks": [CallbackHandler()]})
    return graph


class LangfuseRunMiddleware(AgentMiddleware):
    """Stamp the thread and the account on the trace. No callback when tracing is off."""

    def _stamp(self, runtime: Runtime) -> None:
        config = getattr(runtime, "config", None)
        if not isinstance(config, dict):
            return
        configurable = config.get("configurable") or {}
        if not isinstance(configurable, dict):
            return
        metadata = config.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            return
        thread_id = configurable.get("thread_id")
        user = configurable.get("langgraph_auth_user") or {}
        identity = user.get("identity") if isinstance(user, dict) else getattr(user, "identity", None)
        if thread_id:
            metadata["langfuse_session_id"] = thread_id
        if identity:
            metadata["langfuse_user_id"] = identity
        metadata["langfuse_tags"] = ["clinical-companion"]

    def before_agent(self, state: AgentState, runtime: Runtime) -> None:
        del state
        self._stamp(runtime)

    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> None:
        del state
        self._stamp(runtime)


companion = build_companion()

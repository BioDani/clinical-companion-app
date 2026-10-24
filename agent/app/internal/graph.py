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

from .intake import CLOSE, INTAKE_START, NOTICE, handle_intake, load_progress
from .trace_mask import NOTES_PREFIX, install_langfuse_mask, note_sensitive
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
    publish_message,
    kept_tavily_results,
    latest_human,
    pii_middleware,
    QUIET,
)
from .knowledge_sources import citation_label
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
    "IN_SCOPE is diet and nutrition, exercise and movement, sleep, healthy habits, "
    "the body including anatomy, muscles, and organs, "
    "or what an illness, a symptom, or a health condition is. "
    "OUT_OF_SCOPE is a question that is not about health, including the assistant's "
    "training, dates, coding, news, or trivia. "
    "Examples: "
    "hello -> GREETING. "
    "What is a good breakfast? -> IN_SCOPE. "
    "What are the leg muscles? -> IN_SCOPE. "
    "What is a cold? -> IN_SCOPE. "
    "What is the flu? -> IN_SCOPE. "
    "What is alzheimer? -> IN_SCOPE. "
    "How do I sleep better? -> IN_SCOPE. "
    "Write a python script -> OUT_OF_SCOPE. "
    "What model are you? -> OUT_OF_SCOPE."
)

RELEVANCE_PROMPT = (
    "Judge which numbered book passages are about the question. "
    "Reply with RELEVANT and the 1-based numbers, for example RELEVANT 1, 3. "
    "Reply with NONE when no passage is about the question. "
    "Do not explain."
)

REWRITE_PROMPT = (
    "Rewrite the latest user question as one search query. "
    "Keep the same subject. Do not change an illness question into a diet or exercise query. "
    "Use earlier turns only to resolve a short follow-up. "
    "Reply with only the query."
)

GROUNDED_PROMPT = (
    "Decide if the reply is supported by the evidence. "
    "Reply with only GROUNDED or UNGROUNDED. "
    "GROUNDED means every habit claim is in the evidence. "
    "UNGROUNDED means the reply adds a claim the evidence does not support.\n\n"
    "Evidence:\n{evidence}\n\nReply:\n{reply}"
)

_PAGE = re.compile(r"\bpage (\d+)\b")
_GENERAL_KNOWLEDGE = re.compile(r"^.*general knowledge.*$", re.IGNORECASE | re.MULTILINE)
_DEFINITION = re.compile(
    r"^\s*what(?:'s| is| are)\s+(?:a |an |the )?(.+?)\s*\??\s*$",
    re.IGNORECASE,
)
_DEFINITION_STOP = {"for", "of", "and", "in", "on", "to", "with", "about", "from"}
_CONDITION = re.compile(
    r"\b(?:colds?|flu|influenza|alzheimer(?:'s|s)?|dementia|fevers?|coughs?|"
    r"diabetes|asthma|cancers?|infections?|diseases?|illness(?:es)?|symptoms?|"
    r"viruses?)\b",
    re.IGNORECASE,
)
_HEALTH_TERM = re.compile(
    r"\b(?:colds?|flu|influenza|alzheimer(?:'s|s)?|dementia|fevers?|coughs?|"
    r"diabetes|asthma|cancers?|infections?|diseases?|illness(?:es)?|symptoms?|"
    r"viruses?|nutrition|diets?|exercise|sleep|foods?|meals?|weight|bmi|"
    r"muscles?|organs?|bones?|joints?|anatomy|hearts?|lungs?|brains?)\b",
    re.IGNORECASE,
)
_REFERENCE_MARK = re.compile(
    r"\bdoi\b|\bet al\b|accessed \d{1,2}\b|\(\s*https?://",
    re.IGNORECASE,
)


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
    lines = []
    for source, pages in grouped:
        name = citation_label(source) if source else BOOK_TITLE
        if pages:
            lines.append(f"- {name}, {', '.join(f'page {page}' for page in pages)}")
        else:
            lines.append(f"- {name}")
    return "### Reference\n\n" + "\n".join(lines)


def _pages_by_source(hits: list) -> list[tuple[str, list[str]]]:
    order: list[str] = []
    pages: dict[str, list[str]] = {}
    for hit in hits:
        source = str(getattr(hit, "source", "") or "").strip()
        label = _page_label(hit)
        if not source and not label:
            continue
        if source not in pages:
            order.append(source)
            pages[source] = []
        if label and label not in pages[source]:
            pages[source].append(label)
    return [(source, pages[source]) for source in order]


def _page_label(hit: object) -> str:
    from .vector_store import page_number

    page = page_number(getattr(hit, "page", 0))
    if page:
        return str(page)
    match = _PAGE.search(str(getattr(hit, "text", "") or ""))
    return match.group(1) if match else ""


def definition_terms(question: str) -> list[str]:
    """Words in a 'what is' question, without articles and short fillers."""
    match = _DEFINITION.match(question or "")
    if not match:
        return []
    words = re.findall(r"[a-z0-9]+", match.group(1).lower())
    return [word for word in words if word not in _DEFINITION_STOP and len(word) > 2]


def condition_terms(question: str) -> list[str]:
    """Condition words a definition must actually name. Diet wording is not included."""
    if not _CONDITION.search(question or ""):
        return []
    named = [term for term in definition_terms(question) if _CONDITION.search(term)]
    if named:
        return named
    return [match.group(0).lower().rstrip("s") for match in _CONDITION.finditer(question or "")]


def _mentions_term(text: str, term: str) -> bool:
    return (
        re.search(rf"\b{re.escape(term)}(?:'s|s)?\b", text or "", re.IGNORECASE)
        is not None
    )


def is_reference_list(text: str) -> bool:
    """True when a passage is a bibliography rather than an explanation."""
    body = text or ""
    marks = _REFERENCE_MARK.findall(body)
    if len(marks) < 3:
        return False
    return len(body) / len(marks) < 250


def _answerable(hits: list, terms: list[str]) -> list:
    kept = []
    for hit in hits:
        text = str(getattr(hit, "text", "") or "")
        if is_reference_list(text):
            continue
        if terms and not all(_mentions_term(text, term) for term in terms):
            continue
        kept.append(hit)
    return kept


def _health_question(question: str) -> bool:
    return _HEALTH_TERM.search(question or "") is not None


def _scope_label(verdict: str) -> str:
    """Use the last explicit label. An earlier mention must not override it."""
    text = (verdict or "").upper()
    found = [
        (text.rfind(label), label)
        for label in ("OUT_OF_SCOPE", "GREETING", "IN_SCOPE")
        if label in text
    ]
    if not found:
        return "OUT_OF_SCOPE"
    return max(found)[1]


def _message_text(message: object) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content or "")


def _relevant_indexes(verdict: str, count: int) -> list[int]:
    """1-based passage numbers. NONE, or a reply that is not RELEVANT, keeps nothing."""
    text = (verdict or "").upper()
    if "NONE" in text or "RELEVANT" not in text:
        return []
    kept: list[int] = []
    for number in re.findall(r"\d+", text):
        index = int(number)
        if 1 <= index <= count and index not in kept:
            kept.append(index)
    return kept


def _is_grounded(verdict: str) -> bool:
    text = (verdict or "").upper()
    if "UNGROUNDED" in text:
        return False
    return "GROUNDED" in text


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


def _stands_alone(question: str) -> bool:
    """A question that names its subject is not a follow-up to the previous turn."""
    return bool(_DEFINITION.match(question or "")) or bool(condition_terms(question))


def _keeps_condition(question: str, rewritten: str) -> bool:
    terms = condition_terms(question)
    if not terms:
        return True
    return all(_mentions_term(rewritten, term) for term in terms)


def _evidence_message(book: str, web_text: str, question: str = "") -> str:
    parts = [
        "Use only the evidence below. Do not add facts from memory. "
        "Do not say this is general knowledge. "
        "Answer this question only, not an earlier question."
    ]
    if question.strip():
        parts.append(f"Question:\n{question.strip()}")
    if book.strip() and web_text.strip():
        parts.append(
            "Use the knowledge passages as the base. "
            "Add a fact from the web evidence only when the passages do not already state it."
        )
    if book.strip():
        parts.append(f"Knowledge passages:\n{book.strip()}")
    if web_text.strip():
        parts.append(f"Web evidence:\n{web_text.strip()}")
    return "\n\n".join(parts)


def _web_evidence(search_text: str) -> tuple[list[dict], str]:
    from .web_search import search_web

    web_results = kept_tavily_results(
        search_web(search_text, include_domains=list(HEALTH_DOMAINS))
    )
    web_text = guard_tavily_results(web_results)
    if web_text == NO_WEB_SOURCES:
        return [], ""
    return web_results, web_text


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

    def _label(self, prompt: str, content: str) -> str:
        verdict = self.model.invoke(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            config=QUIET,
        )
        return _message_text(verdict)

    def _keep(self, question: str, hits: list, terms: list[str] | None = None) -> list:
        hits = _answerable(hits, terms or [])
        if not hits:
            return []
        numbered = "\n\n".join(
            f"{index}. {getattr(hit, 'text', '')}" for index, hit in enumerate(hits, start=1)
        )
        verdict = self._label(
            RELEVANCE_PROMPT,
            f"Question:\n{question}\n\nPassages:\n{numbered}",
        )
        return [hits[index - 1] for index in _relevant_indexes(verdict, len(hits))]

    def _rewrite(self, state: CompanionState, question: str) -> str:
        lines: list[str] = []
        if _stands_alone(question):
            lines.append(f"User: {question}")
        else:
            lines.extend(lookup_lines(state.get("messages") or []))
        if not lines:
            return ""
        rewritten = self._label(REWRITE_PROMPT, "\n".join(lines)).strip()
        if not _keeps_condition(question, rewritten):
            return ""
        return rewritten

    def _prepare(self, state: CompanionState) -> dict[str, Any] | None:
        question = latest_habit_human(state) or latest_human(state)
        label = _scope_label(self._label(SCOPE_PROMPT, question))
        if label == "GREETING":
            return _end(INTRODUCTION)
        if label == "OUT_OF_SCOPE" and not _health_question(question):
            return _end(OUT_OF_SCOPE)

        from .vector_store import search_hits

        terms = condition_terms(question)
        hits = search_hits(question)
        kept = self._keep(question, hits, terms)
        search_text = question
        if not kept:
            rewritten = self._rewrite(state, question)
            if rewritten:
                search_text = rewritten
                hits = search_hits(rewritten)
                kept = self._keep(rewritten, hits, terms)
        if kept:
            book = _book_text(kept)
            return {
                "citations": _citation_block(book, [], kept if book else None),
                "messages": [HumanMessage(content=_evidence_message(book, "", question))],
            }
        web_results, web_text = _web_evidence(search_text)
        if not web_results:
            return _end(NOT_COVERED)
        return {
            "citations": _citation_block("", web_results, None),
            "messages": [HumanMessage(content=_evidence_message("", web_text, question))],
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


def _evidence_text(state: CompanionState) -> str:
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") != "human":
            continue
        text = _message_text(message).strip()
        if text.startswith("Use only the evidence below."):
            return text
    return ""


class GroundingMiddleware(AgentMiddleware):
    """Replace an unsupported draft once, then use the not-covered sentence."""

    state_schema = CompanionState

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def _judge(self, evidence: str, reply: str) -> bool:
        verdict = self.model.invoke(
            [
                {
                    "role": "user",
                    "content": GROUNDED_PROMPT.format(evidence=evidence, reply=reply),
                }
            ],
            config=QUIET,
        )
        return _is_grounded(_message_text(verdict))

    def _review(self, state: CompanionState) -> dict[str, Any] | None:
        messages = state.get("messages") or []
        if not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None
        content = _message_text(last_message).strip()
        if not content or _withholds_citations(content):
            return None
        evidence = _evidence_text(state)
        if not evidence:
            return None
        if self._judge(evidence, content):
            return None
        regenerated = self.model.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": evidence},
            ],
            config=QUIET,
        )
        revised = _message_text(regenerated).strip()
        if revised and self._judge(evidence, revised):
            last_message.content = revised
            return None
        last_message.content = NOT_COVERED
        state["citations"] = ""
        return {"citations": ""}

    def after_model(
        self, state: CompanionState, runtime: Runtime
    ) -> dict[str, Any] | None:
        del runtime
        return self._review(state)

    async def aafter_model(
        self, state: CompanionState, runtime: Runtime
    ) -> dict[str, Any] | None:
        del runtime
        return await asyncio.to_thread(self._review, state)


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
        if _withholds_citations(body) or citations in body:
            last_message.content = body
            return
        last_message.content = f"{body}\n\n{citations}"

    def _publish(self, state: CompanionState) -> dict[str, Any] | None:
        self._attach(state)
        messages = state.get("messages") or []
        if not messages:
            return None
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return None
        return publish_message(last_message)

    def after_agent(self, state: CompanionState, runtime: Runtime) -> dict[str, Any] | None:
        del runtime
        return self._publish(state)

    async def aafter_agent(
        self, state: CompanionState, runtime: Runtime
    ) -> dict[str, Any] | None:
        del runtime
        return self._publish(state)


def _skip_lookup(message: Any) -> bool:
    """Intake turns and personal notes are not search text."""
    kwargs = getattr(message, "additional_kwargs", None) or {}
    if kwargs.get("intake"):
        return True
    text = _message_text(message).strip()
    if text == INTAKE_START or text.startswith(NOTES_PREFIX) or text.startswith(NOTICE):
        return True
    if text == CLOSE:
        return True
    return False


def lookup_lines(messages: list[Any]) -> list[str]:
    lines: list[str] = []
    for message in messages:
        if _skip_lookup(message):
            continue
        text = _message_text(message).strip()
        if not text or text.startswith("Use only the evidence below."):
            continue
        kind = getattr(message, "type", "")
        if kind == "system":
            continue
        speaker = "User" if kind == "human" else "Companion"
        lines.append(f"{speaker}: {text}")
    return lines


def latest_habit_human(state: dict[str, Any]) -> str:
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") != "human" or _skip_lookup(message):
            continue
        return _message_text(message)
    return ""


class IntakeMiddleware(AgentMiddleware):
    """Ask the fixed introduction on the signed-in graph, before the body-mass stop."""

    def _apply(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        from .user_memory import store_from_runtime, user_id_from_runtime

        store = store_from_runtime(runtime)
        user_id = user_id_from_runtime(runtime)
        if store is None or not user_id:
            return None
        if load_progress(store, user_id)["status"] != "pending":
            return None
        human = latest_human(state)
        handled, reply, sensitive = handle_intake(store, user_id, human)
        if not handled or not reply:
            return None
        for text in sensitive:
            note_sensitive(text)
        for message in state.get("messages") or []:
            if getattr(message, "type", "") == "human" and _message_text(message).strip() == human.strip():
                kwargs = dict(getattr(message, "additional_kwargs", None) or {})
                kwargs["intake"] = True
                message.additional_kwargs = kwargs
        return {
            "messages": [AIMessage(content=reply, additional_kwargs={"intake": True})],
            "citations": "",
            "jump_to": "end",
        }

    @hook_config(can_jump_to=["end"])
    def before_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return self._apply(state, runtime)

    @hook_config(can_jump_to=["end"])
    async def abefore_agent(self, state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._apply, state, runtime)


def _middleware(chat, *, with_memory: bool) -> list:
    from .user_memory import UserMemoryMiddleware

    stack: list = []
    if with_memory:
        stack.append(IntakeMiddleware())
    stack.extend(
        [
            ContentFilterMiddleware(),
            *pii_middleware(),
        ]
    )
    if with_memory:
        stack.append(UserMemoryMiddleware())
    stack.extend(
        [
            EvidenceMiddleware(chat),
            GroundingMiddleware(chat),
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

        install_langfuse_mask()
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

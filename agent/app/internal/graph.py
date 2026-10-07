"""Clinical Companion: guard, then BMI, the book, the web, or general knowledge."""

from __future__ import annotations

import operator
import re
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .bmi import BmiTool
from .llm import complete
from .vector_store import search_passages


SMALL_TALK_PROMPT = (
    "You are Clinical Companion, a helpful assistant for diet, exercise, "
    "and healthy habits. "
    "You are not a diagnostic device and you do not replace a physician. "
    "Keep answers concise. "
    "If this is a greeting, introduce yourself as Clinical Companion."
)

ASK_BMI_PROMPT = (
    "You are Clinical Companion. "
    "The user wants a BMI calculation but did not give both weight and height. "
    "Ask for weight in kilograms and height in centimeters or meters. "
    "Do not estimate the numbers. "
    "Keep the reply short."
)

PLAN_PROMPT = (
    "The ebook is in Spanish. "
    "Write one line of Spanish search keywords that are likely to appear in the book. "
    "Do not answer the question. Output keywords only."
)

GRADE_PROMPT = (
    "You grade whether the retrieved passages contain enough information "
    "to answer the user's question accurately. "
    "Reply with exactly one word: SUFFICIENT or INSUFFICIENT."
)

ANSWER_PROMPT = (
    "You are Clinical Companion, an informational wellness assistant. "
    "Elaborate a short English answer from the retrieved passages. "
    "Do not invent claims that are not in the retrieved text. "
    "Do not include page numbers or a bibliography. "
    "A reference is added after your reply. "
    "Do not diagnose or prescribe. "
    "Do not name a medication, a dose, or a treatment regimen. "
    "If the passages mention drugs, leave them out. "
    "Remind the user this is informational guidance only."
)

WEB_ANSWER_PROMPT = (
    "You are Clinical Companion, an informational wellness assistant. "
    "Answer the user's question using the retrieved web sources. "
    "Synthesize the information rather than copying or quoting a single source. "
    "Do not invent claims that are not supported by the retrieved sources. "
    "Prefer relevant and authoritative sources when possible. "
    "For current or recent information, consider the source dates when they are available. "
    "Do not diagnose or prescribe. "
    "Do not name a medication, a dose, or a treatment regimen. "
    "If the sources mention drugs, leave them out and keep only lifestyle habits. "
    "Keep the answer concise and clearly distinguish information from medical advice. "
    "This is informational guidance only."
)

GENERAL_PROMPT = (
    "You are Clinical Companion, an informational wellness assistant. "
    "The curated book and the web search did not contain this answer. "
    "Answer from general knowledge. "
    "Say that this is general knowledge, not a citation from the book or the web. "
    "Do not diagnose or prescribe. "
    "Do not name a medication, a dose, or a treatment regimen. "
    "Keep the answer concise. "
    "Remind the user this is informational guidance only."
)

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

BOOK_TITLE = "Dr. Carlos Jaramillo, Pilares"

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

_GREETINGS = {
    "hello",
    "hi",
    "hey",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
    "good morning",
    "good afternoon",
    "good evening",
    "how are you",
    "who are you",
    "what can you do",
}

_PAGE = re.compile(r"\bpage (\d+)\b")
_BMI_WORD = re.compile(r"\bbmi\b|body mass", re.IGNORECASE)


class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: str
    web_sources: list[dict]
    issues: Annotated[list[str], operator.add]
    route: str
    search_query: str
    searches: int
    coverage: str


def guardrail(user_text: str) -> str | None:
    """Return DANGEROUS or TREATMENT before any model or search call."""
    text = user_text.lower()
    if any(phrase in text for phrase in _DANGEROUS):
        return "DANGEROUS"
    if any(phrase in text for phrase in _TREATMENT):
        return "TREATMENT"
    return None


def _latest_human(state: RouterAgentState) -> str:
    """Return the most recent human message."""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content or "")
    return ""


def _normalized(text: str) -> str:
    return text.strip().lower().rstrip("!?. ")


def _is_smalltalk(text: str) -> bool:
    """Greetings, thanks, and questions about the assistant itself."""
    normalized = _normalized(text)
    if not normalized or normalized in _GREETINGS:
        return True
    first = normalized.split()[0]
    if first in {"hello", "hi", "hey", "thanks", "bye", "goodbye"}:
        return True
    return normalized.startswith(
        (
            "thank you",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "who are you",
            "what can you do",
        )
    )


def _asks_bmi(text: str) -> bool:
    return _BMI_WORD.search(text) is not None


def _tool_readings(question: str, tools: list | None = None) -> list[str]:
    """Run configured tools and return any readings they produce."""
    readings: list[str] = []
    for tool in tools if tools is not None else (BmiTool(),):
        reading = tool.from_text(question)
        if reading:
            readings.append(reading)
    return readings


def _first_line(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.splitlines()[0].strip()


def _first_word(text: str) -> str:
    line = _first_line(text)
    if not line:
        return ""
    return line.split()[0].strip(".,:;").upper()


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


def _with_footer(text: str, footer: str) -> str:
    body = text.rstrip()
    if not footer:
        return body
    return f"{body}\n\n{footer}"


def _stop_with(state: RouterAgentState, content: str) -> dict:
    question = _latest_human(state)
    update: dict = {"messages": [AIMessage(content=content)]}
    if question:
        update["issues"] = [question]
    return update


class DangerousAgent:
    """Handle requests that require a diagnosis."""

    def respond(self, state: RouterAgentState) -> dict:
        return _stop_with(state, REFUSAL)


class HabitBoundaryAgent:
    """Refuse medication and treatment, and offer lifestyle habits only."""

    def respond(self, state: RouterAgentState) -> dict:
        return _stop_with(state, HABIT_BOUNDARY)


class RagAgent:
    """Search the ebook once, then answer or leave the question for the web."""

    def search(self, state: RouterAgentState) -> dict:
        """Plan Spanish keywords and search the ebook in Weaviate once."""
        question = _latest_human(state)
        query = _first_line(
            complete(
                [
                    SystemMessage(content=PLAN_PROMPT),
                    HumanMessage(content=question),
                ]
            )
        ) or question
        return {
            "search_query": query,
            "retrieved": search_passages(query),
            "web_sources": [],
            "searches": 1,
            "coverage": "",
        }

    def grade(self, state: RouterAgentState) -> dict:
        """Grade retrieved passages. Empty retrieval is a miss, with no model call."""
        retrieved = (state.get("retrieved") or "").strip()
        if not retrieved:
            return {"coverage": "MISS"}
        text = complete(
            [
                SystemMessage(content=GRADE_PROMPT),
                HumanMessage(content=_latest_human(state)),
                SystemMessage(content=f"Retrieved context:\n{retrieved}"),
            ]
        )
        coverage = "SUFFICIENT" if _first_word(text) == "SUFFICIENT" else "MISS"
        return {"coverage": coverage}

    def choose_after_grade(self, state: RouterAgentState) -> str:
        if (state.get("coverage") or "").strip().upper() == "SUFFICIENT":
            return "rag_answer"
        return "web_search"

    def answer(self, state: RouterAgentState) -> dict:
        """Elaborate from the book and append the page reference."""
        messages: list[AnyMessage] = [SystemMessage(content=ANSWER_PROMPT)]
        messages.extend(state.get("messages") or [])
        retrieved = (state.get("retrieved") or "").strip()
        if not retrieved:
            retrieved = "The book does not cover this question."
        messages.append(SystemMessage(content=f"Retrieved context:\n{retrieved}"))
        text = complete(messages, max_tokens=1024)
        return {
            "messages": [
                AIMessage(
                    content=_with_footer(text, format_book_reference(state.get("retrieved") or ""))
                )
            ]
        }


class ToolsAgent:
    """Handle deterministic calculations such as BMI."""

    def __init__(self, tools: list | None = None):
        self.tools = list(tools) if tools is not None else [BmiTool()]

    def respond(self, state: RouterAgentState) -> dict:
        lines = _tool_readings(_latest_human(state), self.tools)
        if not lines:
            return {}
        return {"messages": [AIMessage(content="\n".join(lines))]}


class WebSearchAgent:
    """Search the public web using Tavily and answer from web sources."""

    def search(self, state: RouterAgentState) -> dict:
        from .web_search import format_search_results, search_web

        question = _latest_human(state)
        results = search_web(question)
        return {
            "retrieved": format_search_results(results),
            "web_sources": results,
        }

    def choose_after_search(self, state: RouterAgentState) -> str:
        if state.get("web_sources"):
            return "web_answer"
        return "general"

    def answer(self, state: RouterAgentState) -> dict:
        from .web_search import format_source_links

        question = _latest_human(state)
        retrieved = (state.get("retrieved") or "").strip()
        if not retrieved:
            retrieved = "No web sources were found for this question."
        text = complete(
            [
                SystemMessage(content=WEB_ANSWER_PROMPT),
                HumanMessage(content=question),
                SystemMessage(content=f"Web evidence:\n{retrieved}"),
            ],
            max_tokens=1024,
        )
        return {
            "messages": [
                AIMessage(
                    content=_with_footer(
                        text,
                        format_source_links(state.get("web_sources") or []),
                    )
                )
            ]
        }


class GeneralAgent:
    """Answer from model knowledge after the book and the web miss."""

    def answer(self, state: RouterAgentState) -> dict:
        text = complete(
            [
                SystemMessage(content=GENERAL_PROMPT),
                HumanMessage(content=_latest_human(state)),
            ],
            max_tokens=1024,
        )
        return {"messages": [AIMessage(content=text)]}


class _ChatModel:
    """`.invoke` adapter so smalltalk can call the smolagents `complete` helper."""

    def invoke(self, messages):
        return AIMessage(content=complete(messages))


class RouterAgent:
    """Build and compile the Clinical Companion graph."""

    def __init__(self, model, smalltalk_prompt, debug=False):
        self.smalltalk_prompt = smalltalk_prompt
        self.model = model
        self.debug = debug
        self.dangerous = DangerousAgent()
        self.habits = HabitBoundaryAgent()
        self.rag = RagAgent()
        self.tools = ToolsAgent()
        self.web = WebSearchAgent()
        self.general = GeneralAgent()

        graph = StateGraph(RouterAgentState)
        graph.add_node("guard", self.apply_guard)
        graph.add_node("refuse", self.dangerous.respond)
        graph.add_node("habits", self.habits.respond)
        graph.add_node("classify", self.classify)
        graph.add_node("smalltalk", self.respond_smalltalk)
        graph.add_node("bmi", self.tools.respond)
        graph.add_node("rag_search", self.rag.search)
        graph.add_node("rag_grade", self.rag.grade)
        graph.add_node("rag_answer", self.rag.answer)
        graph.add_node("web_search", self.web.search)
        graph.add_node("web_answer", self.web.answer)
        graph.add_node("general", self.general.answer)

        graph.add_conditional_edges(
            "guard",
            lambda state: state["route"],
            {"DANGEROUS": "refuse", "TREATMENT": "habits", "SAFE": "classify"},
        )
        graph.add_conditional_edges(
            "classify",
            lambda state: state["route"],
            {"SMALLTALK": "smalltalk", "BMI": "bmi", "KNOWLEDGE": "rag_search"},
        )
        graph.add_edge("rag_search", "rag_grade")
        graph.add_conditional_edges(
            "rag_grade",
            self.rag.choose_after_grade,
            {"rag_answer": "rag_answer", "web_search": "web_search"},
        )
        graph.add_conditional_edges(
            "web_search",
            self.web.choose_after_search,
            {"web_answer": "web_answer", "general": "general"},
        )
        for node in ("refuse", "habits", "smalltalk", "bmi", "rag_answer", "web_answer", "general"):
            graph.add_edge(node, END)
        graph.set_entry_point("guard")
        self.router_graph = graph.compile(checkpointer=MemorySaver())

    def apply_guard(self, state: RouterAgentState) -> dict:
        route = guardrail(_latest_human(state)) or "SAFE"
        if self.debug:
            print(f"Guard route {route}")
        return {"route": route}

    def classify(self, state: RouterAgentState) -> dict:
        """Pick smalltalk, BMI, or the knowledge cascade without a model call."""
        question = _latest_human(state)
        if _tool_readings(question, self.tools.tools):
            route = "BMI"
        elif _is_smalltalk(question) or _asks_bmi(question):
            route = "SMALLTALK"
        else:
            route = "KNOWLEDGE"
        if self.debug:
            print(f"Classify route {route}")
        return {"route": route}

    def respond_smalltalk(self, state: RouterAgentState) -> dict:
        """Greet the user, or ask for the measurements a BMI calculation needs."""
        question = _latest_human(state)
        if _asks_bmi(question) and not _tool_readings(question, self.tools.tools):
            prompt = ASK_BMI_PROMPT
        else:
            prompt = self.smalltalk_prompt
        messages = [SystemMessage(content=prompt), *list(state.get("messages") or [])]
        if self.debug:
            print(f"Small talk received: {messages}")
        result = self.model.invoke(messages)
        return {"messages": [result]}


def build_graph():
    """Build and return the compiled companion graph."""
    return RouterAgent(_ChatModel(), SMALL_TALK_PROMPT, debug=False).router_graph


router_agent = RouterAgent(_ChatModel(), SMALL_TALK_PROMPT, debug=False)
companion = router_agent.router_graph

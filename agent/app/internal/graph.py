"""Clinical Companion router: guardrail, RAG, web search, and tools."""

from __future__ import annotations

import operator
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
from .book import BookIndex
from .llm import complete


ROUTING_PROMPT = (
    "You are a Router that analyzes the input query and chooses one of 5 options: "
    "SMALLTALK: If the user input is small talk, like greetings and goodbyes. "
    "RAG: If the user input is a question about diet, exercise, nutrition, "
    "sleep, healthy habits, or general wellness information. "
    "TOOLS: If the user input asks for a measurement a tool can compute, "
    "such as height and weight. "
    "DANGEROUS: If the user input asks for a diagnosis or a treatment plan. "
    "END: Default when the input is neither SMALLTALK, RAG, TOOLS, nor DANGEROUS. "
    "Output exactly one word: SMALLTALK, RAG, TOOLS, DANGEROUS, or END."
)

SMALL_TALK_PROMPT = (
    "You are Clinical Companion, a helpful assistant for diet, exercise, "
    "and healthy habits. "
    "You are not a diagnostic device and you do not replace a physician. "
    "Keep answers concise. "
    "If this is a greeting, introduce yourself as Clinical Companion."
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
    "Choose the retrieved passage that best matches the question. "
    "Reply with that passage as one short English quotation: a close translation, "
    "or a slightly clearer elaboration of the same lines. "
    "The whole reply is that quotation. "
    "Do not invent claims that are not in the retrieved text. "
    "Do not answer with page N: text, a keyword list, or the Spanish source. "
    "If the retrieved context says the book does not cover the question, "
    "say that in English. "
    "Do not diagnose or prescribe. "
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
    "Keep the answer concise and clearly distinguish information from medical advice. "
    "This is informational guidance only."
)

REFUSAL = (
    "I can't help with diagnosis or treatment plans. "
    "Please contact a clinician or specialist for this question. "
    "This is informational guidance only."
)


_LABELS = {
    "SMALLTALK",
    "RAG",
    "TOOLS",
    "DANGEROUS",
    "END",
}

_DANGEROUS = (
    "diagnos",
    "prescribe",
    "treatment plan",
    "what medication",
    "do i have",
)

_RAG = (
    "diet",
    "exercise",
    "habit",
    "sleep",
    "nutrition",
)


class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: str
    web_sources: list[dict]
    issues: Annotated[list[str], operator.add]
    route: str
    search_query: str
    searches: int
    coverage: str


def _needs_rag(question: str) -> bool:
    """Return True when the question should use the curated knowledge base."""
    text = question.lower()
    return any(phrase in text for phrase in _RAG)


def _tool_readings(
    question: str,
    tools: list | None = None,
) -> list[str]:
    """Run configured tools and return any readings they produce."""
    readings: list[str] = []

    for tool in tools if tools is not None else (BmiTool(),):
        reading = tool.from_text(question)

        if reading:
            readings.append(reading)

    return readings


def guardrail(
    user_text: str,
    tools: list | None = None,
) -> str | None:
    """
    Perform deterministic routing before calling the LLM router.

    Dangerous requests and tool-compatible requests are handled first.
    Wellness questions are sent directly to RAG.
    """
    text = user_text.lower()

    if any(phrase in text for phrase in _DANGEROUS):
        return "DANGEROUS"

    if _tool_readings(user_text, tools):
        return "TOOLS"

    if _needs_rag(user_text):
        return "RAG"

    return None


def _latest_human(state: RouterAgentState) -> str:
    """Return the most recent human message."""
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") == "human":
            content = message.content

            return (
                content
                if isinstance(content, str)
                else str(content or "")
            )

    return ""


class DangerousAgent:
    """Handle requests that require diagnosis or treatment."""

    def respond(self, state: RouterAgentState) -> dict:
        question = _latest_human(state)

        update: dict = {
            "messages": [
                AIMessage(content=REFUSAL)
            ]
        }

        if question:
            update["issues"] = [question]

        return update


def _first_line(text: str) -> str:
    """Return the first non-empty line."""
    stripped = text.strip()

    if not stripped:
        return ""

    return stripped.splitlines()[0].strip()


def _first_word(text: str) -> str:
    """Return the first word from the first line in uppercase."""
    line = _first_line(text)

    if not line:
        return ""

    return line.split()[0].strip(".,:;").upper()


def _merge_passages(*chunks: str) -> str:
    """Merge retrieved passages while removing exact duplicates."""
    seen: list[str] = []

    for chunk in chunks:
        for block in (chunk or "").split("\n\n"):
            text = block.strip()

            if text and text not in seen:
                seen.append(text)

    return "\n\n".join(seen)


class RagAgent:
    """Search the curated knowledge base and determine whether it is sufficient."""

    def __init__(self, book: BookIndex | None = None):
        self.book = book or BookIndex()

    def plan(self, state: RouterAgentState) -> dict:
        """Generate Spanish keywords for the curated ebook search."""
        question = _latest_human(state)

        messages: list[AnyMessage] = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=question),
        ]

        query = _first_line(complete(messages)) or question

        return {
            "search_query": query,
            "retrieved": "",
            "web_sources": [],
            "searches": 0,
            "coverage": "",
        }

    def search(self, state: RouterAgentState) -> dict:
        """Search the curated knowledge base."""
        question = _latest_human(state)

        query = (
            (state.get("search_query") or "").strip()
            or question
        )

        passages = self.book.search(query)

        attempt = int(state.get("searches") or 0) + 1

        retrieved = _merge_passages(
            state.get("retrieved") or "",
            passages,
        )

        return {
            "retrieved": retrieved,
            "searches": attempt,
        }

    def grade(self, state: RouterAgentState) -> dict:
        """Determine whether the RAG results are sufficient."""
        retrieved = (
            (state.get("retrieved") or "").strip()
            or "none"
        )

        text = complete(
            [
                SystemMessage(content=GRADE_PROMPT),
                HumanMessage(content=_latest_human(state)),
                SystemMessage(
                    content=f"Retrieved context:\n{retrieved}"
                ),
            ]
        )

        return {
            "coverage": _first_word(text)
        }

    def choose_after_grade(
        self,
        state: RouterAgentState,
    ) -> str:
        """
        Decide whether to answer from RAG or fall back to web search.
        """
        coverage = (
            state.get("coverage") or ""
        ).strip().upper()

        if coverage == "SUFFICIENT":
            return "answer"

        return "web"

    def answer(self, state: RouterAgentState) -> dict:
        """Generate an answer using the curated knowledge base."""
        messages: list[AnyMessage] = [
            SystemMessage(content=ANSWER_PROMPT)
        ]

        messages.extend(state.get("messages") or [])

        retrieved = (
            state.get("retrieved") or ""
        ).strip()

        if not retrieved:
            retrieved = "The book does not cover this question."

        messages.append(
            SystemMessage(
                content=f"Retrieved context:\n{retrieved}"
            )
        )

        text = complete(
            messages,
            max_tokens=1024,
        )

        return {
            "messages": [
                AIMessage(content=text)
            ]
        }


class ToolsAgent:
    """Handle deterministic calculations such as BMI."""

    def __init__(self, tools: list | None = None):
        self.tools = (
            list(tools)
            if tools is not None
            else [BmiTool()]
        )

    def respond(self, state: RouterAgentState) -> dict:
        """Run the configured tools."""
        lines = _tool_readings(
            _latest_human(state),
            self.tools,
        )

        if not lines:
            return {}

        return {
            "messages": [
                AIMessage(
                    content="\n".join(lines)
                )
            ]
        }

    def choose_next(
        self,
        state: RouterAgentState,
    ) -> str:
        """
        After a tool response, continue to RAG if the question
        also requires wellness information.
        """
        if _needs_rag(_latest_human(state)):
            return "plan"

        return "end"


class WebSearchAgent:
    """Search the public web using Tavily and answer from web sources."""

    def search(self, state: RouterAgentState) -> dict:
        """Search the web using the user's original question."""
        from .web_search import (
            format_search_results,
            search_web,
        )

        question = _latest_human(state)

        results = search_web(question)

        retrieved = format_search_results(results)

        return {
            "retrieved": retrieved,
            "web_sources": results,
            "searches": int(state.get("searches") or 0) + 1,
        }

    def answer(self, state: RouterAgentState) -> dict:
        """Generate an answer using the web evidence."""
        from .web_search import format_source_links

        question = _latest_human(state)

        retrieved = (
            state.get("retrieved") or ""
        ).strip()

        if not retrieved:
            retrieved = "No web sources were found for this question."

        messages: list[AnyMessage] = [
            SystemMessage(content=WEB_ANSWER_PROMPT),
            HumanMessage(content=question),
            SystemMessage(
                content=f"Web evidence:\n{retrieved}"
            ),
        ]

        text = complete(
            messages,
            max_tokens=1024,
        )

        source_links = format_source_links(
            state.get("web_sources") or []
        )

        if source_links:
            text = (
                f"{text.rstrip()}\n\n"
                f"{source_links}"
            )

        return {
            "messages": [
                AIMessage(content=text)
            ]
        }


class _ChatModel:
    """
    `.invoke` adapter so the router can call the
    smolagents `complete` helper.
    """

    def invoke(self, messages):
        return AIMessage(
            content=complete(messages)
        )


class RouterAgent:
    """Build and compile the Clinical Companion router graph."""

    def __init__(
        self,
        model,
        system_prompt,
        smalltalk_prompt,
        debug=False,
    ):
        self.system_prompt = system_prompt
        self.smalltalk_prompt = smalltalk_prompt
        self.model = model
        self.debug = debug

        self.dangerous = DangerousAgent()
        self.rag = RagAgent()
        self.tools = ToolsAgent()
        self.web = WebSearchAgent()

        router_graph = StateGraph(
            RouterAgentState
        )

        router_graph.add_node(
            "guard",
            self.apply_guard,
        )

        router_graph.add_node(
            "Router",
            self.call_llm,
        )

        router_graph.add_node(
            "Small_Talk",
            self.respond_smalltalk,
        )

        router_graph.add_node(
            "Dangerous",
            self.dangerous.respond,
        )

        router_graph.add_node(
            "Tools",
            self.tools.respond,
        )

        router_graph.add_node(
            "plan",
            self.rag.plan,
        )

        router_graph.add_node(
            "search",
            self.rag.search,
        )

        router_graph.add_node(
            "grade",
            self.rag.grade,
        )

        router_graph.add_node(
            "answer",
            self.rag.answer,
        )

        router_graph.add_node(
            "Web_Search",
            self.web.search,
        )

        router_graph.add_node(
            "Web_Answer",
            self.web.answer,
        )

        router_graph.add_conditional_edges(
            "guard",
            lambda state: state["route"],
            {
                "DANGEROUS": "Dangerous",
                "TOOLS": "Tools",
                "RAG": "plan",
                "ASK_MODEL": "Router",
            },
        )

        router_graph.add_conditional_edges(
            "Router",
            self.find_route,
            {
                "SMALLTALK": "Small_Talk",
                "RAG": "plan",
                "TOOLS": "Tools",
                "DANGEROUS": "Dangerous",
                "END": END,
            },
        )

        router_graph.add_conditional_edges(
            "Tools",
            self.tools.choose_next,
            {
                "plan": "plan",
                "end": END,
            },
        )

        router_graph.add_edge(
            "plan",
            "search",
        )

        router_graph.add_edge(
            "search",
            "grade",
        )

        router_graph.add_conditional_edges(
            "grade",
            self.rag.choose_after_grade,
            {
                "answer": "answer",
                "web": "Web_Search",
            },
        )

        router_graph.add_edge(
            "answer",
            END,
        )

        router_graph.add_edge(
            "Web_Search",
            "Web_Answer",
        )

        router_graph.add_edge(
            "Web_Answer",
            END,
        )

        router_graph.add_edge(
            "Dangerous",
            END,
        )

        router_graph.add_edge(
            "Small_Talk",
            END,
        )

        router_graph.set_entry_point(
            "guard"
        )

        self.router_graph = router_graph.compile(
            checkpointer=MemorySaver()
        )

    def apply_guard(
        self,
        state: RouterAgentState,
    ) -> dict:
        """Apply deterministic guardrail routing."""
        route = (
            guardrail(
                _latest_human(state),
                self.tools.tools,
            )
            or "ASK_MODEL"
        )

        if self.debug:
            print(
                f"Guard route {route}"
            )

        return {
            "route": route
        }

    def call_llm(
        self,
        state: RouterAgentState,
    ) -> dict:
        """Call the LLM router."""
        messages = list(
            state["messages"]
        )

        if self.debug:
            print(
                f"Call LLM received {messages}"
            )

        if self.system_prompt:
            messages = [
                SystemMessage(
                    content=self.system_prompt
                )
            ] + messages

        result = self.model.invoke(
            messages
        )

        if self.debug:
            print(
                f"Call LLM result {result}"
            )

        return {
            "messages": [result]
        }

    def respond_smalltalk(
        self,
        state: RouterAgentState,
    ) -> dict:
        """Generate a response to small talk."""
        messages = list(
            state["messages"]
        )

        if self.debug:
            print(
                f"Small talk received: {messages}"
            )

        messages = [
            SystemMessage(
                content=self.smalltalk_prompt
            )
        ] + messages

        result = self.model.invoke(
            messages
        )

        if self.debug:
            print(
                f"Small talk result {result}"
            )

        return {
            "messages": [result]
        }

    def find_route(
        self,
        state: RouterAgentState,
    ) -> str:
        """Validate the route selected by the LLM."""
        content = state["messages"][-1].content

        label = (
            content.strip()
            if isinstance(content, str)
            else ""
        )

        if self.debug:
            print(
                f"Destination chosen : {label}"
            )

        if label not in _LABELS:
            return "DANGEROUS"

        return label


def build_graph():
    """Build and return the compiled router graph."""
    return RouterAgent(
        _ChatModel(),
        ROUTING_PROMPT,
        SMALL_TALK_PROMPT,
        debug=False,
    ).router_graph


router_agent = RouterAgent(
    _ChatModel(),
    ROUTING_PROMPT,
    SMALL_TALK_PROMPT,
    debug=False,
)

companion = router_agent.router_graph

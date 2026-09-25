"""Clinical Companion router: guardrail, book answers, and BMI."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .bmi import BmiTool
from .book import BookIndex
from .llm import complete

ROUTING_PROMPT = (
    "You are a Router, that analyzes the input query and chooses 4 options:"
    "SMALLTALK: If the user input is small talk, like greetings and good byes."
    "INFORMATIONAL: If the user input is a question about diet, exercise, or healthy habits."
    "DANGEROUS: If the user input is a dangerous query, like asking for a diagnosis or a treatment plan."
    "END: Default, when its neither SMALLTALK, INFORMATIONAL, or DANGEROUS."
    "The output should only be just one word out of the possible 4 : SMALLTALK, INFORMATIONAL, DANGEROUS, END."
)

SMALL_TALK_PROMPT = (
    "You are Clinical Companion, a helpful assistant for diet, exercise, and healthy habits. "
    "You are not a diagnostic device and you do not replace a physician. "
    "Keep answers concise. If this is a greeting, introduce yourself as Clinical Companion."
)

PLAN_PROMPT = (
    "The ebook is in Spanish. "
    "Write one line of Spanish search keywords that are likely to appear in the book. "
    "Do not answer the question. Output keywords only."
)

GRADE_PROMPT = (
    "You grade whether the retrieved passages can answer the user's question. "
    "Reply with exactly one word: SUFFICIENT or REVISE."
)

ANSWER_PROMPT = (
    "You are Clinical Companion, an informational wellness assistant. "
    "Explain the answer in the user's words, in a few short paragraphs, using only the retrieved passages. "
    "Then include one or two quotations copied from those passages and name each page. "
    "If a BMI line is present, include it. "
    "If the retrieved context says the book does not cover the question, say that. "
    "Do not diagnose or prescribe. Remind the user this is informational guidance only."
)

REFUSAL = (
    "I can't help with diagnosis or treatment plans. "
    "Please contact a clinician or specialist for this question. "
    "This is informational guidance only."
)

_LABELS = {"SMALLTALK", "INFORMATIONAL", "DANGEROUS", "END"}
_DANGEROUS = ("diagnos", "prescribe", "treatment plan", "what medication", "do i have")
_INFORMATIONAL = ("diet", "exercise", "habit", "sleep", "nutrition")


class RouterAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: str
    issues: Annotated[list[str], operator.add]
    route: str
    search_query: str
    searches: int
    coverage: str


def guardrail(user_text: str) -> str | None:
    text = user_text.lower()
    if any(phrase in text for phrase in _DANGEROUS):
        return "DANGEROUS"
    if any(phrase in text for phrase in _INFORMATIONAL) or BmiTool().from_text(user_text):
        return "INFORMATIONAL"
    return None


def _latest_human(state: RouterAgentState) -> str:
    for message in reversed(state.get("messages") or []):
        if getattr(message, "type", "") == "human":
            content = message.content
            return content if isinstance(content, str) else str(content or "")
    return ""


class DangerousAgent:
    def respond(self, state: RouterAgentState) -> dict:
        question = _latest_human(state)
        update: dict = {"messages": [AIMessage(content=REFUSAL)]}
        if question:
            update["issues"] = [question]
        return update


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


def _merge_passages(*chunks: str) -> str:
    seen: list[str] = []
    for chunk in chunks:
        for block in (chunk or "").split("\n\n"):
            text = block.strip()
            if text and text not in seen:
                seen.append(text)
    return "\n\n".join(seen)


class InformationalAgent:
    def __init__(self, book: BookIndex | None = None, bmi: BmiTool | None = None):
        self.book = book or BookIndex()
        self.bmi = bmi or BmiTool()

    def plan(self, state: RouterAgentState) -> dict:
        question = _latest_human(state)
        continuing = (state.get("coverage") or "").strip().upper() == "REVISE" and int(
            state.get("searches") or 0
        ) < 2
        messages: list[AnyMessage] = [
            SystemMessage(content=PLAN_PROMPT),
            HumanMessage(content=question),
        ]
        already = (state.get("retrieved") or "").strip() if continuing else ""
        if already:
            messages.append(
                SystemMessage(content=f"Already retrieved:\n{already}\nWrite different keywords.")
            )
        query = _first_line(complete(messages)) or question
        update = {"search_query": query}
        if not continuing:
            update["retrieved"] = ""
            update["searches"] = 0
            update["coverage"] = ""
        return update

    def search(self, state: RouterAgentState) -> dict:
        question = _latest_human(state)
        query = (state.get("search_query") or "").strip() or question
        passages = self.book.search(query)
        attempt = int(state.get("searches") or 0) + 1
        prefix = ""
        if attempt == 1:
            prefix = self.bmi.from_text(question) or ""
        retrieved = _merge_passages(prefix, state.get("retrieved") or "", passages)
        return {"retrieved": retrieved, "searches": attempt}

    def grade(self, state: RouterAgentState) -> dict:
        retrieved = (state.get("retrieved") or "").strip() or "none"
        text = complete(
            [
                SystemMessage(content=GRADE_PROMPT),
                HumanMessage(content=_latest_human(state)),
                SystemMessage(content=f"Retrieved context:\n{retrieved}"),
            ]
        )
        return {"coverage": _first_word(text)}

    def choose_after_grade(self, state: RouterAgentState) -> str:
        coverage = (state.get("coverage") or "").strip().upper()
        if coverage == "REVISE" and int(state.get("searches") or 0) < 2:
            return "plan"
        return "answer"

    def answer(self, state: RouterAgentState) -> dict:
        messages: list[AnyMessage] = [SystemMessage(content=ANSWER_PROMPT)]
        messages.extend(state.get("messages") or [])
        retrieved = (state.get("retrieved") or "").strip()
        if not retrieved:
            retrieved = "The book does not cover this question."
        messages.append(SystemMessage(content=f"Retrieved context:\n{retrieved}"))
        text = complete(messages, max_tokens=1024)
        return {"messages": [AIMessage(content=text)]}


class _ChatModel:
    """`.invoke` adapter so the router can call the smolagents `complete` helper."""

    def invoke(self, messages):
        return AIMessage(content=complete(messages))


class RouterAgent:
    def __init__(self, model, system_prompt, smalltalk_prompt, debug=False):
        self.system_prompt = system_prompt
        self.smalltalk_prompt = smalltalk_prompt
        self.model = model
        self.debug = debug
        self.dangerous = DangerousAgent()
        self.informational = InformationalAgent()

        router_graph = StateGraph(RouterAgentState)
        router_graph.add_node("guard", self.apply_guard)
        router_graph.add_node("Router", self.call_llm)
        router_graph.add_node("Small_Talk", self.respond_smalltalk)
        router_graph.add_node("Dangerous", self.dangerous.respond)
        router_graph.add_node("plan", self.informational.plan)
        router_graph.add_node("search", self.informational.search)
        router_graph.add_node("grade", self.informational.grade)
        router_graph.add_node("answer", self.informational.answer)

        router_graph.add_conditional_edges(
            "guard",
            lambda state: state["route"],
            {
                "DANGEROUS": "Dangerous",
                "INFORMATIONAL": "plan",
                "ASK_MODEL": "Router",
            },
        )
        router_graph.add_conditional_edges(
            "Router",
            self.find_route,
            {
                "SMALLTALK": "Small_Talk",
                "INFORMATIONAL": "plan",
                "DANGEROUS": "Dangerous",
                "END": END,
            },
        )
        router_graph.add_edge("plan", "search")
        router_graph.add_edge("search", "grade")
        router_graph.add_conditional_edges(
            "grade",
            self.informational.choose_after_grade,
            {"plan": "plan", "answer": "answer"},
        )
        router_graph.add_edge("answer", END)
        router_graph.add_edge("Dangerous", END)
        router_graph.add_edge("Small_Talk", END)
        router_graph.set_entry_point("guard")
        self.router_graph = router_graph.compile(checkpointer=MemorySaver())

    def apply_guard(self, state: RouterAgentState) -> dict:
        route = guardrail(_latest_human(state)) or "ASK_MODEL"
        if self.debug:
            print(f"Guard route {route}")
        return {"route": route}

    def call_llm(self, state: RouterAgentState) -> dict:
        messages = list(state["messages"])
        if self.debug:
            print(f"Call LLM received {messages}")
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        result = self.model.invoke(messages)
        if self.debug:
            print(f"Call LLM result {result}")
        return {"messages": [result]}

    def respond_smalltalk(self, state: RouterAgentState) -> dict:
        messages = list(state["messages"])
        if self.debug:
            print(f"Small talk received: {messages}")
        messages = [SystemMessage(content=self.smalltalk_prompt)] + messages
        result = self.model.invoke(messages)
        if self.debug:
            print(f"Small talk result {result}")
        return {"messages": [result]}

    def find_route(self, state: RouterAgentState) -> str:
        content = state["messages"][-1].content
        label = content.strip() if isinstance(content, str) else ""
        if self.debug:
            print(f"Destination chosen : {label}")
        if label not in _LABELS:
            return "DANGEROUS"
        return label


def build_graph():
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

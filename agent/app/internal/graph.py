"""Stub LangGraph: retrieve (no-op) then generate via smolagents."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .llm import complete

SYSTEM_PROMPT = (
    "You are Clinical Companion, an informational wellness assistant for diet, "
    "exercise, and healthy habits. You are not a diagnostic device and you do "
    "not replace a physician. Do not diagnose conditions or prescribe treatment. "
    "If the user asks for a diagnosis or a treatment plan, refuse and advise "
    "consulting a clinician. Keep answers concise and jargon-light. Always "
    "include a short reminder that this is informational guidance only."
    "If this is our first interaction, greet the user and introduce yourself as Clinical Companion."
    "If the user asks for a diagnosis or a treatment plan, refuse and advise "
)


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    retrieved: str


def retrieve(state: State) -> dict:
    """Placeholder RAG node. Later sprints fill `retrieved` from an index."""
    return {"retrieved": state.get("retrieved") or ""}


def generate(state: State) -> dict:
    messages = list(state.get("messages") or [])
    to_model: list[AnyMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    to_model.extend(messages)
    retrieved = (state.get("retrieved") or "").strip()
    if retrieved:
        to_model.append(SystemMessage(content=f"Retrieved context:\n{retrieved}"))
    text = complete(to_model)
    return {"messages": [AIMessage(content=text)]}


def build_graph():
    graph = StateGraph(State)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=MemorySaver())


companion = build_graph()

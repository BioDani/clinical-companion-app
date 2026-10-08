"""Clinical Companion: a LangChain agent with guardrail middleware."""

from __future__ import annotations

import json
import re

from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from .guardrails import (
    HABIT_BOUNDARY,
    REFUSAL,
    TAVILY_PAYLOAD,
    ContentFilterMiddleware,
    SafetyGuardrailMiddleware,
    TavilyResultMiddleware,
    pii_middleware,
)
from .llm import HuggingFaceChat

BOOK_TITLE = "Dr. Carlos Jaramillo, Pilares"

SYSTEM_PROMPT = (
    "You are Clinical Companion, a helpful assistant for diet, exercise, "
    "and healthy habits. "
    "You are not a diagnostic device and you do not replace a physician. "
    "Keep answers concise and in English. "
    "If this is a greeting, introduce yourself as Clinical Companion. "
    "If the user wants a BMI calculation but did not give both weight and height, "
    "ask for weight in kilograms and height in centimeters or meters. "
    "Do not estimate the numbers. "
    "For a knowledge question, call search_ebook first. "
    "The ebook is in Spanish, so pass Spanish keywords. "
    "If the passages are enough, answer from them and include the reference block "
    "from the tool. Do not invent claims that are not in the passages. "
    "If the ebook does not cover the question, call search_web. "
    "Answer from the web evidence the tool returns and include its Sources block. "
    "If the tool says no usable web sources were found, answer from general knowledge "
    "and say that this is general knowledge, not a citation from the book or the web. "
    "Do not diagnose or prescribe. "
    "Do not name a medication, a dose, or a treatment regimen. "
    "If a source mentions drugs, leave them out and keep only lifestyle habits. "
    "Remind the user this is informational guidance only."
)

_PAGE = re.compile(r"\bpage (\d+)\b")


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


@tool
def search_ebook(query: str) -> str:
    """Search the Spanish ebook. Pass Spanish keywords likely to appear in the book."""
    from .vector_store import search_passages

    text = (search_passages(query) or "").strip()
    if not text:
        return "The book does not cover this question."
    reference = format_book_reference(text)
    if not reference:
        return text
    return f"{text}\n\n{reference}"


@tool
def search_web(query: str) -> str:
    """Search the public web after the ebook does not cover the question."""
    from .web_search import search_web as run_search

    results = run_search(query)
    return TAVILY_PAYLOAD + json.dumps(results)


def build_companion(model=None):
    """Build the companion agent with the guardrail middleware stack."""
    chat = HuggingFaceChat() if model is None else model
    return create_agent(
        model=chat,
        tools=[search_ebook, search_web],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            ContentFilterMiddleware(),
            *pii_middleware(),
            TavilyResultMiddleware(),
            SafetyGuardrailMiddleware(chat),
        ],
        checkpointer=MemorySaver(),
    )


companion = build_companion()

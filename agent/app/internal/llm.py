"""Hugging Face chat model for the LangChain companion."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI

from ..config import hf_model, hf_token

HF_OPENAI_BASE = "https://router.huggingface.co/v1"


class HuggingFaceChat(BaseChatModel):
    """ChatOpenAI pointed at the Hugging Face router.

    The token is read on the first model call, so importing the agent does not
    require ``HF_TOKEN``.
    """

    @property
    def _llm_type(self) -> str:
        return "huggingface-router"

    def _client(self) -> ChatOpenAI:
        return ChatOpenAI(
            model=hf_model(),
            api_key=hf_token(),
            base_url=HF_OPENAI_BASE,
            temperature=0,
        )

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        return self._client()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )

    def bind_tools(self, tools, **kwargs):
        return self._client().bind_tools(tools, **kwargs)

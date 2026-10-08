"""Runtime config for the Clinical Companion agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Small non-thinking instruct model. SmolLM is not on any Inference Provider.
# Live on featherless and nscale, so a short grade or keyword line stays a direct reply.
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
AGENT_MODEL_ID = "clinical-companion"


def hf_token() -> str:
    token = (os.getenv("HF_TOKEN") or "").strip()
    if not token.startswith("hf_"):
        raise ValueError(
            "Set HF_TOKEN to a Hugging Face token that starts with hf_. "
            "Create one at https://huggingface.co/settings/tokens"
        )
    return token


def hf_model() -> str:
    return (os.getenv("HF_MODEL") or "").strip() or DEFAULT_HF_MODEL


def hf_embedding_model() -> str:
    return (os.getenv("HF_EMBEDDING_MODEL") or "").strip() or DEFAULT_HF_EMBEDDING_MODEL


def weaviate_host() -> str:
    return (os.getenv("WEAVIATE_HOST") or "").strip() or "localhost"


def jwt_secret() -> str:
    secret = (os.getenv("JWT_SECRET") or "").strip()
    if not secret:
        raise ValueError(
            "JWT_SECRET is required so the agent can verify tokens from rbac"
        )
    return secret


def jwt_algorithm() -> str:
    return (os.getenv("JWT_ALGORITHM") or "").strip() or "HS256"


def langfuse_enabled() -> bool:
    """True when both Langfuse keys are set and this process is not pytest."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    public = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public and secret)

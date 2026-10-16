"""Runtime config for the Clinical Companion agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

# Small non-thinking instruct model. SmolLM is not on any Inference Provider.
# Live on featherless and nscale, so a short grade or keyword line stays a direct reply.
DEFAULT_HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Qwen3 non-thinking instruct sampling. Temperature 0 repeats itself on this model.
DEFAULT_HF_TEMPERATURE = 0.7
DEFAULT_HF_TOP_P = 0.8
DEFAULT_HF_TOP_K = 20
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


def hf_temperature() -> float:
    """Sampling temperature for chat completions. 0 is greedy; 2 is the top of the range."""
    return _unit_float(
        "HF_TEMPERATURE",
        DEFAULT_HF_TEMPERATURE,
        low=0.0,
        high=2.0,
    )


def hf_top_p() -> float:
    """Nucleus sampling cutoff. Kept above 0 so the cutoff still includes a token."""
    return _unit_float("HF_TOP_P", DEFAULT_HF_TOP_P, low=0.0, high=1.0, low_exclusive=True)


def hf_top_k() -> int:
    """How many top tokens sampling may draw from. Sent to the router as top_k."""
    raw = (os.getenv("HF_TOP_K") or "").strip()
    if not raw:
        return DEFAULT_HF_TOP_K
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError("HF_TOP_K must be an integer of at least 1") from exc
    if value < 1:
        raise ValueError("HF_TOP_K must be an integer of at least 1")
    return value


def _unit_float(
    name: str,
    default: float,
    *,
    low: float,
    high: float,
    low_exclusive: bool = False,
) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(_sampling_range(name, low, high, low_exclusive)) from exc
    if value < low or value > high or (low_exclusive and value == low):
        raise ValueError(_sampling_range(name, low, high, low_exclusive))
    return value


def _sampling_range(name: str, low: float, high: float, low_exclusive: bool) -> str:
    if low_exclusive:
        return f"{name} must be a number greater than {low:g} and at most {high:g}"
    return f"{name} must be a number from {low:g} to {high:g}"


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
    """True when both Langfuse keys are set and this process is not pytest.

    Compose sets LANGFUSE_ENABLED to 1 only when COMPOSE_PROFILES includes
    langfuse. An empty or false value keeps traces off even if the keys exist.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    if "LANGFUSE_ENABLED" in os.environ:
        switch = os.environ["LANGFUSE_ENABLED"].strip().lower()
        if switch in {"", "0", "false", "off", "no"}:
            return False
    public = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip()
    secret = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip()
    return bool(public and secret)

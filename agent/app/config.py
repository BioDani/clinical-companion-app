"""Runtime config for the Clinical Companion agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-72B-Instruct"
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


def jwt_secret() -> str:
    secret = (os.getenv("JWT_SECRET") or "").strip()
    if not secret:
        raise ValueError(
            "JWT_SECRET is required so the agent can verify tokens from rbac"
        )
    return secret


def jwt_algorithm() -> str:
    return (os.getenv("JWT_ALGORITHM") or "").strip() or "HS256"

"""Runtime config reads the environment and rejects incomplete secrets."""

from __future__ import annotations

import pytest

from app.config import (
    AGENT_MODEL_ID,
    DEFAULT_HF_MODEL,
    DEFAULT_HF_TEMPERATURE,
    DEFAULT_HF_TOP_K,
    DEFAULT_HF_TOP_P,
    hf_model,
    hf_temperature,
    hf_token,
    hf_top_k,
    hf_top_p,
    jwt_algorithm,
    jwt_secret,
)


def test_agent_model_id_is_stable():
    assert AGENT_MODEL_ID == "clinical-companion"


def test_hf_token_returns_stripped_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "  hf_test_token  ")
    assert hf_token() == "hf_test_token"


@pytest.mark.parametrize("raw", ["", "   ", "sk-not-hf", "HF_uppercase"])
def test_hf_token_rejects_missing_or_malformed(monkeypatch: pytest.MonkeyPatch, raw: str):
    monkeypatch.setenv("HF_TOKEN", raw)
    with pytest.raises(ValueError, match="HF_TOKEN"):
        hf_token()


def test_hf_model_defaults_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert hf_model() == DEFAULT_HF_MODEL


def test_hf_model_uses_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_MODEL", "  org/small-model  ")
    assert hf_model() == "org/small-model"


def test_hf_model_blank_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_MODEL", "   ")
    assert hf_model() == DEFAULT_HF_MODEL


def test_sampling_defaults_when_unset(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TEMPERATURE", raising=False)
    monkeypatch.delenv("HF_TOP_P", raising=False)
    monkeypatch.delenv("HF_TOP_K", raising=False)
    assert hf_temperature() == DEFAULT_HF_TEMPERATURE
    assert hf_top_p() == DEFAULT_HF_TOP_P
    assert hf_top_k() == DEFAULT_HF_TOP_K


def test_sampling_uses_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TEMPERATURE", " 0.2 ")
    monkeypatch.setenv("HF_TOP_P", " 0.95 ")
    monkeypatch.setenv("HF_TOP_K", " 40 ")
    assert hf_temperature() == 0.2
    assert hf_top_p() == 0.95
    assert hf_top_k() == 40


def test_sampling_blank_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TEMPERATURE", "  ")
    monkeypatch.setenv("HF_TOP_P", "")
    monkeypatch.setenv("HF_TOP_K", "   ")
    assert hf_temperature() == DEFAULT_HF_TEMPERATURE
    assert hf_top_p() == DEFAULT_HF_TOP_P
    assert hf_top_k() == DEFAULT_HF_TOP_K


@pytest.mark.parametrize(
    ("name", "raw", "reader"),
    [
        ("HF_TEMPERATURE", "hot", hf_temperature),
        ("HF_TEMPERATURE", "2.1", hf_temperature),
        ("HF_TEMPERATURE", "-0.1", hf_temperature),
        ("HF_TOP_P", "wide", hf_top_p),
        ("HF_TOP_P", "0", hf_top_p),
        ("HF_TOP_P", "1.1", hf_top_p),
        ("HF_TOP_K", "many", hf_top_k),
        ("HF_TOP_K", "0", hf_top_k),
        ("HF_TOP_K", "1.5", hf_top_k),
    ],
)
def test_sampling_rejects_out_of_range(monkeypatch: pytest.MonkeyPatch, name, raw, reader):
    monkeypatch.setenv(name, raw)
    with pytest.raises(ValueError, match=name):
        reader()


def test_jwt_secret_returns_stripped_value(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JWT_SECRET", "  lab-secret  ")
    assert jwt_secret() == "lab-secret"


def test_jwt_secret_rejects_blank(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JWT_SECRET", "   ")
    with pytest.raises(ValueError, match="JWT_SECRET"):
        jwt_secret()


def test_jwt_algorithm_defaults_to_hs256(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JWT_ALGORITHM", raising=False)
    assert jwt_algorithm() == "HS256"


def test_jwt_algorithm_uses_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JWT_ALGORITHM", " HS384 ")
    assert jwt_algorithm() == "HS384"

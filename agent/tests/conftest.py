"""Isolate unit tests from a developer .env before the app imports config."""

from __future__ import annotations

import os

os.environ["JWT_SECRET"] = "unit-test-jwt-secret-32-bytes-long"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["HF_TOKEN"] = ""
os.environ["HF_MODEL"] = ""

import pytest
from fastapi.testclient import TestClient

from tests.tokens import make_token


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-optional",
        action="store_true",
        default=False,
        help="Run optional graph and LLM tests",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-optional"):
        return
    skip = pytest.mark.skip(reason="optional test; pass --run-optional to run")
    for item in items:
        if item.get_closest_marker("optional"):
            item.add_marker(skip)


@pytest.fixture
def token() -> str:
    return make_token()


@pytest.fixture
def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client

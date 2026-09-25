"""Open WebUI boot script: wait for rbac, mint a JWT, then exec the UI."""

from __future__ import annotations

import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "open-webui-entrypoint.py"
)


class _Response:
    def __init__(self, payload: bytes | dict, status: int = 200):
        if isinstance(payload, dict):
            payload = json.dumps(payload).encode()
        self._payload = payload
        self.status = status

    def read(self, _n: int = -1) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


@pytest.fixture
def entrypoint():
    spec = importlib.util.spec_from_file_location("open_webui_entrypoint", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wait_healthy_returns_once_status_is_200(monkeypatch, entrypoint, capsys):
    calls = {"n": 0}

    def urlopen(url, timeout=3):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("down")
        assert url == "http://rbac.test/health"
        assert timeout == 3
        return _Response(b"ok", status=200)

    monkeypatch.setattr(entrypoint, "RBAC_URL", "http://rbac.test")
    monkeypatch.setattr(entrypoint.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(entrypoint.time, "sleep", lambda _seconds: None)

    entrypoint._wait_healthy(attempts=3)
    assert calls["n"] == 2
    assert capsys.readouterr().err == ""


def test_wait_healthy_exits_when_rbac_stays_down(monkeypatch, entrypoint, capsys):
    def urlopen(url, timeout=3):
        raise TimeoutError("timed out")

    monkeypatch.setattr(entrypoint, "RBAC_URL", "http://rbac.test")
    monkeypatch.setattr(entrypoint.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(entrypoint.time, "sleep", lambda _seconds: None)

    with pytest.raises(SystemExit) as exc:
        entrypoint._wait_healthy(attempts=2)
    assert exc.value.code == 1
    assert "rbac not healthy at http://rbac.test" in capsys.readouterr().err


def test_login_returns_access_token(monkeypatch, entrypoint):
    captured: dict = {}

    def urlopen(request, timeout=10):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["content_type"] = request.get_header("Content-type")
        assert timeout == 10
        return _Response({"access_token": "jwt-1"})

    monkeypatch.setattr(entrypoint, "RBAC_URL", "http://rbac.test")
    monkeypatch.setattr(entrypoint, "USERNAME", "ada")
    monkeypatch.setattr(entrypoint, "PASSWORD", "secret")
    monkeypatch.setattr(entrypoint.urllib.request, "urlopen", urlopen)

    assert entrypoint._login() == "jwt-1"
    assert captured["url"] == "http://rbac.test/auth/login"
    assert captured["body"] == {"username": "ada", "password": "secret"}
    assert captured["content_type"] == "application/json"


def test_login_exits_on_http_error(monkeypatch, entrypoint, capsys):
    def urlopen(request, timeout=10):
        raise urllib.error.HTTPError(request.full_url, 401, "nope", hdrs=None, fp=None)

    monkeypatch.setattr(entrypoint.urllib.request, "urlopen", urlopen)
    with pytest.raises(SystemExit) as exc:
        entrypoint._login()
    assert exc.value.code == 1
    assert "HTTP 401" in capsys.readouterr().err


def test_login_exits_when_token_missing(monkeypatch, entrypoint, capsys):
    monkeypatch.setattr(
        entrypoint.urllib.request,
        "urlopen",
        lambda request, timeout=10: _Response({"detail": "no"}),
    )
    with pytest.raises(SystemExit) as exc:
        entrypoint._login()
    assert exc.value.code == 1
    assert "no access_token" in capsys.readouterr().err


def test_main_requires_admin_password(monkeypatch, entrypoint, capsys):
    monkeypatch.setattr(entrypoint, "PASSWORD", "")
    with pytest.raises(SystemExit) as exc:
        entrypoint.main()
    assert exc.value.code == 1
    assert "ADMIN_PASSWORD" in capsys.readouterr().err


def test_main_installs_token_and_execs_webui(monkeypatch, entrypoint, capsys):
    recorded: list[tuple] = []

    monkeypatch.setattr(entrypoint, "PASSWORD", "secret")
    monkeypatch.setattr(entrypoint, "_wait_healthy", lambda: None)
    monkeypatch.setattr(entrypoint, "_login", lambda: "minted-token")
    monkeypatch.setattr(
        entrypoint.os,
        "execvp",
        lambda cmd, args: recorded.append((cmd, args)),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEYS", raising=False)

    entrypoint.main()

    assert recorded == [("bash", ["bash", "start.sh"])]
    assert entrypoint.os.environ["OPENAI_API_KEY"] == "minted-token"
    assert entrypoint.os.environ["OPENAI_API_KEYS"] == "minted-token"
    assert "len=12" in capsys.readouterr().out

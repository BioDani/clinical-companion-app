#!/usr/bin/env python3
"""Login to rbac, then start Open WebUI with that JWT as OPENAI_API_KEY."""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

RBAC_URL = os.environ.get("RBAC_URL", "http://rbac:8000").rstrip("/")
USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
PASSWORD = os.environ.get("ADMIN_PASSWORD", "")


def _wait_healthy(attempts: int = 60) -> None:
    last_error = "unreachable"
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(f"{RBAC_URL}/health", timeout=3) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(1)
    print(f"rbac not healthy at {RBAC_URL}: {last_error}", file=sys.stderr)
    raise SystemExit(1)


def _login() -> str:
    body = json.dumps({"username": USERNAME, "password": PASSWORD}).encode()
    request = urllib.request.Request(
        f"{RBAC_URL}/auth/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        print(f"rbac login failed: HTTP {exc.code}", file=sys.stderr)
        raise SystemExit(1) from exc
    except urllib.error.URLError as exc:
        print(f"rbac login failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    token = payload.get("access_token")
    if not token:
        print("rbac login returned no access_token", file=sys.stderr)
        raise SystemExit(1)
    return str(token)


def main() -> None:
    if not PASSWORD:
        print("ADMIN_PASSWORD is required to mint an Open WebUI agent token", file=sys.stderr)
        raise SystemExit(1)
    _wait_healthy()
    token = _login()
    os.environ["OPENAI_API_KEY"] = token
    os.environ["OPENAI_API_KEYS"] = token
    print(f"minted agent API token (len={len(token)})", flush=True)
    os.execvp("bash", ["bash", "start.sh"])


if __name__ == "__main__":
    main()

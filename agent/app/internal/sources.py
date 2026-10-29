"""Admin-edited citation title and URL, layered over the built-in catalog."""

from __future__ import annotations

import json
from pathlib import Path

from .inclusion import inclusion_file
from .knowledge_sources import KNOWLEDGE_SOURCES

_SOURCES_NAME = "sources.json"


def sources_file() -> Path:
    return inclusion_file().with_name(_SOURCES_NAME)


def load_sources() -> dict[str, dict[str, str]]:
    """Return overlay entries. A missing file means the built-in catalog is used."""
    path = sources_file()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    sources: dict[str, dict[str, str]] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        name = Path(key).name
        if not name:
            continue
        title = value.get("title")
        url = value.get("url")
        sources[name] = {
            "title": title if isinstance(title, str) else "",
            "url": url if isinstance(url, str) else "",
        }
    return sources


def save_sources(sources: dict[str, dict[str, str]]) -> None:
    path = sources_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        name: {"title": fields.get("title") or "", "url": fields.get("url") or ""}
        for name, fields in sorted(sources.items())
    }
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def public_source(filename: str) -> dict[str, str]:
    """Overlay wins, including an empty URL. Otherwise the built-in catalog."""
    overlay = load_sources().get(filename)
    if overlay is not None:
        return overlay
    entry = KNOWLEDGE_SOURCES.get(filename) or {}
    return {
        "title": str(entry.get("title") or ""),
        "url": str(entry.get("url") or ""),
    }


def citation_label(filename: str) -> str:
    """Return a Markdown link for a known PDF, or the filename itself."""
    entry = public_source(filename)
    url = entry["url"].strip()
    if not url:
        return filename
    title = entry["title"].strip() or filename
    return f"[{title}]({url})"

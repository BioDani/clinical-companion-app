"""Persisted Indexed-off filenames for the knowledge folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .book import list_knowledge_pdfs

_INCLUSION_DIR = Path("/data/knowledge-inclusion")
_INCLUSION_NAME = "disabled.json"


def inclusion_file() -> Path:
    return _INCLUSION_DIR / _INCLUSION_NAME


def load_disabled() -> set[str]:
    """Return stored disabled basenames. A missing file means none are disabled."""
    path = inclusion_file()
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    names = payload.get("disabled") if isinstance(payload, dict) else None
    if not isinstance(names, list):
        return set()
    disabled: set[str] = set()
    for item in names:
        if not isinstance(item, str):
            continue
        name = Path(item).name
        if name:
            disabled.add(name)
    return disabled


def save_disabled(names: Iterable[str]) -> None:
    """Write unique basenames as the disabled list."""
    unique = sorted({Path(name).name for name in names if str(name).strip()})
    path = inclusion_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps({"disabled": unique}, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def included_pdfs(root: Path | None = None) -> list[Path]:
    """PDFs in the folder that are not in the disabled list."""
    disabled = load_disabled()
    return [path for path in list_knowledge_pdfs(root) if path.name not in disabled]

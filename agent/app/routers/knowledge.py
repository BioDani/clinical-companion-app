"""Admin catalog of knowledge PDFs and the Indexed toggle."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from starlette.datastructures import UploadFile as FormUpload

from ..auth import Principal, require_principal
from ..internal.book import list_knowledge_pdfs
from ..internal.inclusion import load_disabled, save_disabled
from ..internal.sources import load_sources, public_source, save_sources
from ..internal.vector_store import ensure_indexed

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

_MAX_PDF_BYTES = 64 * 1024 * 1024


def _index_soon() -> None:
    """Store the file first. Rebuilding every PDF must not hold the upload open."""
    threading.Thread(target=ensure_indexed, name="knowledge-index", daemon=True).start()


def _folder() -> Path:
    from ..internal import book

    return book._KNOWLEDGE


def _require_admin(principal: Annotated[Principal, Depends(require_principal)]) -> Principal:
    if principal.role != "admin":
        raise HTTPException(status_code=403, detail="forbidden")
    return principal


def _pdf_name(file_name: str) -> str | None:
    name = Path(file_name).name
    if name != file_name or name in {".", ".."} or "/" in file_name or "\\" in file_name:
        return None
    if not name.lower().endswith(".pdf") or name == ".pdf":
        return None
    return name


def _pdf_by_name(file_name: str) -> Path | None:
    name = _pdf_name(file_name)
    if name is None:
        return None
    for path in list_knowledge_pdfs():
        if path.name == name:
            return path
    return None


def _row(path: Path, disabled: set[str]) -> dict[str, str | bool]:
    name = path.name
    entry = public_source(name)
    return {
        "file_name": name,
        "local_url": f"/knowledge/files/{name}",
        "title": entry["title"],
        "web_url": entry["url"],
        "indexed": name not in disabled,
    }


def _pdf_bytes(data: bytes) -> None:
    if not data.startswith(b"%PDF") or len(data) > _MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="file must be a PDF")


def _write_pdf(dest: Path, data: bytes) -> None:
    _pdf_bytes(data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)


def _store_source(name: str, title: str | None, web_url: str | None, *, previous: str | None = None) -> None:
    sources = load_sources()
    old = previous if previous and previous != name else None
    carried = sources.pop(old, None) if old else None
    if title is None and web_url is None:
        if carried is not None:
            sources[name] = carried
        elif old:
            inherited = public_source(old)
            if inherited["title"] or inherited["url"]:
                sources[name] = inherited
        if carried is not None or old:
            save_sources(sources)
        return
    base = carried or sources.get(name) or public_source(old or name)
    if title is not None:
        base = {**base, "title": title}
    if web_url is not None:
        base = {**base, "url": web_url}
    sources[name] = {"title": base.get("title") or "", "url": base.get("url") or ""}
    save_sources(sources)


def _move_disabled(old: str, new: str) -> None:
    if old == new:
        return
    disabled = load_disabled()
    if old not in disabled:
        return
    disabled.discard(old)
    disabled.add(new)
    save_disabled(disabled)


@router.get("")
def list_files(_admin: Annotated[Principal, Depends(_require_admin)]) -> dict[str, list]:
    disabled = load_disabled()
    return {"files": [_row(path, disabled) for path in list_knowledge_pdfs()]}


@router.post("/files", status_code=201)
async def create_file(
    _admin: Annotated[Principal, Depends(_require_admin)],
    file: Annotated[UploadFile, File()],
    title: Annotated[str, Form()] = "",
    web_url: Annotated[str, Form()] = "",
) -> dict[str, str | bool]:
    name = _pdf_name(Path(file.filename or "").name)
    if name is None:
        raise HTTPException(status_code=400, detail="file must be a PDF")
    if _pdf_by_name(name) is not None:
        raise HTTPException(status_code=409, detail="already exists")
    data = await file.read()
    _write_pdf(_folder() / name, data)
    if title or web_url:
        _store_source(name, title, web_url)
    _index_soon()
    path = _pdf_by_name(name)
    if path is None:
        raise HTTPException(status_code=500, detail="file was not stored")
    return _row(path, load_disabled())


@router.patch("/files/{file_name}")
async def update_file(
    file_name: str,
    request: Request,
    _admin: Annotated[Principal, Depends(_require_admin)],
) -> dict[str, str | bool]:
    form = await request.form()
    current = _pdf_by_name(file_name)
    if current is None:
        raise HTTPException(status_code=404, detail="not found")
    raw_name = form.get("file_name")
    new_name = raw_name if isinstance(raw_name, str) else None
    title = form.get("title") if "title" in form else None
    web_url = form.get("web_url") if "web_url" in form else None
    title = title if isinstance(title, str) else None
    web_url = web_url if isinstance(web_url, str) else None
    upload = form.get("file")
    target_name = current.name if new_name is None else _pdf_name(new_name)
    if target_name is None:
        raise HTTPException(status_code=400, detail="file must be a PDF")
    existing = _pdf_by_name(target_name)
    if existing is not None and existing.name != current.name:
        raise HTTPException(status_code=409, detail="already exists")
    data = b""
    if isinstance(upload, FormUpload) and upload.filename:
        data = await upload.read()
    renamed = target_name != current.name
    replaced = bool(data)
    if replaced:
        _pdf_bytes(data)
    if renamed or replaced:
        dest = _folder() / target_name
        if replaced:
            _write_pdf(dest, data)
            if renamed and current.exists():
                current.unlink()
        else:
            current.replace(dest)
        _move_disabled(current.name, target_name)
        _store_source(target_name, title, web_url, previous=current.name)
        _index_soon()
    elif title is not None or web_url is not None:
        _store_source(current.name, title, web_url)
    path = _pdf_by_name(target_name)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    return _row(path, load_disabled())


@router.get("/files/{file_name}")
def read_file(
    file_name: str,
    _admin: Annotated[Principal, Depends(_require_admin)],
) -> FileResponse:
    path = _pdf_by_name(file_name)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path, media_type="application/pdf", filename=path.name, content_disposition_type="inline")


@router.delete("/files/{file_name}", status_code=204)
def delete_file(
    file_name: str,
    _admin: Annotated[Principal, Depends(_require_admin)],
) -> Response:
    path = _pdf_by_name(file_name)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    path.unlink()
    disabled = load_disabled()
    if path.name in disabled:
        disabled.discard(path.name)
        save_disabled(disabled)
    sources = load_sources()
    if path.name in sources:
        sources.pop(path.name, None)
        save_sources(sources)
    _index_soon()
    return Response(status_code=204)


@router.put("/files/{file_name}")
def set_indexed(
    file_name: str,
    body: Annotated[dict[str, Any], Body(...)],
    _admin: Annotated[Principal, Depends(_require_admin)],
) -> dict[str, str | bool]:
    path = _pdf_by_name(file_name)
    if path is None:
        raise HTTPException(status_code=404, detail="not found")
    if not isinstance(body, dict) or not isinstance(body.get("indexed"), bool):
        raise HTTPException(status_code=400, detail="indexed must be a boolean")
    disabled = load_disabled()
    if body["indexed"]:
        disabled.discard(path.name)
    else:
        disabled.add(path.name)
    save_disabled(disabled)
    _index_soon()
    return _row(path, load_disabled())

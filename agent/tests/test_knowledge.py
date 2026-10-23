"""Admin knowledge catalog routes."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.tokens import make_token


def _write_pdfs(folder: Path, names: list[str]) -> None:
    for name in names:
        (folder / name).write_bytes(name.encode())


@pytest.fixture
def knowledge_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _write_pdfs(tmp_path, ["DGA.pdf", "orphan.pdf"])
    monkeypatch.setattr("app.internal.inclusion._INCLUSION_DIR", tmp_path / "_inclusion")
    monkeypatch.setattr("app.internal.book._KNOWLEDGE", tmp_path)
    monkeypatch.setattr("app.routers.knowledge.ensure_indexed", lambda root=None: None)
    return tmp_path


def test_knowledge_requires_bearer_token(client: TestClient, knowledge_dir: Path):
    response = client.get("/knowledge")
    assert response.status_code == 401
    assert response.json()["detail"] == "unauthorized"


def test_knowledge_rejects_non_admin(client: TestClient, knowledge_dir: Path):
    token = make_token(role="user")
    response = client.get("/knowledge", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 403


def test_knowledge_lists_pdfs(client: TestClient, knowledge_dir: Path, auth_header: dict[str, str]):
    response = client.get("/knowledge", headers=auth_header)
    assert response.status_code == 200
    files = response.json()["files"]
    assert [row["file_name"] for row in files] == ["DGA.pdf", "orphan.pdf"]
    dga = files[0]
    assert dga["local_url"] == "/knowledge/files/DGA.pdf"
    assert dga["web_url"] == "https://cdn.realfood.gov/DGA.pdf"
    assert dga["title"] == "Dietary Guidelines for Americans, 2025–2030"
    assert dga["indexed"] is True
    orphan = files[1]
    assert orphan["web_url"] == ""
    assert orphan["indexed"] is True


def test_local_pdf_requires_a_signed_in_admin(
    client: TestClient, knowledge_dir: Path, auth_header: dict[str, str]
):
    missing_token = client.get("/knowledge/files/DGA.pdf")
    assert missing_token.status_code == 401
    user = client.get(
        "/knowledge/files/DGA.pdf",
        headers={"Authorization": f"Bearer {make_token(role='user')}"},
    )
    assert user.status_code == 403
    opened = client.get("/knowledge/files/DGA.pdf", headers=auth_header)
    assert opened.status_code == 200
    assert opened.headers["content-type"].startswith("application/pdf")
    assert opened.content == b"DGA.pdf"
    absent = client.get("/knowledge/files/missing.pdf", headers=auth_header)
    assert absent.status_code == 404


def test_knowledge_toggle_persists(
    client: TestClient, knowledge_dir: Path, auth_header: dict[str, str]
):
    off = client.put(
        "/knowledge/files/DGA.pdf",
        headers=auth_header,
        json={"indexed": False},
    )
    assert off.status_code == 200
    assert off.json()["indexed"] is False
    listed = client.get("/knowledge", headers=auth_header).json()["files"]
    assert next(row for row in listed if row["file_name"] == "DGA.pdf")["indexed"] is False

    missing = client.put(
        "/knowledge/files/missing.pdf",
        headers=auth_header,
        json={"indexed": False},
    )
    assert missing.status_code == 404


def _pdf(name: str, body: bytes = b"%PDF-1.4 test") -> tuple:
    return ("file", (name, body, "application/pdf"))


def test_create_uploads_a_pdf(client: TestClient, knowledge_dir: Path, auth_header: dict[str, str]):
    response = client.post(
        "/knowledge/files",
        headers=auth_header,
        files=[_pdf("notes.pdf")],
        data={"title": "Notes", "web_url": "https://example.test/notes"},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["file_name"] == "notes.pdf"
    assert body["title"] == "Notes"
    assert body["web_url"] == "https://example.test/notes"
    assert body["indexed"] is True
    assert (knowledge_dir / "notes.pdf").read_bytes().startswith(b"%PDF")

    duplicate = client.post(
        "/knowledge/files",
        headers=auth_header,
        files=[_pdf("notes.pdf")],
        data={"title": "", "web_url": ""},
    )
    assert duplicate.status_code == 409


def test_create_rejects_a_non_pdf(client: TestClient, knowledge_dir: Path, auth_header: dict[str, str]):
    response = client.post(
        "/knowledge/files",
        headers=auth_header,
        files=[_pdf("notes.pdf", b"not a pdf")],
        data={"title": "", "web_url": ""},
    )
    assert response.status_code == 400
    assert not (knowledge_dir / "notes.pdf").exists()


def test_update_can_replace_rename_and_edit_citation(
    client: TestClient, knowledge_dir: Path, auth_header: dict[str, str]
):
    replaced = client.patch(
        "/knowledge/files/orphan.pdf",
        headers=auth_header,
        files=[_pdf("orphan.pdf", b"%PDF-1.4 replaced")],
        data={"title": "Orphan title", "web_url": "https://example.test/orphan"},
    )
    assert replaced.status_code == 200
    assert (knowledge_dir / "orphan.pdf").read_bytes() == b"%PDF-1.4 replaced"
    assert replaced.json()["title"] == "Orphan title"

    citation = client.patch(
        "/knowledge/files/DGA.pdf",
        headers=auth_header,
        data={"title": "Edited title", "web_url": ""},
    )
    assert citation.status_code == 200
    assert citation.json()["web_url"] == ""
    assert citation.json()["title"] == "Edited title"
    assert (knowledge_dir / "DGA.pdf").exists()

    renamed = client.patch(
        "/knowledge/files/orphan.pdf",
        headers=auth_header,
        data={"file_name": "renamed.pdf"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["file_name"] == "renamed.pdf"
    assert renamed.json()["web_url"] == "https://example.test/orphan"
    assert not (knowledge_dir / "orphan.pdf").exists()
    assert (knowledge_dir / "renamed.pdf").exists()


def test_delete_removes_the_pdf(client: TestClient, knowledge_dir: Path, auth_header: dict[str, str]):
    response = client.delete("/knowledge/files/orphan.pdf", headers=auth_header)
    assert response.status_code == 204
    assert not (knowledge_dir / "orphan.pdf").exists()
    listed = client.get("/knowledge", headers=auth_header).json()["files"]
    assert [row["file_name"] for row in listed] == ["DGA.pdf"]

    missing = client.delete("/knowledge/files/orphan.pdf", headers=auth_header)
    assert missing.status_code == 404


def test_citation_label_follows_the_admin_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("app.internal.inclusion._INCLUSION_DIR", tmp_path)
    from app.internal.knowledge_sources import citation_label
    from app.internal.sources import save_sources

    assert citation_label("DGA.pdf") == (
        "[Dietary Guidelines for Americans, 2025–2030](https://cdn.realfood.gov/DGA.pdf)"
    )
    save_sources({"DGA.pdf": {"title": "Edited title", "url": "https://example.test/dga"}})
    assert citation_label("DGA.pdf") == "[Edited title](https://example.test/dga)"
    save_sources({"DGA.pdf": {"title": "Edited title", "url": ""}})
    assert citation_label("DGA.pdf") == "DGA.pdf"

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, createKnowledge, deleteKnowledge, fetchKnowledgePdf, listKnowledge, setKnowledgeIndexed, updateKnowledge } from "../api";
import { catalogRows, type KnowledgeFile } from "../knowledge";

type Props = {
  token: string;
  onUnauthorized: () => void;
};

type Drawer =
  | { kind: "create" }
  | { kind: "edit"; file: KnowledgeFile }
  | null;

const emptyForm = { fileName: "", title: "", webUrl: "" };

function upsertFile(current: KnowledgeFile[], saved: KnowledgeFile): KnowledgeFile[] {
  const without = current.filter((item) => item.file_name !== saved.file_name);
  return [...without, saved].sort((left, right) => left.file_name.localeCompare(right.file_name));
}

export function Resources({ token, onUnauthorized }: Props) {
  const [files, setFiles] = useState<KnowledgeFile[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [pending, setPending] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const [form, setForm] = useState(emptyForm);
  const [upload, setUpload] = useState<File | null>(null);
  const [loadState, setLoadState] = useState<"loading" | "loaded" | "error">("loading");
  const [loadedCount, setLoadedCount] = useState(0);
  const listGeneration = useRef(0);

  const refresh = useCallback(async () => {
    const generation = ++listGeneration.current;
    setLoadState("loading");
    setError(null);
    try {
      const rows = catalogRows({ files: await listKnowledge(token) });
      if (generation !== listGeneration.current) return;
      setFiles(rows);
      setLoadedCount(rows.length);
      setLoadState("loaded");
    } catch (err) {
      if (generation !== listGeneration.current) return;
      if (err instanceof ApiError && err.status === 401) {
        onUnauthorized();
        return;
      }
      setLoadState("error");
      setError("The resource list did not load.");
    }
  }, [token, onUnauthorized]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  function openCreate() {
    setForm(emptyForm);
    setUpload(null);
    setFormError(null);
    setDrawer({ kind: "create" });
  }

  function openEdit(file: KnowledgeFile) {
    setForm({ fileName: file.file_name, title: file.title, webUrl: file.web_url });
    setUpload(null);
    setFormError(null);
    setDrawer({ kind: "edit", file });
  }

  function correction(err: unknown, fallback: string, fileName = ""): string {
    if (!(err instanceof ApiError)) return fallback;
    if (err.status === 409 || err.detail === "already exists") {
      const named = fileName ? `A PDF named “${fileName}”` : "A PDF with that name";
      return `${named} is already in the folder. Change the file name and try again.`;
    }
    if (err.detail === "file must be a PDF") return "Choose a PDF file. The file name must end in .pdf.";
    if (err.detail === "not found") return "That PDF is no longer in the folder. Reload the list and try again.";
    if (err.detail === "file was not stored") return "The PDF was not stored. Choose the file again.";
    if (err.detail === "indexed must be a boolean") return "The Indexed choice did not save. Try the toggle again.";
    if (err.detail) return err.detail;
    return fallback;
  }

  function fail(err: unknown, fallback: string, fileName = "") {
    if (err instanceof ApiError && err.status === 401) {
      onUnauthorized();
      return;
    }
    setFormError(correction(err, fallback, fileName));
  }

  async function submitCreate() {
    if (!upload) {
      setFormError("Choose a PDF file.");
      return;
    }
    setBusy(true);
    setFormError(null);
    setNotice(null);
    let saved: KnowledgeFile | undefined;
    try {
      saved = catalogRows({
        files: [await createKnowledge(token, upload, form.title.trim(), form.webUrl.trim())],
      })[0];
    } catch (err) {
      fail(err, "The PDF was not added. Change the file and try again.", upload.name);
      return;
    } finally {
      setBusy(false);
    }
    setDrawer(null);
    if (saved) {
      setFiles((current) => upsertFile(current, saved));
      setLoadState("loaded");
      setNotice(`${saved.file_name} was added. Answers include it after indexing finishes.`);
    }
    await refresh();
  }

  async function submitEdit(file: KnowledgeFile) {
    const nextName = form.fileName.trim();
    if (!nextName.toLowerCase().endsWith(".pdf")) {
      setFormError("The file name must end in .pdf.");
      return;
    }
    setBusy(true);
    setFormError(null);
    setNotice(null);
    let saved: KnowledgeFile | undefined;
    try {
      saved = catalogRows({
        files: [
          await updateKnowledge(token, file.file_name, {
            file: upload,
            file_name: nextName,
            title: form.title.trim(),
            web_url: form.webUrl.trim(),
          }),
        ],
      })[0];
    } catch (err) {
      fail(err, "The PDF was not saved. Correct the form and try again.", nextName);
      return;
    } finally {
      setBusy(false);
    }
    setDrawer(null);
    if (saved) {
      setFiles((current) => upsertFile(current.filter((item) => item.file_name !== file.file_name), saved));
      setLoadState("loaded");
      setNotice(`${saved.file_name} was saved.`);
    }
    await refresh();
  }

  async function toggle(file: KnowledgeFile, indexed: boolean) {
    setPending(file.file_name);
    setError(null);
    try {
      const row = await setKnowledgeIndexed(token, file.file_name, indexed);
      setFiles((current) =>
        current.map((item) => (item.file_name === row.file_name ? catalogRows({ files: [row] })[0]! : item)),
      );
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onUnauthorized();
        return;
      }
      setError(correction(err, "The Indexed choice did not save."));
    } finally {
      setPending(null);
    }
  }

  async function openLocal(file: KnowledgeFile) {
    setError(null);
    try {
      const blob = await fetchKnowledgePdf(token, file.local_url);
      const url = URL.createObjectURL(blob);
      window.open(url, "_blank", "noopener");
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onUnauthorized();
        return;
      }
      setError("The PDF did not open.");
    }
  }

  function remove(file: KnowledgeFile) {
    const confirmed = window.confirm(`Delete ${file.file_name}? This removes the PDF from the knowledge folder and from answers.`);
    if (!confirmed) return;
    setPending(file.file_name);
    setError(null);
    deleteKnowledge(token, file.file_name)
      .then(() => {
        setFiles((current) => current.filter((item) => item.file_name !== file.file_name));
        setLoadState("loaded");
        return refresh();
      })
      .catch((err: unknown) => {
        if (err instanceof ApiError && err.status === 401) onUnauthorized();
        else setError(correction(err, "The PDF was not deleted. The file is still in the list."));
      })
      .finally(() => setPending(null));
  }

  return (
    <section>
      <div className="toolbar">
        <div>
          <h1>Resources</h1>
          <p className="help">PDFs in the knowledge folder. Indexed off leaves the file on disk and out of answers.</p>
        </div>
        <button className="btn" type="button" onClick={openCreate}>
          Add resource
        </button>
      </div>
      {notice ? <p className="status" role="status">{notice}</p> : null}
      {loadState === "loading" ? <p className="help">Loading resources…</p> : null}
      {loadState === "loaded" ? (
        <p className="help">{loadedCount === 1 ? "1 PDF loaded." : `${loadedCount} PDFs loaded.`}</p>
      ) : null}
      {error ? <p className="banner">{error}</p> : null}
      <div className="table-card">
        <table>
          <thead>
            <tr>
              <th>File name</th>
              <th>Title</th>
              <th>Local URL</th>
              <th>Web URL</th>
              <th>Indexed</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {loadState === "loaded" && files.length === 0 ? (
              <tr>
                <td colSpan={6} className="muted">
                  No PDFs in the knowledge folder.
                </td>
              </tr>
            ) : (
              files.map((file) => (
                <tr key={file.file_name}>
                  <td>{file.file_name}</td>
                  <td>{file.title || <span className="muted">—</span>}</td>
                  <td>
                    <a
                      href={file.local_url}
                      onClick={(event) => {
                        event.preventDefault();
                        void openLocal(file);
                      }}
                    >
                      {file.local_url}
                    </a>
                  </td>
                  <td>
                    {file.web_url ? (
                      <a href={file.web_url} target="_blank" rel="noreferrer">
                        {file.web_url}
                      </a>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    <label className="indexed">
                      <input
                        type="checkbox"
                        checked={file.indexed}
                        disabled={pending === file.file_name}
                        aria-label={`Indexed ${file.file_name}`}
                        onChange={(event) => void toggle(file, event.target.checked)}
                      />
                      {file.indexed ? "On" : "Off"}
                    </label>
                  </td>
                  <td>
                    <button className="icon-btn" type="button" aria-label={`Edit ${file.file_name}`} onClick={() => openEdit(file)}>
                      ✎
                    </button>
                    <button
                      className="icon-btn danger"
                      type="button"
                      aria-label={`Delete ${file.file_name}`}
                      disabled={pending === file.file_name}
                      onClick={() => remove(file)}
                    >
                      ⌫
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {drawer ? <div className="scrim" onClick={() => { if (!busy) setDrawer(null); }} /> : null}
      {drawer ? (
        <aside className="drawer" role="dialog" aria-busy={busy} aria-label={drawer.kind === "create" ? "Add resource" : `Edit ${drawer.file.file_name}`}>
          <h2>{drawer.kind === "create" ? "Add resource" : `Edit ${drawer.file.file_name}`}</h2>
          <p className="help">
            {busy
              ? "Uploading the PDF…"
              : drawer.kind === "create"
                ? "Upload a PDF. It starts included in answers."
                : "Leave the file empty to keep the current PDF."}
          </p>
          {drawer.kind === "edit" ? (
            <label>
              File name
              <input value={form.fileName} onChange={(event) => setForm({ ...form, fileName: event.target.value })} />
            </label>
          ) : null}
          <label>
            Title
            <input value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} />
          </label>
          <label>
            Web URL
            <input value={form.webUrl} onChange={(event) => setForm({ ...form, webUrl: event.target.value })} />
          </label>
          <label>
            PDF
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => setUpload(event.target.files?.[0] ?? null)}
            />
          </label>
          {busy ? (
            <p className="status" role="status">
              {drawer.kind === "create" ? "Uploading…" : "Saving…"}
            </p>
          ) : null}
          {formError ? <p className="form-error" role="alert">{formError}</p> : null}
          <div className="actions">
            <button className="btn ghost" type="button" disabled={busy} onClick={() => setDrawer(null)}>Cancel</button>
            <button
              className="btn"
              type="button"
              disabled={busy}
              onClick={() => void (drawer.kind === "create" ? submitCreate() : submitEdit(drawer.file))}
            >
              {busy ? (drawer.kind === "create" ? "Uploading…" : "Saving…") : drawer.kind === "create" ? "Add resource" : "Save"}
            </button>
          </div>
        </aside>
      ) : null}
    </section>
  );
}

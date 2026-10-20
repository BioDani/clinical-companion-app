import { useEffect, useState } from "react";
import {
  ApiError,
  createUser,
  deleteUser,
  grantUserPermission,
  listUsers,
  updateUser,
} from "../api";
import { isCatalogName } from "../catalog";
import {
  DEFAULT_PAGE_SIZE,
  PAGE_SIZES,
  deleteGuard,
  filterAccounts,
  pageAccounts,
  summaries,
  type Account,
} from "../directory";
import {
  alreadyGranted,
  alreadyOnUser,
  cannotDeleteSelf,
  chooseRole,
  duplicateAccount,
  grantAdded,
  grantNextSignIn,
  keepOneAdministrator,
  notInCatalog,
  unknownPermission,
  validateIdentity,
} from "../messages";

type Props = {
  token: string;
  selfId: number;
  onUnauthorized: () => void;
  onLoaded: (accounts: Account[]) => void;
};

type Drawer =
  | { kind: "create" }
  | { kind: "edit"; account: Account }
  | { kind: "delete"; account: Account }
  | null;

const emptyForm = { username: "", email: "", password: "", role: "user" };

export function Users({ token, selfId, onUnauthorized, onLoaded }: Props) {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [query, setQuery] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState<number>(DEFAULT_PAGE_SIZE);
  const [drawer, setDrawer] = useState<Drawer>(null);
  const [form, setForm] = useState(emptyForm);
  const [formError, setFormError] = useState<string | null>(null);
  const [grantName, setGrantName] = useState("");
  const [grantNote, setGrantNote] = useState<{ title: string; body: string; tone: "good" | "bad" } | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    const rows = await listUsers(token);
    setAccounts(rows);
    onLoaded(rows);
  }

  useEffect(() => {
    let cancelled = false;
    listUsers(token)
      .then((rows) => {
        if (cancelled) return;
        setAccounts(rows);
        onLoaded(rows);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 401) onUnauthorized();
        else setLoadError("The account list did not load.");
      });
    return () => {
      cancelled = true;
    };
  }, [token, onUnauthorized, onLoaded]);

  const totals = summaries(accounts);
  const filtered = filterAccounts(accounts, query);
  const paged = pageAccounts(filtered, page, pageSize);

  function explain(err: unknown): boolean {
    if (err instanceof ApiError && err.status === 401) {
      onUnauthorized();
      return true;
    }
    if (err instanceof ApiError && err.status === 409) {
      setFormError(duplicateAccount);
      return true;
    }
    if (err instanceof ApiError && err.status === 422 && err.detail === "Invalid role") {
      setFormError(chooseRole);
      return true;
    }
    return false;
  }

  function openCreate() {
    setForm(emptyForm);
    setFormError(null);
    setDrawer({ kind: "create" });
  }

  function openEdit(account: Account) {
    setForm({ username: account.username, email: account.email, password: "", role: account.role });
    setFormError(null);
    setGrantName("");
    setGrantNote(null);
    setDrawer({ kind: "edit", account });
  }

  async function submitCreate() {
    const problem = validateIdentity({ ...form, passwordRequired: true });
    if (problem) {
      setFormError(problem);
      return;
    }
    setBusy(true);
    try {
      await createUser(token, {
        username: form.username.trim(),
        email: form.email.trim(),
        password: form.password,
        role: form.role,
      });
      await refresh();
      setDrawer(null);
    } catch (err) {
      if (!explain(err)) setFormError("The account was not created.");
    } finally {
      setBusy(false);
    }
  }

  async function submitEdit(account: Account) {
    const problem = validateIdentity({ ...form, passwordRequired: false });
    if (problem) {
      setFormError(problem);
      return;
    }
    const body: { username?: string; email?: string; password?: string; role?: string } = {};
    if (form.username.trim() !== account.username) body.username = form.username.trim();
    if (form.email.trim() !== account.email) body.email = form.email.trim();
    if (form.role !== account.role) body.role = form.role;
    if (form.password) body.password = form.password;
    setBusy(true);
    try {
      if (Object.keys(body).length > 0) await updateUser(token, account.id, body);
      await refresh();
      setDrawer(null);
    } catch (err) {
      if (!explain(err)) setFormError("The account was not saved.");
    } finally {
      setBusy(false);
    }
  }

  async function submitGrant(account: Account) {
    const name = grantName.trim();
    if (!isCatalogName(name)) {
      setGrantNote({ tone: "bad", title: unknownPermission, body: notInCatalog });
      return;
    }
    try {
      await grantUserPermission(token, account.id, name);
      setGrantName("");
      setGrantNote({ tone: "good", title: grantAdded, body: grantNextSignIn });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onUnauthorized();
        return;
      }
      if (err instanceof ApiError && err.status === 409) {
        setGrantNote({ tone: "bad", title: alreadyGranted, body: alreadyOnUser });
        return;
      }
      setGrantNote({ tone: "bad", title: "Could not add that permission.", body: "" });
    }
  }

  async function submitDelete(account: Account) {
    const block = deleteGuard(account, selfId, accounts);
    if (block === "self") {
      setFormError(cannotDeleteSelf);
      return;
    }
    if (block === "last-admin") {
      setFormError(keepOneAdministrator);
      return;
    }
    setBusy(true);
    try {
      await deleteUser(token, account.id);
      await refresh();
      setDrawer(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) onUnauthorized();
      else if (err instanceof ApiError && err.status === 404) await refresh().catch(() => undefined);
      else setFormError("The account was not deleted.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <h1>Users</h1>
      <div className="toolbar">
        <div className="search">
          <span aria-hidden="true">⌕</span>
          <input
            aria-label="Search users"
            placeholder="Search users"
            value={query}
            onChange={(event) => {
              setQuery(event.target.value);
              setPage(1);
            }}
          />
        </div>
        <button className="btn" type="button" onClick={openCreate}>
          Add user
        </button>
      </div>
      <div className="stats">
        <div className="stat"><span>Total users</span><strong>{totals.total}</strong></div>
        <div className="stat"><span>Administrators</span><strong>{totals.administrators}</strong></div>
        <div className="stat"><span>App users</span><strong>{totals.appUsers}</strong></div>
        <div className="stat"><span>Roles</span><strong>{totals.roles}</strong></div>
      </div>
      {loadError ? <p className="banner">{loadError}</p> : null}
      <div className="table-card">
        <div className="table-head">
          <strong>All users</strong>
          <span>
            {paged.from}–{paged.to} of {paged.total}
          </span>
        </div>
        <table>
          <thead>
            <tr>
              <th>User</th>
              <th>Role</th>
              <th>Id</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {paged.rows.map((account) => (
              <tr key={account.id}>
                <td>
                  <div className="person">
                    <div className="avatar" aria-hidden="true">{account.username.slice(0, 1)}</div>
                    <div>
                      {account.username}
                      <small>{account.email}</small>
                    </div>
                  </div>
                </td>
                <td>
                  <span className={account.role === "admin" ? "badge admin" : "badge"}>{account.role}</span>
                </td>
                <td>{account.id}</td>
                <td>
                  <button className="icon-btn" type="button" aria-label={`Edit ${account.username}`} onClick={() => openEdit(account)}>
                    ✎
                  </button>
                  <button className="icon-btn danger" type="button" aria-label={`Delete ${account.username}`} onClick={() => { setFormError(null); setDrawer({ kind: "delete", account }); }}>
                    ⌫
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="pager">
          <span>
            {paged.from}–{paged.to} of {paged.total}
          </span>
          <div className="pager-controls">
            <label>
              Rows per page
              <select
                value={pageSize}
                onChange={(event) => {
                  setPageSize(Number(event.target.value));
                  setPage(1);
                }}
              >
                {PAGE_SIZES.map((size) => (
                  <option key={size} value={size}>{size}</option>
                ))}
              </select>
            </label>
            <button className="btn ghost" type="button" disabled={paged.page <= 1} onClick={() => setPage((current) => current - 1)}>
              Previous
            </button>
            <button className="btn ghost" type="button" disabled={paged.page >= paged.pageCount} onClick={() => setPage((current) => current + 1)}>
              Next
            </button>
          </div>
        </div>
      </div>

      {drawer ? <div className="scrim" onClick={() => setDrawer(null)} /> : null}

      {drawer?.kind === "create" || drawer?.kind === "edit" ? (
        <aside className="drawer" role="dialog" aria-label={drawer.kind === "create" ? "Add user" : `Edit ${drawer.account.username}`}>
          <h2>{drawer.kind === "create" ? "Add user" : `Edit ${drawer.account.username}`}</h2>
          <p className="help">
            {drawer.kind === "create"
              ? "Creates an account with one role, user or admin."
              : "Leave the password blank to keep the current one."}
          </p>
          <label>
            Username
            <input value={form.username} onChange={(event) => setForm({ ...form, username: event.target.value })} />
          </label>
          <label>
            Email
            <input value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} />
          </label>
          <label>
            Password
            <input
              type="password"
              value={form.password}
              placeholder={drawer.kind === "edit" ? "Leave blank to keep" : undefined}
              onChange={(event) => setForm({ ...form, password: event.target.value })}
            />
          </label>
          <label>
            Role
            <select value={form.role} onChange={(event) => setForm({ ...form, role: event.target.value })}>
              <option value="user">user</option>
              <option value="admin">admin</option>
            </select>
          </label>
          {formError ? <p className="form-error">{formError}</p> : null}
          <div className="actions">
            <button className="btn ghost" type="button" onClick={() => setDrawer(null)}>Cancel</button>
            <button
              className="btn"
              type="button"
              disabled={busy}
              onClick={() => void (drawer.kind === "create" ? submitCreate() : submitEdit(drawer.account))}
            >
              {drawer.kind === "create" ? "Create user" : "Save"}
            </button>
          </div>
          {drawer.kind === "edit" ? (
            <div>
              <h2>Add extra permission</h2>
              <label>
                Permission name
                <input value={grantName} onChange={(event) => setGrantName(event.target.value)} />
              </label>
              <button className="btn" type="button" onClick={() => void submitGrant(drawer.account)}>
                Add permission
              </button>
              <p className="help">Takes effect the next time {drawer.account.username} signs in.</p>
              {grantNote ? (
                <div className={`callout ${grantNote.tone}`}>
                  <strong>{grantNote.title}</strong>
                  <div>{grantNote.body}</div>
                </div>
              ) : null}
            </div>
          ) : null}
        </aside>
      ) : null}

      {drawer?.kind === "delete" ? (
        <div className="dialog" role="dialog" aria-label={`Delete ${drawer.account.username}`}>
          <h2>Delete {drawer.account.username}?</h2>
          <p>{drawer.account.username} stays in the list until this delete is confirmed.</p>
          {formError ? <p className="form-error">{formError}</p> : null}
          <div className="actions">
            <button className="btn ghost" type="button" onClick={() => setDrawer(null)}>Cancel</button>
            <button className="btn danger" type="button" disabled={busy} onClick={() => void submitDelete(drawer.account)}>
              Delete
            </button>
          </div>
        </div>
      ) : null}
    </section>
  );
}

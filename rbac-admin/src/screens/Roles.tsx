import { useEffect, useState } from "react";
import { ApiError, grantRolePermission, listRoles, type RoleRecord } from "../api";
import { CATALOG, isCatalogName } from "../catalog";
import { alreadyGranted, alreadyOnRole, notInCatalog, unknownPermission } from "../messages";

type Props = {
  token: string;
  onUnauthorized: () => void;
};

export function Roles({ token, onUnauthorized }: Props) {
  const [roles, setRoles] = useState<RoleRecord[]>([]);
  const [names, setNames] = useState<Record<number, string>>({});
  const [notes, setNotes] = useState<Record<number, { tone: "good" | "bad"; title: string; body: string }>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    listRoles(token)
      .then((rows) => {
        if (!cancelled) setRoles(rows);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 401) onUnauthorized();
        else setError("The role list did not load.");
      });
    return () => {
      cancelled = true;
    };
  }, [token, onUnauthorized]);

  async function addPermission(role: RoleRecord) {
    const name = (names[role.id] ?? "").trim();
    if (!isCatalogName(name)) {
      setNotes((current) => ({
        ...current,
        [role.id]: { tone: "bad", title: unknownPermission, body: notInCatalog },
      }));
      return;
    }
    try {
      await grantRolePermission(token, role.id, name);
      setNames((current) => ({ ...current, [role.id]: "" }));
      setNotes((current) => {
        const next = { ...current };
        delete next[role.id];
        return next;
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) {
        onUnauthorized();
        return;
      }
      if (err instanceof ApiError && err.status === 409) {
        setNotes((current) => ({
          ...current,
          [role.id]: { tone: "bad", title: alreadyGranted, body: alreadyOnRole },
        }));
        return;
      }
      setNotes((current) => ({
        ...current,
        [role.id]: { tone: "bad", title: "Could not add that permission.", body: "" },
      }));
    }
  }

  return (
    <section>
      <h1>Roles</h1>
      <p className="help">A permission added here is included the next time that account signs in.</p>
      {error ? <p className="banner">{error}</p> : null}
      <div className="catalog">
        <strong>Names init seeds</strong>
        <p className="help">
          These names are created when the service is initialized. This list is a reference, not the grants
          currently stored on a role.
        </p>
        <div className="chips">
          {CATALOG.map((name) => (
            <span className="chip" key={name}>
              {name}
            </span>
          ))}
        </div>
      </div>
      <div className="role-grid">
        {roles.map((role) => {
          const note = notes[role.id];
          return (
            <article className="role-card" key={role.id}>
              <h2>{role.name}</h2>
              <p className="muted">Id {role.id}</p>
              <label>
                Permission name
                <input
                  value={names[role.id] ?? ""}
                  onChange={(event) => setNames((current) => ({ ...current, [role.id]: event.target.value }))}
                />
              </label>
              <button className="btn" type="button" onClick={() => void addPermission(role)}>
                Add permission
              </button>
              <p className="help">
                Takes effect the next time an account with this role signs in. There is no control to rename this
                role or to add a third one.
              </p>
              {note ? (
                <div className={`callout ${note.tone}`}>
                  <strong>{note.title}</strong>
                  {note.body ? <div>{note.body}</div> : null}
                </div>
              ) : null}
            </article>
          );
        })}
      </div>
    </section>
  );
}

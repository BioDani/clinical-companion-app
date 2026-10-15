import { useCallback, useState } from "react";
import type { Account } from "./directory";
import { physicianNotice } from "./messages";
import { clearSession, loadSession, type Session } from "./session";
import { Roles } from "./screens/Roles";
import { SignIn } from "./screens/SignIn";
import { Users } from "./screens/Users";

type Screen = "users" | "roles";

export function App() {
  const [session, setSession] = useState<Session | null>(() => loadSession());
  const [screen, setScreen] = useState<Screen>("users");
  const [username, setUsername] = useState("admin");

  const onLoaded = useCallback((accounts: Account[]) => {
    setSession((current) => {
      if (!current) return current;
      const mine = accounts.find((account) => account.id === current.userId);
      if (mine) setUsername(mine.username);
      return current;
    });
  }, []);

  function signOut() {
    clearSession();
    setSession(null);
    setScreen("users");
  }

  if (!session) {
    return <SignIn onSuccess={setSession} />;
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="mark" aria-hidden="true" />
          <div>
            <strong>Clinical Companion</strong>
            <span>RBAC admin</span>
          </div>
        </div>
        <nav className="nav">
          <button className={screen === "users" ? "active" : ""} type="button" onClick={() => setScreen("users")}>
            Users
          </button>
          <button className={screen === "roles" ? "active" : ""} type="button" onClick={() => setScreen("roles")}>
            Roles
          </button>
        </nav>
        <div className="sidebar-foot">
          <div className="who">
            <div className="avatar" aria-hidden="true">{username.slice(0, 1)}</div>
            <div>
              <strong>{username}</strong>
              <button className="linkish" type="button" onClick={signOut}>Sign out</button>
            </div>
          </div>
          <p className="notice">{physicianNotice}</p>
        </div>
      </aside>
      <main className="main">
        {screen === "users" ? (
          <Users token={session.token} selfId={session.userId} onUnauthorized={signOut} onLoaded={onLoaded} />
        ) : (
          <Roles token={session.token} onUnauthorized={signOut} />
        )}
      </main>
    </div>
  );
}

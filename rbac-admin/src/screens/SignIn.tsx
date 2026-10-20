import { useState, type FormEvent } from "react";
import { ApiError, login } from "../api";
import {
  administratorsOnly,
  physicianNotice,
  signInFailed,
  validateSignIn,
} from "../messages";
import { saveSession, type Session } from "../session";

type Props = {
  onSuccess: (session: Session) => void;
};

export function SignIn({ onSuccess }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const invalid = validateSignIn(username, password);
    if (invalid) {
      setError(invalid);
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const result = await login(username.trim(), password);
      const session = saveSession(result.access_token);
      if (session === "not-admin") {
        setError(administratorsOnly);
        return;
      }
      if (!session) {
        setError(signInFailed);
        return;
      }
      onSuccess(session);
    } catch (err) {
      if (err instanceof ApiError && err.status === 401) setError(signInFailed);
      else setError(signInFailed);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="sign-in-page">
      <form className="sign-in-card" onSubmit={onSubmit}>
        <div className="brand">
          <div className="mark" aria-hidden="true" />
          <div>
            <strong>Clinical Companion</strong>
            <span>RBAC admin</span>
          </div>
        </div>
        <h1>Sign in</h1>
        <p className="help">Use the username and password on your account.</p>
        <label>
          Username
          <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
        </label>
        <label>
          Password
          <input
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
            autoComplete="current-password"
          />
        </label>
        {error ? <p className="form-error">{error}</p> : null}
        <button className="btn wide" type="submit" disabled={busy}>
          Sign in
        </button>
        <p className="notice">{physicianNotice}</p>
      </form>
    </main>
  );
}

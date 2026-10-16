import { FormEvent, useState } from "react";
import { emptyFieldMessage, physicianNotice, signInFailed } from "../composer";
import { acceptLogin, type Session } from "../session";

type Props = {
  onSignedIn: (session: Session) => void;
};

export function SignIn({ onSignedIn }: Props) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const missing = emptyFieldMessage(username, password);
    if (missing) {
      setError(missing);
      return;
    }
    setError(null);
    const response = await fetch("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      setPassword("");
      setError(signInFailed);
      return;
    }
    const body = (await response.json()) as { access_token?: string };
    const session = acceptLogin(response.status, body, username);
    setPassword("");
    if (!session) {
      setError(signInFailed);
      return;
    }
    onSignedIn(session);
  }

  return (
    <main className="sign-in-page">
      <form className="card" onSubmit={onSubmit}>
        <p className="product">Clinical Companion</p>
        <h1>Sign in</h1>
        <label>
          Username
          <input
            name="username"
            autoComplete="username"
            value={username}
            onChange={(event) => setUsername(event.target.value)}
          />
        </label>
        <label>
          Password
          <input
            name="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        {error ? <p className="error">{error}</p> : null}
        <button type="submit">Sign in</button>
        <p className="notice">{physicianNotice}</p>
      </form>
    </main>
  );
}

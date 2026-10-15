import { useCallback, useState } from "react";
import { Home } from "./screens/Home";
import { SignIn } from "./screens/SignIn";
import { clearSession, loadSession, loadUsername, type Session } from "./session";

export function App() {
  const [session, setSession] = useState<Session | null>(() => loadSession());
  const [username, setUsername] = useState<string | null>(() =>
    loadSession() ? loadUsername() : null,
  );
  const signOut = useCallback(() => {
    clearSession();
    setSession(null);
    setUsername(null);
  }, []);
  const onSignedIn = useCallback((next: Session) => {
    setSession(next);
    setUsername(loadUsername());
  }, []);

  if (!session) return <SignIn onSignedIn={onSignedIn} />;
  return <Home onSignOut={signOut} username={username} />;
}

import { useCallback, useState } from "react";
import { Home } from "./screens/Home";
import { SignIn } from "./screens/SignIn";
import { clearSession, loadSession, type Session } from "./session";

export function App() {
  const [session, setSession] = useState<Session | null>(() => loadSession());
  const signOut = useCallback(() => {
    clearSession();
    setSession(null);
  }, []);

  if (!session) return <SignIn onSignedIn={setSession} />;
  return <Home onSignOut={signOut} />;
}

import { Component, StrictMode, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "./styles.css";

class Boundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null };

  static getDerivedStateFromError(error: unknown) {
    const message = error instanceof Error ? error.stack ?? error.message : String(error);
    return { error: message };
  }

  render() {
    if (this.state.error) return <pre className="error">{this.state.error}</pre>;
    return this.props.children;
  }
}

const root = document.getElementById("root");
if (!root) throw new Error("missing root");
createRoot(root).render(
  <StrictMode>
    <Boundary>
      <App />
    </Boundary>
  </StrictMode>,
);

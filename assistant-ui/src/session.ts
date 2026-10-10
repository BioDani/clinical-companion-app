const STORAGE_KEY = "assistant-ui-token";

export type Session = {
  token: string;
  userId: string;
  role: "user" | "admin";
};

type Payload = {
  sub?: string | number;
  role?: string;
};

export type TokenStore = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
};

const memory: TokenStore = {
  getItem: () => null,
  setItem: () => undefined,
  removeItem: () => undefined,
};

let store: TokenStore = typeof sessionStorage === "undefined" ? memory : sessionStorage;

export function bindStore(next: TokenStore): void {
  store = next;
}

export function decodePayload(token: string): Payload | null {
  const segment = token.split(".")[1];
  if (!segment) return null;
  try {
    const padded = segment.replace(/-/g, "+").replace(/_/g, "/");
    const pad = padded.length % 4 === 0 ? "" : "=".repeat(4 - (padded.length % 4));
    const json = atob(padded + pad);
    return JSON.parse(json) as Payload;
  } catch {
    return null;
  }
}

export function sessionFromToken(token: string): Session | null {
  const payload = decodePayload(token);
  if (!payload || payload.sub == null) return null;
  if (payload.role !== "user" && payload.role !== "admin") return null;
  return { token, userId: String(payload.sub), role: payload.role };
}

export function loadSession(): Session | null {
  const token = store.getItem(STORAGE_KEY);
  if (!token) return null;
  const session = sessionFromToken(token);
  if (!session) {
    store.removeItem(STORAGE_KEY);
    return null;
  }
  return session;
}

export function saveSession(token: string): Session | null {
  const session = sessionFromToken(token);
  if (session) store.setItem(STORAGE_KEY, token);
  return session;
}

export function acceptLogin(
  status: number,
  body: { access_token?: string },
): Session | null {
  if (status !== 200 || !body.access_token) return null;
  return saveSession(body.access_token);
}

export function clearSession(): void {
  store.removeItem(STORAGE_KEY);
}

export function storedValues(): string[] {
  const token = store.getItem(STORAGE_KEY);
  return token ? [token] : [];
}

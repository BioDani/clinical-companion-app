const STORAGE_KEY = "rbac-admin-token";

export type Session = {
  token: string;
  userId: number;
  role: "admin";
};

type Payload = {
  sub?: string | number;
  role?: string;
};

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

export function sessionFromToken(token: string): Session | "not-admin" | null {
  const payload = decodePayload(token);
  if (!payload || payload.role == null || payload.sub == null) return null;
  if (payload.role !== "admin") return "not-admin";
  const userId = Number(payload.sub);
  if (!Number.isFinite(userId)) return null;
  return { token, userId, role: "admin" };
}

export function loadSession(): Session | null {
  const token = sessionStorage.getItem(STORAGE_KEY);
  if (!token) return null;
  const session = sessionFromToken(token);
  if (!session || session === "not-admin") {
    sessionStorage.removeItem(STORAGE_KEY);
    return null;
  }
  return session;
}

export function saveSession(token: string): Session | "not-admin" | null {
  const session = sessionFromToken(token);
  if (session && session !== "not-admin") {
    sessionStorage.setItem(STORAGE_KEY, token);
  }
  return session;
}

export function clearSession(): void {
  sessionStorage.removeItem(STORAGE_KEY);
}

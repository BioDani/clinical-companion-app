import type { TokenStore } from "./session";

const TIMES_KEY = "assistant-ui-message-times";

const memory: TokenStore = {
  getItem: () => null,
  setItem: () => undefined,
  removeItem: () => undefined,
};

let store: TokenStore = typeof sessionStorage === "undefined" ? memory : sessionStorage;

export function bindMessageTimeStore(next: TokenStore): void {
  store = next;
}

function readTimes(): Record<string, string> {
  const raw = store.getItem(TIMES_KEY);
  if (!raw) return {};
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return {};
    return parsed as Record<string, string>;
  } catch {
    return {};
  }
}

export function rememberedMessageTime(id: string): Date | null {
  const iso = readTimes()[id];
  if (!iso) return null;
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function rememberMessageTime(id: string, createdAt: Date): void {
  if (!id || Number.isNaN(createdAt.getTime())) return;
  const times = readTimes();
  if (times[id]) return;
  times[id] = createdAt.toISOString();
  store.setItem(TIMES_KEY, JSON.stringify(times));
}

export function formatMessageTime(date: Date, now = new Date()): string {
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: date.getFullYear() === now.getFullYear() ? undefined : "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

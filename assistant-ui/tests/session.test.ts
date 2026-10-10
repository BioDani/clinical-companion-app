import { beforeEach, describe, expect, it } from "vitest";
import {
  acceptLogin,
  bindStore,
  clearSession,
  loadSession,
  storedValues,
  type TokenStore,
} from "../src/session";

function memoryStore(): TokenStore & { dump(): string[] } {
  const rows = new Map<string, string>();
  return {
    getItem: (key) => rows.get(key) ?? null,
    setItem: (key, value) => {
      rows.set(key, value);
    },
    removeItem: (key) => {
      rows.delete(key);
    },
    dump: () => [...rows.values()],
  };
}

function token(role: "user" | "admin", sub = "7"): string {
  const payload = btoa(JSON.stringify({ sub, role, permissions: [], exp: 9, iat: 1 }))
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
  return `header.${payload}.sig`;
}

describe("session", () => {
  let store: ReturnType<typeof memoryStore>;

  beforeEach(() => {
    store = memoryStore();
    bindStore(store);
  });

  it("keeps a user or admin token after 200", () => {
    expect(acceptLogin(200, { access_token: token("user") })?.role).toBe("user");
    expect(loadSession()?.userId).toBe("7");
    clearSession();
    expect(acceptLogin(200, { access_token: token("admin", "3") })?.role).toBe("admin");
    expect(loadSession()?.userId).toBe("3");
  });

  it("does not store a rejected login", () => {
    expect(acceptLogin(401, {})).toBeNull();
    expect(loadSession()).toBeNull();
  });

  it("clears the token on sign out and on a later 401", () => {
    acceptLogin(200, { access_token: token("user") });
    clearSession();
    expect(loadSession()).toBeNull();
    acceptLogin(200, { access_token: token("admin") });
    acceptLogin(401, { access_token: token("admin") });
    expect(loadSession()?.role).toBe("admin");
    clearSession();
    expect(storedValues()).toEqual([]);
  });

  it("never stores the password", () => {
    const password = "s3cret-password";
    acceptLogin(200, { access_token: token("user") });
    expect(store.dump().join(" ")).not.toContain(password);
  });
});

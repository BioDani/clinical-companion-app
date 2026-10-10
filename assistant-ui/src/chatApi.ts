import { Client } from "@langchain/langgraph-sdk";
import { loadSession } from "./session";

export const ASSISTANT_ID = "companion";

export function langGraphApiUrl(): string {
  if (typeof window === "undefined") return "/api";
  return new URL("/api/", window.location.href).href.replace(/\/$/, "");
}

export function createClient(): Client {
  const session = loadSession();
  return new Client({
    apiUrl: langGraphApiUrl(),
    defaultHeaders: session ? { Authorization: `Bearer ${session.token}` } : {},
  });
}

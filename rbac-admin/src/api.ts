import type { Account } from "./directory";
import type { KnowledgeFile } from "./knowledge";

export type { Account };

export type RoleRecord = {
  id: number;
  name: string;
};

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

export type LoginResult = {
  access_token: string;
  token_type: string;
  user_id: number;
};

function errorDetail(detail: unknown): string {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return "";
  return detail
    .map((item) => {
      if (typeof item === "string") return item;
      if (item && typeof item === "object" && "msg" in item && typeof item.msg === "string") return item.msg;
      return "";
    })
    .filter(Boolean)
    .join(" ");
}

async function readError(response: Response): Promise<ApiError> {
  let detail = "";
  try {
    const body = (await response.json()) as { detail?: unknown };
    detail = errorDetail(body.detail);
  } catch {
    detail = "";
  }
  return new ApiError(response.status, detail);
}

function bearer(token: string): HeadersInit {
  return {
    Authorization: `Bearer ${token}`,
    "Content-Type": "application/json",
  };
}

export async function login(username: string, password: string): Promise<LoginResult> {
  const response = await fetch("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as LoginResult;
}

export async function listUsers(token: string): Promise<Account[]> {
  const response = await fetch("/users", { headers: bearer(token) });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as Account[];
}

export async function createUser(
  token: string,
  body: { username: string; email: string; password: string; role: string },
): Promise<Account> {
  const response = await fetch("/users", {
    method: "POST",
    headers: bearer(token),
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as Account;
}

export async function updateUser(
  token: string,
  id: number,
  body: { username?: string; email?: string; password?: string; role?: string },
): Promise<Account> {
  const response = await fetch(`/users/${id}`, {
    method: "PATCH",
    headers: bearer(token),
    body: JSON.stringify(body),
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as Account;
}

export async function deleteUser(token: string, id: number): Promise<void> {
  const response = await fetch(`/users/${id}`, {
    method: "DELETE",
    headers: bearer(token),
  });
  if (!response.ok) throw await readError(response);
}

export async function grantUserPermission(token: string, id: number, name: string): Promise<void> {
  const response = await fetch(`/users/${id}/permissions`, {
    method: "POST",
    headers: bearer(token),
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw await readError(response);
}

export async function listRoles(token: string): Promise<RoleRecord[]> {
  const response = await fetch("/roles", { headers: bearer(token) });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as RoleRecord[];
}

function authOnly(token: string): HeadersInit {
  return { Authorization: `Bearer ${token}` };
}

export async function createKnowledge(
  token: string,
  file: File,
  title: string,
  webUrl: string,
): Promise<KnowledgeFile> {
  const body = new FormData();
  body.set("file", file);
  body.set("title", title);
  body.set("web_url", webUrl);
  const response = await fetch("/api/knowledge/files", {
    method: "POST",
    headers: authOnly(token),
    body,
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as KnowledgeFile;
}

export async function updateKnowledge(
  token: string,
  fileName: string,
  fields: { file?: File | null; file_name?: string; title?: string; web_url?: string },
): Promise<KnowledgeFile> {
  const body = new FormData();
  if (fields.file) body.set("file", fields.file);
  if (fields.file_name != null) body.set("file_name", fields.file_name);
  if (fields.title != null) body.set("title", fields.title);
  if (fields.web_url != null) body.set("web_url", fields.web_url);
  const response = await fetch(`/api/knowledge/files/${encodeURIComponent(fileName)}`, {
    method: "PATCH",
    headers: authOnly(token),
    body,
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as KnowledgeFile;
}

export async function fetchKnowledgePdf(token: string, localUrl: string): Promise<Blob> {
  const response = await fetch(localUrl, {
    headers: authOnly(token),
    cache: "no-store",
  });
  if (!response.ok) throw await readError(response);
  return response.blob();
}

export async function deleteKnowledge(token: string, fileName: string): Promise<void> {
  const response = await fetch(`/api/knowledge/files/${encodeURIComponent(fileName)}`, {
    method: "DELETE",
    headers: authOnly(token),
  });
  if (!response.ok) throw await readError(response);
}

export async function listKnowledge(token: string): Promise<KnowledgeFile[]> {
  const response = await fetch(`/api/knowledge?t=${Date.now()}`, {
    headers: bearer(token),
    cache: "no-store",
  });
  if (!response.ok) throw await readError(response);
  const body = (await response.json()) as { files?: KnowledgeFile[] };
  return body.files ?? [];
}

export async function setKnowledgeIndexed(
  token: string,
  fileName: string,
  indexed: boolean,
): Promise<KnowledgeFile> {
  const response = await fetch(`/api/knowledge/files/${encodeURIComponent(fileName)}`, {
    method: "PUT",
    headers: bearer(token),
    body: JSON.stringify({ indexed }),
  });
  if (!response.ok) throw await readError(response);
  return (await response.json()) as KnowledgeFile;
}

export async function grantRolePermission(token: string, id: number, name: string): Promise<void> {
  const response = await fetch(`/roles/${id}/permissions`, {
    method: "POST",
    headers: bearer(token),
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw await readError(response);
}

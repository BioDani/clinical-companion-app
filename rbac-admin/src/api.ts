import type { Account } from "./directory";

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

async function readError(response: Response): Promise<ApiError> {
  let detail = "";
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") detail = body.detail;
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

export async function grantRolePermission(token: string, id: number, name: string): Promise<void> {
  const response = await fetch(`/roles/${id}/permissions`, {
    method: "POST",
    headers: bearer(token),
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw await readError(response);
}

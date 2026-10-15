export const CATALOG = [
  "users:create",
  "users:read",
  "users:update",
  "users:delete",
  "users:read_self",
  "users:update_self",
  "users:delete_self",
  "roles:read",
  "roles:grant",
  "users:grant",
] as const;

export type CatalogName = (typeof CATALOG)[number];

export function isCatalogName(name: string): name is CatalogName {
  return (CATALOG as readonly string[]).includes(name.trim());
}

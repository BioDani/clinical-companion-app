import { describe, expect, it } from "vitest";
import { CATALOG, isCatalogName } from "../src/catalog";

describe("catalog", () => {
  it("accepts only the ten seeded names", () => {
    expect(CATALOG).toEqual([
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
    ]);
    for (const name of CATALOG) expect(isCatalogName(name)).toBe(true);
  });

  it("rejects names outside the catalog", () => {
    expect(isCatalogName("not-a-permission")).toBe(false);
    expect(isCatalogName("users:read ")).toBe(true);
    expect(isCatalogName("")).toBe(false);
  });
});

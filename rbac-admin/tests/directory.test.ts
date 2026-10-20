import { describe, expect, it } from "vitest";
import {
  DEFAULT_PAGE_SIZE,
  PAGE_SIZES,
  deleteGuard,
  filterAccounts,
  pageAccounts,
  summaries,
  type Account,
} from "../src/directory";

const accounts: Account[] = [
  { id: 1, username: "admin", email: "admin@example.com", role: "admin" },
  { id: 2, username: "Maria", email: "maria@example.com", role: "user" },
  ...Array.from({ length: 10 }, (_, index) => ({
    id: index + 3,
    username: `user${index}`,
    email: `user${index}@example.com`,
    role: "user",
  })),
];

describe("directory", () => {
  it("searches username and email without regard to case", () => {
    expect(filterAccounts(accounts, "MARIA").map((account) => account.id)).toEqual([2]);
    expect(filterAccounts(accounts, "admin@example").map((account) => account.id)).toEqual([1]);
    expect(filterAccounts(accounts, "zzz-no-such-user")).toEqual([]);
  });

  it("counts the full list and always reports two roles", () => {
    expect(summaries(accounts)).toEqual({
      total: 12,
      administrators: 1,
      appUsers: 11,
      roles: 2,
    });
    expect(summaries(filterAccounts(accounts, "maria")).total).toBe(1);
    expect(summaries(accounts).roles).toBe(2);
  });

  it("pages ten rows by default and offers 10, 20, and 50", () => {
    expect(DEFAULT_PAGE_SIZE).toBe(10);
    expect(PAGE_SIZES).toEqual([10, 20, 50]);
    const first = pageAccounts(accounts, 1, 10);
    expect(first.rows).toHaveLength(10);
    expect(first).toMatchObject({ from: 1, to: 10, total: 12, page: 1, pageCount: 2 });
    const second = pageAccounts(accounts, 2, 10);
    expect(second.rows).toHaveLength(2);
    expect(second.from).toBe(11);
  });

  it("refuses deleting the signed-in account before the last administrator", () => {
    expect(deleteGuard(accounts[0], 1, accounts)).toBe("self");
    expect(deleteGuard(accounts[1], 1, accounts)).toBe(null);
    const onlyAdmin = [accounts[0]];
    expect(deleteGuard(onlyAdmin[0], 1, onlyAdmin)).toBe("self");
    const otherAdmin = { id: 9, username: "lucia", email: "lucia@example.com", role: "admin" };
    expect(deleteGuard(otherAdmin, 1, [accounts[0]])).toBe("last-admin");
  });
});

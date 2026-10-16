export type Account = {
  id: number;
  username: string;
  email: string;
  role: string;
};

export const PAGE_SIZES = [10, 20, 50] as const;
export const DEFAULT_PAGE_SIZE = 10;

export type Summaries = {
  total: number;
  administrators: number;
  appUsers: number;
  roles: 2;
};

export function summaries(accounts: Account[]): Summaries {
  return {
    total: accounts.length,
    administrators: accounts.filter((account) => account.role === "admin").length,
    appUsers: accounts.filter((account) => account.role === "user").length,
    roles: 2,
  };
}

export function filterAccounts(accounts: Account[], query: string): Account[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return accounts;
  return accounts.filter(
    (account) =>
      account.username.toLowerCase().includes(needle) ||
      account.email.toLowerCase().includes(needle),
  );
}

export type Page = {
  rows: Account[];
  from: number;
  to: number;
  total: number;
  page: number;
  pageCount: number;
};

export function pageAccounts(rows: Account[], page: number, pageSize: number): Page {
  const size = (PAGE_SIZES as readonly number[]).includes(pageSize) ? pageSize : DEFAULT_PAGE_SIZE;
  const total = rows.length;
  const pageCount = Math.max(1, Math.ceil(total / size));
  const current = Math.min(Math.max(page, 1), pageCount);
  const start = (current - 1) * size;
  const slice = rows.slice(start, start + size);
  return {
    rows: slice,
    from: total === 0 ? 0 : start + 1,
    to: total === 0 ? 0 : start + slice.length,
    total,
    page: current,
    pageCount,
  };
}

export type DeleteBlock = "self" | "last-admin";

export function deleteGuard(account: Account, selfId: number, all: Account[]): DeleteBlock | null {
  if (account.id === selfId) return "self";
  const administrators = all.filter((row) => row.role === "admin").length;
  if (account.role === "admin" && administrators <= 1) return "last-admin";
  return null;
}

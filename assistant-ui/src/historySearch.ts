/** Typed text with spaces removed from the ends only. A space between words stays. */
export function comparedHistoryQuery(raw: string): string {
  return raw.trim();
}

/** True when the compared text is long enough to filter History. */
export function historySearchActive(compared: string): boolean {
  return compared.length >= 3;
}

/** Case-insensitive sequence match. The history name is not an input. */
export function messageMatchesQuery(text: string, compared: string): boolean {
  return text.toLowerCase().includes(compared.toLowerCase());
}

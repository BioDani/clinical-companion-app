export type LinkPart =
  | { kind: "text"; text: string }
  | { kind: "link"; label: string; url: string };

const LINK_PATTERN =
  /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)|\(([^)]+)\)\[(https?:\/\/[^\s\]]+)\]|(https?:\/\/[^\s)]+)/g;

export function linkParts(text: string): LinkPart[] {
  const parts: LinkPart[] = [];
  const pattern = new RegExp(LINK_PATTERN.source, "g");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ kind: "text", text: text.slice(last, index) });
    const bare = match[5];
    const url = match[2] ?? match[4] ?? bare ?? "";
    const trimmed = bare ? url.replace(/[.,)]+$/, "") : url;
    const label = match[1] ?? match[3] ?? trimmed;
    parts.push({ kind: "link", label, url: trimmed });
    last = index + match[0].length - (bare ? url.length - trimmed.length : 0);
  }
  if (last < text.length) parts.push({ kind: "text", text: text.slice(last) });
  return parts;
}

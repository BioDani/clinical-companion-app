import { linkParts } from "./links";

export type Inline =
  | { type: "text"; text: string }
  | { type: "bold"; text: string }
  | { type: "italic"; text: string }
  | { type: "link"; label: string; url: string };

export type Block =
  | { type: "paragraph"; inlines: Inline[] }
  | { type: "heading"; level: number; inlines: Inline[] }
  | { type: "list"; items: Inline[][] };

const CITATION_HEADING = /^### (?:Reference|Sources)\s*$/;
const BULLET = /^\s*[-*]\s+/;
const HEADING = /^(#{1,3})\s+(.*)$/;
const EMPHASIS = /\*\*([^*\n]+)\*\*|\*([^*\n]+)\*/g;

export function splitAnswer(text: string): { body: string; sources: string[] } {
  const lines = text.split("\n");
  let start = citationStart(lines);
  if (start < 0) {
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      if (CITATION_HEADING.test(lines[index].trim())) {
        start = index;
        break;
      }
    }
  }
  if (start < 0) return { body: text.trim(), sources: [] };
  const sources = lines
    .slice(start)
    .filter((line) => !CITATION_HEADING.test(line.trim()))
    .map((line) => line.replace(BULLET, "").trim())
    .filter(Boolean);
  return { body: lines.slice(0, start).join("\n").trim(), sources };
}

function citationStart(lines: string[]): number {
  for (let index = 0; index < lines.length; index += 1) {
    if (!CITATION_HEADING.test(lines[index].trim())) continue;
    const citationOnly = lines.slice(index).every((line) => {
      const trimmed = line.trim();
      return trimmed === "" || CITATION_HEADING.test(trimmed) || BULLET.test(line);
    });
    if (citationOnly) return index;
  }
  return -1;
}

export function parseInlines(text: string): Inline[] {
  const inlines: Inline[] = [];
  for (const part of linkParts(text)) {
    if (part.kind === "link") {
      inlines.push({ type: "link", label: part.label, url: part.url });
      continue;
    }
    inlines.push(...emphasisParts(part.text));
  }
  return inlines;
}

function emphasisParts(text: string): Inline[] {
  const parts: Inline[] = [];
  const pattern = new RegExp(EMPHASIS.source, "g");
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ type: "text", text: text.slice(last, index) });
    if (match[1] != null) parts.push({ type: "bold", text: match[1] });
    else if (match[2] != null) parts.push({ type: "italic", text: match[2] });
    last = index + match[0].length;
  }
  if (last < text.length) parts.push({ type: "text", text: text.slice(last) });
  return parts;
}

export function parseMarkdown(text: string): Block[] {
  const lines = text.split("\n");
  const blocks: Block[] = [];
  let index = 0;
  while (index < lines.length) {
    if (!lines[index].trim()) {
      index += 1;
      continue;
    }
    const heading = HEADING.exec(lines[index].trim());
    if (heading) {
      blocks.push({
        type: "heading",
        level: heading[1].length,
        inlines: parseInlines(heading[2]),
      });
      index += 1;
      continue;
    }
    if (BULLET.test(lines[index])) {
      const items: Inline[][] = [];
      while (index < lines.length && BULLET.test(lines[index])) {
        items.push(parseInlines(lines[index].replace(BULLET, "")));
        index += 1;
      }
      blocks.push({ type: "list", items });
      continue;
    }
    const paragraph: string[] = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !HEADING.test(lines[index].trim()) &&
      !BULLET.test(lines[index])
    ) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    blocks.push({ type: "paragraph", inlines: parseInlines(paragraph.join(" ")) });
  }
  return blocks;
}

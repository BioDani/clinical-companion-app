import { describe, expect, it } from "vitest";
import type { LangChainMessage } from "@assistant-ui/react-langgraph";
import { linkParts } from "../src/links";
import { parseMarkdown, splitAnswer } from "../src/markdown";
import { formatMessageTime, rememberMessageTime, rememberedMessageTime, bindMessageTimeStore } from "../src/messageTime";
import { isVisibleTranscriptMessage, messagesFromUpdates } from "../src/runtime";
import type { TokenStore } from "../src/session";

function human(content: string): LangChainMessage {
  return { type: "human", content };
}

function ai(content: string): LangChainMessage {
  return { type: "ai", content };
}

describe("transcript", () => {
  it("turns a citation update into the finished assistant message", () => {
    const finished = messagesFromUpdates({
      "CitationMiddleware.after_agent": {
        messages: [
          human("Use only the evidence below. Knowledge passages follow."),
          ai("Walk daily.\n\n### Reference\n\n- DGA.pdf, page 12"),
        ],
      },
    });
    expect(finished).toEqual([
      ai("Walk daily.\n\n### Reference\n\n- DGA.pdf, page 12"),
    ]);
    expect(messagesFromUpdates(null)).toEqual([]);
  });

  it("hides the evidence note and internal labels", () => {
    expect(
      isVisibleTranscriptMessage(
        human("Use only the evidence below. Do not add facts from memory."),
      ),
    ).toBe(false);
    expect(isVisibleTranscriptMessage(ai("GREETING"))).toBe(false);
    expect(isVisibleTranscriptMessage(ai("SAFE"))).toBe(false);
    expect(isVisibleTranscriptMessage(ai("NONE"))).toBe(false);
    expect(isVisibleTranscriptMessage(ai("RELEVANT 3"))).toBe(false);
    expect(isVisibleTranscriptMessage(ai("GROUNDED"))).toBe(false);
    expect(isVisibleTranscriptMessage(ai("UNGROUNDED"))).toBe(false);
    expect(isVisibleTranscriptMessage(human("Hello"))).toBe(true);
    expect(isVisibleTranscriptMessage(ai("A short answer."))).toBe(true);
  });

  it("turns citation markdown and bare urls into links", () => {
    const parts = linkParts(
      "See [MedlinePlus](https://medlineplus.gov/dizziness.html) and (CDC)[https://www.cdc.gov/] plus https://example.com/a.",
    );
    expect(parts).toEqual([
      { kind: "text", text: "See " },
      {
        kind: "link",
        label: "MedlinePlus",
        url: "https://medlineplus.gov/dizziness.html",
      },
      { kind: "text", text: " and " },
      { kind: "link", label: "CDC", url: "https://www.cdc.gov/" },
      { kind: "text", text: " plus " },
      { kind: "link", label: "https://example.com/a", url: "https://example.com/a" },
      { kind: "text", text: "." },
    ]);
  });

  it("splits trailing reference and source headings into a citation list", () => {
    const answer = splitAnswer(
      "Walk daily.\n\n### Reference\n\n- Dr. Carlos Jaramillo, Pilares, page 12\n\n### Sources\n\n- [MedlinePlus](https://medlineplus.gov/)",
    );
    expect(answer.body).toBe("Walk daily.");
    expect(answer.sources).toEqual([
      "Dr. Carlos Jaramillo, Pilares, page 12",
      "[MedlinePlus](https://medlineplus.gov/)",
    ]);
  });

  it("renders bold and lists without raw markdown markers", () => {
    const blocks = parseMarkdown("**Walk** daily and *rest*.\n\n- Water\n- Sleep");
    expect(JSON.stringify(blocks)).not.toMatch(/\*\*/);
    expect(JSON.stringify(blocks)).not.toMatch(/(^|[^\\])\*(?!\*)/);
    expect(blocks).toEqual([
      {
        type: "paragraph",
        inlines: [
          { type: "bold", text: "Walk" },
          { type: "text", text: " daily and " },
          { type: "italic", text: "rest" },
          { type: "text", text: "." },
        ],
      },
      {
        type: "list",
        items: [[{ type: "text", text: "Water" }], [{ type: "text", text: "Sleep" }]],
      },
    ]);
  });

  it("includes the date on every stamp and keeps the first time for a message", () => {
    const rows = new Map<string, string>();
    const store: TokenStore = {
      getItem: (key) => rows.get(key) ?? null,
      setItem: (key, value) => {
        rows.set(key, value);
      },
      removeItem: (key) => {
        rows.delete(key);
      },
    };
    bindMessageTimeStore(store);
    const now = new Date(2026, 9, 15, 14, 5);
    expect(formatMessageTime(now, now)).toContain("Oct");
    expect(formatMessageTime(now, now)).toContain("15");
    const yesterday = new Date(2026, 9, 14, 9, 30);
    expect(formatMessageTime(yesterday, now)).toContain("Oct");
    expect(formatMessageTime(yesterday, now)).toContain("14");
    const lastYear = new Date(2025, 9, 15, 9, 30);
    expect(formatMessageTime(lastYear, now)).toContain("2025");
    rememberMessageTime("m1", yesterday);
    rememberMessageTime("m1", now);
    expect(rememberedMessageTime("m1")?.toISOString()).toBe(yesterday.toISOString());
    expect(rememberedMessageTime("missing")).toBeNull();
  });
});
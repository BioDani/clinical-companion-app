import { describe, expect, it } from "vitest";
import type { LangChainMessage } from "@assistant-ui/react-langgraph";
import { linkParts } from "../src/links";
import { isVisibleTranscriptMessage } from "../src/runtime";

function human(content: string): LangChainMessage {
  return { type: "human", content };
}

function ai(content: string): LangChainMessage {
  return { type: "ai", content };
}

describe("transcript", () => {
  it("hides the evidence note and internal labels", () => {
    expect(
      isVisibleTranscriptMessage(
        human("Use only the evidence below. Do not add facts from memory."),
      ),
    ).toBe(false);
    expect(isVisibleTranscriptMessage(ai("GREETING"))).toBe(false);
    expect(isVisibleTranscriptMessage(ai("SAFE"))).toBe(false);
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
});
import { describe, expect, it } from "vitest";
import type { LangChainMessage } from "@assistant-ui/react-langgraph";
import {
  comparedHistoryQuery,
  historySearchActive,
  messageMatchesQuery,
} from "../src/historySearch";
import { visibleSearchTexts } from "../src/runtime";

function human(content: string): LangChainMessage {
  return { type: "human", content };
}

function ai(content: string): LangChainMessage {
  return { type: "ai", content };
}

describe("history search", () => {
  it("ignores spaces at the ends and stays inactive under three characters", () => {
    expect(comparedHistoryQuery("  br")).toBe("br");
    expect(historySearchActive(comparedHistoryQuery("  br"))).toBe(false);
    expect(comparedHistoryQuery("a b")).toBe("a b");
    expect(historySearchActive("a b")).toBe(true);
  });

  it("matches a fragment inside a visible message without using a history name", () => {
    expect(messageMatchesQuery("What is an evidence-based breakfast?", "bre")).toBe(true);
    expect(messageMatchesQuery("Breakfast", "breakfast")).toBe(true);
    expect(messageMatchesQuery("Health introduction", "vegetarian")).toBe(false);
    expect(messageMatchesQuery("I follow a vegetarian diet", "vegetarian")).toBe(true);
  });

  it("leaves internal labels and the evidence instruction out of the searchable text", () => {
    const texts = visibleSearchTexts([
      human("Use only the evidence below. Do not add facts from memory."),
      ai("GREETING"),
      ai("SAFE"),
      human("What is an evidence-based breakfast?"),
      ai("A short answer."),
    ]);
    expect(texts).toEqual([
      "What is an evidence-based breakfast?",
      "A short answer.",
    ]);
  });
});

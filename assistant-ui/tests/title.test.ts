import { describe, expect, it } from "vitest";
import { HEALTH_INTRODUCTION, historyCreatedAt, historyTitle, withRememberedThreads } from "../src/runtime";

const close = "Thank you. I will use what you chose to share on later habit questions.";

function user(text: string) {
  return { role: "user", content: [{ type: "text", text }] };
}

function assistant(text: string) {
  return { role: "assistant", content: [{ type: "text", text }] };
}

describe("history titles", () => {
  it("names an introduction-only chat for the health conversation", () => {
    const title = historyTitle([
      assistant(
        "Hi, I’m wellnest, your wellness companion 👋\nTo give you more relevant answers, you can choose to share a bit about yourself, like health conditions or daily habits. This is completely optional, and anything you share will be used only to tailor my answers to you.\nYour information is kept safe and handled according to our Privacy Policy. It will never be used to train AI models.\n\nWhat is your age?",
      ),
      user("42"),
      assistant("How many hours do you sleep at night?"),
      user("7 hours a night"),
      assistant(close),
    ]);
    expect(title).toBe(HEALTH_INTRODUCTION);
    expect(title).not.toBe("42");
    expect(title).not.toBe("7 hours a night");
  });

  it("follows a later habit question", () => {
    const habit = "What is an evidence-based pattern for a vegetarian breakfast?";
    const title = historyTitle([
      assistant("What is your age?"),
      user("42"),
      assistant(close),
      user(habit),
    ]);
    expect(title).toBe(habit);
    expect(title).not.toBe("42");
  });

  it("leaves a chat with no subject untitled", () => {
    expect(historyTitle([])).toBe("");
  });

  it("reads the chat creation time from history metadata", () => {
    const created = historyCreatedAt({ createdAt: "2026-10-24T19:09:00.000Z" });
    expect(created?.toISOString()).toBe("2026-10-24T19:09:00.000Z");
    expect(historyCreatedAt(undefined)).toBeNull();
    expect(historyCreatedAt({ createdAt: "not-a-date" })).toBeNull();
  });

  it("trims a long habit question to 80 characters", () => {
    const habit = "What is an evidence-based breakfast pattern I can follow on weekdays and weekends too?";
    expect(historyTitle([user(habit)])).toBe(habit.slice(0, 80));
  });

  it("keeps a chat titled in this session when the server list omits it", () => {
    const listed = [
      {
        status: "regular" as const,
        remoteId: "older",
        externalId: "older",
        title: "Older chat",
      },
    ];
    const remembered = new Map([
      ["fresh", { title: "Morning walk", custom: { createdAt: "2026-10-27T12:00:00.000Z" } }],
      ["older", { title: "Older chat" }],
    ]);
    const threads = withRememberedThreads(listed, remembered);
    expect(threads.map((thread) => thread.remoteId)).toEqual(["fresh", "older"]);
    expect(threads[0]?.title).toBe("Morning walk");
  });
});

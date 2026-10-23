import { describe, expect, it } from "vitest";
import { HEALTH_INTRODUCTION, historyTitle } from "../src/runtime";

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
        "Sharing these answers is optional. This companion is not a diagnostic device. These answers are not used to train a model.\n\nWhat is your age?",
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

  it("trims a long habit question to 80 characters", () => {
    const habit = "What is an evidence-based breakfast pattern I can follow on weekdays and weekends too?";
    expect(historyTitle([user(habit)])).toBe(habit.slice(0, 80));
  });
});

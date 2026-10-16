import { describe, expect, it } from "vitest";
import { mediaControls, shouldStartRun } from "../src/composer";

describe("composer", () => {
  it("does not start a run for an empty message", () => {
    expect(shouldStartRun("")).toBe(false);
    expect(shouldStartRun("   ")).toBe(false);
    expect(shouldStartRun("What about breakfast?")).toBe(true);
  });

  it("offers no attachment, image, or speech control", () => {
    expect(mediaControls).toEqual([]);
    expect(mediaControls).not.toContain("attachment");
    expect(mediaControls).not.toContain("image");
    expect(mediaControls).not.toContain("speech");
  });
});

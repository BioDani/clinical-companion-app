import { describe, expect, it } from "vitest";
import { catalogRows, localUrlFor } from "../src/knowledge";

describe("knowledge catalog rows", () => {
  it("maps one row per file and keeps an empty web URL empty", () => {
    const rows = catalogRows({
      files: [
        {
          file_name: "DGA.pdf",
          local_url: "/knowledge/files/DGA.pdf",
          web_url: "https://cdn.realfood.gov/DGA.pdf",
          indexed: true,
        },
        {
          file_name: "orphan.pdf",
          local_url: "/knowledge/files/orphan.pdf",
          web_url: "",
          indexed: false,
        },
      ],
    });

    expect(rows).toHaveLength(2);
    expect(rows[0]?.file_name).toBe("DGA.pdf");
    expect(rows[0]?.web_url).toBe("https://cdn.realfood.gov/DGA.pdf");
    expect(rows[1]?.file_name).toBe("orphan.pdf");
    expect(rows[1]?.web_url).toBe("");
    expect(rows[0]?.title).toBe("");
  });

  it("keeps a citation title on the row", () => {
    const rows = catalogRows({
      files: [
        {
          file_name: "DGA.pdf",
          local_url: "/knowledge/files/DGA.pdf",
          web_url: "https://cdn.realfood.gov/DGA.pdf",
          title: "Dietary Guidelines for Americans",
          indexed: true,
        },
      ],
    });
    expect(rows[0]?.title).toBe("Dietary Guidelines for Americans");
  });

  it("uses /knowledge/files/{file_name} as the local URL", () => {
    expect(localUrlFor("DGA.pdf")).toBe("/knowledge/files/DGA.pdf");
    const rows = catalogRows({
      files: [
        {
          file_name: "DGA.pdf",
          local_url: "",
          web_url: "",
          indexed: true,
        },
      ],
    });
    expect(rows[0]?.local_url).toBe("/knowledge/files/DGA.pdf");
  });
});

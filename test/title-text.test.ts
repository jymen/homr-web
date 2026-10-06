/**
 * The title text homr-web keeps. Two deliberate differences from homr's
 * title_detection, both pinned here with plain strings (no page is needed,
 * and none of a published score may be committed): letters of any alphabet
 * survive the clean-up, and count as letters for the tempo-marking test.
 */
import { describe, expect, it } from "vitest";
import { cleanupText, isTempoMarking } from "../src/ocr/page.js";

describe("cleanupText", () => {
  it("keeps accented letters, apostrophes and hyphens", () => {
    expect(cleanupText("Marche des élèves")).toBe("Marche des élèves");
    expect(cleanupText("Le p'tit Sarny")).toBe("Le p'tit Sarny");
    expect(cleanupText("Reel à Bouchard")).toBe("Reel à Bouchard");
    expect(cleanupText("Saint-Hyacinthe")).toBe("Saint-Hyacinthe");
    expect(cleanupText("Port na bPúcaí")).toBe("Port na bPúcaí");
  });

  it("composes an accent read as a separate combining mark", () => {
    expect(cleanupText("élèves")).toBe("élèves");
  });

  it("still reduces punctuation and runs of space to single spaces, as homr does", () => {
    expect(cleanupText("  The Kesh!  ")).toBe("The Kesh");
    expect(cleanupText("CHORD STUDY")).toBe("CHORD STUDY");
    expect(cleanupText("Grand  Staff · Study")).toBe("Grand Staff Study");
    expect(cleanupText("Jig (No. 2)")).toBe("Jig No 2");
  });
});

describe("isTempoMarking", () => {
  it("drops tempo markings, including a note read as a CJK letter", () => {
    expect(isTempoMarking("小=85")).toBe(true);
    expect(isTempoMarking("= 120")).toBe(true);
    expect(isTempoMarking("jig")).toBe(true);
  });

  it("keeps titles in any alphabet", () => {
    expect(isTempoMarking("Été")).toBe(true); // three characters: too short, as in homr
    expect(isTempoMarking("Marche des élèves")).toBe(false);
    expect(isTempoMarking("Ελληνικός χορός")).toBe(false);
    expect(isTempoMarking("Калинка")).toBe(false);
  });
});

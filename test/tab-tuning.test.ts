/**
 * The tuning and capo parser (src/tab/tuning.ts) on the phrasings tab pages
 * print, English and French, banjo, guitar and mandolin, and on the header
 * text that must not read as either.
 */

import { describe, expect, it } from "vitest";
import type { TabLineCount } from "../src/tab/detect.js";
import {
  parseLetterLabels,
  parseTabText,
  resolveTuning,
  standardStrings,
  type TabTuning,
} from "../src/tab/tuning.js";

const BANJO_G = ["D4", "B3", "G3", "D3", "G4"];
const GUITAR = ["E4", "B3", "G3", "D3", "A2", "E2"];
const DADGAD = ["D4", "A3", "G3", "D3", "A2", "D2"];

function tuningOf(text: string, lines: TabLineCount): TabTuning | undefined {
  const { tuning } = parseTabText(text);
  return tuning === undefined
    ? undefined
    : resolveTuning(tuning, lines, text, 0.9);
}

const stringsOf = (text: string, lines: TabLineCount) => {
  const tuning = tuningOf(text, lines);
  return tuning?.status === "read" ? tuning.strings : tuning?.status;
};

describe("parseTabText: tunings", () => {
  const read: readonly [string, TabLineCount, readonly string[]][] = [
    ["aDADE tuning", 5, ["E4", "D4", "A3", "D3", "A4"]],
    ["aDADE tuning, Brainjo level 3", 5, ["E4", "D4", "A3", "D3", "A4"]],
    ["aEAC#E tuning, Brainjo level 3", 5, ["E4", "C#4", "A3", "E3", "A4"]],
    ["gDGBD", 5, BANJO_G],
    ["GDGBD", 5, BANJO_G],
    ["gCGCD", 5, ["D4", "C4", "G3", "C3", "G4"]],
    ["f#DF#AD", 5, ["D4", "A3", "F#3", "D3", "F#4"]],
    ["gDGBbD", 5, ["D4", "Bb3", "G3", "D3", "G4"]],
    ["g♯DGBD", 5, ["D4", "B3", "G3", "D3", "G#4"]],
    ["DADGAD", 6, DADGAD],
    ["D A D G B E", 6, ["E4", "B3", "G3", "D3", "A2", "D2"]],
    ["D-A-D-G-A-D", 6, DADGAD],
    ["Tuning: E A D G B E", 6, GUITAR],
    ["Accord : DADGAD", 6, DADGAD],
    ["Accordage : Ré La Ré Sol La Ré", 6, DADGAD],
    ["Standard (EADGBE)", 6, GUITAR],
    ["(6)=D", 6, ["E4", "B3", "G3", "D3", "A2", "D2"]],
    ["6=D 5=G", 6, ["E4", "B3", "G3", "D3", "G2", "D2"]],
    ["⑥=D ⑤=G", 6, ["E4", "B3", "G3", "D3", "G2", "D2"]],
    ["(6) = Ré", 6, ["E4", "B3", "G3", "D3", "A2", "D2"]],
    ["Open G", 6, ["D4", "B3", "G3", "D3", "G2", "D2"]],
    ["Open G", 5, BANJO_G],
    ["Accordage en sol ouvert", 6, ["D4", "B3", "G3", "D3", "G2", "D2"]],
    ["Drop D", 6, ["E4", "B3", "G3", "D3", "A2", "D2"]],
    ["Drop D tuning", 6, ["E4", "B3", "G3", "D3", "A2", "D2"]],
    ["Standard tuning", 6, GUITAR],
    ["Standard tuning", 4, ["E5", "A4", "D4", "G3"]],
    ["Standard tuning (GDAE)", 4, ["E5", "A4", "D4", "G3"]],
    ["Double C", 5, ["D4", "C4", "G3", "C3", "G4"]],
    ["Accordage : Double C", 5, ["D4", "C4", "G3", "C3", "G4"]],
    ["Sawmill", 5, ["D4", "C4", "G3", "D3", "G4"]],
    ["Capo 2 Sawmill tuning", 5, ["D4", "C4", "G3", "D3", "G4"]],
    ["Key of Am - Standard Open G Tuning", 5, BANJO_G],
    ["Key of Am - Štandard Open G Tuning", 5, BANJO_G],
    ["Double drop D", 6, ["D4", "B3", "G3", "D3", "A2", "D2"]],
  ];
  for (const [text, lines, strings] of read) {
    it(`${JSON.stringify(text)} on ${lines} lines`, () => {
      expect(stringsOf(text, lines)).toEqual(strings);
    });
  }

  it("keeps the source: letters printed, or a name looked up", () => {
    expect(tuningOf("gDGBD", 5)).toMatchObject({ source: "text" });
    expect(tuningOf("Open G", 5)).toMatchObject({ source: "named" });
    expect(tuningOf("Standard (EADGBE)", 6)).toMatchObject({ source: "text" });
    expect(tuningOf("gDGBD", 5)).toMatchObject({
      confidence: 0.9,
      text: "gDGBD",
    });
  });

  it("reports a tuning for another string count without forcing it", () => {
    expect(tuningOf("DADGAD tuning", 5)).toEqual({
      confidence: 0.9,
      source: "text",
      status: "string_count",
      strings: DADGAD,
      text: "DADGAD tuning",
    });
    expect(tuningOf("Sawmill", 6)).toMatchObject({
      source: "named",
      status: "string_count",
      strings: ["D4", "C4", "G3", "D3", "G4"],
    });
    expect(tuningOf("gDGBD", 4)).toMatchObject({ status: "string_count" });
  });

  it("keeps an unknown name as text, never a guess", () => {
    expect(tuningOf("Open Zeta tuning", 6)).toEqual({
      confidence: 0.9,
      status: "unknown_name",
      text: "Open Zeta tuning",
    });
    expect(tuningOf("Accordage : spécial", 6)).toMatchObject({
      status: "unknown_name",
    });
  });

  const nothing = [
    "Chicken reel",
    "Banjo: Cold Frosty Morning",
    "www.pickinlessons.com",
    "G Major",
    "E minor",
    "Am",
    "Jazz Standard",
    "Key Of A (Capo 2)",
    "Traditional Arr. by Mike Hedding",
    "TAB",
    "Bjo 1",
    "s.guit.",
    "Mandolin",
    "A B C D in a sentence",
    "Clawhammer Banjo",
    "TITMTITM",
  ];
  for (const text of nothing) {
    it(`${JSON.stringify(text)} names no tuning`, () => {
      expect(parseTabText(text).tuning).toBeUndefined();
    });
  }
});

describe("parseTabText: capos", () => {
  const capos: readonly [string, number][] = [
    ["Capo 2", 2],
    ["Capo II", 2],
    ["capo IV", 4],
    ["capo on 2nd fret", 2],
    ["Capo on the 5th fret", 5],
    ["Capo 3rd", 3],
    ["Capo 1st fret", 1],
    ["Capodastre 2", 2],
    ["Capodastre en 3e case", 3],
    ["Capodastre case 4", 4],
    ["Capo : 5", 5],
    ["capo 2 (sounds in A)", 2],
    ["Key Of A (Capo 2)", 2],
    ["Capo 2 Sawmill tuning", 2],
    ["No capo", 0],
    ["sans capodastre", 0],
  ];
  for (const [text, fret] of capos) {
    it(`${JSON.stringify(text)} is capo ${fret}`, () => {
      expect(parseTabText(text).capo).toBe(fret);
    });
  }

  for (const text of ["Capo", "Capo 30", "Capone", "Key of A", "Capo 2nd"]) {
    it(`${JSON.stringify(text)} ${text === "Capo 2nd" ? "is capo 2" : "names no capo"}`, () => {
      expect(parseTabText(text).capo).toBe(text === "Capo 2nd" ? 2 : undefined);
    });
  }
});

describe("parseLetterLabels and standardStrings", () => {
  it("reads one string name per line, top line first", () => {
    const parsed = parseLetterLabels(["e", "B", "G", "D", "A", "D"]);
    expect(
      parsed === undefined ? undefined : resolveTuning(parsed, 6, "", 1)
    ).toMatchObject({
      status: "read",
      strings: ["E4", "B3", "G3", "D3", "A2", "D2"],
    });
    expect(parseLetterLabels(["e", "B", "G", "T"])).toBeUndefined();
  });

  it("gives the standard tuning per line count, top line first", () => {
    expect(standardStrings(4)).toEqual(["E5", "A4", "D4", "G3"]);
    expect(standardStrings(5)).toEqual(BANJO_G);
    expect(standardStrings(6)).toEqual(GUITAR);
  });
});

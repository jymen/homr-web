/**
 * A PDF's text layer read in place of the recogniser (src/tab/layer.ts):
 * runs to glyph boxes, frets on a system's lines, a crop's text, and header
 * lines for the tuning reader.
 */

import { describe, expect, it } from "vitest";
import type { LineGroup } from "../src/tab/detect.js";
import {
  glyphsOf,
  linesIn,
  onLines,
  type PageTextItem,
  textIn,
} from "../src/tab/layer.js";

const WIDTH = 2000;
const HEIGHT = 1000;

/** A run in page pixels, as page-normalised as an app hands it over. */
const item = (
  text: string,
  x: number,
  baseline: number,
  size: number,
  width = 0.6 * size * text.length
): PageTextItem => ({
  text,
  x0: x / WIDTH,
  x1: (x + width) / WIDTH,
  y0: (baseline - size) / HEIGHT,
  y1: baseline / HEIGHT,
});

const group: LineGroup = {
  lines: [300, 330, 360, 390, 420].map((y) => ({
    breakLength: 0,
    x0: 100,
    x1: 1900,
    y,
    y0: y - 1,
    y1: y + 1,
  })),
  spacing: 30,
  x0: 100,
  x1: 1900,
};

/** A fret set in a 40 px font sits on its line: its baseline 14 px under it. */
const fretOn = (text: string, x: number, line: number) =>
  item(text, x, line + 14, 40);

describe("the text layer", () => {
  it("gives each run a box from cap height to just under the baseline", () => {
    expect(glyphsOf([item("0", 500, 314, 40, 22)], WIDTH, HEIGHT)).toEqual([
      { box: { h: 34, w: 22, x: 500, y: 283 }, text: "0" },
    ]);
  });

  it("puts a fret on the line it is centred on, and nothing taller than a fret or off the lines", () => {
    const glyphs = glyphsOf(
      [
        fretOn("0", 500, 300),
        fretOn("12", 600, 390),
        item("G", 500, 250, 60),
        item("4", 150, 330, 70),
        fretOn("3", 700, 345),
      ],
      WIDTH,
      HEIGHT
    );
    expect(onLines(glyphs, group).map((g) => [g.text, g.line])).toEqual([
      ["0", 0],
      ["12", 3],
    ]);
  });

  it("reads a crop as the glyphs centred in it, left to right", () => {
    const glyphs = glyphsOf(
      [
        item("P", 520, 520, 30),
        item("o", 540, 520, 30),
        item("H", 900, 520, 30),
      ],
      WIDTH,
      HEIGHT
    );
    expect(textIn(glyphs, { h: 30, w: 40, x: 515, y: 495 }, 4)).toBe("Po");
    expect(textIn(glyphs, { h: 30, w: 40, x: 700, y: 495 }, 4)).toBe("");
  });

  it("joins the runs of a header line, and keeps two lines apart", () => {
    const glyphs = glyphsOf(
      [
        item("Tuning:", 100, 100, 30, 100),
        item("G", 210, 100, 30, 20),
        item("DGBD", 236, 100, 30, 80),
        item("Clawhammer Banjo", 100, 50, 30, 250),
      ],
      WIDTH,
      HEIGHT
    );
    const lines = linesIn(glyphs, { x0: 50, x1: 1000, y0: 0, y1: 200 });
    expect(lines.map((l) => [l.text, l.score])).toEqual([
      ["Clawhammer Banjo", 1],
      ["Tuning: G DGBD", 1],
    ]);
    expect(lines[1]?.box[0]).toEqual([50, 77]);
  });
});

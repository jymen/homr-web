/**
 * The rhythm geometry (src/tab/rhythm.ts) on a tab system drawn pixel by
 * pixel: stems and beams under fret columns, barlines with repeat dots, an
 * arc between two frets with its letter under the beams, and a time
 * signature's two digits.
 */

import { describe, expect, it } from "vitest";
import type { Box, LineGroup } from "../src/tab/detect.js";
import { type FretColumn, rhythmGeometry } from "../src/tab/rhythm.js";
import { testOpenCv } from "./support/opencv.js";

const WIDTH = 1200;
const HEIGHT = 420;
const LINE_Y = [100, 120, 140, 160, 180];

function page() {
  const ink = new Uint8Array(WIDTH * HEIGHT);
  const fill = (x0: number, x1: number, y0: number, y1: number) => {
    for (let y = y0; y <= y1; y += 1) {
      ink.fill(1, y * WIDTH + x0, y * WIDTH + x1 + 1);
    }
  };
  return { fill, ink };
}

const group: LineGroup = {
  lines: LINE_Y.map((y) => ({
    breakLength: 0,
    x0: 50,
    x1: 1150,
    y: y + 0.5,
    y0: y,
    y1: y + 1,
  })),
  spacing: 20,
  x0: 50,
  x1: 1150,
};

/** A fret's box on 1-based `string`, centred at `x`. */
const fret = (x: number, string: number): Box => ({
  h: 15,
  w: 10,
  x: x - 5,
  y: (LINE_Y[string - 1] ?? 0) - 7,
});

const column = (x: number, string: number): FretColumn => {
  const box = fret(x, string);
  return { bottom: box.y + box.h, centre: x, notes: [{ box, string }] };
};

function drawn() {
  const { fill, ink } = page();
  for (const y of LINE_Y) {
    fill(50, 1150, y, y + 1);
  }
  fill(50, 51, 100, 181);
  fill(60, 65, 100, 181);
  fill(68, 69, 100, 181);
  fill(75, 79, 128, 132);
  fill(75, 79, 148, 152);
  fill(85, 100, 104, 129);
  fill(85, 100, 134, 157);
  fill(600, 601, 100, 181);
  fill(1130, 1134, 128, 132);
  fill(1130, 1134, 148, 152);
  fill(1140, 1141, 100, 181);
  fill(1144, 1149, 100, 181);
  const columns = [
    column(150, 1),
    column(200, 5),
    column(300, 1),
    column(400, 3),
    column(480, 4),
    column(520, 4),
    column(700, 2),
    column(760, 2),
  ];
  for (const { notes } of columns) {
    for (const { box } of notes) {
      fill(box.x, box.x + box.w - 1, box.y, box.y + box.h - 1);
    }
  }
  fill(150, 150, 110, 215);
  fill(200, 200, 190, 215);
  fill(150, 200, 210, 215);
  fill(300, 300, 110, 160);
  fill(480, 480, 170, 215);
  fill(520, 520, 170, 215);
  fill(480, 520, 210, 214);
  fill(480, 520, 202, 206);
  fill(703, 757, 106, 108);
  fill(696, 704, 240, 254);
  return { columns, ink };
}

describe("rhythmGeometry", () => {
  it("reads a stem's beams as its value, no stem as none, and barlines with their repeats", async () => {
    const cv = await testOpenCv();
    const { columns, ink } = drawn();
    const geometry = rhythmGeometry(cv, ink, WIDTH, HEIGHT, group, columns);
    expect(
      geometry.stems.map((s) => (s.kind === "none" ? "none" : s.beams))
    ).toEqual([1, 1, 0, "none", 2, 2, "none", "none"]);
    expect(
      geometry.barlines.map((b) => ({
        end: b.repeatEnd,
        start: b.repeatStart,
        thick: b.thick,
        x: Math.round(b.px),
      }))
    ).toEqual([
      { end: false, start: true, thick: true, x: 60 },
      { end: false, start: false, thick: false, x: 601 },
      { end: true, start: false, thick: true, x: 1145 },
    ]);
  });

  it("ties an arc to the two frets on the string under it, with the letter by the first", async () => {
    const cv = await testOpenCv();
    const { columns, ink } = drawn();
    const { slurs } = rhythmGeometry(cv, ink, WIDTH, HEIGHT, group, columns);
    expect(slurs.map(({ from, string, to }) => ({ from, string, to }))).toEqual(
      [{ from: 6, string: 2, to: 7 }]
    );
    expect(slurs[0]?.letters[0]).toEqual({ h: 15, w: 9, x: 696, y: 240 });
  });

  it("offers the two stacked digits before the first fret as the time signature", async () => {
    const cv = await testOpenCv();
    const { columns, ink } = drawn();
    const { timeDigits } = rhythmGeometry(
      cv,
      ink,
      WIDTH,
      HEIGHT,
      group,
      columns
    );
    expect(timeDigits[0]).toEqual([
      { h: 26, w: 16, x: 85, y: 104 },
      { h: 24, w: 16, x: 85, y: 134 },
    ]);
  });
});

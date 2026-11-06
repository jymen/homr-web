/**
 * The button tab (CADB) fixtures' truth (tools/typeset-tab.mjs, written to
 * test/fixtures/button-tab/truth.json): every reading has the H2
 * ButtonTabReading shape (docs/homr-web-plan/button-tab.md in the app), every
 * mark and letter stands inside its staff's horizontal extent, and the page
 * really carries ink at each one's column inside its band. No model runs.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import type { ButtonTabReading } from "../src/tab/button-read.js";

const FIXTURES = join(
  dirname(fileURLToPath(import.meta.url)),
  "fixtures/button-tab"
);

interface TruthStaff {
  readonly bandBottom: number;
  readonly bandTop: number;
  readonly bottom: number;
  readonly rule: number;
  readonly staff: number;
  readonly top: number;
  readonly x0: number;
  readonly x1: number;
}

interface TruthPage {
  readonly buttonTabs: readonly unknown[];
  readonly file: string;
  readonly height: number;
  readonly rotation: number;
  readonly staves: readonly TruthStaff[];
  readonly width: number;
}

const truth: { readonly pages: readonly TruthPage[] } = JSON.parse(
  readFileSync(join(FIXTURES, "truth.json"), "utf8")
);

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const hasKeys = (value: Record<string, unknown>, keys: readonly string[]) =>
  Object.keys(value).sort().join() === [...keys].sort().join();

const isUnit = (value: unknown): value is number =>
  typeof value === "number" && value >= 0 && value <= 1;

/** A runtime check of the ButtonTabReading shape, refusing any extra key. */
function isButtonTabReading(value: unknown): value is ButtonTabReading {
  if (
    !(
      isRecord(value) &&
      hasKeys(value, ["staff", "marks", "leftHand", "unread"]) &&
      Number.isInteger(value.staff) &&
      Number.isInteger(value.unread) &&
      Array.isArray(value.marks) &&
      Array.isArray(value.leftHand)
    )
  ) {
    return false;
  }
  const marksOk = value.marks.every(
    (mark: unknown) =>
      isRecord(mark) &&
      hasKeys(mark, ["x", "bellows", "button", "row", "held"]) &&
      isUnit(mark.x) &&
      (mark.bellows === "push" || mark.bellows === "pull") &&
      Number.isInteger(mark.button) &&
      (mark.button as number) >= 1 &&
      (mark.row === 0 || mark.row === 1 || mark.row === 2) &&
      typeof mark.held === "boolean"
  );
  const lettersOk = value.leftHand.every(
    (item: unknown) =>
      isRecord(item) &&
      hasKeys(item, ["x", "text"]) &&
      isUnit(item.x) &&
      typeof item.text === "string" &&
      item.text !== ""
  );
  return marksOk && lettersOk;
}

function readings(page: TruthPage): ButtonTabReading[] {
  return page.buttonTabs.map((reading, k) => {
    if (!isButtonTabReading(reading)) {
      throw new Error(
        `${page.file}: buttonTabs[${k}] is not a ButtonTabReading`
      );
    }
    return reading;
  });
}

/** Dark pixels within 12 px of page x, between two page rows: wide enough to reach both digits of an 11, whose ink centre is the gap between them. */
function inkAt(png: PNG, x: number, y0: number, y1: number): number {
  let dark = 0;
  for (let y = Math.floor(y0); y <= Math.ceil(y1); y += 1) {
    for (let dx = -12; dx <= 12; dx += 1) {
      const column = Math.round(x) + dx;
      if ((png.data[(y * png.width + column) * 4] ?? 255) < 128) {
        dark += 1;
      }
    }
  }
  return dark;
}

describe("button tab fixture truth", () => {
  it("holds a straight page and one turned 0.3 degrees, with the same marks", () => {
    expect(truth.pages.map((page) => page.rotation)).toEqual([0, 0.3]);
    const [straight, rotated] = truth.pages.map(readings);
    const strip = (reading: ButtonTabReading) => ({
      ...reading,
      leftHand: reading.leftHand.map(({ text }) => text),
      marks: reading.marks.map(({ x: _x, ...rest }) => rest),
    });
    expect(rotated?.map(strip)).toEqual(straight?.map(strip));
  });

  it("covers what the plan asks of a fixture", () => {
    const marks = readings(truth.pages[0] as TruthPage).flatMap((r) => r.marks);
    expect(new Set(marks.map((m) => m.bellows))).toEqual(
      new Set(["push", "pull"])
    );
    expect(new Set(marks.map((m) => m.row))).toEqual(new Set([0, 1, 2]));
    expect(marks.filter((m) => m.button >= 10).length).toBeGreaterThan(2);
    expect(marks.filter((m) => m.held).length).toBeGreaterThan(1);
  });

  for (const page of truth.pages) {
    it(`${page.file}: every reading is a ButtonTabReading on its own staff, marks inside the staff and on ink`, () => {
      const png = PNG.sync.read(readFileSync(join(FIXTURES, page.file)));
      expect([png.width, png.height]).toEqual([page.width, page.height]);
      const list = readings(page);
      expect(list.map((r) => r.staff)).toEqual(page.staves.map((s) => s.staff));
      for (const reading of list) {
        const staff = page.staves[reading.staff] as TruthStaff;
        expect(reading.unread).toBe(0);
        expect(staff.top).toBeLessThan(staff.bottom);
        expect(staff.bottom).toBeLessThan(staff.bandTop);
        expect(staff.bandTop).toBeLessThan(staff.rule);
        expect(staff.rule).toBeLessThan(staff.bandBottom);
        const xs = [...reading.marks, ...reading.leftHand].map((m) => m.x);
        for (const x of xs) {
          expect(x).toBeGreaterThan(staff.x0);
          expect(x).toBeLessThan(staff.x1);
        }
        for (const bellows of ["push", "pull"] as const) {
          const band = reading.marks
            .filter((m) => m.bellows === bellows)
            .map((m) => m.x);
          expect(band).toEqual([...band].sort((a, b) => a - b));
        }
        // Clear of the rule, which drifts up to 6 px from its mid-page row on the turned page.
        const margin = 8 / page.height;
        for (const mark of reading.marks) {
          const [top, bottom] =
            mark.bellows === "push"
              ? [staff.bandTop, staff.rule]
              : [staff.rule, staff.bandBottom];
          expect(
            inkAt(
              png,
              mark.x * page.width,
              (top + margin) * page.height,
              (bottom - margin) * page.height
            ),
            `${page.file} staff ${reading.staff} mark ${mark.button} at ${mark.x}`
          ).toBeGreaterThan(20);
        }
      }
    });
  }
});

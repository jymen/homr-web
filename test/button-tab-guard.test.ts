/**
 * The button tab guard (src/tab/button.ts): the push/pull bands under a
 * staff found and painted out before homr reads the page.
 *
 * - Lines are traced across a turned page, which whole-row cover cannot do.
 * - On the typeset CADB fixtures (test/fixtures/button-tab/), every band of
 *   truth.json is found where the truth puts it, under its staff, and the
 *   bands, the margin letters and the letter line come out white while the
 *   staff is untouched.
 * - No band is found on the other public pages, nor on the local pages of
 *   staves and tablature where present.
 *
 * BUTTON_TAB_PAGES, a ":"-separated list of `path#bands` entries, adds pages
 * that may not be committed (the findings' book photo is one): each must
 * give exactly that many bands, and is named in the output by its index
 * only.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import {
  type ColorImage,
  colorImageFromRgba,
  createColor,
} from "../src/image/plane.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { recognizeCrops } from "../src/ocr/rapid-ocr.js";
import {
  type DetectedButtonTab,
  detectButtonTabs,
  lineYAt,
  type TracedLine,
  traceLines,
  whitenButtonTabs,
  withStaves,
} from "../src/tab/button.js";
import type { ReadCrops } from "../src/tab/detect.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const PAGE_TIMEOUT_MS = 300_000;
const FIXTURES = join(import.meta.dirname, "fixtures");
const BUTTON = join(FIXTURES, "button-tab");

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
  readonly file: string;
  readonly height: number;
  readonly staves: readonly TruthStaff[];
  readonly width: number;
}

const truth: { readonly pages: readonly TruthPage[] } = JSON.parse(
  readFileSync(join(BUTTON, "truth.json"), "utf8")
);

const pngPage = (path: string): ColorImage => {
  const png = PNG.sync.read(readFileSync(path));
  return colorImageFromRgba(png.width, png.height, png.data);
};

const grayAt = (page: ColorImage, x: number, y: number): number => {
  const at = (y * page.width + x) * 3;
  return (
    ((page.data[at] ?? 0) +
      (page.data[at + 1] ?? 0) +
      (page.data[at + 2] ?? 0)) /
    3
  );
};

/** A white page with the given lines drawn black, each `thickness` rows from `(x0, y0)` to `(x1, y1)`. */
function linesPage(
  width: number,
  height: number,
  lines: readonly (readonly [number, number, number, number])[],
  thickness = 3
): ColorImage {
  const page = createColor(width, height);
  page.data.fill(255);
  for (const [x0, y0, x1, y1] of lines) {
    for (let x = x0; x <= x1; x += 1) {
      const y = Math.round(y0 + ((y1 - y0) * (x - x0)) / (x1 - x0));
      for (let row = y; row < y + thickness; row += 1) {
        page.data.fill(0, (row * width + x) * 3, (row * width + x + 1) * 3);
      }
    }
  }
  return page;
}

describe("tracing lines", () => {
  it("follows a staff and a rule turned 3 degrees, which whole-row cover loses", () => {
    const slope = 0.055;
    const staff = [0, 1, 2, 3, 4].map(
      (k) => [100, 200 + 20 * k, 1100, 200 + 20 * k + 1000 * slope] as const
    );
    const rule = [100, 380, 1100, 380 + 1000 * slope] as const;
    const page = linesPage(1200, 600, [...staff, rule]);
    const ink = new Uint8Array(page.width * page.height);
    for (let k = 0; k < ink.length; k += 1) {
      ink[k] =
        grayAt(page, k % page.width, Math.floor(k / page.width)) < 128 ? 1 : 0;
    }
    const lines = traceLines(ink, page.width, page.height).sort(
      (a, b) => lineYAt(a, 600) - lineYAt(b, 600)
    );
    expect(lines).toHaveLength(6);
    for (const [k, [x0, y0]] of [...staff, rule].entries()) {
      for (const x of [150, 600, 1050]) {
        const expected = y0 + 1 + slope * (x - x0);
        const line = lines[k] as TracedLine;
        expect(Math.abs(lineYAt(line, x) - expected)).toBeLessThan(1.5);
      }
    }
  });

  it("never consults the recogniser for a page of staves", async () => {
    const cv = await testOpenCv();
    const page = linesPage(
      1200,
      600,
      [0, 1, 2, 3, 4].flatMap((k) => [
        [100, 100 + 20 * k, 1100, 140 + 20 * k] as const,
        [100, 350 + 20 * k, 1100, 390 + 20 * k] as const,
      ])
    );
    const unreachable: ReadCrops = () =>
      Promise.reject(new Error("the recogniser was consulted"));
    expect(await detectButtonTabs(cv, page, unreachable)).toEqual([]);
  });
});

describe("withStaves", () => {
  const tab = (rule: number): DetectedButtonTab => ({
    band: {
      bandBottom: rule + 0.01,
      bandTop: rule - 0.01,
      rule,
      x0: 0.1,
      x1: 0.9,
    },
    down: 10,
    letterLine: [],
    letters: {
      p: { h: 10, w: 8, x: 0, y: 0 },
      t: { h: 10, w: 8, x: 0, y: 20 },
    },
    marginX: 0,
    rule: { points: [{ x: 0, y: 0 }] },
    spacing: 5,
    up: 10,
    x0: 100,
    x1: 900,
  });
  it("takes the lowest staff above the rule that overlaps it", () => {
    const staves = [
      { cx: 0.5, cy: 0.1, h: 0.03, index: 0, w: 0.8 },
      { cx: 0.5, cy: 0.3, h: 0.03, index: 1, w: 0.8 },
      { cx: 0.5, cy: 0.5, h: 0.03, index: 2, w: 0.8 },
    ];
    expect(
      withStaves([tab(0.15), tab(0.36), tab(0.05)], staves).map(
        ({ band }) => band.staff
      )
    ).toEqual([0, 1, undefined]);
  });
});

/** The recogniser on crops, opened once per test. */
async function recogniser() {
  const cv = await testOpenCv();
  const store = await storeOn(CPU);
  const session = await store.open("ocrRecognize");
  const characters = ctcCharacters(session.metadata.get("character") ?? "");
  const calls = { crops: 0 };
  const read: ReadCrops = async (crops) => {
    calls.crops += crops.length;
    return (await recognizeCrops(cv, session, characters, crops)).map(
      (r) => r.text
    );
  };
  return { calls, close: () => store.close(), cv, read };
}

describeWithModels("the button tab guard on the CADB fixtures", () => {
  for (const sheet of truth.pages) {
    it(
      `${sheet.file}: every band found, under its staff, painted out`,
      async () => {
        const { close, cv, read } = await recogniser();
        try {
          const page = pngPage(join(BUTTON, sheet.file));
          const tabs = await detectButtonTabs(cv, page, read);
          expect(tabs).toHaveLength(sheet.staves.length);
          const staves = sheet.staves.map((s) => ({
            cx: (s.x0 + s.x1) / 2,
            cy: (s.top + s.bottom) / 2,
            h: s.bottom - s.top,
            index: s.staff,
            w: s.x1 - s.x0,
          }));
          const placed = withStaves(tabs, staves);
          const painted = whitenButtonTabs(page, tabs);
          for (const [k, staff] of sheet.staves.entries()) {
            const { band } = placed[k] as DetectedButtonTab;
            // Half a line spacing, in page-normalised units.
            const dy = (staff.bottom - staff.top) / 8;
            const dx = (dy * sheet.height) / sheet.width;
            expect(band.staff).toBe(staff.staff);
            expect(Math.abs(band.bandTop - staff.bandTop)).toBeLessThan(dy);
            expect(Math.abs(band.rule - staff.rule)).toBeLessThan(dy);
            expect(Math.abs(band.bandBottom - staff.bandBottom)).toBeLessThan(
              dy
            );
            expect(Math.abs(band.x0 - staff.x0)).toBeLessThan(dx);
            expect(Math.abs(band.x1 - staff.x1)).toBeLessThan(dx);

            // White from the band's top to two spacings under it (the letter
            // line), from three spacings left of the rule (P and T) to its
            // end; 8 px off the band's top for the turned page's drift.
            const spacing = ((staff.bottom - staff.top) * sheet.height) / 4;
            const top = Math.round(staff.bandTop * sheet.height) + 8;
            const bottom = Math.round(
              staff.bandBottom * sheet.height + 1.8 * spacing
            );
            const left = Math.round(staff.x0 * sheet.width - 3 * spacing);
            const right = Math.round(staff.x1 * sheet.width);
            let dark = 0;
            for (let y = top; y <= bottom; y += 1) {
              for (let x = left; x <= right; x += 1) {
                dark += grayAt(painted, x, y) < 128 ? 1 : 0;
              }
            }
            expect(dark).toBe(0);

            // The staff itself, down to its lowest stem, untouched.
            for (
              let y = Math.round((staff.top - 4 * dy) * sheet.height);
              y <= Math.round((staff.bottom + 4 * dy) * sheet.height);
              y += 1
            ) {
              for (let x = left; x <= right; x += 1) {
                if (grayAt(painted, x, y) !== grayAt(page, x, y)) {
                  throw new Error(`staff ${k} painted at (${x}, ${y})`);
                }
              }
            }
          }
        } finally {
          await close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

const pngsIn = (dir: string): string[] =>
  existsSync(dir)
    ? readdirSync(dir)
        .filter((name) => name.endsWith(".png"))
        .map((name) => join(dir, name))
    : [];

const OTHER_PAGES = [
  join(FIXTURES, "the-kesh-300dpi.png"),
  join(FIXTURES, "chord-study-300dpi.png"),
  join(FIXTURES, "grand-staff-300dpi.png"),
  ...pngsIn(join(FIXTURES, "tab")),
  ...pngsIn(join(FIXTURES, "local", "staves")),
  ...pngsIn(join(FIXTURES, "local", "tablature", ".pages")),
];

describeWithModels("the button tab guard elsewhere", () => {
  it(
    `finds no band on ${OTHER_PAGES.length} pages of staves and string tab`,
    async () => {
      const { close, cv, read } = await recogniser();
      try {
        const found: string[] = [];
        for (const path of OTHER_PAGES) {
          // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, a page at a time
          const tabs = await detectButtonTabs(cv, pngPage(path), read);
          if (tabs.length > 0) {
            found.push(`${path}: ${tabs.length}`);
          }
        }
        expect(found).toEqual([]);
      } finally {
        await close();
      }
    },
    PAGE_TIMEOUT_MS
  );

  const local = (process.env.BUTTON_TAB_PAGES ?? "")
    .split(":")
    .filter((entry) => entry !== "")
    .map((entry) => {
      const hash = entry.lastIndexOf("#");
      return {
        bands: Number(entry.slice(hash + 1)),
        path: entry.slice(0, hash),
      };
    });
  for (const [k, { bands, path }] of local.entries()) {
    it(
      `local page ${k + 1}: ${bands} bands`,
      async () => {
        const { close, cv, read } = await recogniser();
        try {
          const tabs = await detectButtonTabs(cv, pngPage(path), read);
          expect(tabs).toHaveLength(bands);
          for (const [j, tab] of tabs.entries()) {
            const next = tabs[j + 1];
            expect(tab.band.bandTop).toBeLessThan(tab.band.rule);
            expect(tab.band.rule).toBeLessThan(tab.band.bandBottom);
            if (next !== undefined) {
              expect(tab.band.bandBottom).toBeLessThan(next.band.bandTop);
            }
          }
        } finally {
          await close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

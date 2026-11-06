/**
 * The button tab reader (src/tab/button-read.ts): the numbers in the
 * push/pull bands the guard found, read with their suffix and column.
 *
 * - `readButton` turns a recogniser's text into a button, a row mark and a
 *   held dash, and refuses what is no button.
 * - On the typeset CADB fixtures (test/fixtures/button-tab/), the reading
 *   equals truth.json mark for mark, x within 3 px, on the straight page
 *   and on the one turned 0.3 degrees, and every two-digit number is the
 *   recogniser's, not the single-digit classifier's. The left-hand
 *   letters equal truth.json's too, x within 3 px.
 *
 * BUTTON_TAB_READ_PAGES, a ":"-separated list of `page.png#truth.json`
 * entries, adds pages that may not be committed (the findings' book photo
 * is one). Their truth lists, per system, bars of marks left to right (band
 * "P" or "T", button, row, held), with no x. Each system's reading is
 * aligned with its truth in order, and the output gives counts and miss
 * classes only, never the marks themselves: every truth mark must be read.
 * The left-hand line is checked the same way against each system's bars'
 * `leftHand` letters.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { detectButtonTabs, localInkOf } from "../src/tab/button.js";
import { readLeftHand } from "../src/tab/button-letters.js";
import {
  type ButtonTabNumber,
  type ButtonTabReading,
  readButton,
  readButtonTabNumbers,
  readButtonTabs,
} from "../src/tab/button-read.js";
import type { ReadCrops } from "../src/tab/detect.js";
import { classifyDigits } from "../src/tab/digits.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const PAGE_TIMEOUT_MS = 300_000;
const BUTTON = join(import.meta.dirname, "fixtures", "button-tab");

interface TruthPage {
  readonly buttonTabs: readonly ButtonTabReading[];
  readonly file: string;
  readonly width: number;
}

const truth: { readonly pages: readonly TruthPage[] } = JSON.parse(
  readFileSync(join(BUTTON, "truth.json"), "utf8")
);

const pngPage = (path: string): ColorImage => {
  const png = PNG.sync.read(readFileSync(path));
  return colorImageFromRgba(png.width, png.height, png.data);
};

describe("readButton", () => {
  it("reads a button with its row mark and dash", () => {
    expect(readButton("7")).toEqual({ button: 7, held: false, row: 0 });
    expect(readButton("7'")).toEqual({ button: 7, held: false, row: 1 });
    expect(readButton("10’")).toEqual({ button: 10, held: false, row: 1 });
    expect(readButton("3''")).toEqual({ button: 3, held: false, row: 2 });
    expect(readButton('3"')).toEqual({ button: 3, held: false, row: 2 });
    expect(readButton("8'-")).toEqual({ button: 8, held: true, row: 1 });
    expect(readButton(" 1O ")).toEqual({ button: 10, held: false, row: 0 });
  });

  it("refuses what is no button", () => {
    for (const text of ["", "T", "P", "0", "13", "7a", "123"]) {
      expect(readButton(text), text).toBeUndefined();
    }
  });
});

/** The recogniser and the tab digit classifier on crops, opened once per test. */
async function readers() {
  const cv = await testOpenCv();
  const store = await storeOn(CPU);
  const session = await store.open("ocrRecognize");
  const digits = await store.open("tabDigits");
  const characters = ctcCharacters(session.metadata.get("character") ?? "");
  const read: ReadCrops = async (crops) =>
    (await recognizeCrops(cv, session, characters, crops)).map((r) => r.text);
  const classify = (crops: readonly ColorImage[]) =>
    classifyDigits(digits, crops);
  return { classify, close: () => store.close(), cv, read };
}

describeWithModels("the button tab reader on the CADB fixtures", () => {
  for (const sheet of truth.pages) {
    it(
      `${sheet.file}: every mark of truth.json, x within 3 px`,
      async () => {
        const { classify, close, cv, read } = await readers();
        try {
          const page = pngPage(join(BUTTON, sheet.file));
          const tabs = await detectButtonTabs(cv, page, read);
          const got = await readButtonTabs(cv, page, tabs, read, classify);
          const ink = localInkOf(cv, page);
          for (const tab of tabs) {
            // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, a tab at a time
            const numbers = await readButtonTabNumbers(
              cv,
              page,
              ink,
              tab,
              read,
              classify
            );
            // The classifier reads one digit; every 10 and 11 is the recogniser's.
            for (const number of numbers) {
              if ((number.mark?.button ?? 0) >= 10) {
                expect(number.readBy).toBe("recogniser");
              }
            }
          }
          expect(got).toHaveLength(sheet.buttonTabs.length);
          for (const [k, want] of sheet.buttonTabs.entries()) {
            const reading = got[k] as ButtonTabReading;
            expect(reading.staff).toBe(want.staff);
            expect(reading.unread).toBe(0);
            expect(reading.leftHand.map(({ text }) => text)).toEqual(
              want.leftHand.map(({ text }) => text)
            );
            for (const [j, letter] of reading.leftHand.entries()) {
              const dx = (letter.x - (want.leftHand[j]?.x ?? 0)) * sheet.width;
              expect(Math.abs(dx), `staff ${k} letter ${j}`).toBeLessThan(3);
            }
            const strip = ({ marks }: ButtonTabReading) =>
              marks.map(({ x: _x, ...rest }) => rest);
            expect(strip(reading)).toEqual(strip(want));
            for (const [j, mark] of reading.marks.entries()) {
              const dx = (mark.x - (want.marks[j]?.x ?? 0)) * sheet.width;
              expect(Math.abs(dx), `staff ${k} mark ${j}`).toBeLessThan(3);
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

/** A local page's truth: per system, its bars' marks and left-hand letters left to right. */
interface LocalTruth {
  readonly systems: readonly {
    readonly bars: readonly {
      readonly leftHand: readonly string[];
      readonly marks: readonly {
        readonly band: "P" | "T";
        readonly button: number;
        readonly held: boolean;
        readonly row: number;
      }[];
    }[];
  }[];
}

type LocalMark = LocalTruth["systems"][number]["bars"][number]["marks"][number];

const MISS_CLASSES = [
  "ok",
  "digit",
  "apostrophe",
  "dash",
  "bellows",
  "unread",
  "missed",
  "extra",
] as const;
type MissClass = (typeof MISS_CLASSES)[number];

const bandOf = (number: ButtonTabNumber): LocalMark["band"] =>
  number.bellows === "push" ? "P" : "T";

/** What one truth mark and the number aligned with it make; a merge or a split shows as a digit, an unread or an extra next to a missed. */
function classOf(
  want: LocalMark | undefined,
  got: ButtonTabNumber | undefined
): MissClass {
  if (want === undefined) {
    return "extra";
  }
  if (got === undefined) {
    return "missed";
  }
  if (got.mark === undefined) {
    return "unread";
  }
  if (bandOf(got) !== want.band) {
    return "bellows";
  }
  if (got.mark.button !== want.button) {
    return "digit";
  }
  if (got.mark.row !== want.row) {
    return "apostrophe";
  }
  return got.mark.held === want.held ? "ok" : "dash";
}

/** Truth and reading aligned in order (Needleman-Wunsch, gap 1, a near miss in the right band 0.5), each pair classed. */
function alignedClasses(
  want: readonly LocalMark[],
  got: readonly ButtonTabNumber[]
): Record<MissClass, number> {
  const cost = (a: LocalMark, b: ButtonTabNumber) => {
    const kind = classOf(a, b);
    if (kind === "ok") {
      return 0;
    }
    return kind === "bellows" ? 1.2 : 0.5;
  };
  const n = want.length;
  const m = got.length;
  // table[r][c]: the cheapest alignment of want's first r marks with got's first c.
  const table = Array.from({ length: n + 1 }, () =>
    new Array<number>(m + 1).fill(0)
  );
  const at = (r: number, c: number) => table[r]?.[c] ?? 0;
  for (let r = 0; r <= n; r += 1) {
    for (let c = 0; c <= m; c += 1) {
      const row = table[r] as number[];
      if (r === 0 || c === 0) {
        row[c] = r + c;
        continue;
      }
      row[c] = Math.min(
        at(r - 1, c) + 1,
        at(r, c - 1) + 1,
        at(r - 1, c - 1) +
          cost(want[r - 1] as LocalMark, got[c - 1] as ButtonTabNumber)
      );
    }
  }
  const counts = Object.fromEntries(MISS_CLASSES.map((c) => [c, 0])) as Record<
    MissClass,
    number
  >;
  let i = n;
  let j = m;
  while (i > 0 || j > 0) {
    const a = want[i - 1];
    const b = got[j - 1];
    let kind: MissClass;
    if (
      a !== undefined &&
      b !== undefined &&
      at(i, j) === at(i - 1, j - 1) + cost(a, b)
    ) {
      kind = classOf(a, b);
      i -= 1;
      j -= 1;
    } else if (a !== undefined && at(i, j) === at(i - 1, j) + 1) {
      kind = "missed";
      i -= 1;
    } else {
      kind = "extra";
      j -= 1;
    }
    counts[kind] += 1;
  }
  return counts;
}

const LOCAL = (process.env.BUTTON_TAB_READ_PAGES ?? "")
  .split(":")
  .filter((entry) => entry !== "")
  .map((entry) => {
    const hash = entry.lastIndexOf("#");
    return { page: entry.slice(0, hash), truth: entry.slice(hash + 1) };
  });

const LETTER_CLASSES = ["ok", "letter", "case", "missed", "extra"] as const;
type LetterClass = (typeof LETTER_CLASSES)[number];

const letterClass = (want: string, got: string): LetterClass => {
  if (want === got) {
    return "ok";
  }
  return want.toLowerCase() === got.toLowerCase() ? "case" : "letter";
};

/** Truth letters and letters read aligned in order (gap 1, a wrong letter 0.5), each pair classed. */
function alignedLetters(
  want: readonly string[],
  got: readonly string[]
): Record<LetterClass, number> {
  const cost = (a: string, b: string) => (a === b ? 0 : 0.5);
  const table = Array.from({ length: want.length + 1 }, () =>
    new Array<number>(got.length + 1).fill(0)
  );
  const at = (i: number, j: number) => table[i]?.[j] ?? 0;
  for (let i = 0; i <= want.length; i += 1) {
    for (let j = 0; j <= got.length; j += 1) {
      const row = table[i] as number[];
      row[j] =
        i === 0 || j === 0
          ? i + j
          : Math.min(
              at(i - 1, j) + 1,
              at(i, j - 1) + 1,
              at(i - 1, j - 1) +
                cost(want[i - 1] as string, got[j - 1] as string)
            );
    }
  }
  const counts = Object.fromEntries(
    LETTER_CLASSES.map((kind) => [kind, 0])
  ) as Record<LetterClass, number>;
  let r = want.length;
  let c = got.length;
  while (r > 0 || c > 0) {
    const a = want[r - 1];
    const b = got[c - 1];
    if (
      a !== undefined &&
      b !== undefined &&
      at(r, c) === at(r - 1, c - 1) + cost(a, b)
    ) {
      counts[letterClass(a, b)] += 1;
      r -= 1;
      c -= 1;
    } else if (a !== undefined && at(r, c) === at(r - 1, c) + 1) {
      counts.missed += 1;
      r -= 1;
    } else {
      counts.extra += 1;
      c -= 1;
    }
  }
  return counts;
}

// Declared only with pages to read: vitest fails a suite with no test in it.
const describeLocal = LOCAL.length === 0 ? describe.skip : describeWithModels;

describeLocal("the button tab reader on local pages", () => {
  const local = LOCAL;
  for (const [k, paths] of local.entries()) {
    it(
      `local page ${k + 1}: every truth mark read, system by system`,
      async () => {
        const { classify, close, cv, read } = await readers();
        try {
          const want: LocalTruth = JSON.parse(
            readFileSync(paths.truth, "utf8")
          );
          const page = pngPage(paths.page);
          const started = performance.now();
          const tabs = await detectButtonTabs(cv, page, read);
          const found = performance.now();
          expect(tabs).toHaveLength(want.systems.length);
          const ink = localInkOf(cv, page);
          const rows: string[] = [];
          let missed = 0;
          let recognised = 0;
          let total = 0;
          for (const [s, tab] of tabs.entries()) {
            // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, a tab at a time
            const got = await readButtonTabNumbers(
              cv,
              page,
              ink,
              tab,
              read,
              classify
            );
            const marks = (want.systems[s]?.bars ?? []).flatMap(
              (bar) => bar.marks
            );
            const counts = alignedClasses(marks, got);
            missed += marks.length - counts.ok;
            recognised += got.filter((n) => n.readBy === "recogniser").length;
            total += got.length;
            rows.push(
              `system ${s + 1}: ${counts.ok}/${marks.length} read; ${MISS_CLASSES.slice(
                1
              )
                .map((c) => `${c} ${counts[c]}`)
                .join(", ")}`
            );
          }
          const done = performance.now();
          process.stdout.write(
            `local page ${k + 1}, bands ${Math.round(found - started)} ms, numbers ${Math.round(done - found)} ms, ${recognised} of ${total} marks read by the recogniser\n${rows.join("\n")}\n`
          );
          expect(missed).toBe(0);
        } finally {
          await close();
        }
      },
      PAGE_TIMEOUT_MS
    );
    it(
      `local page ${k + 1}: every left-hand letter read, system by system`,
      async () => {
        const { close, cv, read } = await readers();
        try {
          const want: LocalTruth = JSON.parse(
            readFileSync(paths.truth, "utf8")
          );
          const page = pngPage(paths.page);
          const tabs = await detectButtonTabs(cv, page, read);
          expect(tabs).toHaveLength(want.systems.length);
          const ink = localInkOf(cv, page);
          const started = performance.now();
          const rows: string[] = [];
          let wrong = 0;
          for (const [s, tab] of tabs.entries()) {
            // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, a tab at a time
            const { letters, readAs } = await readLeftHand(
              cv,
              page,
              ink,
              tab,
              read
            );
            const truthLetters = (want.systems[s]?.bars ?? []).flatMap(
              (bar) => bar.leftHand
            );
            const counts = alignedLetters(
              truthLetters,
              letters.map(({ text }) => text)
            );
            wrong += truthLetters.length - counts.ok;
            rows.push(
              `system ${s + 1}: ${counts.ok}/${truthLetters.length} letters read (${readAs}); ${LETTER_CLASSES.slice(
                1
              )
                .map((c) => `${c} ${counts[c]}`)
                .join(", ")}`
            );
          }
          process.stdout.write(
            `local page ${k + 1}, letters ${Math.round(performance.now() - started)} ms\n${rows.join("\n")}\n`
          );
          expect(wrong).toBe(0);
        } finally {
          await close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

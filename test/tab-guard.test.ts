/**
 * The tab guard on pages drawn in the test (test/support/synthetic-page.ts):
 * which line groups are tablature, when the recogniser is consulted, what the
 * guard paints out, and the `tablature_only` result.
 */

import { describe, expect, it } from "vitest";
import type { ModelStore } from "../src/models/store.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import {
  detectTablature,
  isFretText,
  type ReadCrops,
  whitenTablature,
} from "../src/tab/detect.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";
import {
  type Drawn,
  drawnFrets,
  drawPage,
  PAGE_HEIGHT,
  PAGE_WIDTH,
} from "./support/synthetic-page.js";

const PAGE_TIMEOUT_MS = 300_000;
const TAB = 21;
const STAFF = 12;

/** A reader that must not be reached. */
const unreachable: ReadCrops = () =>
  Promise.reject(new Error("the recogniser was consulted"));

/** A reader answering every crop with `text`, counting the crops. */
const answering = (text: string) => {
  const seen = { crops: 0 };
  const read: ReadCrops = (crops) => {
    seen.crops += crops.length;
    return Promise.resolve(crops.map(() => text));
  };
  return { read, seen };
};

const detect = async (systems: readonly Drawn[], read: ReadCrops) =>
  await detectTablature(
    await testOpenCv(),
    drawPage(await testOpenCv(), systems),
    read
  );

describe("isFretText", () => {
  it("takes a fret with a technique letter stuck to it, and an O for a zero", () => {
    for (const text of ["0", "7", "12", "24", "S0", "7h", "p3", "O", "1O"]) {
      expect(isFretText(text), text).toBe(true);
    }
  });

  it("refuses letters, frets past 24 and three digits", () => {
    for (const text of ["", "e", "Harm.", "25", "123", "#", "x"]) {
      expect(isFretText(text), text).toBe(false);
    }
  });
});

describe("detectTablature without a recogniser", () => {
  it("calls a four-line group and a six-line group tablature on their line count", async () => {
    const tabs = await detect(
      [
        { kind: "tab", lines: 6, spacing: TAB, top: 300 },
        { kind: "tab", lines: 4, spacing: TAB, top: 800 },
      ],
      unreachable
    );
    expect(tabs.map((t) => t.system.lines)).toEqual([6, 4]);
    expect(tabs.map((t) => t.system.index)).toEqual([0, 1]);
    const [six] = tabs;
    expect(six?.system.cy).toBeCloseTo((300 + (5 * TAB) / 2) / PAGE_HEIGHT, 2);
    expect(six?.system.h).toBeCloseTo((5 * TAB) / PAGE_HEIGHT, 2);
    expect(six?.system.w).toBeCloseTo((PAGE_WIDTH - 180) / PAGE_WIDTH, 2);
  });

  it("never reads a standard staff: its noteheads are wider than a line space", async () => {
    const tabs = await detect(
      [
        { kind: "staff", spacing: STAFF, top: 300 },
        { kind: "staff", spacing: STAFF, top: 600 },
      ],
      unreachable
    );
    expect(tabs).toEqual([]);
  });

  it("asks about a five-line group with numbers on its lines, and reads one batch when it decides", async () => {
    const numbers = answering("7");
    const tab: Drawn = { kind: "tab", lines: 5, spacing: TAB, top: 300 };
    expect(await detect([tab], numbers.read)).toHaveLength(1);
    expect(numbers.seen.crops).toBe(6);
    const letters = answering("e");
    expect(await detect([tab], letters.read)).toEqual([]);
    expect(letters.seen.crops).toBe(6);
  });

  it("paints out the tab band and nothing else, on a copy", async () => {
    const cv = await testOpenCv();
    const page = drawPage(cv, [
      { kind: "staff", spacing: STAFF, top: 300 },
      { kind: "tab", lines: 6, spacing: TAB, top: 500 },
    ]);
    const before = Uint8Array.from(page.data);
    const tabs = await detectTablature(cv, page, unreachable);
    const white = whitenTablature(page, tabs, 0.75);
    expect(Buffer.from(page.data).equals(Buffer.from(before))).toBe(true);
    const inkIn = (rowFrom: number, rowTo: number) => {
      let ink = 0;
      for (
        let i = rowFrom * PAGE_WIDTH * 3;
        i < rowTo * PAGE_WIDTH * 3;
        i += 1
      ) {
        ink += (white.data[i] ?? 255) < 128 ? 1 : 0;
      }
      return ink;
    };
    expect(inkIn(480, 640)).toBe(0);
    expect(inkIn(250, 360)).toBeGreaterThan(0);
  });
});

describeWithModels("detectTablature with RapidOCR's recogniser", () => {
  const withRecogniser = async (
    body: (read: ReadCrops) => Promise<void>
  ): Promise<void> => {
    const store: ModelStore = await storeOn(CPU);
    try {
      const cv = await testOpenCv();
      const session = await store.open("ocrRecognize");
      const characters = ctcCharacters(session.metadata.get("character") ?? "");
      await body(async (crops) =>
        (await recognizeCrops(cv, session, characters, crops)).map(
          (r) => r.text
        )
      );
    } finally {
      await store.close();
    }
  };

  it("finds the five-line tab under a staff and leaves the staff alone", async () => {
    await withRecogniser(async (read) => {
      const tabs = await detect(
        [
          { kind: "staff", spacing: STAFF, top: 250 },
          { kind: "tab", lines: 5, spacing: TAB, top: 380 },
          { kind: "staff", spacing: STAFF, top: 750 },
          { kind: "tab", lines: 5, spacing: TAB, top: 880 },
        ],
        read
      );
      expect(tabs.map((t) => t.system.lines)).toEqual([5, 5]);
      expect(tabs[0]?.system.cy).toBeCloseTo((380 + 2 * TAB) / PAGE_HEIGHT, 2);
    });
  });

  it("finds four-, five- and six-line tabs on one page", async () => {
    await withRecogniser(async (read) => {
      const tabs = await detect(
        [
          { kind: "tab", lines: 4, spacing: TAB, top: 300 },
          { kind: "tab", lines: 5, spacing: TAB, top: 600 },
          { kind: "tab", lines: 6, spacing: TAB, top: 900 },
        ],
        read
      );
      expect(tabs.map((t) => t.system.lines)).toEqual([4, 5, 6]);
    });
  });
});

describeWithModels("recognizePage behind the tab guard", () => {
  it(
    "answers tablature_only, with its systems, for a page of tablature alone",
    async () => {
      const cv = await testOpenCv();
      const store = await storeOn(CPU);
      try {
        const progress: string[] = [];
        const result = await recognizePage(
          drawPage(cv, [
            { kind: "tab", lines: 5, spacing: TAB, top: 300 },
            { kind: "tab", lines: 6, spacing: TAB, top: 700 },
          ]),
          {
            backend: "wasm",
            cv,
            open: (role, batch) =>
              store.open(role, batch === undefined ? {} : { batch }),
          },
          {
            ocr: false,
            onProgress: ({ done, stage, total }) =>
              progress.push(`${stage} ${done}/${total}`),
          }
        );
        expect(result).toMatchObject({
          error: "tablature_only",
          musicXml: "",
          ok: false,
          staves: [],
          texts: [],
        });
        expect(result.tablature.map((t) => t.lines)).toEqual([5, 6]);
        // The first seven columns are single digits. Hershey's two digits stand
        // apart at 150 dpi and read as two frets; test/tab-read.test.ts covers
        // two-digit frets on typeset pages.
        expect(
          result.tablature.map((t) => t.events.slice(0, 7).map((e) => e.notes))
        ).toEqual(
          [5, 6].map((lines) =>
            drawnFrets(lines)
              .slice(0, 7)
              .map((n) => [n])
          )
        );
        expect(progress).toEqual(
          expect.arrayContaining(["tab 1/2", "tab 2/2"])
        );
        expect(result.log).toContain("Every system on the page is tablature");
      } finally {
        await store.close();
      }
    },
    PAGE_TIMEOUT_MS
  );
});

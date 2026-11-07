/**
 * The tab guard on pages drawn in the test (test/support/synthetic-page.ts):
 * which line groups are tablature, when the recogniser is consulted, what the
 * guard paints out, and the `tablature_only` result.
 */

import { describe, expect, it } from "vitest";
import { withMatScope } from "../src/cv/opencv.js";
import type { ModelStore } from "../src/models/store.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import {
  bandWithoutLines,
  componentBoxes,
  detectTablature,
  grayOf,
  inkOf,
  isFretText,
  marksOf,
  penLevelOf,
  READ_MARK,
  type ReadCrops,
  whitenTablature,
} from "../src/tab/detect.js";
import { type ClassifyDigits, NOT_A_DIGIT } from "../src/tab/digits.js";
import { readTablature } from "../src/tab/read.js";
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
    for (const text of [
      "0",
      "7",
      "12",
      "24",
      "S0",
      "7h",
      "p3",
      "O",
      "1O",
      "θ",
    ]) {
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

  it("finds a tab whose lines a turned scan spreads over several rows each", async () => {
    const tabs = await detect(
      [
        { kind: "tab", lines: 6, spacing: TAB, tilt: 6, top: 300 },
        { kind: "tab", lines: 4, spacing: TAB, tilt: -5, top: 800 },
      ],
      unreachable
    );
    expect(tabs.map((t) => t.system.lines)).toEqual([6, 4]);
  });

  it("takes no group of rules too thick for lines, level or not", async () => {
    const tabs = await detect(
      [
        { kind: "tab", lines: 4, spacing: TAB, thickness: 7, top: 300 },
        {
          kind: "tab",
          lines: 4,
          spacing: TAB,
          thickness: 7,
          tilt: 4,
          top: 800,
        },
      ],
      unreachable
    );
    expect(tabs).toEqual([]);
  });

  it("takes a hand's wide zeros for marks when reading a tab, not when deciding whether a group is one", async () => {
    const cv = await testOpenCv();
    const page = drawPage(cv, [
      { kind: "tab", lines: 6, spacing: TAB, top: 300, wideZeros: true },
    ]);
    const [tab] = await detectTablature(cv, page, unreachable);
    expect(tab).toBeDefined();
    if (tab === undefined) {
      return;
    }
    const ink = withMatScope(
      (scope) => new Uint8Array(inkOf(cv, scope, page).data)
    );
    const count = (size?: typeof READ_MARK) =>
      marksOf(cv, ink, page.width, page.height, tab.group, size).marks.length;
    expect(count()).toBeLessThan(4);
    expect(count(READ_MARK)).toBe(24);
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

describe("five-line groups drawn like the page's five-line tabs", () => {
  const fiveLineTabs: readonly Drawn[] = [
    { kind: "tab", lines: 5, spacing: TAB, top: 250 },
    { kind: "tab", lines: 5, spacing: TAB, top: 550 },
  ];

  it("takes a refused five-line group of the tabs' line spacing once two five-line tabs read as numbers", async () => {
    const tabs = await detect(
      [
        ...fiveLineTabs,
        { empty: true, kind: "tab", lines: 5, spacing: TAB, top: 850 },
      ],
      answering("7").read
    );
    expect(tabs.map((t) => t.system.lines)).toEqual([5, 5, 5]);
    expect(tabs[2]?.system.cy).toBeCloseTo((850 + 2 * TAB) / PAGE_HEIGHT, 2);
  });

  it("does not take it beside a single five-line tab", async () => {
    const tabs = await detect(
      [
        fiveLineTabs[0] as Drawn,
        { empty: true, kind: "tab", lines: 5, spacing: TAB, top: 850 },
      ],
      answering("7").read
    );
    expect(tabs).toHaveLength(1);
  });

  it("does not take a group of another line spacing", async () => {
    const tabs = await detect(
      [
        ...fiveLineTabs,
        { empty: true, kind: "tab", lines: 5, spacing: 15, top: 850 },
      ],
      answering("7").read
    );
    expect(tabs).toHaveLength(2);
  });

  it("does not take a staff of the tabs' line spacing: its noteheads stand in its spaces", async () => {
    const tabs = await detect(
      [...fiveLineTabs, { kind: "staff", spacing: TAB, top: 850 }],
      answering("7").read
    );
    expect(tabs).toHaveLength(2);
  });
});

describe("reading the marks of a hand", () => {
  const top = 400;
  const line = (k: number) => top + k * TAB;

  const readMarks = async (systems: readonly Drawn[], jogs = false) => {
    const cv = await testOpenCv();
    const page = drawPage(cv, systems);
    const [tab] = await detectTablature(cv, page, unreachable);
    if (tab === undefined) {
      throw new Error("the four-line tab was not found");
    }
    const ink = withMatScope(
      (scope) => new Uint8Array(inkOf(cv, scope, page).data)
    );
    return marksOf(
      cv,
      ink,
      page.width,
      page.height,
      tab.group,
      READ_MARK,
      jogs
    );
  };

  it("takes no piece of a pen's stem for a fret: a stem a little wider at its top has no head", async () => {
    // Stems from the second line down past the tab, five pixels wide for
    // 0.6 spacing then three, leaning: they used to pass for frets hanging
    // their stems, which cut every stem between the lines.
    const xs = [200, 330, 460, 590, 720, 850];
    const stems = xs.map((x) => ({
      points: [
        [x, line(1)],
        [x + 3, line(1) + Math.round(0.6 * TAB)],
      ] as const,
      thickness: 3,
    }));
    const tails = xs.map((x) => ({
      points: [
        [x + 3, line(1) + Math.round(0.6 * TAB)],
        [x + 7, line(3) + 3 * TAB],
      ] as const,
      thickness: 2,
    }));
    const { marks, short } = await readMarks([
      { empty: true, kind: "tab", lines: 4, spacing: TAB, top },
      { kind: "strokes", strokes: [...stems, ...tails] },
    ]);
    expect(marks).toEqual([]);
    expect(short).toEqual([]);
  });

  it("keeps a 5 whole whose bowl starts a pen's width along the line from its stroke", async () => {
    // The stroke comes down to the third line; the bowl leaves it five
    // pixels to the right, going down (strokes three pixels wide).
    const x = 600;
    const y = line(2);
    const cv = await testOpenCv();
    const page = drawPage(cv, [
      { empty: true, kind: "tab", lines: 4, spacing: TAB, top },
      {
        kind: "strokes",
        strokes: [
          {
            points: [
              [x + 10, y - 10],
              [x, y - 10],
              [x, y - 1],
            ],
            thickness: 2,
          },
          {
            points: [
              [x + 5, y + 1],
              [x + 10, y + 3],
              [x + 10, y + 7],
              [x + 2, y + 9],
            ],
            thickness: 2,
          },
        ],
      },
    ]);
    const [tab] = await detectTablature(cv, page, unreachable);
    if (tab === undefined) {
      throw new Error("the four-line tab was not found");
    }
    const ink = withMatScope(
      (scope) => new Uint8Array(inkOf(cv, scope, page).data)
    );
    const glyphs = (jogs: boolean) => {
      const band = bandWithoutLines(
        ink,
        page.width,
        page.height,
        tab.group,
        undefined,
        jogs
      );
      const cols = tab.group.x1 - tab.group.x0 + 1;
      return componentBoxes(cv, band.data, cols, band.rows).filter(
        (box) => Math.abs(box.x + tab.group.x0 - x) < TAB
      );
    };
    expect(glyphs(true)).toHaveLength(1);
    expect(glyphs(false)).toHaveLength(2);
  });
});

describe("a pen over printed rules", () => {
  const top = 400;
  const line = (k: number) => top + k * TAB;
  // A 4 whose crossbar lies along the third line: its diagonal comes down to
  // the line from above, its upright crosses it.
  const four = (x: number): Drawn => ({
    kind: "strokes",
    strokes: [
      {
        points: [
          [x + 2, line(2) - 12],
          [x, line(2) - 2],
        ],
        thickness: 2,
      },
      {
        points: [
          [x, line(2)],
          [x + 13, line(2)],
        ],
        thickness: 2,
      },
      {
        points: [
          [x + 10, line(2) - 12],
          [x + 10, line(2) + 9],
        ],
        thickness: 2,
      },
    ],
  });

  const marksNear = async (lineGray: number, x: number) => {
    const cv = await testOpenCv();
    const page = drawPage(cv, [
      { empty: true, kind: "tab", lineGray, lines: 4, spacing: TAB, top },
      four(x),
    ]);
    const [tab] = await detectTablature(cv, page, unreachable);
    if (tab === undefined) {
      throw new Error("the four-line tab was not found");
    }
    const ink = withMatScope(
      (scope) => new Uint8Array(inkOf(cv, scope, page).data)
    );
    const gray = grayOf(cv, page);
    const below = penLevelOf(gray, page.width, tab.group);
    const near = (pen: boolean) => {
      const { marks, short } = marksOf(
        cv,
        ink,
        page.width,
        page.height,
        tab.group,
        READ_MARK,
        true,
        pen && below !== undefined ? { below, gray } : undefined
      );
      return [...marks, ...short].filter((m) => Math.abs(m.x - x) < TAB);
    };
    return { below, withoutPen: near(false), withPen: near(true) };
  };

  it("keeps a 4 whole whose crossbar lies along a line lighter than the pen", async () => {
    const { below, withPen, withoutPen } = await marksNear(140, 600);
    expect(below).toBeDefined();
    expect(withPen).toHaveLength(1);
    expect(withPen[0]?.w).toBeGreaterThanOrEqual(13);
    expect(withoutPen.length).toBeGreaterThan(1);
  });

  it("finds no pen on black rules: a printed page erases its lines as before", async () => {
    const { below } = await marksNear(0, 600);
    expect(below).toBeUndefined();
  });

  it("takes no stem the band's edge cuts for a 1 on the bottom line", async () => {
    const cv = await testOpenCv();
    const x = 700;
    const page = drawPage(cv, [
      { empty: true, kind: "tab", lines: 4, spacing: TAB, top },
      {
        kind: "strokes",
        strokes: [
          {
            points: [
              [x, line(3) - Math.round(0.3 * TAB)],
              [x, line(3) + 3 * TAB],
            ],
            thickness: 3,
          },
        ],
      },
    ]);
    const [tab] = await detectTablature(cv, page, unreachable);
    if (tab === undefined) {
      throw new Error("the four-line tab was not found");
    }
    const ink = withMatScope(
      (scope) => new Uint8Array(inkOf(cv, scope, page).data)
    );
    const { marks, short } = marksOf(
      cv,
      ink,
      page.width,
      page.height,
      tab.group,
      READ_MARK,
      true
    );
    expect([...marks, ...short].filter((m) => Math.abs(m.x - x) < TAB)).toEqual(
      []
    );
  });

  it("takes a hand's digit up to 1.45 spacings tall and 1.55 wide", async () => {
    const cv = await testOpenCv();
    const x = 800;
    // 1.43 tall by 1.48 wide with the stroke: over the 1.3 by 1.35 bound.
    const h = Math.round(1.3 * TAB);
    const w = Math.round(1.35 * TAB);
    const page = drawPage(cv, [
      { empty: true, kind: "tab", lines: 4, spacing: TAB, top },
      {
        kind: "strokes",
        strokes: [
          {
            points: [
              [x, line(1) - h / 2],
              [x + w, line(1) - h / 2],
              [x + w, line(1) + h / 2],
              [x, line(1) + h / 2],
              [x, line(1) - h / 2],
            ],
            thickness: 2,
          },
        ],
      },
    ]);
    const [tab] = await detectTablature(cv, page, unreachable);
    if (tab === undefined) {
      throw new Error("the four-line tab was not found");
    }
    const ink = withMatScope(
      (scope) => new Uint8Array(inkOf(cv, scope, page).data)
    );
    const { marks } = marksOf(
      cv,
      ink,
      page.width,
      page.height,
      tab.group,
      READ_MARK,
      true
    );
    expect(marks.filter((m) => Math.abs(m.x - x) < TAB)).toHaveLength(1);
  });
});

describe("notes whose fret was not read", () => {
  const top = 400;
  const line = (k: number) => top + k * TAB;
  // A mark on the top line that reads as no fret, over a pen's stem down to a
  // beam; a second beamed stem with nothing over it.
  const drawn: readonly Drawn[] = [
    { empty: true, kind: "tab", lines: 4, spacing: TAB, top },
    {
      kind: "strokes",
      strokes: [
        {
          points: [
            [592, line(0) - 7],
            [608, line(0) - 7],
            [608, line(0) + 7],
            [592, line(0) + 7],
            [592, line(0) - 7],
          ],
          thickness: 2,
        },
        {
          points: [
            [601, line(1) + 4],
            [601, line(3) + 2 * TAB],
          ],
          thickness: 2,
        },
        {
          points: [
            [661, line(1) + 4],
            [661, line(3) + 2 * TAB],
          ],
          thickness: 2,
        },
        {
          points: [
            [601, line(3) + 2 * TAB - 2],
            [661, line(3) + 2 * TAB - 2],
          ],
          thickness: 4,
        },
      ],
    },
  ];

  const readAs = async (text: string) => {
    const cv = await testOpenCv();
    const page = drawPage(cv, drawn);
    const tabs = await detectTablature(cv, page, unreachable);
    return readTablature(cv, page, tabs, answering(text).read);
  };

  it("reports a beamed stem under no fret, with the line and text of the mark over it", async () => {
    const [reading] = await readAs("u");
    expect(reading?.events).toEqual([]);
    expect(
      reading?.unreadFrets?.map((u) => ({
        beams: u.stem.kind === "stem" ? u.stem.beams : undefined,
        reason: u.reason,
        string: u.string,
        text: u.text,
        x: Math.round(u.x * PAGE_WIDTH),
      }))
    ).toEqual([
      { beams: 1, reason: "stem_without_fret", string: 1, text: "u", x: 601 },
      {
        beams: 1,
        reason: "stem_without_fret",
        string: undefined,
        text: undefined,
        x: 661,
      },
    ]);
  });

  it("takes a stem under a dead note's x for that note's, not an unread fret", async () => {
    const [reading] = await readAs("x");
    expect(
      reading?.unreadFrets?.map((u) => Math.round(u.x * PAGE_WIDTH))
    ).toEqual([661]);
  });

  const classifying =
    (digit: number, probability: number): ClassifyDigits =>
    (crops) =>
      Promise.resolve(crops.map(() => ({ digit, probability })));

  const readWith = async (
    systems: readonly Drawn[],
    text: string,
    classify: ClassifyDigits
  ) => {
    const cv = await testOpenCv();
    const page = drawPage(cv, systems);
    const tabs = await detectTablature(cv, page, unreachable);
    const [reading] = await readTablature(
      cv,
      page,
      tabs,
      answering(text).read,
      undefined,
      [],
      classify
    );
    return {
      events: reading?.events.map((e) => ({
        notes: e.notes,
        stem: e.stem.kind,
        x: Math.round(e.x * PAGE_WIDTH),
      })),
      unread: reading?.unreadFrets?.map((u) => Math.round(u.x * PAGE_WIDTH)),
    };
  };

  it("reads the mark over a stem as the classifier's digit when it is fairly sure, the stem proving a note", async () => {
    // RapidOCR reads a hand's 2 as "Z"; the classifier says 2, under
    // DIGIT_SURE but over STEM_DIGIT_SURE.
    expect(await readWith(drawn, "Z", classifying(2, 0.9))).toEqual({
      events: [{ notes: [{ fret: 2, string: 1 }], stem: "stem", x: 601 }],
      unread: [661],
    });
    expect(await readWith(drawn, "Z", classifying(2, 0.7))).toMatchObject({
      events: [],
      unread: [601, 661],
    });
  });

  it("reads a tick over a stem as a 1", async () => {
    // The mark over the first stem is a short slanted stroke.
    const ticked = drawn.map(
      (d): Drawn =>
        d.kind === "strokes"
          ? {
              ...d,
              strokes: [
                {
                  points: [
                    [603, line(0) - 8],
                    [599, line(0) + 8],
                  ],
                  thickness: 3,
                },
                ...d.strokes.slice(1),
              ],
            }
          : d
    );
    const { events } = await readWith(
      ticked,
      "f",
      classifying(NOT_A_DIGIT, 0.9)
    );
    expect(events?.map((e) => e.notes)).toEqual([[{ fret: 1, string: 1 }]]);
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
          expect.arrayContaining(["tab 1/3", "tab 2/3", "tab 3/3"])
        );
        // the page prints no tuning or capo, so none is read and none is assumed
        expect(
          result.tablature.map(({ capo, tuning }) => ({ capo, tuning }))
        ).toEqual([{}, {}]);
        expect(result.log).toContain(
          "Tab text: no tuning, no capo; no tuning, no capo"
        );
        expect(result.log).toContain("Every system on the page is tablature");
      } finally {
        await store.close();
      }
    },
    PAGE_TIMEOUT_MS
  );

  it(
    "paints out only the systems it read notes on, and says so",
    async () => {
      const cv = await testOpenCv();
      const store = await storeOn(CPU);
      try {
        const result = await recognizePage(
          drawPage(cv, [
            { kind: "tab", lines: 5, spacing: TAB, top: 300 },
            { empty: true, kind: "tab", lines: 4, spacing: TAB, top: 700 },
          ]),
          {
            backend: "wasm",
            cv,
            open: (role, batch) =>
              store.open(role, batch === undefined ? {} : { batch }),
          },
          { ocr: false }
        );
        // The empty four-line group passes for a tab on its line count; with
        // no event read on it, it is left on the page for the staff reader.
        expect(result.tablature.map((t) => t.events.length > 0)).toEqual([
          true,
          false,
        ]);
        expect(result.log).toContain(
          "Tab guard: 2 tablature systems (5, 4 lines), 1 painted out, 1 with no event left to the staff reader"
        );
      } finally {
        await store.close();
      }
    },
    PAGE_TIMEOUT_MS
  );
});

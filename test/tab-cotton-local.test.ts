/**
 * Cotton-Eyed Joe, Peghead Nation's clawhammer banjo page, end to end: a
 * tab-only page read to a playable tune, every fret, every duration and every
 * slur against the page, once from the image alone and once with the PDF's
 * text layer. The PDF may not be redistributed: it lives in the git-ignored
 * test/fixtures/local/tablature/banjo/, and this suite is skipped wherever it
 * is absent, CI included.
 *
 * The truth was read off the page by eye (docs/design/tab-4-findings.md):
 * eight bars of 4/4, parts A and B repeated, gDGBD printed as "Tuning: G
 * DGBD", no capo, four pull-offs and a hammer-on.
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { beforeAll, describe, expect, it } from "vitest";
import { colorImageFromRgba } from "../src/image/plane.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import type { PageTextItem } from "../src/tab/layer.js";
import { resolveTab, type TabTune, ticksOf } from "../src/tab/tune.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";
import { textLayerOf } from "./support/pdf-text.js";

const ROOT = join(import.meta.dirname, "fixtures", "local", "tablature");
const PDF = join(ROOT, "banjo", "ClawCottonEyedJoe.pdf");
const PNG_FILE = join(ROOT, ".pages", "banjo_ClawCottonEyedJoe_pdf-p1.png");
const PAGE_TIMEOUT_MS = 300_000;

const CHORD = "1/0+2/0+3/0:e";
const A3 = `2/0:q ${CHORD} 5/0:e 3/2:e 3/0:e ${CHORD} 5/0:e`;
const A4 = `3/0:q 4/2:q 4/0:q ${CHORD} 5/0:e`;
/** Each bar's events as the page prints them: frets, then q or e. */
const BARS = [
  `2/0:q 1/0:q 1/0:q ${CHORD} 5/0:e`,
  `1/2:e 1/0:e ${CHORD} 5/0:e 2/0:q ${CHORD} 5/0:e`,
  A3,
  A4,
  `4/0:e 4/2:e ${CHORD} 5/0:e 3/0:q ${CHORD} 5/0:e`,
  `3/2:e 3/0:e ${CHORD} 5/0:e 2/0:q ${CHORD} 5/0:e`,
  A3,
  A4,
];

/** The slurs, as bar:event:string:technique. */
const SLURS = [
  "1:0:1:pull-off",
  "2:3:3:pull-off",
  "4:0:4:hammer-on",
  "5:0:3:pull-off",
  "6:3:3:pull-off",
];

const VALUE = { eighth: "e", quarter: "q" } as const;

const barsOf = (tune: TabTune) =>
  tune.bars.map((bar) =>
    bar.events
      .map(
        (e) =>
          `${e.notes.map((n) => `${n.string}/${n.fret}`).join("+")}:${VALUE[e.duration.value as keyof typeof VALUE] ?? e.duration.value}`
      )
      .join(" ")
  );

const slursOf = (tune: TabTune) =>
  tune.bars.flatMap((bar, b) =>
    bar.events.flatMap((e, k) =>
      e.notes.flatMap((n) =>
        n.techniques.map((t) => `${b}:${k}:${n.string}:${t}`)
      )
    )
  );

function rasterised() {
  mkdirSync(join(ROOT, ".pages"), { recursive: true });
  if (existsSync(PNG_FILE)) {
    return;
  }
  const tool = join(ROOT, ".pages", "rasterise-pdf");
  const source = join(
    import.meta.dirname,
    "..",
    "tools",
    "rasterise-pdf.swift"
  );
  if (!existsSync(tool) || statSync(tool).mtimeMs < statSync(source).mtimeMs) {
    execFileSync("swiftc", ["-O", source, "-o", tool]);
  }
  execFileSync(tool, [PDF, PNG_FILE, "300", "1"]);
}

async function readPage(textLayer: readonly PageTextItem[]) {
  const cv = await testOpenCv();
  const store = await storeOn(CPU);
  try {
    const png = PNG.sync.read(readFileSync(PNG_FILE));
    return await recognizePage(
      colorImageFromRgba(png.width, png.height, png.data),
      {
        backend: "wasm",
        cv,
        open: (role, batch) =>
          store.open(role, batch === undefined ? {} : { batch }),
      },
      { ocr: false, textLayer }
    );
  } finally {
    await store.close();
  }
}

const suite = existsSync(PDF) ? describeWithModels : describe.skip;

suite("Cotton-Eyed Joe, tab only, to a tune", () => {
  beforeAll(rasterised, 120_000);

  for (const source of ["the image", "the text layer"] as const) {
    it(
      `from ${source}: every fret, duration and slur of the eight bars, pitched in gDGBD`,
      async () => {
        const layer = source === "the image" ? [] : await textLayerOf(PDF, 1);
        const result = await readPage(layer);
        expect(result.error).toBe("tablature_only");
        expect(result.tablature.map((t) => t.readFrom)).toEqual(
          result.tablature.map(() =>
            source === "the image" ? "image" : "text_layer"
          )
        );
        const resolution = resolveTab(result.tablature);
        const { prerequisites } = resolution;
        expect(prerequisites.instrument).toMatchObject({
          status: "read",
          value: { kind: "banjo5" },
        });
        expect(prerequisites.tuning).toMatchObject({
          status: "read",
          value: ["D4", "B3", "G3", "D3", "G4"],
        });
        expect(prerequisites.capo).toMatchObject({
          status: "assumed",
          value: 0,
        });
        expect(prerequisites.meter).toMatchObject({
          status: "read",
          value: { den: 4, num: 4 },
        });
        if (!resolution.ok) {
          throw new Error(
            `needs ${resolution.needs.map((n) => `${n.field}: ${n.evidence}`).join("; ")}`
          );
        }
        const { tune } = resolution;
        expect(barsOf(tune)).toEqual(BARS);
        expect(
          tune.bars.map((bar) =>
            bar.events.reduce((sum, e) => sum + ticksOf(e.duration), 0)
          )
        ).toEqual(BARS.map(() => 3840));
        expect(slursOf(tune)).toEqual(SLURS);
        expect(
          tune.bars.map((bar) => [bar.repeatStart, bar.repeatEnd])
        ).toEqual([
          [true, false],
          [false, false],
          [false, false],
          [false, true],
          [true, false],
          [false, false],
          [false, false],
          [false, true],
        ]);
        expect(
          tune.bars[0]?.events.map((e) => e.notes.map((n) => n.midi))
        ).toEqual([[59], [62], [62], [62, 59, 55], [67]]);
      },
      PAGE_TIMEOUT_MS
    );
  }
});

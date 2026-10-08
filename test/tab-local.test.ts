/**
 * The tab guard on the owner's tablature samples in
 * test/fixtures/local/tablature/, which may not be redistributed and are
 * git-ignored: skipped wherever they are absent, CI included. Each page is
 * rasterised at 300 dpi by tools/rasterise-pdf.swift (macOS CoreGraphics)
 * into the ignored `.pages/` beside them, once.
 *
 * The truth was counted by eye on every rasterised page: the banjo and
 * mandolin PDFs by the prototype (.scratch/tab-sketch/pages.ts, 2026-10-08),
 * the three guitar PDFs for this guard (seven pages; the Borus March file has
 * no extension and is read as a PDF all the same). 56 tab systems over 15 pages.
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { beforeAll, describe, expect, it } from "vitest";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import { detectTablature } from "../src/tab/detect.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const ROOT = join(import.meta.dirname, "fixtures", "local", "tablature");
const PAGES = join(ROOT, ".pages");
const PAGE_TIMEOUT_MS = 300_000;

interface Sample {
  readonly file: string;
  readonly lines: 4 | 5 | 6;
  /** 1-based; page 1 when absent. */
  readonly page?: number;
  /** Standard staves on the page besides the tabs; 0 is a tab-only page. */
  readonly staves: number;
  readonly tabs: number;
}

const SAMPLES: readonly Sample[] = [
  { file: "banjo/Banjo-Cold Frosty Morning.pdf", lines: 5, staves: 0, tabs: 6 },
  { file: "banjo/Chicken's reel.pdf", lines: 5, staves: 0, tabs: 3 },
  { file: "banjo/Cluck old hen clawhammer.pdf", lines: 5, staves: 0, tabs: 2 },
  {
    file: "banjo/Cripple Creek Beginner Banjo Tab.tef.pdf",
    lines: 5,
    staves: 0,
    tabs: 4,
  },
  {
    file: "banjo/Cripple Creek Beginner Banjo Tab_with_Score.tef.pdf",
    lines: 5,
    staves: 4,
    tabs: 4,
  },
  {
    file: "banjo/Sourwood Mountain clawhammer.pdf",
    lines: 5,
    staves: 0,
    tabs: 3,
  },
  {
    file: "mandolin/Arkansas_Traveler_mandolin.pdf",
    lines: 4,
    staves: 4,
    tabs: 4,
  },
  { file: "mandolin/Killdare_fancy.pdf", lines: 4, staves: 4, tabs: 4 },
  { file: "guitar/AMAZING GRACE.pdf", lines: 6, staves: 4, tabs: 2 },
  { file: "guitar/AMAZING GRACE.pdf", lines: 6, page: 2, staves: 6, tabs: 3 },
  { file: "guitar/AMAZING GRACE.pdf", lines: 6, page: 3, staves: 6, tabs: 3 },
  { file: "guitar/MOON OVER SHANGAI.pdf", lines: 6, staves: 4, tabs: 4 },
  {
    file: "guitar/MOON OVER SHANGAI.pdf",
    lines: 6,
    page: 2,
    staves: 5,
    tabs: 5,
  },
  {
    file: "guitar/MOON OVER SHANGAI.pdf",
    lines: 6,
    page: 3,
    staves: 5,
    tabs: 5,
  },
  { file: "guitar/DADGADTab-Brian-Borus-March", lines: 6, staves: 4, tabs: 4 },
];

const nameOf = (sample: Sample) => `${sample.file} p${sample.page ?? 1}`;

const PRESENT = SAMPLES.every((s) => existsSync(join(ROOT, s.file)));
const pngOf = (sample: Sample) =>
  join(
    PAGES,
    `${sample.file.replace(/[^A-Za-z0-9]/g, "_")}-p${sample.page ?? 1}.png`
  );

function rasteriseMissing(): void {
  mkdirSync(PAGES, { recursive: true });
  const tool = join(PAGES, "rasterise-pdf");
  const source = join(
    import.meta.dirname,
    "..",
    "tools",
    "rasterise-pdf.swift"
  );
  if (!existsSync(tool) || statSync(tool).mtimeMs < statSync(source).mtimeMs) {
    execFileSync("swiftc", ["-O", source, "-o", tool]);
  }
  for (const sample of SAMPLES) {
    if (!existsSync(pngOf(sample))) {
      execFileSync(tool, [
        join(ROOT, sample.file),
        pngOf(sample),
        "300",
        String(sample.page ?? 1),
      ]);
    }
  }
}

function pageOf(sample: Sample): ColorImage {
  const png = PNG.sync.read(readFileSync(pngOf(sample)));
  return colorImageFromRgba(png.width, png.height, png.data);
}

const localSuite = (title: string, suite: () => void) =>
  PRESENT
    ? describeWithModels(title, () => {
        beforeAll(rasteriseMissing, 120_000);
        suite();
      })
    : describe.skip(`${title} (no test/fixtures/local/tablature)`, suite);

localSuite("the tab guard on the local tablature samples", () => {
  it(
    "finds every tab system of the fifteen pages, 56 in all, with its line count",
    async () => {
      const cv = await testOpenCv();
      const store = await storeOn(CPU);
      try {
        const session = await store.open("ocrRecognize");
        const characters = ctcCharacters(
          session.metadata.get("character") ?? ""
        );
        const found: Record<string, number[]> = {};
        for (const sample of SAMPLES) {
          // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, one page at a time
          const tabs = await detectTablature(
            cv,
            pageOf(sample),
            async (crops) =>
              (await recognizeCrops(cv, session, characters, crops)).map(
                (r) => r.text
              )
          );
          found[nameOf(sample)] = tabs.map((t) => t.system.lines);
        }
        expect(found).toEqual(
          Object.fromEntries(
            SAMPLES.map((s) => [
              nameOf(s),
              Array.from({ length: s.tabs }, () => s.lines),
            ])
          )
        );
      } finally {
        await store.close();
      }
    },
    PAGE_TIMEOUT_MS
  );

  /** Every tab-only page, and page 1 of each file with staves over its tabs. */
  const read = SAMPLES.filter(
    (s) => s.staves === 0 || (s.page ?? 1) === 1
  ).filter((s) => s.staves === 0 || !s.file.startsWith("mandolin/"));
  for (const sample of read) {
    it(
      `${nameOf(sample)}: ${sample.staves === 0 ? "tablature_only" : "no staff read on a tab"}`,
      async () => {
        const cv = await testOpenCv();
        const store = await storeOn(CPU);
        try {
          const result = await recognizePage(
            pageOf(sample),
            {
              backend: "wasm",
              cv,
              open: (role, batch) =>
                store.open(role, batch === undefined ? {} : { batch }),
            },
            { ocr: false }
          );
          process.stdout.write(
            `${nameOf(sample)}: ${result.durationMs} ms\n${result.log}\n`
          );
          expect(result.tablature).toHaveLength(sample.tabs);
          if (sample.staves === 0) {
            expect(result.error).toBe("tablature_only");
            return;
          }
          expect(result.ok).toBe(true);
          expect(result.staves.length).toBeGreaterThan(0);
          // A staff box reaching a tab's centre line is homr reading the tab.
          const onTab = result.staves.filter((staff) =>
            result.tablature.some(
              (tab) => Math.abs(staff.cy - tab.cy) < staff.h / 2
            )
          );
          expect(onTab).toEqual([]);
        } finally {
          await store.close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

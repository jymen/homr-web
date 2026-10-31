/**
 * Pages of staves the tab guard must leave alone, from the owner's samples in
 * test/fixtures/local/staves/ (git-ignored, skipped where absent):
 *
 * - Reel de Mattawa, one engraved page of four staff systems in sixteenths,
 *   rasterised by tools/rasterise-pdf.swift at 200, 300 and 400 dpi. Under
 *   0.5.2 the guard took systems 1 and 2 for four-line tabs at 300 and 400
 *   dpi (a beam row two pixels from a top line), read no event on them and
 *   painted them white, so homr read bars 10 to 18 only (2026-10-31).
 * - Marche des élèves et p'tit Sarny, a web image 820 by 1061 enlarged to
 *   2711 by 3508 as the app enlarges it: nine staves, two of them pale enough
 *   for the guard to paint them out before the faint-ink checks (2026-10-28).
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import { detectTablature } from "../src/tab/detect.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const ROOT = join(import.meta.dirname, "fixtures", "local", "staves");
const PAGE_TIMEOUT_MS = 300_000;

const PAGES = [
  { file: "reel-de-mattawa-200.png", read: false, staves: 4 },
  { file: "reel-de-mattawa-300.png", read: true, staves: 4 },
  { file: "reel-de-mattawa-400.png", read: false, staves: 4 },
  { file: "marche-des-eleves-2711x3508.png", read: true, staves: 9 },
] as const;

const PRESENT = PAGES.every((p) => existsSync(join(ROOT, p.file)));

function pageOf(file: string): ColorImage {
  const png = PNG.sync.read(readFileSync(join(ROOT, file)));
  return colorImageFromRgba(png.width, png.height, png.data);
}

const suite = PRESENT
  ? describeWithModels
  : (title: string, body: () => void) =>
      describe.skip(`${title} (no test/fixtures/local/staves)`, body);

suite("the tab guard on pages of staves", () => {
  for (const sample of PAGES) {
    it(
      `${sample.file}: no tab system`,
      async () => {
        const cv = await testOpenCv();
        const store = await storeOn(CPU);
        try {
          const session = await store.open("ocrRecognize");
          const characters = ctcCharacters(
            session.metadata.get("character") ?? ""
          );
          const tabs = await detectTablature(
            cv,
            pageOf(sample.file),
            async (crops) =>
              (await recognizeCrops(cv, session, characters, crops)).map(
                (r) => r.text
              )
          );
          expect(tabs.map((t) => t.system.lines)).toEqual([]);
        } finally {
          await store.close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
  for (const sample of PAGES.filter((p) => p.read)) {
    it(
      `${sample.file}: homr reads all ${sample.staves} staves`,
      async () => {
        const cv = await testOpenCv();
        const store = await storeOn(CPU);
        try {
          const result = await recognizePage(
            pageOf(sample.file),
            {
              backend: "wasm",
              cv,
              open: (role, batch) =>
                store.open(role, batch === undefined ? {} : { batch }),
            },
            { ocr: false }
          );
          process.stdout.write(`${sample.file}:\n${result.log}\n`);
          expect(result.ok).toBe(true);
          expect(result.tablature).toEqual([]);
          expect(result.staves).toHaveLength(sample.staves);
        } finally {
          await store.close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

/**
 * A page of plain staves whose stems the tab guard read as frets, from the
 * owner's own scan, named by SAILORS_BONNET_PDF and never copied here: skipped
 * wherever the variable is unset, CI included.
 *
 * "The Sailor's Bonnet", a printed strip of three staves (12 bars, D major)
 * pasted on an A4 sheet and scanned at 300 dpi. At the app's raster, 2479 by
 * 3271, the guard sampled six marks on the third staff, stems cut between its
 * lines, and the recogniser read four of them "1": the staff passed as a
 * five-line tab, was painted white, and homr read bars 1 to 8 only
 * (2026-11-19). At 2480 by 3272 other marks were sampled and the page read
 * whole, so both rasters are asserted.
 *
 * Its title starts left of the indented first staff, outside homr's title
 * crop, and was read "he Sailor's Bonnet" until the cut line was read again
 * from a crop run on to the page edge (2026-11-19).
 */

import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { PNG } from "pngjs";
import { beforeAll, describe, expect, it } from "vitest";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const PDF = process.env.SAILORS_BONNET_PDF ?? "";
/** Outside the repository: a render of the page is the owner's too. */
const PAGES = join(tmpdir(), "homr-web-tab-stems-local");
const PAGE_TIMEOUT_MS = 300_000;

/** The page is 594.96 by 785.04 points; 300.13 dpi is the first scale that rounds to 2480 wide. */
const RASTERS = [
  { dpi: "300", height: 3271, width: 2479 },
  { dpi: "300.13", height: 3272, width: 2480 },
] as const;

const pngOf = (dpi: string) => join(PAGES, `page-${dpi}.png`);

function rasterise(): void {
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
  for (const { dpi } of RASTERS) {
    execFileSync(tool, [PDF, pngOf(dpi), dpi]);
  }
}

function pageOf(dpi: string): ColorImage {
  const png = PNG.sync.read(readFileSync(pngOf(dpi)));
  return colorImageFromRgba(png.width, png.height, png.data);
}

const suite =
  PDF !== "" && existsSync(PDF)
    ? describeWithModels
    : (title: string, body: () => void) =>
        describe.skip(`${title} (SAILORS_BONNET_PDF unset)`, body);

suite("stems the tab guard must not read as frets", () => {
  beforeAll(rasterise, 120_000);
  for (const { dpi, height, width } of RASTERS) {
    it(
      `${width} by ${height}: three staves, twelve bars, no tab`,
      async () => {
        const page = pageOf(dpi);
        expect([page.width, page.height]).toEqual([width, height]);
        const cv = await testOpenCv();
        const store = await storeOn(CPU);
        try {
          const result = await recognizePage(page, {
            backend: "wasm",
            cv,
            open: (role, batch) =>
              store.open(role, batch === undefined ? {} : { batch }),
          });
          expect(result.ok).toBe(true);
          expect(result.tablature).toEqual([]);
          expect(result.staves).toHaveLength(3);
          expect(result.musicXml.match(/<measure\b/g)).toHaveLength(12);
          expect(result.musicXml).toContain(
            "<work-title>The Sailor's Bonnet</work-title>"
          );
        } finally {
          await store.close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

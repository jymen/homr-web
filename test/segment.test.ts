/**
 * Phase 3's own criterion: the five masks this port computes against the five
 * `tools/dump-golden.py` dumped out of homr, per class, at testing.md's bar.
 *
 * Segnet is the slowest thing in the library on Node. Measured here over six
 * runs, 54 tiles of the Kesh page at a batch of 8 on the wasm provider with one
 * thread: **55 to 105 seconds**, 1.0 to 1.9 s a tile depending on what else the
 * machine is doing, plus half a second to open the session. The 159 ms a tile
 * recorded in `src/models/manifest.ts`'s comment does not reproduce and should be
 * read as unverified.
 *
 * So the whole page goes through segnet once by default, in the golden test,
 * from the Python preprocessed page; the port's own preprocess feeds it in
 * test/recognize-golden.test.ts.
 */

import { describe, expect, it } from "vitest";
import { planeAgreement } from "../src/image/plane.js";
import {
  MASK_CLASS_NAMES,
  type SegmentationMasks,
} from "../src/model/pipeline.js";
import { segmentPage } from "../src/segmentation/segment.js";
import { tileGrid } from "../src/segmentation/tiles.js";
import {
  type GoldenFixture,
  goldenPageOf,
  listGoldenFixtures,
} from "./support/golden.js";
import {
  CPU,
  describeWithModels,
  FP16_ON_WASM,
  required,
  storeOn,
} from "./support/models.js";

/** homr's own batch_size, and what the segnet sessions below are opened with. */
const BATCH = 8;
/** testing.md's mask criterion, per class. */
const MASK_AGREEMENT = 0.999;
const PAGE_TIMEOUT_MS = 600_000;
/** Straight to stdout rather than through vitest's per-test console buffer, so the figures appear in the order they were measured. */
const report = (line: string): void => {
  process.stdout.write(`${line}\n`);
};

const firstFixture = (): GoldenFixture =>
  required(listGoldenFixtures()[0], "a golden fixture to segment");

/**
 * Every per-class agreement printed before any of them is asserted, so the five
 * numbers are on the record whether they pass or fail. A boolean tells the next
 * reader that the masks were once good enough and nothing about how close they
 * were.
 */
function reportAgreements(
  label: string,
  fixture: GoldenFixture,
  masks: SegmentationMasks
): number[] {
  const golden = goldenPageOf(fixture);
  const agreements: number[] = [];
  for (const name of MASK_CLASS_NAMES) {
    const mine = masks[name];
    const agreement = planeAgreement(mine, golden.mask(name));
    agreements.push(agreement);
    const wrong = Math.round((1 - agreement) * mine.data.length);
    report(
      `  ${label} ${name}: ${agreement.toFixed(9)} (${wrong} px of ${mine.data.length})`
    );
  }
  return agreements;
}

describeWithModels("segnet over the whole golden page", () => {
  it(
    "reproduces every Python mask from the Python preprocessed page",
    async () => {
      const fixture = firstFixture();
      const golden = goldenPageOf(fixture);
      // The *Python* stage output, never this port's own preprocess: phase 4 reads
      // the golden the same way, so a preprocess regression cannot hide here.
      const page = golden.preprocessed();
      const tiles = tileGrid(page.width, page.height).length;
      const store = await storeOn(CPU);
      try {
        const segnet = await store.open("segnet", { batch: BATCH });
        const began = performance.now();
        const result = await segmentPage(segnet, page, { batch: BATCH });
        const ms = performance.now() - began;
        report(
          `fp32 segnet on ${fixture.name}: ${page.width}x${page.height}, ${tiles} tiles, batch ${BATCH}, ${Math.round(ms)} ms (${Math.round(ms / tiles)} ms/tile)`
        );
        expect(result.width).toBe(page.width);
        expect(result.height).toBe(page.height);

        const agreements = reportAgreements("fp32", fixture, result.masks);
        for (const agreement of agreements) {
          expect(agreement).toBeGreaterThanOrEqual(MASK_AGREEMENT);
        }
      } finally {
        await store.close();
      }
    },
    PAGE_TIMEOUT_MS
  );
});

/**
 * The phase's real risk, and measurable here with no GPU: `Placement` splits the
 * artifacts from the provider so the fp16 files run on the wasm EP. It is two
 * more page runs, 135 to 180 s measured, so it is opt-in, and the skip says how
 * to run it because a silent skip proves nothing.
 */
function describeFp16Page(title: string, suite: () => void): void {
  if (process.env.HOMR_FP16_PAGE === "1") {
    describeWithModels(title, suite);
    return;
  }
  describe.skip(
    `${title} (set HOMR_FP16_PAGE=1 to run it: two more page runs, about 135 to 180 s)`,
    suite
  );
}

describeFp16Page("the fp16 segnet against the fp32 one, page-wide", () => {
  it(
    "agrees with the fp32 artifact and with the Python masks on every class",
    async () => {
      const fixture = firstFixture();
      const page = goldenPageOf(fixture).preprocessed();
      const fp32Store = await storeOn(CPU);
      const fp16Store = await storeOn(FP16_ON_WASM);
      try {
        const fp32 = await fp32Store.open("segnet", { batch: BATCH });
        const fp16 = await fp16Store.open("segnet", { batch: BATCH });
        expect(fp16.inputSpec("input").type).toBe("float16");

        const plain = await segmentPage(fp32, page, { batch: BATCH });
        const half = await segmentPage(fp16, page, { batch: BATCH });

        const classMaps = planeAgreement(plain.classes, half.classes);
        report(
          `fp16 vs fp32 class maps, whole page: ${classMaps.toFixed(9)} (${Math.round((1 - classMaps) * page.data.length)} px of ${page.data.length})`
        );
        // session.test.ts measured 0.999961 on the inkiest single tile of this
        // page. Page-wide, measured 2026-09-28: 0.999990, 51 px of 5.2 M, so that
        // tile was the worst case rather than a typical one.
        const agreements = [
          ...reportAgreements("fp32", fixture, plain.masks),
          ...reportAgreements("fp16", fixture, half.masks),
          classMaps,
        ];
        for (const agreement of agreements) {
          expect(agreement).toBeGreaterThanOrEqual(MASK_AGREEMENT);
        }
      } finally {
        await Promise.all([fp32Store.close(), fp16Store.close()]);
      }
    },
    PAGE_TIMEOUT_MS
  );
});

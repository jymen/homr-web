/**
 * predict_symbols against Python, list by list, from the Python filtered masks.
 *
 * Every check but the rect is exact, and the one that catches most is the
 * cheapest: a merged group's concatenated contour is byte-identical or it is
 * not, so a wrong grouping shows up there rather than as a drifted rect three
 * steps later. The one-pixel polygon allowance is conditioned on the same
 * entry's rect being inexact, never on a magnitude, and how often it is used is
 * printed rather than asserted -- an opencv.js patch release could move which
 * corners sit on an integer boundary without any of them becoming wrong.
 */

import { describe, expect, it } from "vitest";
import type { AngledBox } from "../src/geometry/boxes.js";
import {
  assertBoxListMatches,
  compareBoxLists,
  describeBoxComparison,
} from "../src/golden/box-tolerance.js";
import type { GoldenPage } from "../src/golden/page.js";
import type {
  PredictedSymbols,
  SegmentationMasks,
} from "../src/model/pipeline.js";
import { predictSymbols } from "../src/pipeline/predict-symbols.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";

/** The golden file each list is compared against, and the name Python logs it under. */
const LISTS = [
  ["noteheads", (page: GoldenPage) => page.noteheads()],
  ["staff_fragments", (page: GoldenPage) => page.boxes("staffFragments")],
  ["clefs_keys", (page: GoldenPage) => page.boxes("clefsKeys")],
  ["stems_rest", (page: GoldenPage) => page.boxes("stemsRest")],
  ["bar_lines", (page: GoldenPage) => page.boxes("barLines")],
] as const satisfies ReadonlyArray<
  readonly [keyof never & string, (page: GoldenPage) => AngledBox[]]
>;

const GOT_BY_LABEL: Readonly<
  Record<string, (symbols: PredictedSymbols) => readonly AngledBox[]>
> = {
  bar_lines: (symbols) => symbols.barLines,
  clefs_keys: (symbols) => symbols.clefsKeys,
  noteheads: (symbols) => symbols.noteheads,
  staff_fragments: (symbols) => symbols.staffFragments,
  stems_rest: (symbols) => symbols.stemsRest,
};

const filteredMasks = (page: GoldenPage): SegmentationMasks => ({
  clefsKeys: page.mask("clefsKeys", true),
  notehead: page.mask("notehead", true),
  staff: page.mask("staff", true),
  stemsRest: page.mask("stemsRest", true),
  symbols: page.mask("symbols", true),
});

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);
  let computed: PredictedSymbols | undefined;
  const symbolsOf = async (): Promise<PredictedSymbols> => {
    if (computed === undefined) {
      const cv = await testOpenCv();
      const started = Date.now();
      computed = predictSymbols(cv, filteredMasks(page));
      process.stdout.write(
        `predictSymbols ${fixture.name}: ${Date.now() - started} ms\n`
      );
    }
    return computed;
  };

  describe(`predict_symbols ${fixture.name}`, () => {
    for (const [label, expected] of LISTS) {
      it(`reproduces ${label}`, async () => {
        const symbols = await symbolsOf();
        const report = compareBoxLists(
          label,
          GOT_BY_LABEL[label]?.(symbols) ?? [],
          expected(page)
        );
        process.stdout.write(`${describeBoxComparison(report)}\n`);
        assertBoxListMatches(report);
        expect(report.contoursExact).toBe(report.count.want);
        expect(report.debugIdsExact).toBe(report.count.want);
      });
    }
  });
}

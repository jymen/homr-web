/**
 * noise_filtering.py against Python. Both public pages are clean, so the page
 * rows prove the grid values and the identity; the masked and skipped outcomes
 * are test/golden/vectors/noise.json's, in vectors.test.ts, and the masking of
 * the seven planes is checked here on one of its cases.
 */

import { describe, expect, it } from "vitest";
import {
  createNoiseGrid,
  filterPredictions,
} from "../src/geometry/noise-filter.js";
import {
  createColor,
  createGray,
  grayFromMask,
  type Mask,
} from "../src/image/plane.js";
import { createInputPredictions } from "../src/model/pipeline.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { maskOfRows, vectorSet } from "./support/vectors.js";

const pagePredictions = (staff: Mask, fill: number) => {
  const ones: Mask = {
    ...staff,
    data: new Uint8Array(staff.data.length).fill(1),
  };
  return createInputPredictions(
    createColor(staff.width, staff.height, fill),
    createGray(staff.width, staff.height, fill),
    { clefsKeys: ones, notehead: ones, staff, stemsRest: ones, symbols: ones }
  );
};

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`filter_predictions ${fixture.name}`, () => {
    it("reproduces create_grid", () => {
      const noise = page.noise();
      const grid = createNoiseGrid(grayFromMask(page.mask("staff")));
      expect([grid.tileHeight, grid.tileWidth]).toEqual([
        noise.tile.height,
        noise.tile.width,
      ]);
      expect([grid.rows, grid.cols]).toEqual([
        noise.grid.length,
        noise.grid[0]?.length,
      ]);
      expect(Array.from(grid.values)).toEqual(noise.grid.flat());
    });

    it("reproduces the outcome, and a page not masked is the page it was given", () => {
      const input = pagePredictions(page.mask("staff"), 200);
      const { outcome, predictions } = filterPredictions(input);
      expect(outcome.kind).toBe(page.noise().outcome);
      expect(outcome.totalTiles).toBe(page.noise().total);
      expect(predictions).toBe(input);
    });
  });
}

describe("filterPredictions on a masked page", () => {
  const one = vectorSet("noise").cases.find(
    (entry) => entry.outcome === "masked"
  );

  it("zeroes the filtered tiles in all seven planes and nothing else", () => {
    expect(one).toBeDefined();
    const staff = maskOfRows(one?.staff, "staff");
    const keep = maskOfRows(one?.mask, "mask");
    const input = pagePredictions(staff, 200);
    const { outcome, predictions } = filterPredictions(input);
    expect(outcome.kind).toBe("masked");
    const expected = (value: number): number[] =>
      Array.from(keep.data, (kept) => kept * value);
    expect(Array.from(predictions.preprocessed.data)).toEqual(expected(200));
    expect(Array.from(predictions.original.data)).toEqual(
      expected(200).flatMap((value) => [value, value, value])
    );
    for (const name of [
      "clefsKeys",
      "notehead",
      "stemsRest",
      "symbols",
    ] as const) {
      expect(Array.from(predictions.masks[name].data)).toEqual(expected(1));
    }
    expect(Array.from(predictions.masks.staff.data)).toEqual(
      Array.from(staff.data, (value, i) => value * (keep.data[i] ?? 0))
    );
    expect(input.preprocessed.data.every((value) => value === 200)).toBe(true);
  });
});

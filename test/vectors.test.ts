/**
 * The numeric helpers and the pure algorithms against test/golden/vectors/:
 * inputs no page reaches, with the pinned Python's answers as data. Every
 * comparison here is exact.
 */

import { describe, expect, it } from "vitest";
import { rotatedBoxFromRect } from "../src/geometry/box-transforms.js";
import {
  legacyConventionRectOf,
  pointListFromPairs,
  type RotatedBox,
} from "../src/geometry/boxes.js";
import {
  createNoiseGrid,
  noiseOutcomeOf,
} from "../src/geometry/noise-filter.js";
import { findHorizontalLines } from "../src/geometry/other-clefs.js";
import {
  areLinesParallel,
  beginsOrEndsOnOneStaffLine,
  connectStaffLines,
} from "../src/geometry/staff-lines.js";
import { npArgsort } from "../src/image/argsort.js";
import { type FindPeaksOptions, findPeaks } from "../src/image/find-peaks.js";
import { floorDiv, mean, std, sum } from "../src/image/numeric.js";
import { createMask, grayFromMask } from "../src/image/plane.js";
import { DetectionError } from "../src/model/pipeline.js";
import { maskOfRows, type VectorCase, vectorSet } from "./support/vectors.js";

const numbersOf = (one: VectorCase, key: string): number[] =>
  one[key] as number[];

const numberOrNaN = (value: unknown): number =>
  value === null ? Number.NaN : (value as number);

describe("pairwise.json", () => {
  const { cases } = vectorSet("pairwise");

  it("holds arrays on both sides of numpy's 8 and 128 thresholds", () => {
    const lengths = new Set(
      cases.map((one) => numbersOf(one, "values").length)
    );
    for (const n of [0, 7, 8, 9, 128, 129, 2718]) {
      expect(lengths.has(n), `no array of ${n} values`).toBe(true);
    }
  });

  it.each([
    ["sum", sum],
    ["mean", mean],
    ["std", std],
  ] as const)("%s is numpy's on every array", (name, reduce) => {
    const wrong = cases.flatMap((one, i) => {
      const got = reduce(numbersOf(one, "values"));
      return Object.is(got, numberOrNaN(one[name]))
        ? []
        : [`cases[${i}] n=${numbersOf(one, "values").length}: ${got}`];
    });
    expect(wrong).toEqual([]);
  });
});

describe("floor-div.json", () => {
  it("floorDiv is Python's a // b on every pair", () => {
    const wrong = vectorSet("floor-div").cases.flatMap((one, i) => {
      const got = floorDiv(one.a as number, one.b as number);
      return got === one.q
        ? []
        : [`cases[${i}] ${one.a} // ${one.b}: ${got}, Python ${one.q}`];
    });
    expect(wrong).toEqual([]);
  });
});

describe("argsort.json", () => {
  const { cases } = vectorSet("argsort");

  it("npArgsort is np.argsort on every tied and random array", () => {
    const quick = cases.filter((one) => one.heapsorted === 0);
    expect(quick.length).toBeGreaterThan(0);
    for (const [i, one] of quick.entries()) {
      expect(
        Array.from(npArgsort(numbersOf(one, "values"))),
        `quicksorted case ${i}`
      ).toEqual(one.order);
    }
  });

  it("npArgsort is np.argsort where numpy falls back to heapsort", () => {
    const heaped = cases.filter((one) => (one.heapsorted as number) > 0);
    expect(heaped.length).toBeGreaterThan(0);
    for (const one of heaped) {
      expect(
        Array.from(npArgsort(numbersOf(one, "values"))),
        `input of ${numbersOf(one, "values").length}`
      ).toEqual(one.order);
    }
  });
});

describe("find-peaks.json", () => {
  const optionsOf = (one: VectorCase): FindPeaksOptions => ({
    ...(one.distance === null ? {} : { distance: one.distance as number }),
    ...(one.height === null ? {} : { height: one.height as number }),
    ...(one.prominence === null
      ? {}
      : { prominence: one.prominence as number }),
  });

  it("findPeaks is homr's find_peaks on every case", () => {
    const { cases } = vectorSet("find-peaks");
    expect(
      Math.max(...cases.map((one) => numbersOf(one, "peaks").length))
    ).toBeGreaterThanOrEqual(17);
    for (const [i, one] of cases.entries()) {
      expect(
        Array.from(
          findPeaks(Float64Array.from(numbersOf(one, "x")), optionsOf(one))
        ),
        `cases[${i}]`
      ).toEqual(one.peaks);
    }
  });
});

describe("noise.json", () => {
  const { cases } = vectorSet("noise");

  it("holds the three outcomes, a wrapped tile and a short last tile", () => {
    expect(new Set(cases.map((one) => one.outcome))).toEqual(
      new Set(["clean", "masked", "skipped"])
    );
    expect(cases.some((one) => (one.height as number) % 20 !== 0)).toBe(true);
    const grids = cases
      .filter((one) => String(one.name).includes("above 255"))
      .map((one) => (one.grid as number[][]).flat());
    expect(grids).toHaveLength(1);
    expect(grids[0]).toContain(2040 % 256);
  });

  it("the grid, the counts, the outcome and the mask are homr's on every case", () => {
    for (const one of cases) {
      const staff = maskOfRows(one.staff, `${one.name}.staff`);
      const grid = createNoiseGrid(grayFromMask(staff));
      expect(Array.from(grid.values), `${one.name}: grid`).toEqual(
        (one.grid as number[][]).flat()
      );
      const outcome = noiseOutcomeOf(grid, staff);
      expect(
        {
          filtered: outcome.kind === "clean" ? 0 : outcome.filteredTiles,
          kind: outcome.kind,
          total: outcome.totalTiles,
        },
        String(one.name)
      ).toEqual({
        filtered: one.filtered,
        kind: one.outcome,
        total: one.total,
      });
      if (outcome.kind === "masked") {
        expect(Array.from(outcome.keep.data), `${one.name}: mask`).toEqual(
          Array.from(maskOfRows(one.mask, `${one.name}.mask`).data)
        );
      }
    }
  });
});

/** A vector file's `[[cx, cy], [w, h], angle]` as the contourless box Python built from it. */
function boxOfRect(value: unknown): RotatedBox {
  const [[cx, cy], [w, h], angle] = value as [
    [number, number],
    [number, number],
    number,
  ];
  return rotatedBoxFromRect(
    legacyConventionRectOf({ angle, cx, cy, h, w }),
    pointListFromPairs([]),
    0
  );
}

describe("connect-lines.json", () => {
  const { cases } = vectorSet("connect-lines");

  it("holds a fragment in two chains, a non-parallel set and tied left edges", () => {
    const names = cases.map((one) => String(one.name));
    for (const part of [
      "two chains",
      "off the mean angle",
      "equal bottom_left",
    ]) {
      expect(names.some((name) => name.includes(part))).toBe(true);
    }
  });

  it("connectStaffLines, areLinesParallel and beginsOrEndsOnOneStaffLine are homr's on every case", () => {
    for (const one of cases) {
      const fragments = (one.fragments as unknown[]).map(boxOfRect);
      const unitSize = one.unitSize as number;
      const lines = connectStaffLines(fragments, unitSize);
      expect(
        lines.map((line) =>
          line.fragments.map((fragment) => fragments.indexOf(fragment))
        ),
        `${one.name}: lines`
      ).toEqual(one.lines);
      expect(areLinesParallel(lines, unitSize), `${one.name}: parallel`).toBe(
        one.parallel
      );
      for (const [i, probe] of (one.probes as VectorCase[]).entries()) {
        expect(
          beginsOrEndsOnOneStaffLine(boxOfRect(probe.line), lines, unitSize),
          `${one.name}: probes[${i}]`
        ).toBe(probe.onOneLine);
      }
    }
  });
});

describe("line-groups.json", () => {
  const { cases } = vectorSet("line-groups");

  const columnsOf = (one: VectorCase) => {
    const mask = createMask(one.width as number, one.height as number);
    for (const [y, n] of one.rowCounts as [number, number][]) {
      mask.data.fill(1, y * mask.width, y * mask.width + n);
    }
    return mask;
  };

  it("findHorizontalLines is homr's find_horizontal_lines on every case that returns", () => {
    const returning = cases.filter((one) => "groups" in one);
    expect(returning.length).toBeGreaterThan(0);
    for (const one of returning) {
      expect(
        findHorizontalLines(columnsOf(one), one.unitSize as number),
        String(one.name)
      ).toEqual(one.groups);
    }
  });

  it("a zone with no peak is a DetectionError where homr raises IndexError", () => {
    const raising = cases.filter((one) => one.error === "IndexError");
    expect(raising.map((one) => one.name)).toContain("all zero");
    for (const one of raising) {
      let thrown: unknown;
      try {
        findHorizontalLines(columnsOf(one), one.unitSize as number);
      } catch (error) {
        thrown = error;
      }
      expect(thrown, String(one.name)).toBeInstanceOf(DetectionError);
      expect((thrown as DetectionError).code).toBe("zone-without-lines");
    }
  });
});

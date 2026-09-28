import { describe, expect, it } from "vitest";
import {
  assertNormalizedRect,
  axisBoxOf,
  centerOf,
  concatPointLists,
  cornersOf,
  createAxisBox,
  filterPoints,
  GeometryError,
  hasValidRectSize,
  legacyConventionRectOf,
  legacyFromFitEllipse,
  normalizeRotatedRect,
  pointAt,
  pointCount,
  pointListFromInt32,
  pointListFromPairs,
  polygonOf,
  rawFitEllipseRectOf,
  rawMinAreaRectOf,
  rectKey,
  rotatedBoxFromParts,
  rotatedRectOf,
  sameRect,
  sortPointsByX,
  toLegacyAngleConvention,
} from "../src/geometry/boxes.js";

describe("normalizeRotatedRect (AngledBoundingBox.__init__)", () => {
  it.each([
    [
      { angle: 30, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: 30, h: 4, w: 10 },
    ],
    [
      { angle: 90, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: 0, h: 10, w: 4 },
    ],
    [
      { angle: 46, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: -44, h: 10, w: 4 },
    ],
    [
      { angle: -50, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: 40, h: 10, w: 4 },
    ],
    [
      { angle: 170, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: -10, h: 4, w: 10 },
    ],
    [
      { angle: -170, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: 10, h: 4, w: 10 },
    ],
    [
      { angle: 45, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: 45, h: 4, w: 10 },
    ],
    [
      { angle: -45, cx: 1, cy: 2, h: 4, w: 10 },
      { angle: -45, h: 4, w: 10 },
    ],
  ])("normalises %o", (raw, expected) => {
    const rect = normalizeRotatedRect(legacyConventionRectOf(raw));
    expect([rect.w, rect.h, rect.angle]).toEqual([
      expected.w,
      expected.h,
      expected.angle,
    ]);
    expect(normalizeRotatedRect(rect)).toEqual(rect);
  });
  it("refuses a rect outside the normalised range at the golden boundary", () => {
    expect(() =>
      assertNormalizedRect({ angle: 90, cx: 0, cy: 0, h: 1, w: 1 })
    ).toThrow(GeometryError);
  });
});

/**
 * The sweep phase-4-findings.md measured: identical integer corner sets given
 * to both builds at a known true rotation, `js` what @techstark/opencv-js
 * 4.12.0 answered and `python` what opencv-python 4.14.0 did. The conversion
 * has to turn the first into the second, dimensions included.
 */
const CONVENTION_SWEEP = [
  { js: 90, python: -90, rotation: 0, swapped: false },
  { js: 9.926, python: -80.074, rotation: 10, swapped: true },
  { js: 45, python: -45, rotation: 45, swapped: true },
  { js: 80.074, python: -9.926, rotation: 80, swapped: true },
  { js: 90, python: -90, rotation: 90, swapped: false },
] as const;

const jsMinAreaRect = (angle: number) =>
  rawMinAreaRectOf({
    angle,
    center: { x: 3, y: 5 },
    size: { height: 20, width: 7 },
  });

describe("toLegacyAngleConvention", () => {
  it.each(CONVENTION_SWEEP)(
    "at a true rotation of $rotation turns $js into $python",
    ({ js, python, swapped }) => {
      const rect = toLegacyAngleConvention(jsMinAreaRect(js));
      expect(rect.angle).toBeCloseTo(python, 10);
      expect([rect.w, rect.h]).toEqual(swapped ? [20, 7] : [7, 20]);
      expect([rect.cx, rect.cy]).toEqual([3, 5]);
    }
  );

  it.each([
    [45, -45, true],
    [90, -90, false],
    [-45, -45, false],
    [-90, -90, false],
  ])(
    "converts %d to exactly %d, swapping the dimensions: %s",
    (js, expected, swapped) => {
      const rect = toLegacyAngleConvention(jsMinAreaRect(js));
      expect(rect.angle).toBe(expected);
      expect([rect.w, rect.h]).toEqual(swapped ? [20, 7] : [7, 20]);
    }
  );

  it("leaves a fitEllipse rect alone, angle and dimensions", () => {
    const raw = rawFitEllipseRectOf({
      angle: 67.928,
      center: { x: 3, y: 5 },
      size: { height: 20, width: 7 },
    });
    expect(legacyFromFitEllipse(raw)).toEqual({
      angle: 67.928,
      cx: 3,
      cy: 5,
      h: 20,
      w: 7,
    });
  });
});

describe("hasValidRectSize (_has_box_valid_size)", () => {
  it.each([
    [{ angle: 0, cx: 0, cy: 0, h: 2, w: 1 }, true],
    [{ angle: 0, cx: 0, cy: 0, h: 2, w: 0 }, false],
    [{ angle: 0, cx: 0, cy: 0, h: 0, w: 1 }, false],
    [{ angle: 0, cx: 0, cy: 0, h: 2, w: Number.NaN }, false],
    [{ angle: 0, cx: 0, cy: 0, h: Number.NaN, w: 1 }, false],
  ])("reads %o as %s", (rect, expected) => {
    expect(hasValidRectSize(rect)).toBe(expected);
  });
});

describe("derived geometry", () => {
  const rect = normalizeRotatedRect(
    legacyConventionRectOf({ angle: 30, cx: 10, cy: 20, h: 6, w: 4 })
  );
  it("corners ignore the angle, as homr's do", () => {
    expect(cornersOf(rect)).toEqual({
      bottomLeft: { x: 8, y: 23 },
      bottomRight: { x: 12, y: 23 },
      topLeft: { x: 8, y: 17 },
      topRight: { x: 12, y: 17 },
    });
  });
  it("equality and hashing follow the triple", () => {
    const same = normalizeRotatedRect(
      legacyConventionRectOf({ angle: 30, cx: 10, cy: 20, h: 6, w: 4 })
    );
    expect(sameRect(rect, same)).toBe(true);
    expect(rectKey(rect)).toBe(rectKey(same));
    expect(sameRect(rect, { ...rect, angle: 31 })).toBe(false);
  });
  it("axis boxes derive rect, polygon and centre; angled boxes project to axis boxes", () => {
    const axis = createAxisBox(1, 2, 5, 8, pointListFromPairs([]));
    expect(rotatedRectOf(axis)).toEqual({ angle: 0, cx: 3, cy: 5, h: 6, w: 4 });
    expect(Array.from(polygonOf(axis))).toEqual([1, 2, 5, 2, 5, 8, 1, 8]);
    expect(centerOf(axis)).toEqual({ x: 3, y: 5 });
    const box = rotatedBoxFromParts(
      rect,
      pointListFromPairs([[0, 0]]),
      pointListFromPairs([]),
      7
    );
    expect(axisBoxOf(box)).toMatchObject({
      debugId: 7,
      kind: "axis",
      x1: 8,
      x2: 12,
      y1: 17,
      y2: 23,
    });
  });
});

describe("point lists", () => {
  it("wrap, index, concatenate, filter and sort", () => {
    expect(() => pointListFromInt32(new Int32Array(3))).toThrow(GeometryError);
    const list = pointListFromPairs([
      [5, 1],
      [2.9, 7],
      [9, 0],
    ]);
    expect(pointCount(list)).toBe(3);
    expect(pointAt(list, 1)).toEqual({ x: 2, y: 7 });
    expect(() => pointAt(list, 3)).toThrow(GeometryError);
    expect(Array.from(sortPointsByX(list))).toEqual([2, 7, 5, 1, 9, 0]);
    expect(Array.from(filterPoints(list, (x) => x > 4))).toEqual([5, 1, 9, 0]);
    expect(
      Array.from(concatPointLists([list, pointListFromPairs([[1, 1]])]))
    ).toHaveLength(8);
  });
});

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
  normalizeRotatedRect,
  pointAt,
  pointCount,
  pointListFromInt32,
  pointListFromPairs,
  polygonOf,
  rectKey,
  rotatedBoxFromParts,
  rotatedRectOf,
  sameRect,
  sortPointsByX,
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
    const rect = normalizeRotatedRect(raw);
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

describe("derived geometry", () => {
  const rect = normalizeRotatedRect({ angle: 30, cx: 10, cy: 20, h: 6, w: 4 });
  it("corners ignore the angle, as homr's do", () => {
    expect(cornersOf(rect)).toEqual({
      bottomLeft: { x: 8, y: 23 },
      bottomRight: { x: 12, y: 23 },
      topLeft: { x: 8, y: 17 },
      topRight: { x: 12, y: 17 },
    });
  });
  it("equality and hashing follow the triple", () => {
    const same = normalizeRotatedRect({
      angle: 30,
      cx: 10,
      cy: 20,
      h: 6,
      w: 4,
    });
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

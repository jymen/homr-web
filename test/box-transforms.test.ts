/**
 * The per-box operations, whose only production callers are phase 5's, so this
 * file is their whole verification. The two that would be silent if they went
 * wrong: makeBoxTaller rectangularises an ellipse on purpose, and every one of
 * them recomputes its outline rather than carrying the old one forward.
 */

import { describe, expect, it } from "vitest";
import {
  fitEllipseForGating,
  fitRotatedRect,
  fitRotatedRectUnchecked,
  polygonViaBoxPoints,
} from "../src/cv/box-fitting.js";
import {
  ensureMinDimension,
  getCenterExtrapolated,
  isOverlappingExtrapolated,
  makeBoxTaller,
  makeBoxTallerKeepCenter,
  makeBoxThicker,
  moveToXHorizontalBy,
} from "../src/cv/box-transforms.js";
import { withMatScope } from "../src/cv/opencv.js";
import { prepareBarLineImage } from "../src/geometry/barlines.js";
import {
  type Ellipse,
  ellipseFromParts,
  legacyConventionRectOf,
  normalizeRotatedRect,
  pointCount,
  pointListFromPairs,
  type RotatedBox,
  rotatedBoxFromParts,
} from "../src/geometry/boxes.js";
import { createMask } from "../src/image/plane.js";
import { testOpenCv } from "./support/opencv.js";

const NO_POINTS = pointListFromPairs([]);

const rect = (cx: number, cy: number, w: number, h: number, angle = 0) =>
  normalizeRotatedRect(legacyConventionRectOf({ angle, cx, cy, h, w }));

const box = (cx: number, cy: number, w: number, h: number, angle = 0) =>
  rotatedBoxFromParts(
    rect(cx, cy, w, h, angle),
    pointListFromPairs([[0, 0]]),
    pointListFromPairs([[1, 1]]),
    7
  ) satisfies RotatedBox;

const ellipse = (cx: number, cy: number, w: number, h: number) =>
  ellipseFromParts(
    rect(cx, cy, w, h),
    pointListFromPairs([[0, 0]]),
    pointListFromPairs([[1, 1]]),
    7
  ) satisfies Ellipse;

describe("makeBoxThicker", () => {
  it("grows both dimensions without moving the centre, and keeps a rotated box rotated", async () => {
    const cv = await testOpenCv();
    const thicker = withMatScope((scope) =>
      makeBoxThicker(cv, scope, box(10, 20, 4, 6), 2)
    );
    expect(thicker.kind).toBe("rotated");
    expect(thicker.rect).toEqual({ angle: 0, cx: 10, cy: 20, h: 8, w: 6 });
    expect(Array.from(thicker.polygon)).toEqual([7, 24, 7, 16, 13, 16, 13, 24]);
    expect(thicker.debugId).toBe(7);
  });

  it("keeps an ellipse an ellipse, with an ellipse2Poly outline", async () => {
    const cv = await testOpenCv();
    const thicker = withMatScope((scope) =>
      makeBoxThicker(cv, scope, ellipse(30, 40, 10, 6), 4)
    );
    expect(thicker.kind).toBe("ellipse");
    expect(thicker.rect).toEqual({ angle: 0, cx: 30, cy: 40, h: 10, w: 14 });
    expect(pointCount(thicker.polygon)).toBe(49);
  });

  it("returns the box itself for a thickness of zero or less", async () => {
    const cv = await testOpenCv();
    const original = box(10, 20, 4, 6);
    withMatScope((scope) => {
      expect(makeBoxThicker(cv, scope, original, 0)).toBe(original);
      expect(makeBoxThicker(cv, scope, original, -3)).toBe(original);
    });
  });
});

describe("the taller family", () => {
  it("turns an ellipse into a rotated box, as BoundingEllipse.make_box_taller does", async () => {
    const cv = await testOpenCv();
    const taller = makeBoxTaller(cv, ellipse(30, 40, 10, 6), 4);
    expect(taller.kind).toBe("rotated");
    expect(taller.rect).toEqual({ angle: 0, cx: 30, cy: 40, h: 10, w: 10 });
    expect(Array.from(taller.polygon)).toEqual([
      25, 45, 25, 35, 35, 35, 35, 45,
    ]);
  });

  it("keeps the centre by flooring the half thickness, also for a negative one", async () => {
    const cv = await testOpenCv();
    expect(makeBoxTallerKeepCenter(cv, box(10, 20, 4, 6), 5).rect).toEqual({
      angle: 0,
      cx: 10,
      cy: 18,
      h: 11,
      w: 4,
    });
    expect(makeBoxTallerKeepCenter(cv, box(10, 20, 4, 6), -5).rect).toEqual({
      angle: 0,
      cx: 10,
      cy: 23,
      h: 1,
      w: 4,
    });
  });
});

describe("the remaining mutators", () => {
  it("moves the centre horizontally and redraws the outline", async () => {
    const cv = await testOpenCv();
    const moved = moveToXHorizontalBy(cv, box(10, 20, 4, 6), -10);
    expect(moved.rect.cx).toBe(0);
    expect(Array.from(moved.polygon)).toEqual([-2, 23, -2, 17, 2, 17, 2, 23]);
  });

  it("raises each dimension to its floor and leaves the rest alone", async () => {
    const cv = await testOpenCv();
    const grown = ensureMinDimension(cv, box(10, 20, 2, 9, 30), 3, 3);
    expect(grown.rect).toEqual({ angle: 30, cx: 10, cy: 20, h: 9, w: 3 });
  });
});

describe("extrapolation", () => {
  it("reads the centre line at a given x", () => {
    expect(getCenterExtrapolated(box(0, 10, 100, 2, 45), 5)).toBeCloseTo(15, 9);
    expect(getCenterExtrapolated(box(0, 10, 100, 2, 0), 5)).toBe(10);
  });

  it("accepts two fragments of one line and rejects a gap wider than five unit sizes", () => {
    const unitSize = 10;
    expect(
      isOverlappingExtrapolated(box(0, 50, 20, 2), box(40, 50, 20, 2), unitSize)
    ).toBe(true);
    expect(
      isOverlappingExtrapolated(
        box(0, 50, 20, 2),
        box(400, 50, 20, 2),
        unitSize
      )
    ).toBe(false);
    expect(
      isOverlappingExtrapolated(box(0, 50, 20, 2), box(40, 90, 20, 2), unitSize)
    ).toBe(false);
  });
});

describe("the fitting entry points", () => {
  const contour = pointListFromPairs([
    [10, 10],
    [30, 10],
    [30, 20],
    [10, 20],
  ]);

  /**
   * Cross-checked against homr on 2026-09-28: RotatedBoundingBox of
   * cv2.minAreaRect on this contour holds box ((20, 15), (20, 10), 0) and
   * polygon [[30, 20], [10, 20], [10, 10], [30, 10]]. opencv.js answers
   * ((20, 15), (10, 20), 90) for the same contour, so the conversion is what
   * turns it into opencv-python's -90.
   */
  it("fits a rotated rect through the convention conversion", async () => {
    const cv = await testOpenCv();
    const fitted = withMatScope((scope) =>
      fitRotatedRect(cv, scope, contour, 3)
    );
    expect(fitted?.rect).toEqual({ angle: 0, cx: 20, cy: 15, h: 10, w: 20 });
    expect(fitted?.debugId).toBe(3);
    expect(Array.from(fitted?.polygon ?? new Int32Array())).toEqual([
      30, 20, 10, 20, 10, 10, 30, 10,
    ]);
  });

  it("refuses a degenerate contour, where the singular constructor does not", async () => {
    const cv = await testOpenCv();
    const line = pointListFromPairs([
      [10, 10],
      [30, 10],
    ]);
    withMatScope((scope) => {
      expect(fitRotatedRect(cv, scope, line, 0)).toBeNull();
      expect(fitRotatedRectUnchecked(cv, scope, line, 0).rect.h).toBe(0);
    });
  });

  it("gates fitEllipse below five points", async () => {
    const cv = await testOpenCv();
    withMatScope((scope) => {
      expect(fitEllipseForGating(cv, scope, contour, 0)).toBeNull();
      const fitted = fitEllipseForGating(
        cv,
        scope,
        pointListFromPairs([
          [10, 10],
          [30, 11],
          [31, 20],
          [11, 21],
          [20, 15],
        ]),
        1
      );
      expect(fitted?.kind).toBe("ellipse");
      expect(fitted?.rect.angle).toBeCloseTo(2.056_144_714_355_468_8, 9);
      expect(pointCount(fitted?.polygon ?? NO_POINTS)).toBe(71);
    });
  });

  /**
   * The same quadrilateral as the fitted box above, starting at a different
   * corner. That is why the polygon is stored rather than derived: recomputing
   * it from the normalised rect draws the right shape in the wrong sequence.
   */
  it("starts the point sequence elsewhere when handed the normalised rect", async () => {
    const cv = await testOpenCv();
    expect(Array.from(polygonViaBoxPoints(cv, rect(20, 15, 20, 10)))).toEqual([
      10, 20, 10, 10, 30, 10, 30, 20,
    ]);
  });
});

describe("prepareBarLineImage", () => {
  it("dilates with a five-by-three kernel of ones", async () => {
    const cv = await testOpenCv();
    const mask = createMask(7, 9);
    mask.data[4 * 7 + 3] = 1;
    const grown = prepareBarLineImage(cv, mask);
    const set: Array<readonly [number, number]> = [];
    for (let y = 0; y < grown.height; y += 1) {
      for (let x = 0; x < grown.width; x += 1) {
        if (grown.data[y * grown.width + x] === 1) {
          set.push([x, y]);
        }
      }
    }
    expect(set).toHaveLength(15);
    expect(set[0]).toEqual([2, 2]);
    expect(set.at(-1)).toEqual([4, 6]);
  });
});

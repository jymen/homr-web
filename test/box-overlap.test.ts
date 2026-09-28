/**
 * The two weak tests homr relies on, pinned as weak: a plus sign does not
 * overlap and a shared corner does. The last test is about the memo, because a
 * Mat per comparison is what makes the merge unusable rather than slow.
 */

import { describe, expect, it, vi } from "vitest";
import { doPolygonsOverlap, OverlapTester } from "../src/cv/box-overlap.js";
import { MatScope, withMatScope } from "../src/cv/opencv.js";
import {
  canShapesPossiblyTouch,
  legacyConventionRectOf,
  normalizeRotatedRect,
  pointListFromPairs,
  type RotatedBox,
  rotatedBoxFromParts,
} from "../src/geometry/boxes.js";
import { testOpenCv } from "./support/opencv.js";

const quad = (x1: number, y1: number, x2: number, y2: number): RotatedBox =>
  rotatedBoxFromParts(
    normalizeRotatedRect(
      legacyConventionRectOf({
        angle: 0,
        cx: (x1 + x2) / 2,
        cy: (y1 + y2) / 2,
        h: y2 - y1,
        w: x2 - x1,
      })
    ),
    pointListFromPairs([
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ]),
    pointListFromPairs([]),
    0
  );

/** Crossing like a plus sign: neither has a vertex inside the other. */
const WIDE = quad(-10, -2, 10, 2);
const TALL = quad(-2, -10, 2, 10);

describe("canShapesPossiblyTouch", () => {
  it("rejects only what is further apart than the two longer sides together", () => {
    expect(canShapesPossiblyTouch(quad(0, 0, 10, 10), quad(5, 5, 15, 15))).toBe(
      true
    );
    expect(
      canShapesPossiblyTouch(quad(0, 0, 10, 10), quad(100, 0, 110, 10))
    ).toBe(false);
  });
});

describe("doPolygonsOverlap", () => {
  it("reports two quads crossing like a plus sign as not overlapping", async () => {
    const cv = await testOpenCv();
    expect(
      withMatScope((scope) =>
        doPolygonsOverlap(cv, scope, WIDE.polygon, TALL.polygon)
      )
    ).toBe(false);
  });

  it("accepts a vertex that only sits on the other polygon's edge", async () => {
    const cv = await testOpenCv();
    expect(
      withMatScope((scope) =>
        doPolygonsOverlap(
          cv,
          scope,
          quad(0, 0, 10, 10).polygon,
          quad(10, 10, 20, 20).polygon
        )
      )
    ).toBe(true);
  });

  it("finds a vertex genuinely inside", async () => {
    const cv = await testOpenCv();
    expect(
      withMatScope((scope) =>
        doPolygonsOverlap(
          cv,
          scope,
          quad(0, 0, 10, 10).polygon,
          quad(5, 5, 15, 15).polygon
        )
      )
    ).toBe(true);
  });
});

describe("OverlapTester", () => {
  it("answers the whole relation, weaknesses included", async () => {
    const cv = await testOpenCv();
    withMatScope((scope) => {
      const tester = new OverlapTester(cv, scope);
      expect(tester.overlaps(WIDE, TALL)).toBe(false);
      expect(tester.overlaps(quad(0, 0, 10, 10), quad(5, 5, 15, 15))).toBe(
        true
      );
      expect(tester.overlaps(quad(0, 0, 10, 10), quad(100, 0, 110, 10))).toBe(
        false
      );
      expect(
        tester.overlapsAny(quad(0, 0, 10, 10), [
          quad(100, 0, 110, 10),
          quad(5, 5, 15, 15),
        ])
      ).toBe(true);
      expect(tester.overlapsAny(quad(0, 0, 10, 10), [])).toBe(false);
    });
  });

  it("builds one polygon Mat per box, not one per comparison", async () => {
    const cv = await testOpenCv();
    const scope = new MatScope();
    const kept = vi.spyOn(scope, "keep");
    try {
      const boxes = [
        quad(0, 0, 10, 10),
        quad(5, 5, 15, 15),
        quad(9, 9, 19, 19),
        quad(12, 12, 22, 22),
      ];
      const tester = new OverlapTester(cv, scope);
      let comparisons = 0;
      for (const [i, left] of boxes.entries()) {
        for (const [j, right] of boxes.entries()) {
          if (j > i) {
            comparisons += 1;
            tester.overlaps(left, right);
          }
        }
      }
      expect(comparisons).toBe(6);
      expect(kept).toHaveBeenCalledTimes(boxes.length);
    } finally {
      scope.release();
    }
  });
});

/**
 * The wasm boundary of phase 4: a plane in and a mask out, a point list in and
 * a contour out, and findContours on the real filtered masks. The contour
 * counts are the measured ones -- findContours is the input to every box in the
 * phase and it carries debug_id, so a build that disagreed here would make
 * nothing else comparable.
 */

import { describe, expect, it } from "vitest";
import { maskFromMat, planeToMat } from "../src/cv/mat-plane.js";
import {
  findContoursOf,
  pointListOfMat,
  pointListToMat,
} from "../src/cv/mat-points.js";
import { withMatScope } from "../src/cv/opencv.js";
import { pointCount, pointListFromPairs } from "../src/geometry/boxes.js";
import { createMask } from "../src/image/plane.js";
import type { MaskClass } from "../src/model/pipeline.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";

/** Measured 2026-09-28 on opencv-python 4.14.0, same count, order and points. */
const FILTERED_CONTOUR_COUNTS: Record<
  string,
  ReadonlyArray<readonly [MaskClass, number]>
> = {
  "the-kesh-300dpi": [
    ["staff", 188],
    ["stemsRest", 387],
    ["clefsKeys", 19],
    ["notehead", 82],
  ],
};

describe("the Mat boundary", () => {
  it("round-trips a mask, thresholding what cv2 wrote as 255", async () => {
    const cv = await testOpenCv();
    const mask = createMask(3, 2);
    mask.data.set([1, 0, 1, 0, 1, 0]);
    const back = withMatScope((scope) => {
      const mat = planeToMat(cv, scope, mask);
      const grown = scope.keep(new cv.Mat());
      const kernel = scope.keep(cv.Mat.ones(1, 3, cv.CV_8U));
      cv.dilate(mat, grown, kernel);
      return maskFromMat(grown);
    });
    expect(back.kind).toBe("mask");
    expect([back.width, back.height]).toEqual([3, 2]);
    expect(Array.from(back.data)).toEqual([1, 1, 1, 1, 1, 1]);
  });

  it("gives cv2 a contour as (n, 1, 2) and reads one back", async () => {
    const cv = await testOpenCv();
    const points = pointListFromPairs([
      [10, 10],
      [30, 10],
      [30, 20],
      [10, 20],
    ]);
    const read = withMatScope((scope) => {
      const mat = pointListToMat(cv, scope, points);
      expect([mat.rows, mat.cols]).toEqual([4, 1]);
      expect(mat.type()).toBe(cv.CV_32SC2);
      return pointListOfMat(mat);
    });
    expect(Array.from(read)).toEqual(Array.from(points));
  });

  it("finds the one contour of a filled rectangle, at CHAIN_APPROX_SIMPLE's four corners", async () => {
    const cv = await testOpenCv();
    const mask = createMask(20, 20);
    for (let y = 5; y < 10; y += 1) {
      for (let x = 5; x < 12; x += 1) {
        mask.data[y * 20 + x] = 1;
      }
    }
    const contours = withMatScope((scope) => findContoursOf(cv, scope, mask));
    expect(contours).toHaveLength(1);
    expect(Array.from(contours[0] ?? new Int32Array())).toEqual([
      5, 5, 5, 9, 11, 9, 11, 5,
    ]);
  });
});

for (const fixture of listGoldenFixtures()) {
  const expected = FILTERED_CONTOUR_COUNTS[fixture.name];
  if (expected === undefined) {
    continue;
  }
  const page = goldenPageOf(fixture);
  describe(`findContours on ${fixture.name}`, () => {
    it.each(expected)(
      "finds %s's measured %d contours in the filtered mask",
      async (name, count) => {
        const cv = await testOpenCv();
        const contours = withMatScope((scope) =>
          findContoursOf(cv, scope, page.mask(name, true))
        );
        expect(contours).toHaveLength(count);
        for (const contour of contours) {
          expect(pointCount(contour)).toBeGreaterThan(0);
        }
      }
    );
  });
}

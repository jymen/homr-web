/**
 * BoxOps on opencv.js: the one place detection touches a Mat for a box.
 */

import type { Mat } from "@techstark/opencv-js";
import type { BoxOps } from "../geometry/box-ops.js";
import {
  canShapesPossiblyTouch,
  ellipseFromParts,
  normalizeRotatedRect,
  type RotatedBox,
  type RotatedRect,
} from "../geometry/boxes.js";
import {
  fitRotatedRectUnchecked,
  polygonViaEllipse2Poly,
} from "./box-fitting.js";
import { OverlapTester } from "./box-overlap.js";
import { makeBoxThicker } from "./box-transforms.js";
import { type MatScope, type OpenCv, withMatScope } from "./opencv.js";

const nativeRectOf = (rect: RotatedRect) => ({
  angle: rect.angle,
  center: { x: rect.cx, y: rect.cy },
  size: { height: rect.h, width: rect.w },
});

/**
 * `cv2.rotatedRectangleIntersection(a.box, b.box)[0] != INTERSECT_NONE`, after
 * the same _can_shapes_possibly_touch prefilter Python applies. Both rects are
 * the stored, normalised ones, which is what Python passes.
 *
 * opencv.js takes three arguments and throws a BindingError on two; the third
 * is an output Mat nothing here reads. `region` is the caller's so that every
 * call of a stage shares one allocation.
 */
export function isIntersecting(
  cv: OpenCv,
  region: Mat,
  box: RotatedBox,
  other: RotatedBox
): boolean {
  if (!canShapesPossiblyTouch(box, other)) {
    return false;
  }
  return (
    cv.rotatedRectangleIntersection(
      nativeRectOf(box.rect),
      nativeRectOf(other.rect),
      region
    ) !== cv.INTERSECT_NONE
  );
}

/**
 * Invariant: every Mat the returned object allocates is in `scope`, so the
 * object must not outlive it. The overlap memo is keyed on box identity
 * (OverlapTester), so callers reuse box objects across calls where they can.
 */
export function createCvBoxOps(cv: OpenCv, scope: MatScope): BoxOps {
  const tester = new OverlapTester(cv, scope);
  const region = scope.keep(new cv.Mat());
  return {
    ellipseFromRect: (rect, contour, debugId) =>
      ellipseFromParts(
        normalizeRotatedRect(rect),
        polygonViaEllipse2Poly(cv, scope, rect),
        contour,
        debugId
      ),
    fitRotatedBox: (contour, debugId) =>
      fitRotatedRectUnchecked(cv, scope, contour, debugId),
    intersects: (box, other) => isIntersecting(cv, region, box, other),
    overlaps: (box, other) => tester.overlaps(box, other),
    thicker: (box, thickness) => makeBoxThicker(cv, scope, box, thickness),
  };
}

/** One scope and one overlap memo for one synchronous stage. */
export function withBoxOps<T>(cv: OpenCv, body: (ops: BoxOps) => T): T {
  return withMatScope((scope) => body(createCvBoxOps(cv, scope)));
}

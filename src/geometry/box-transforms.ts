/**
 * homr's RotatedBoundingBox mutators and line-extrapolation questions that
 * need no Mat: box in, box or number out.
 *
 * None of the mutators touches the angle, so the result is still inside
 * [-45, 45] and normalizeRotatedRect is never called again -- the new rect is
 * the old one spread with a changed dimension or centre. Each does recompute
 * the outline with a fresh boxPoints, because Python's constructor recomputes
 * it every time and a shape whose polygon did not follow its rect is a shape
 * whose overlap answers are stale.
 *
 * make_box_thicker is the one mutator not here: an ellipse's outline is
 * cv.ellipse2Poly, so it lives in src/cv/box-transforms.ts.
 */

import { floorDiv, toFloat32 } from "../image/numeric.js";
import {
  maxLineGapSize,
  toleranceForStaffLineDetection,
} from "../model/constants.js";
import {
  type AngledBox,
  type LegacyConventionRect,
  normalizeRotatedRect,
  type PointList,
  pointListFromPairs,
  type RotatedBox,
  type RotatedRect,
  type RotatedRectParams,
  rotatedBoxFromParts,
} from "./boxes.js";

/**
 * `cv2.boxPoints(box).astype(np.int64)`, reimplemented in float32 rather than
 * called, because the two builds do not compute it the same way.
 *
 * OpenCV's RotatedRect::points derives the first two corners from the centre,
 * the angle and the size; the older form then *reflects* the other two through
 * the centre, `pt[2] = 2 * centre - pt[0]`, and opencv-python 4.14.0 derives all
 * four directly. On a bit-identical rect the reflection can land exactly on an
 * integer where the direct form lands 3e-5 below it, and Python's int() then
 * differs by one: 4 coordinates of 5072 on the Kesh page, 3 of them in
 * staff_fragments entries whose *stored* rect is bit-exact, so nothing in the
 * stored data explains the difference and no rect-conditioned tolerance can
 * admit it.
 *
 * Measured 2026-09-28 over every entry of the four rotated golden lists whose
 * raw rect is bit-identical between the builds, 595 of 634: this reproduces
 * opencv-python 4.14.0's floats 595 of 595, and cv.boxPoints reproduces the
 * older reflecting form 595 of 595. phase-4-findings.md's "hand-rolling
 * boxPoints from OpenCV's C++ formula in float32 matched 0 of 967" does not
 * hold.
 *
 * The rect may be pre- or post-normalisation: this has no convention of its
 * own, it draws a rectangle from concrete numbers. float32 at every step, as the
 * C++ is; halving a float32 is exact, so only the products and sums are rounded.
 */
export function polygonViaBoxPoints(rect: RotatedRectParams): PointList {
  return pointListFromPairs(boxPointsFloat32(rect));
}

/** `cv2.boxPoints(rect)` itself, the four float32 corners, before any int conversion. */
export function boxPointsFloat32(
  rect: RotatedRectParams
): readonly [Corner, Corner, Corner, Corner] {
  const radians = (rect.angle * Math.PI) / 180;
  const halfSin = toFloat32(Math.sin(radians)) * 0.5;
  const halfCos = toFloat32(Math.cos(radians)) * 0.5;
  const sinH = toFloat32(halfSin * rect.h);
  const cosW = toFloat32(halfCos * rect.w);
  const cosH = toFloat32(halfCos * rect.h);
  const sinW = toFloat32(halfSin * rect.w);
  const left = toFloat32(rect.cx - sinH);
  const right = toFloat32(rect.cx + sinH);
  const upper = toFloat32(rect.cy - cosH);
  const lower = toFloat32(rect.cy + cosH);
  return [
    [toFloat32(left - cosW), toFloat32(lower - sinW)],
    [toFloat32(right - cosW), toFloat32(upper - sinW)],
    [toFloat32(right + cosW), toFloat32(upper + sinW)],
    [toFloat32(left + cosW), toFloat32(lower + sinW)],
  ];
}

export type Corner = readonly [number, number];

/**
 * `RotatedBoundingBox(box, contours, debug_id)` from numbers Python wrote
 * itself rather than fitted: predict_other_anchors_from_clefs builds
 * `((int(cx), int(cy)), (zone_width, int(span)), 0)` with an empty contour.
 * The outline comes from the rect as handed over, before normalisation, which
 * is the constructor's rule everywhere.
 */
export function rotatedBoxFromRect(
  rect: LegacyConventionRect,
  contour: PointList,
  debugId: number
): RotatedBox {
  return rotatedBoxFromParts(
    normalizeRotatedRect(rect),
    polygonViaBoxPoints(rect),
    contour,
    debugId
  );
}

/**
 * make_box_taller. **Always a RotatedBox**, an Ellipse included:
 * BoundingEllipse.make_box_taller returns a RotatedBoundingBox, silently
 * rectangularising the shape, and staff_detection depends on it because
 * is_intersecting exists only on the rotated class. The return type carries
 * that so no comment has to defend it at the call site.
 */
export function makeBoxTaller(box: AngledBox, thickness: number): RotatedBox {
  const rect = { ...box.rect, h: box.rect.h + thickness } as RotatedRect;
  return rotatedBoxFromParts(
    rect,
    polygonViaBoxPoints(rect),
    box.contour,
    box.debugId
  );
}

/** make_box_taller_keep_center: `cy - thickness // 2`, floor division, so -5 gives -3. */
export function makeBoxTallerKeepCenter(
  box: RotatedBox,
  thickness: number
): RotatedBox {
  const rect = {
    ...box.rect,
    cy: box.rect.cy - floorDiv(thickness, 2),
    h: box.rect.h + thickness,
  } as RotatedRect;
  return rotatedBoxFromParts(
    rect,
    polygonViaBoxPoints(rect),
    box.contour,
    box.debugId
  );
}

export function moveToXHorizontalBy(
  box: RotatedBox,
  xDelta: number
): RotatedBox {
  const rect = { ...box.rect, cx: box.rect.cx + xDelta } as RotatedRect;
  return rotatedBoxFromParts(
    rect,
    polygonViaBoxPoints(rect),
    box.contour,
    box.debugId
  );
}

export function ensureMinDimension(
  box: RotatedBox,
  minWidth: number,
  minHeight: number
): RotatedBox {
  const rect = {
    ...box.rect,
    h: Math.max(box.rect.h, minHeight),
    w: Math.max(box.rect.w, minWidth),
  } as RotatedRect;
  return rotatedBoxFromParts(
    rect,
    polygonViaBoxPoints(rect),
    box.contour,
    box.debugId
  );
}

/**
 * get_center_extrapolated: `(x - cx) * tan(angle / 180 * pi) + cy`.
 *
 * The multiplication order is kept as written. is_overlapping_extrapolated
 * inlines the same formula as `angle * pi / 180.0`, and `(a / 180) * pi` is not
 * `(a * pi) / 180` in float64, so there is deliberately no shared
 * degrees-to-radians helper in this port.
 */
export function getCenterExtrapolated(box: RotatedBox, x: number): number {
  return (
    (x - box.rect.cx) * Math.tan((box.rect.angle / 180) * Math.PI) + box.rect.cy
  );
}

/**
 * is_overlapping_extrapolated. Pure. `size[0] // 2` is Python floor division on
 * a float32 width, so floorDiv and not truncToInt.
 */
export function isOverlappingExtrapolated(
  a: RotatedBox,
  b: RotatedBox,
  unitSize: number
): boolean {
  const [left, right] = a.rect.cx > b.rect.cx ? [b, a] : [a, b];
  const centerX = (left.rect.cx + right.rect.cx) * 0.5;
  const maxGap = maxLineGapSize(unitSize);
  if (
    centerX - left.rect.cx - floorDiv(left.rect.w, 2) > maxGap ||
    right.rect.cx - centerX - floorDiv(right.rect.w, 2) > maxGap
  ) {
    return false;
  }
  const leftY =
    (centerX - left.rect.cx) * Math.tan((left.rect.angle * Math.PI) / 180) +
    left.rect.cy;
  const rightY =
    (centerX - right.rect.cx) * Math.tan((right.rect.angle * Math.PI) / 180) +
    right.rect.cy;
  return Math.abs(leftY - rightY) <= toleranceForStaffLineDetection(unitSize);
}

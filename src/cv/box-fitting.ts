/**
 * A contour in, a shape out, and the one place that decides which rect gets
 * which treatment.
 *
 * Both constructors compute their outline from the rect they are *handed*,
 * before homr's normalisation, and that is not an implementation detail: a
 * swapped (w, h) with a rotated angle draws the same quadrilateral but starts
 * the point sequence elsewhere, 19 of 377 entries' worth on the Kesh page.
 *
 * create_bounding_ellipses fits each contour twice, for two different reasons.
 * fitEllipse gates a contour of fewer than five points and shapes the polygon
 * the merge's overlap test reads; the rect a returned notehead actually carries
 * is a minAreaRect refit of its merged group's contours, singleton groups
 * included. So fitEllipse decides which boxes merge and minAreaRect decides
 * what the merged box is.
 */

import {
  type AngledBox,
  concatPointLists,
  type Ellipse,
  ellipseFromParts,
  hasValidRectSize,
  type LegacyConventionRect,
  legacyFromFitEllipse,
  normalizeRotatedRect,
  type PointList,
  pointCount,
  pointListFromPairs,
  type RawMinAreaRect,
  type RotatedBox,
  type RotatedRectParams,
  rawFitEllipseRectOf,
  rawMinAreaRectOf,
  rotatedBoxFromParts,
  toLegacyAngleConvention,
} from "../geometry/boxes.js";
import { toFloat32, truncToInt } from "../image/numeric.js";
import { pointListToMat } from "./mat-points.js";
import type { MatScope, OpenCv } from "./opencv.js";

/** homr's min_length_to_fit_ellipse: "this is a requirement by opencv". */
export const MIN_POINTS_TO_FIT_ELLIPSE = 5;

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
  return pointListFromPairs([
    [toFloat32(left - cosW), toFloat32(lower - sinW)],
    [toFloat32(right - cosW), toFloat32(upper - sinW)],
    [toFloat32(right + cosW), toFloat32(upper + sinW)],
    [toFloat32(left + cosW), toFloat32(lower + sinW)],
  ]);
}

/**
 * `cv2.ellipse2Poly((int(cx), int(cy)), (int(w / 2), int(h / 2)), int(angle),
 * 0, 360, 1)`. Every parameter truncated toward zero as Python's int() is, the
 * angle included, so -80.07 becomes -80 and not -81.
 *
 * The seventh argument is required here and must be a PointVector: a Mat throws
 * "Expected null or instance of PointVector".
 */
export function polygonViaEllipse2Poly(
  cv: OpenCv,
  scope: MatScope,
  rect: RotatedRectParams
): PointList {
  const sampled = scope.keep(new cv.PointVector());
  cv.ellipse2Poly(
    { x: truncToInt(rect.cx), y: truncToInt(rect.cy) },
    { height: truncToInt(rect.h / 2), width: truncToInt(rect.w / 2) },
    truncToInt(rect.angle),
    0,
    360,
    1,
    sampled
  );
  const corners: Array<readonly [number, number]> = [];
  for (let i = 0; i < sampled.size(); i += 1) {
    const point = sampled.get(i);
    corners.push([point.x, point.y]);
  }
  return pointListFromPairs(corners);
}

function minAreaRectOf(
  cv: OpenCv,
  scope: MatScope,
  contour: PointList
): RawMinAreaRect {
  return rawMinAreaRectOf(cv.minAreaRect(pointListToMat(cv, scope, contour)));
}

function rotatedBoxOf(
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

function ellipseOf(
  cv: OpenCv,
  scope: MatScope,
  rect: LegacyConventionRect,
  contour: PointList,
  debugId: number
): Ellipse {
  return ellipseFromParts(
    normalizeRotatedRect(rect),
    polygonViaEllipse2Poly(cv, scope, rect),
    contour,
    debugId
  );
}

/** create_rotated_bounding_boxes' per-contour fit; null where Python continues. */
export function fitRotatedRect(
  cv: OpenCv,
  scope: MatScope,
  contour: PointList,
  debugId: number
): RotatedBox | null {
  const raw = minAreaRectOf(cv, scope, contour);
  if (!hasValidRectSize(raw)) {
    return null;
  }
  return rotatedBoxOf(toLegacyAngleConvention(raw), contour, debugId);
}

/**
 * create_rotated_bounding_box, singular. It skips the _has_box_valid_size check
 * the plural one applies, so a degenerate contour yields a box with a NaN or
 * zero dimension rather than nothing. Preserved: break_wide_fragments is the
 * only caller and it relies on always getting a box back.
 */
export function fitRotatedRectUnchecked(
  cv: OpenCv,
  scope: MatScope,
  contour: PointList,
  debugId: number
): RotatedBox {
  return rotatedBoxOf(
    toLegacyAngleConvention(minAreaRectOf(cv, scope, contour)),
    contour,
    debugId
  );
}

/**
 * create_bounding_ellipses' per-contour fit, the gating half: the five-point
 * minimum, then fitEllipse with no angle conversion. This rect is replaced by
 * the refit below; its polygon and its normalised size are what survive, the
 * first read by the overlap test and the second by the min/max filter.
 */
export function fitEllipseForGating(
  cv: OpenCv,
  scope: MatScope,
  contour: PointList,
  debugId: number
): Ellipse | null {
  if (pointCount(contour) < MIN_POINTS_TO_FIT_ELLIPSE) {
    return null;
  }
  const raw = rawFitEllipseRectOf(
    cv.fitEllipse(pointListToMat(cv, scope, contour))
  );
  if (!hasValidRectSize(raw)) {
    return null;
  }
  return ellipseOf(cv, scope, legacyFromFitEllipse(raw), contour, debugId);
}

/**
 * _get_box_for_whole_group: the group's contours concatenated in group order,
 * refitted with minAreaRect, debug_id discarded. No validity check here, as in
 * Python.
 */
export function refitRotatedBoxFromGroup(
  cv: OpenCv,
  scope: MatScope,
  group: readonly AngledBox[]
): RotatedBox {
  const contour = concatPointLists(group.map((box) => box.contour));
  return rotatedBoxOf(
    toLegacyAngleConvention(minAreaRectOf(cv, scope, contour)),
    contour,
    0
  );
}

/** _get_ellipse_for_whole_group, which also refits with minAreaRect, not fitEllipse. */
export function refitEllipseFromGroup(
  cv: OpenCv,
  scope: MatScope,
  group: readonly AngledBox[]
): Ellipse {
  const contour = concatPointLists(group.map((box) => box.contour));
  return ellipseOf(
    cv,
    scope,
    toLegacyAngleConvention(minAreaRectOf(cv, scope, contour)),
    contour,
    0
  );
}

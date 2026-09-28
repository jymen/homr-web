/**
 * homr's RotatedBoundingBox and BoundingEllipse mutators: box in, box out.
 *
 * None of them touches the angle, so the result is still inside [-45, 45] and
 * normalizeRotatedRect is never called again -- the new rect is the old one
 * spread with a changed dimension or centre. Each does recompute the outline
 * with a fresh boxPoints or ellipse2Poly call, because Python's constructor
 * recomputes it every time and a shape whose polygon did not follow its rect is
 * a shape whose overlap answers are stale.
 */

import {
  type AngledBox,
  type RotatedBox,
  type RotatedRect,
  rotatedBoxFromParts,
} from "../geometry/boxes.js";
import { floorDiv } from "../image/numeric.js";
import {
  maxLineGapSize,
  toleranceForStaffLineDetection,
} from "../model/constants.js";
import { polygonViaBoxPoints, polygonViaEllipse2Poly } from "./box-fitting.js";
import type { MatScope, OpenCv } from "./opencv.js";

function rebuilt<B extends AngledBox>(
  cv: OpenCv,
  scope: MatScope,
  box: B,
  rect: RotatedRect
): B {
  const polygon =
    box.kind === "ellipse"
      ? polygonViaEllipse2Poly(cv, scope, rect)
      : polygonViaBoxPoints(cv, rect);
  return { ...box, polygon, rect } as B;
}

/**
 * make_box_thicker: both dimensions grow, the centre does not move. homr's own
 * comment says moving it by thickness / 2 gave much worse results on some
 * examples and downstream code depends on today's behaviour, so this is not a
 * bug to fix.
 *
 * Kind-preserving through generic inference, which is homr's two identical
 * overrides without an isinstance chain. The thickness is a number and not an
 * int: brace_dot_detection passes a float despite the Python annotation.
 *
 * One deliberate unification: RotatedBoundingBox returns itself for a thickness
 * of zero or less and BoundingEllipse has no such guard, so a non-positive
 * thickness would rebuild an ellipse's polygon from its normalised rect instead
 * of leaving it alone. No caller anywhere passes one -- the three real call
 * sites pass 15, 30 and a positive tolerance -- so the guard covers both here.
 */
export function makeBoxThicker<B extends AngledBox>(
  cv: OpenCv,
  scope: MatScope,
  box: B,
  thickness: number
): B {
  if (thickness <= 0) {
    return box;
  }
  return rebuilt(cv, scope, box, {
    ...box.rect,
    h: box.rect.h + thickness,
    w: box.rect.w + thickness,
  } as RotatedRect);
}

/**
 * make_box_taller. **Always a RotatedBox**, an Ellipse included:
 * BoundingEllipse.make_box_taller returns a RotatedBoundingBox, silently
 * rectangularising the shape, and staff_detection depends on it because
 * is_intersecting exists only on the rotated class. The return type carries
 * that so no comment has to defend it at the call site.
 */
export function makeBoxTaller(
  cv: OpenCv,
  box: AngledBox,
  thickness: number
): RotatedBox {
  const rect = { ...box.rect, h: box.rect.h + thickness } as RotatedRect;
  return rotatedBoxFromParts(
    rect,
    polygonViaBoxPoints(cv, rect),
    box.contour,
    box.debugId
  );
}

/** make_box_taller_keep_center: `cy - thickness // 2`, floor division, so -5 gives -3. */
export function makeBoxTallerKeepCenter(
  cv: OpenCv,
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
    polygonViaBoxPoints(cv, rect),
    box.contour,
    box.debugId
  );
}

export function moveToXHorizontalBy(
  cv: OpenCv,
  box: RotatedBox,
  xDelta: number
): RotatedBox {
  const rect = { ...box.rect, cx: box.rect.cx + xDelta } as RotatedRect;
  return rotatedBoxFromParts(
    rect,
    polygonViaBoxPoints(cv, rect),
    box.contour,
    box.debugId
  );
}

export function ensureMinDimension(
  cv: OpenCv,
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
    polygonViaBoxPoints(cv, rect),
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

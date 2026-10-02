/**
 * The one per-box mutator that needs opencv.js: an ellipse's outline is
 * cv.ellipse2Poly. The others are in src/geometry/box-transforms.ts, and the
 * rule they share holds here too: the angle is untouched, so the rect is never
 * normalised again, and the outline is recomputed from the new rect.
 */

import { polygonViaBoxPoints } from "../geometry/box-transforms.js";
import type { AngledBox, RotatedRect } from "../geometry/boxes.js";
import { polygonViaEllipse2Poly } from "./box-fitting.js";
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
      : polygonViaBoxPoints(rect);
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

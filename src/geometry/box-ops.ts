/**
 * The five things homr's detection asks of OpenCV about a box, as an interface
 * the algorithms take instead of a `cv` handle.
 *
 * mergeOverlappingGroups already takes its overlap test as a predicate so the
 * grouping can be tested without opencv.js. Detection has the same reason many
 * times over, so the predicate becomes one object. src/cv/box-ops.ts holds the
 * only implementation that runs in production; a test of an ordering rule
 * passes a literal.
 */

import type {
  AngledBox,
  AnyBox,
  Ellipse,
  LegacyConventionRect,
  PointList,
  RotatedBox,
} from "./boxes.js";

export interface BoxOps {
  /**
   * `BoundingEllipse((centre, size, angle), contours, debug_id)` from numbers
   * Python wrote itself (split_clumps_of_noteheads). Never a fitted rect.
   */
  readonly ellipseFromRect: (
    rect: LegacyConventionRect,
    contour: PointList,
    debugId: number
  ) => Ellipse;
  /** create_rotated_bounding_box, singular: minAreaRect, no size check. */
  readonly fitRotatedBox: (contour: PointList, debugId: number) => RotatedBox;
  /** RotatedBoundingBox.is_intersecting. Touching counts. */
  readonly intersects: (box: RotatedBox, other: RotatedBox) => boolean;
  /** is_overlapping. The left operand is never an AxisBox, as in homr. */
  readonly overlaps: (box: AngledBox, other: AnyBox) => boolean;
  /** make_box_thicker, kind-preserving. */
  readonly thicker: <B extends AngledBox>(box: B, thickness: number) => B;
}

/** is_overlapping_with_any. */
export function overlapsAny(
  ops: BoxOps,
  box: AngledBox,
  others: readonly AnyBox[]
): boolean {
  for (const other of others) {
    if (ops.overlaps(box, other)) {
      return true;
    }
  }
  return false;
}

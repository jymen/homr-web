/** Port of staff_detection.break_wide_fragments. */

import type { BoxOps } from "./box-ops.js";
import {
  concatPointLists,
  filterPoints,
  type PointList,
  pointCount,
  type RotatedBox,
  sortPointsByX,
} from "./boxes.js";

/** homr's default `limit`; the one caller passes none. */
export const WIDE_FRAGMENT_LIMIT = 100;

function minContourX(contour: PointList): number {
  let min = Number.POSITIVE_INFINITY;
  for (let i = 0; i < contour.length; i += 2) {
    min = Math.min(min, contour[i] ?? min);
  }
  return min;
}

/**
 * break_wide_fragments: a fragment wider than 100 px is cut at
 * `minX + 100` of its contour, repeatedly, each piece refitted.
 *
 * A new list in the input's order, pieces left to right, every piece keeping
 * the fragment's debugId. A piece can be degenerate (zero width); no size
 * check applies here and connect_staff_lines drops it later.
 */
export function breakWideFragments(
  ops: BoxOps,
  fragments: readonly RotatedBox[]
): RotatedBox[] {
  const result: RotatedBox[] = [];
  for (const fragment of fragments) {
    let remaining = fragment;
    while (remaining.rect.w > WIDE_FRAGMENT_LIMIT) {
      const cut = minContourX(remaining.contour) + WIDE_FRAGMENT_LIMIT;
      const left = sortPointsByX(
        filterPoints(remaining.contour, (x) => x < cut)
      );
      const right = sortPointsByX(
        filterPoints(remaining.contour, (x) => x >= cut)
      );
      if (pointCount(left) === 0 || pointCount(right) === 0) {
        break;
      }
      const joint = right.subarray(0, 2) as PointList;
      result.push(
        ops.fitRotatedBox(concatPointLists([left, joint]), remaining.debugId)
      );
      // staff_detection.py:683 appends contours_left[-1], which line 682 has just made contours_right[0]: the right piece ends on a copy of its own first point.
      remaining = ops.fitRotatedBox(
        concatPointLists([right, joint]),
        remaining.debugId
      );
    }
    result.push(remaining);
  }
  return result;
}

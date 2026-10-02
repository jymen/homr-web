/**
 * staff_detection.detect_staff: the ten calls, in homr's order. Each callee is
 * exported and tested against Python's value for its own input, so this file
 * is the order and nothing else.
 */

import type { BoxOps } from "../geometry/box-ops.js";
import type { RotatedBox } from "../geometry/boxes.js";
import { predictOtherAnchorsFromClefs } from "../geometry/other-clefs.js";
import {
  findRawStaffsByConnectingLineFragments,
  removeDuplicateStaffs,
} from "../geometry/raw-staffs.js";
import {
  filterEdgeOfVision,
  resampleStaffs,
  sortStaffsTopToBottom,
} from "../geometry/resample.js";
import {
  filterUnusualAnchors,
  findStaffAnchors,
} from "../geometry/staff-anchors.js";
import type { Mask } from "../image/plane.js";
import type { Staff } from "../model/staff.js";

/**
 * `strongStaff` is the staff mask after makeLinesStronger and `fragments` the
 * list after breakWideFragments; homr reads the mask for its size and for the
 * columns under the clefs, nothing else.
 *
 * Top to bottom. May be empty; the caller decides what that means.
 */
export function detectStaff(
  ops: BoxOps,
  strongStaff: Mask,
  fragments: readonly RotatedBox[],
  clefsKeys: readonly RotatedBox[],
  barLines: readonly RotatedBox[]
): Staff[] {
  const clefAnchors = findStaffAnchors(ops, fragments, clefsKeys, "clef");
  const otherClefs = predictOtherAnchorsFromClefs(
    ops,
    clefAnchors,
    strongStaff
  );
  const anchors = filterUnusualAnchors([
    ...clefAnchors,
    ...findStaffAnchors(ops, fragments, otherClefs, "clef"),
    ...findStaffAnchors(ops, fragments, barLines, "barLine"),
  ]);
  const rawStaffs = removeDuplicateStaffs(
    ops,
    findRawStaffsByConnectingLineFragments(ops, anchors, fragments)
  );
  return sortStaffsTopToBottom(
    filterEdgeOfVision(resampleStaffs(rawStaffs), strongStaff)
  );
}

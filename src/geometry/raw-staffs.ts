/**
 * Anchors that share a staff line become one RawStaff. Port of
 * staff_detection.RawStaff, get_staff_for_anchor,
 * find_raw_staffs_by_connecting_line_fragments and remove_duplicate_staffs.
 */

import type { BoxOps } from "./box-ops.js";
import {
  concatPointLists,
  type PointList,
  type RotatedBox,
  sameRect,
} from "./boxes.js";
import type { StaffAnchor } from "./staff-anchors.js";
import {
  connectStaffLines,
  type FiveLines,
  holdsAllFragmentsOf,
  mapFiveLines,
  mergeSegments,
} from "./staff-lines.js";

/**
 * Python's RawStaff *is* a RotatedBoundingBox. Here it *has* one, because the
 * box transforms spread their input (`{ ...box, rect }`) and would carry
 * `lines` and `anchors` onto a moved copy, and because a RawStaff must not be
 * accepted where a list of fragments is.
 *
 * `box.debugId` is the staff id. homr reads it nowhere.
 */
export interface RawStaff {
  readonly anchors: readonly [StaffAnchor, ...StaffAnchor[]];
  /**
   * minAreaRect over every contour point of every fragment of the five lines.
   * min_x and max_x are `cornersOf(box.rect)`, of the *normalised* rect.
   */
  readonly box: RotatedBox;
  readonly lines: FiveLines;
}

/** _get_all_contours, concatenated: line by line, fragment by fragment. A fragment two lines hold appears twice. */
export function rawStaffContour(lines: FiveLines): PointList {
  return concatPointLists(
    lines.flatMap((line) => line.fragments.map((fragment) => fragment.contour))
  );
}

/** A RawStaff around a box already fitted: the golden decoder's entry, which has Python's rect and no opencv.js. */
export function rawStaffFromParts(
  box: RotatedBox,
  lines: FiveLines,
  anchors: readonly [StaffAnchor, ...StaffAnchor[]]
): RawStaff {
  return { anchors, box, lines };
}

/** RawStaff(staff_id, lines, anchors). */
export function createRawStaff(
  ops: BoxOps,
  staffId: number,
  lines: FiveLines,
  anchors: readonly [StaffAnchor, ...StaffAnchor[]]
): RawStaff {
  return rawStaffFromParts(
    ops.fitRotatedBox(rawStaffContour(lines), staffId),
    lines,
    anchors
  );
}

/**
 * RawStaff.merge: line i is `other.lines[i]` merged with `self.lines[i]`, in
 * that order, so other's fragments come first; anchors self then other; self's
 * id.
 */
export function mergeRawStaffs(
  ops: BoxOps,
  self: RawStaff,
  other: RawStaff
): RawStaff {
  return createRawStaff(
    ops,
    self.box.debugId,
    mapFiveLines(self.lines, (line, i) => mergeSegments(other.lines[i], line)),
    [...self.anchors, ...other.anchors]
  );
}

/**
 * get_staff_for_anchor: the first staff, in list order, where **any one** of
 * the anchor's lines is contained in the staff's line at the same index. One
 * line, not five; that is how 120 Kesh anchors become 4 staffs.
 */
export function staffForAnchor(
  anchor: StaffAnchor,
  staffs: readonly RawStaff[]
): RawStaff | null {
  // staff_detection.py:172-178 returns from inside the loop over line indices.
  return (
    staffs.find((staff) =>
      staff.lines.some((staffLine, i) => {
        const line = anchor.lines[i];
        return line !== undefined && holdsAllFragmentsOf(staffLine, line);
      })
    ) ?? null
  );
}

/**
 * find_raw_staffs_by_connecting_line_fragments.
 *
 * Order is part of the result: a staff an anchor merges into is removed (the
 * first whose box rect equals it) and the merged staff appended at the end.
 */
export function findRawStaffsByConnectingLineFragments(
  ops: BoxOps,
  anchors: readonly StaffAnchor[],
  fragments: readonly RotatedBox[]
): RawStaff[] {
  const staffs: RawStaff[] = [];
  for (const [staffId, anchor] of anchors.entries()) {
    const existing = staffForAnchor(anchor, staffs);
    // staff_detection.py:195 reads range.stop as a bound the centre may equal.
    const connected = connectStaffLines(
      fragments.filter(
        (fragment) =>
          fragment.rect.cy >= anchor.zone.start &&
          fragment.rect.cy <= anchor.zone.stop
      ),
      anchor.averageUnitSize
    );
    const lines = mapFiveLines(anchor.lines, (anchorLine) => {
      const [match, ...others] = connected.filter((line) =>
        holdsAllFragmentsOf(line, anchorLine)
      );
      return match === undefined || others.length > 0 ? anchorLine : match;
    });
    const fresh = createRawStaff(ops, staffId, lines, [anchor]);
    if (existing === null) {
      staffs.push(fresh);
    } else {
      // staff_detection.py:209 list.remove takes the first staff *equal* to it, by box value.
      staffs.splice(
        staffs.findIndex((staff) =>
          sameRect(staff.box.rect, existing.box.rect)
        ),
        1
      );
      staffs.push(mergeRawStaffs(ops, existing, fresh));
    }
  }
  return staffs;
}

/**
 * remove_duplicate_staffs. A staff overlapping two or more kept ones is
 * dropped; overlapping exactly one, it replaces that one only when it has
 * strictly more anchors, and the replacement goes to the end.
 */
export function removeDuplicateStaffs(
  ops: BoxOps,
  staffs: readonly RawStaff[]
): RawStaff[] {
  let result: RawStaff[] = [];
  for (const staff of staffs) {
    const [rival, ...others] = result.filter((other) =>
      ops.overlaps(staff.box, other.box)
    );
    if (rival === undefined) {
      result.push(staff);
    } else if (
      others.length === 0 &&
      rival.anchors.length < staff.anchors.length
    ) {
      // staff_detection.py:234 drops every kept staff whose box equals the rival's, not the rival alone.
      result = result.filter(
        (kept) => !sameRect(kept.box.rect, rival.box.rect)
      );
      result.push(staff);
    }
  }
  return result;
}

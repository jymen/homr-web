/**
 * Which staffs play together. Port of brace_dot_detection.py from
 * _filter_for_tall_elements on, and of MultiStaff.create_grandstaffs with its
 * scoring.
 *
 * The scoring is a MultiStaff method in model.py. It is here rather than in
 * src/model/staff.ts because it has one caller, this file, and its subject is
 * what a brace means; mergeMultiStaffs and mergeStaffs, which know nothing of
 * braces, are with their types.
 *
 * homr has two more connection finders, at bar lines and at clefs. Both read
 * symbols no code ever puts on a staff, so both return [] on every page and
 * neither is ported.
 */

import { median } from "../image/numeric.js";
import {
  GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR,
  GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR,
  MIN_CONNECTIONS_TO_FORM_COMBINED_STAFF,
  maxWidthForBraceRough,
  minHeightForBrace,
  minHeightForBraceRough,
  toleranceForTouchingClefs,
} from "../model/constants.js";
import {
  createMultiStaff,
  type MultiStaff,
  mergeMultiStaffs,
  mergeStaffs,
  type Staff,
  staffPointAt,
  staffPointToAxisBox,
  yDistanceTo,
} from "../model/staff.js";
import type { BoxOps } from "./box-ops.js";
import type { RotatedBox } from "./boxes.js";

/** `min(staffs, key=...)`: the first of the smallest. */
function closestStaff(
  symbol: RotatedBox,
  staffs: readonly [Staff, ...Staff[]]
): Staff {
  const center = { x: symbol.rect.cx, y: symbol.rect.cy };
  let [closest] = staffs;
  let distance = yDistanceTo(closest, center);
  for (const staff of staffs) {
    const candidate = yDistanceTo(staff, center);
    if (candidate < distance) {
      closest = staff;
      distance = candidate;
    }
  }
  return closest;
}

/**
 * _filter_for_tall_elements. The rough pass measures against `staffs[0]`
 * alone; the exact pass against the staff nearest in y, the first on a tie,
 * and a symbol more than 50 px in x from every staff ties them all at 1e10.
 */
export function filterForTallElements(
  braceDot: readonly RotatedBox[],
  staffs: readonly [Staff, ...Staff[]]
): RotatedBox[] {
  const rough = staffs[0].averageUnitSize;
  return braceDot.filter(
    (symbol) =>
      symbol.rect.h > minHeightForBraceRough(rough) &&
      symbol.rect.w < maxWidthForBraceRough(rough) &&
      symbol.rect.h >
        minHeightForBrace(closestStaff(symbol, staffs).averageUnitSize)
  );
}

/**
 * _get_connections_between_staffs_at_lines: the symbols which, thickened by
 * toleranceForTouchingClefs(staff1.averageUnitSize), overlap both staffs'
 * line span at the symbol's x.
 *
 * Not symmetric: the thickness is staff1's, so (a, b) and (b, a) can differ.
 */
export function connectionsBetweenStaffs(
  ops: BoxOps,
  staff1: Staff,
  staff2: Staff,
  braceDot: readonly RotatedBox[]
): RotatedBox[] {
  const thickness = toleranceForTouchingClefs(staff1.averageUnitSize);
  return braceDot.filter((symbol) => {
    const point1 = staffPointAt(staff1, symbol.rect.cx);
    const point2 = staffPointAt(staff2, symbol.rect.cx);
    if (point1 === null || point2 === null) {
      return false;
    }
    const thicker = ops.thicker(symbol, thickness);
    return (
      ops.overlaps(thicker, staffPointToAxisBox(point1)) &&
      ops.overlaps(thicker, staffPointToAxisBox(point2))
    );
  });
}

/**
 * _merge_multi_staff_if_they_share_a_staff. Sharing is by identity. A merged
 * entry is removed and appended at the end, so the output order is not the
 * input order once anything merges.
 */
export function mergeMultiStaffsSharingAStaff(
  multiStaffs: readonly MultiStaff[]
): MultiStaff[] {
  const result: MultiStaff[] = [];
  for (const multiStaff of multiStaffs) {
    const at = result.findIndex((one) =>
      one.staffs.some((staff) => multiStaff.staffs.includes(staff))
    );
    const [existing] = at < 0 ? [] : result.splice(at, 1);
    result.push(
      existing === undefined
        ? multiStaff
        : mergeMultiStaffs(existing, multiStaff)
    );
  }
  return result;
}

/**
 * MultiStaff._score_brace_with_staff_pair: `yOverlap - xDistance` when the
 * brace's centre is within 5 units of the pair's left edge and it spans more
 * than half its own height of the pair; otherwise 0. The unit is the mean of
 * the two staffs' units.
 */
export function scoreBraceWithStaffPair(
  brace: RotatedBox,
  upper: Staff,
  lower: Staff
): number {
  const unit = median([upper.averageUnitSize, lower.averageUnitSize]);
  const { cx, cy, h } = brace.rect;
  const xDistance = Math.abs(Math.min(upper.minX, lower.minX) - cx);
  const yOverlap =
    Math.min(cy + h / 2, lower.maxY) - Math.max(cy - h / 2, upper.minY);
  return xDistance < GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR * unit &&
    yOverlap > GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR * h &&
    yOverlap > xDistance
    ? yOverlap - xDistance
    : 0;
}

/**
 * MultiStaff._select_grandstaffs: the upper index of each pair taken. Pairs
 * are tried by best score, the earlier pair first among equals, and a pair is
 * skipped once either of its staffs is in a taken one.
 */
function selectGrandstaffs(
  staffs: readonly Staff[],
  braces: readonly RotatedBox[]
): Set<number> {
  const scored: { readonly score: number; readonly upper: number }[] = [];
  for (let i = 0; i + 1 < staffs.length; i += 1) {
    const [upper, lower] = [staffs[i], staffs[i + 1]];
    if (upper === undefined || lower === undefined) {
      continue;
    }
    const score = Math.max(
      ...braces.map((brace) => scoreBraceWithStaffPair(brace, upper, lower))
    );
    if (score > 0) {
      scored.push({ score, upper: i });
    }
  }
  const taken = new Set<number>();
  const used = new Set<number>();
  for (const { upper } of scored.sort((a, b) => b.score - a.score)) {
    if (!(used.has(upper) || used.has(upper + 1))) {
      taken.add(upper);
      used.add(upper).add(upper + 1);
    }
  }
  return taken;
}

/**
 * MultiStaff.create_grandstaffs: adjacent pairs scored against every brace,
 * taken greedily by score, each taken pair fused by mergeStaffs into one
 * ten-line staff. The connections are kept.
 *
 * Returns `multiStaff` itself when it has one staff or no pair scores. With
 * two staffs and no brace Python's `max()` raises; a multi staff of two was
 * joined by a brace, so that list is never empty, and here it scores nothing.
 */
export function createGrandstaffs(
  multiStaff: MultiStaff,
  braces: readonly RotatedBox[]
): MultiStaff {
  const { staffs } = multiStaff;
  if (staffs.length < 2) {
    return multiStaff;
  }
  const pairs = selectGrandstaffs(staffs, braces);
  if (pairs.size === 0) {
    return multiStaff;
  }
  const merged: Staff[] = [];
  for (let i = 0; i < staffs.length; i += 1) {
    const [staff, below] = [staffs[i], staffs[i + 1]];
    if (staff === undefined) {
      continue;
    }
    if (pairs.has(i) && below !== undefined) {
      merged.push(mergeStaffs(staff, below));
      i += 1;
    } else {
      merged.push(staff);
    }
  }
  return createMultiStaff(merged, multiStaff.connections);
}

/**
 * find_braces_brackets_and_grand_staff_lines.
 *
 * `staffs` must be the list addNotesToStaffs returned, top to bottom, and
 * every Staff in the result is one of those objects or a merge of two.
 */
export function findBracesBracketsAndGrandStaffLines(
  ops: BoxOps,
  staffs: readonly [Staff, ...Staff[]],
  braceDot: readonly RotatedBox[]
): MultiStaff[] {
  const tall = filterForTallElements(braceDot, staffs);
  const result: MultiStaff[] = [];
  for (const [i, staff] of staffs.entries()) {
    let connected = false;
    for (const neighbour of [staffs[i - 1], staffs[i + 1]]) {
      if (neighbour === undefined) {
        continue;
      }
      const connections = connectionsBetweenStaffs(ops, staff, neighbour, tall);
      if (connections.length >= MIN_CONNECTIONS_TO_FORM_COMBINED_STAFF) {
        result.push(createMultiStaff([staff, neighbour], connections));
        connected = true;
      }
    }
    if (!connected) {
      result.push(createMultiStaff([staff]));
    }
  }
  return mergeMultiStaffsSharingAStaff(result).map((multiStaff) =>
    createGrandstaffs(multiStaff, tall)
  );
}

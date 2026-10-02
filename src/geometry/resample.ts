/**
 * A RawStaff has gaps and ragged ends; a Staff knows all five line ordinates
 * every 10 px. Port of resample_staff_segment, resample_staff,
 * resample_staffs, filter_edge_of_vision and sort_staffs_top_to_bottom.
 *
 * No BoxOps: this file is arithmetic on rects.
 */

import { floorDiv, mean, roundHalfEven, truncToInt } from "../image/numeric.js";
import { DetectionError } from "../model/pipeline.js";
import {
  createStaff,
  createStaffPoint,
  type Staff,
  type StaffPoint,
} from "../model/staff.js";
import { getCenterExtrapolated } from "./box-transforms.js";
import { cornersOf } from "./boxes.js";
import type { RawStaff } from "./raw-staffs.js";
import type { StaffAnchor } from "./staff-anchors.js";
import { fragmentAt } from "./staff-lines.js";

/** homr's `staff_density`. */
export const STAFF_DENSITY = 10;

/** Python's `range(start, stop, STAFF_DENSITY)` as a list. */
function densityRange(start: number, stop: number): number[] {
  const xs: number[] = [];
  for (let x = start; x < stop; x += STAFF_DENSITY) {
    xs.push(x);
  }
  return xs;
}

/** The dummy point resample_staff_segment starts from: the anchor's own lines at the anchor symbol's x. */
function anchorPoint(anchor: StaffAnchor): StaffPoint {
  const x = anchor.symbol.rect.cx;
  const firsts = anchor.lines.map((line) => line.fragments[0]);
  return createStaffPoint(
    x,
    firsts.map((fragment) => getCenterExtrapolated(fragment, x)),
    mean(firsts.map((fragment) => fragment.rect.angle))
  );
}

/** Lines closer than half a unit to the one above, and lines that jumped half a unit since the previous point, are forgotten. */
function dropUntrustedCenters(
  centers: (number | null)[],
  previous: StaffPoint,
  unit: number
): void {
  const present = centers.filter((center) => center !== null);
  for (let j = 0; j + 1 < present.length; j += 1) {
    // staff_detection.py:409-414: j counts the lines that were found, and is used as an index into all five.
    if ((present[j + 1] ?? 0) - (present[j] ?? 0) < 0.5 * unit) {
      centers[j] = null;
      centers[j + 1] = null;
    }
  }
  for (const [i, previousY] of previous.y.entries()) {
    const center = centers[i];
    if (
      center !== null &&
      center !== undefined &&
      Math.abs(center - previousY) > 0.5 * unit
    ) {
      centers[i] = null;
    }
  }
}

/** A missing line is placed whole units from a known one: from the nearest above it, and failing that from the nearest below. */
function fillMissingCenters(centers: (number | null)[], unit: number): void {
  const topDown = centers.map((_, i) => i);
  // staff_detection.py:424-431: one prev_center for both passes, so the second starts from the last line the first one found.
  let known = -1;
  for (const i of [...topDown, ...[...topDown].reverse()]) {
    if (centers[i] !== null) {
      known = i;
      continue;
    }
    const anchor = centers[known];
    if (anchor !== null && anchor !== undefined) {
      centers[i] = anchor + unit * (i - known);
    }
  }
}

/**
 * resample_staff_segment, as a list: both callers drain the generator.
 *
 * `xs` is walked in the order given and the continuity reference moves with
 * it, so the left side of an anchor is passed right-to-left and reversed by
 * the caller afterwards.
 */
export function resampleStaffSegment(
  anchor: StaffAnchor,
  staff: RawStaff,
  xs: readonly number[]
): StaffPoint[] {
  const unit = anchor.averageUnitSize;
  const points: StaffPoint[] = [];
  let previous = anchorPoint(anchor);
  for (const x of xs) {
    const found = staff.lines.map((line) => fragmentAt(line, x));
    const centers = found.map((fragment) =>
      fragment === null ? null : getCenterExtrapolated(fragment, x)
    );
    if (centers.every((center) => center === null)) {
      continue;
    }
    dropUntrustedCenters(centers, previous, unit);
    fillMissingCenters(centers, unit);
    const ys = centers.filter((center) => center !== null);
    if (ys.length !== centers.length) {
      continue;
    }
    previous = createStaffPoint(
      x,
      ys,
      mean(found.flatMap((fragment) => fragment?.rect.angle ?? []))
    );
    points.push(previous);
  }
  return points;
}

/** staff_detection.py:445 `int(round(x / 10)) * 10`: half to even, so 125 gives 120. */
function roundToDensity(x: number): number {
  return truncToInt(roundHalfEven(x / STAFF_DENSITY)) * STAFF_DENSITY;
}

/**
 * resample_staff.
 *
 * The grid is returned as built: not sorted, with duplicate x values where two
 * anchors' ranges meet. Staff.minX and maxX are its first and last points.
 * Throws DetectionError("staff-without-points") on an empty grid, where
 * Python's Staff([]) raises IndexError.
 */
export function resampleStaff(staff: RawStaff): Staff {
  const anchors = [...staff.anchors].sort(
    (a, b) => a.symbol.rect.cx - b.symbol.rect.cx
  );
  const { bottomRight, topLeft } = cornersOf(staff.box.rect);
  const stop = (floorDiv(bottomRight.x, STAFF_DENSITY) + 1) * STAFF_DENSITY;
  const grid: StaffPoint[] = [];
  let x = floorDiv(topLeft.x, STAFF_DENSITY) * STAFF_DENSITY;
  for (const [i, anchor] of anchors.entries()) {
    const center = anchor.symbol.rect.cx;
    const next = anchors[i + 1];
    const toLeft = densityRange(roundToDensity(x), roundToDensity(center));
    // staff_detection.py:457-466: between two anchors the right side starts at the un-aligned int(centre), and x becomes the range's stop argument, not its last value.
    const rightStart =
      next === undefined ? roundToDensity(center) : truncToInt(center);
    x =
      next === undefined
        ? roundToDensity(stop)
        : truncToInt((center + next.symbol.rect.cx) / 2);
    grid.push(
      ...resampleStaffSegment(anchor, staff, toLeft.reverse()).reverse(),
      ...resampleStaffSegment(anchor, staff, densityRange(rightStart, x))
    );
  }
  if (grid.length === 0) {
    throw new DetectionError("staff-without-points");
  }
  return createStaff(grid);
}

export function resampleStaffs(staffs: readonly RawStaff[]): Staff[] {
  return staffs.map(resampleStaff);
}

/**
 * filter_edge_of_vision: drops a staff reaching past the top or bottom of the
 * image, and a staff narrower than half the mean width that starts in the
 * first or ends in the last 1 % of the width.
 */
export function filterEdgeOfVision(
  staffs: readonly Staff[],
  image: { readonly height: number; readonly width: number }
): Staff[] {
  const usualWidth = mean(staffs.map((staff) => staff.maxX - staff.minX));
  return staffs.filter((staff) => {
    if (staff.maxY >= image.height || staff.minY < 0) {
      return false;
    }
    const atAnEdge =
      staff.minX < 0.01 * image.width || staff.maxX > 0.99 * image.width;
    return !(atAnEdge && staff.maxX - staff.minX < usualWidth / 2);
  });
}

/** sort_staffs_top_to_bottom: stable, by minY. */
export function sortStaffsTopToBottom(staffs: readonly Staff[]): Staff[] {
  return [...staffs].sort((a, b) => a.minY - b.minY);
}

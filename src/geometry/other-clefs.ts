/**
 * Staffs whose clef segnet missed: project the columns under each found clef
 * onto rows, find five evenly spaced peaks, and offer the span as a symbol to
 * find_staff_anchors. Port of predict_other_anchors_from_clefs, init_zone,
 * find_horizontal_lines and the live part of filter_line_peaks.
 */

import { findPeaks } from "../image/find-peaks.js";
import { mean, roundHalfEven, std, truncToInt } from "../image/numeric.js";
import { type Mask, planeFromBytes, rowNonzeroCounts } from "../image/plane.js";
import { LINES_PER_STAFF } from "../model/constants.js";
import { DetectionError } from "../model/pipeline.js";
import { type BoxOps, overlapsAny } from "./box-ops.js";
import { rotatedBoxFromRect } from "./box-transforms.js";
import {
  cornersOf,
  legacyConventionRectOf,
  pointListFromPairs,
  type RotatedBox,
} from "./boxes.js";
import type { StaffAnchor } from "./staff-anchors.js";

/** A half-open column range, `range(start, stop)`. Empty when stop is not beyond start. */
export interface ColumnZone {
  readonly start: number;
  readonly stop: number;
}

/** Staff lines are expected right of a clef only, so the zone grows on that side alone. */
const ZONE_MARGIN_RIGHT = 10;

/** filter_line_peaks' max_gap_ratio. */
const MAX_GAP_RATIO = 1.5;

/**
 * init_zone. Overlapping zones merge, and the merge takes the *later* zone's
 * stop even when it is smaller, so a nested zone shrinks the one around it.
 */
export function initZones(
  clefAnchors: readonly StaffAnchor[],
  imageWidth: number
): ColumnZone[] {
  const ranges = clefAnchors
    .map((anchor): ColumnZone => {
      const corners = cornersOf(anchor.symbol.rect);
      return {
        start: Math.max(truncToInt(corners.bottomLeft.x), 0),
        stop: Math.min(
          truncToInt(corners.topRight.x + ZONE_MARGIN_RIGHT),
          imageWidth
        ),
      };
    })
    .sort((a, b) => a.start - b.start);
  const zones: ColumnZone[] = [];
  for (const range of ranges) {
    const last = zones.at(-1);
    if (last !== undefined && range.start < last.stop) {
      // staff_detection.py:549 takes r.stop, not the larger of the two stops.
      zones[zones.length - 1] = { start: last.start, stop: range.stop };
    } else {
      zones.push(range);
    }
  }
  return zones;
}

/** `image[:, zone]`: the zone's columns of every row, as a mask of its own. */
export function zoneColumns(image: Mask, zone: ColumnZone): Mask {
  const width = Math.max(zone.stop - zone.start, 0);
  const data = new Uint8Array(width * image.height);
  for (let y = 0; y < image.height; y += 1) {
    const from = y * image.width + zone.start;
    data.set(image.data.subarray(from, from + width), y * width);
  }
  return planeFromBytes("mask", width, image.height, data);
}

/**
 * filter_line_peaks, reduced to the one value its caller reads: the group
 * index of each peak, a new group starting where the gap exceeds 1.5 times the
 * mean of the smallest `max(5, round(0.2 * n))` gaps. The rest of the Python
 * function computes a list the caller discards.
 *
 * One peak gives a NaN gap and the group -1, as in Python.
 * Throws DetectionError("zone-without-lines") on no peaks, where Python raises
 * IndexError on `peaks[0]`.
 */
export function groupLinePeaks(peaks: ArrayLike<number>): number[] {
  const [first, ...rest] = Array.from(peaks);
  if (first === undefined) {
    throw new DetectionError("zone-without-lines");
  }
  const gaps = rest.map(
    (peak, i) => peak - (i === 0 ? first : (rest[i - 1] ?? 0))
  );
  const smallest = gaps
    .sort((a, b) => a - b)
    .slice(0, Math.max(LINES_PER_STAFF, roundHalfEven(peaks.length * 0.2)));
  const maxGap = mean(smallest) * MAX_GAP_RATIO;
  const groups: number[] = [];
  let group = -1;
  let previous = first - maxGap - 1;
  for (const peak of [first, ...rest]) {
    if (peak - previous > maxGap) {
      group += 1;
    }
    groups.push(group);
    previous = peak;
  }
  return groups;
}

/**
 * find_horizontal_lines over the columns of one zone: the row ordinates of
 * every group of exactly five peaks, each ascending. A staff whose projection
 * yields four or six peaks is dropped without a trace.
 *
 * `unitSize` is passed to findPeaks as a float distance.
 */
export function findHorizontalLines(
  columns: Mask,
  unitSize: number
): number[][] {
  const count = [0, ...rowNonzeroCounts(columns), 0];
  // staff_detection.py:618 np.std over a page's worth of rows: numeric.std sums pairwise, as numpy does.
  const average = mean(count);
  const deviation = std(count);
  // A zone with every row alike has deviation 0 and a norm of NaN throughout, which holds no peak.
  const norm = Float64Array.from(count, (n) => (n - average) / deviation);
  const centers = Array.from(
    findPeaks(norm, { distance: unitSize, height: 0, prominence: 1 }),
    (peak) => peak - 1
  );
  const byGroup = new Map<number, number[]>();
  for (const [i, group] of groupLinePeaks(centers).entries()) {
    const members = byGroup.get(group) ?? [];
    members.push(centers[i] ?? 0);
    byGroup.set(group, members);
  }
  return [...byGroup.values()].filter(
    (members) => members.length === LINES_PER_STAFF
  );
}

/**
 * predict_other_anchors_from_clefs. Empty in, empty out. The boxes are
 * `((int(cx), int(cy)), (zoneWidth, int(span)), 0)` with an empty contour, and
 * the ones overlapping any clef anchor's symbol are dropped, which on a page
 * where every staff has its clef is all of them.
 *
 * `staff` is the strengthened staff mask; only its columns are read.
 */
export function predictOtherAnchorsFromClefs(
  ops: BoxOps,
  clefAnchors: readonly StaffAnchor[],
  staff: Mask
): RotatedBox[] {
  if (clefAnchors.length === 0) {
    return [];
  }
  const unit = mean(clefAnchors.map((anchor) => anchor.averageUnitSize));
  const symbols = clefAnchors.map((anchor) => anchor.symbol);
  const result: RotatedBox[] = [];
  for (const zone of initZones(clefAnchors, staff.width)) {
    const width = zone.stop - zone.start;
    for (const rows of findHorizontalLines(zoneColumns(staff, zone), unit)) {
      const top = Math.min(...rows);
      const bottom = Math.max(...rows);
      result.push(
        rotatedBoxFromRect(
          legacyConventionRectOf({
            angle: 0,
            cx: truncToInt(zone.start + width / 2),
            cy: truncToInt((top + bottom) / 2),
            h: truncToInt(bottom - top),
            w: width,
          }),
          pointListFromPairs([]),
          0
        )
      );
    }
  }
  return result.filter((box) => !overlapsAny(ops, box, symbols));
}

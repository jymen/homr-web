/**
 * What a staff line is before there is a staff: fragments of the staff mask
 * chained left to right. Port of staff_detection.StaffLineSegment,
 * connect_staff_lines and the three predicates find_staff_anchors asks of five
 * such lines.
 *
 * Fragments are compared by rect value, as Python's __eq__ and __hash__ do.
 * debugId cannot stand in: break_wide_fragments gives every piece of one
 * fragment the same id (147 distinct ids over the 340 Kesh fragments).
 */

import { mean } from "../image/numeric.js";
import {
  isShortConnectedLine,
  isShortLine,
  MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL,
  maxLineGapSize,
  STAFF_LINE_SEGMENT_X_TOLERANCE,
} from "../model/constants.js";
import type { BoxOps } from "./box-ops.js";
import {
  getCenterExtrapolated,
  isOverlappingExtrapolated,
} from "./box-transforms.js";
import { cornersOf, GeometryError, type RotatedBox, rectKey } from "./boxes.js";

export interface StaffLineSegment {
  /**
   * Ascending by centre x, ties in input order. Never empty: Python takes
   * min() over it in the constructor.
   */
  readonly fragments: readonly [RotatedBox, ...RotatedBox[]];
  /**
   * rectKey of every fragment. Stored because the subset test
   * `set(a.fragments) <= set(b.fragments)` runs per anchor, per connected
   * line, per staff line, and rebuilding five-float strings each time is the
   * only cost in it. Written by createStaffLineSegment alone.
   */
  readonly keys: ReadonlySet<string>;
  /** max over fragments of `cx + w / 2`. */
  readonly maxX: number;
  /** max over fragments of `cy + h / 2`. */
  readonly maxY: number;
  readonly minX: number;
  readonly minY: number;
}

/** Exactly five lines, top to bottom: what an anchor and a raw staff hold. */
export type FiveLines = readonly [
  StaffLineSegment,
  StaffLineSegment,
  StaffLineSegment,
  StaffLineSegment,
  StaffLineSegment,
];

/** StaffLineSegment(debug_id, fragments), without the id only homr's drawing reads. Throws GeometryError on an empty list. */
export function createStaffLineSegment(
  fragments: readonly RotatedBox[]
): StaffLineSegment {
  const [first, ...rest] = [...fragments].sort((a, b) => a.rect.cx - b.rect.cx);
  if (first === undefined) {
    throw new GeometryError("a staff line needs at least one fragment");
  }
  const extents = fragments.map((fragment) => cornersOf(fragment.rect));
  return {
    fragments: [first, ...rest],
    keys: new Set(fragments.map((fragment) => rectKey(fragment.rect))),
    maxX: Math.max(...extents.map((one) => one.bottomRight.x)),
    maxY: Math.max(...extents.map((one) => one.bottomRight.y)),
    minX: Math.min(...extents.map((one) => one.topLeft.x)),
    minY: Math.min(...extents.map((one) => one.topLeft.y)),
  };
}

/** Narrows a list of exactly five; null otherwise. */
export function asFiveLines(
  lines: readonly StaffLineSegment[]
): FiveLines | null {
  const [a, b, c, d, e, ...rest] = lines;
  if (
    a === undefined ||
    b === undefined ||
    c === undefined ||
    d === undefined ||
    e === undefined ||
    rest.length > 0
  ) {
    return null;
  }
  return [a, b, c, d, e];
}

/** One line out per line in, which a plain `map` cannot promise the compiler. */
export function mapFiveLines(
  lines: FiveLines,
  change: (line: StaffLineSegment, index: 0 | 1 | 2 | 3 | 4) => StaffLineSegment
): FiveLines {
  const [a, b, c, d, e] = lines;
  return [change(a, 0), change(b, 1), change(c, 2), change(d, 3), change(e, 4)];
}

/**
 * StaffLineSegment.merge: self's fragments, then each of other's whose rect is
 * not already present, re-sorted.
 */
export function mergeSegments(
  self: StaffLineSegment,
  other: StaffLineSegment
): StaffLineSegment {
  const merged: RotatedBox[] = [...self.fragments];
  const present = new Set(self.keys);
  for (const fragment of other.fragments) {
    const key = rectKey(fragment.rect);
    if (!present.has(key)) {
      present.add(key);
      merged.push(fragment);
    }
  }
  return createStaffLineSegment(merged);
}

/**
 * StaffLineSegment.get_at: the first fragment, in stored order, whose extent
 * widened by STAFF_LINE_SEGMENT_X_TOLERANCE holds x. Fragments of one line
 * can overlap in x, so "first" is part of the answer.
 */
export function fragmentAt(
  line: StaffLineSegment,
  x: number
): RotatedBox | null {
  for (const fragment of line.fragments) {
    const { cx, w } = fragment.rect;
    if (
      x >= cx - w / 2 - STAFF_LINE_SEGMENT_X_TOLERANCE &&
      x <= cx + w / 2 + STAFF_LINE_SEGMENT_X_TOLERANCE
    ) {
      return fragment;
    }
  }
  return null;
}

/** `set(required.fragments).issubset(set(line.fragments))`. */
export function holdsAllFragmentsOf(
  line: StaffLineSegment,
  required: StaffLineSegment
): boolean {
  for (const key of required.keys) {
    if (!line.keys.has(key)) {
      return false;
    }
  }
  return true;
}

/** StaffLineSegment.is_overlapping: any fragment pair overlaps. */
export function segmentsOverlap(
  ops: BoxOps,
  a: StaffLineSegment,
  b: StaffLineSegment
): boolean {
  return a.fragments.some((fragment) =>
    b.fragments.some((other) => ops.overlaps(fragment, other))
  );
}

/**
 * connect_staff_lines. `unitSize` is an int from find_staff_anchors and a
 * float from find_raw_staffs_by_connecting_line_fragments; both are passed
 * through as they come.
 */
export function connectStaffLines(
  fragments: readonly RotatedBox[],
  unitSize: number
): StaffLineSegment[] {
  const leftOf = (box: RotatedBox): number => cornersOf(box.rect).bottomLeft.x;
  // staff_detection.py:248 sorts with reverse=True, which keeps ties in input order, and line 253 pops from the end: left to right, and among equal x the last given first.
  const queue = [...fragments].sort((a, b) => leftOf(b) - leftOf(a));
  const maxGap = maxLineGapSize(unitSize);
  const chains: RotatedBox[][] = [];
  let active: RotatedBox[][] = [];
  let lastCleanupAtX = 0;
  for (
    let current = queue.pop();
    current !== undefined;
    current = queue.pop()
  ) {
    const x = leftOf(current);
    if (x - lastCleanupAtX > maxGap) {
      active = active.filter((chain) => {
        const last = chain.at(-1);
        return (
          last !== undefined && x - cornersOf(last.rect).bottomRight.x < maxGap
        );
      });
      lastCleanupAtX = x;
    }
    if (current.rect.w < isShortLine(unitSize)) {
      continue;
    }
    let connected = false;
    // staff_detection.py:272-275 has no break: one fragment joins every active chain it extends.
    for (const chain of active) {
      const last = chain.at(-1);
      if (
        last !== undefined &&
        isOverlappingExtrapolated(last, current, unitSize)
      ) {
        chain.push(current);
        connected = true;
      }
    }
    if (!connected) {
      const chain = [current];
      chains.push(chain);
      active.push(chain);
    }
  }
  return chains
    .sort((a, b) => (a[0]?.rect.cy ?? 0) - (b[0]?.rect.cy ?? 0))
    .map(createStaffLineSegment);
}

/** are_lines_crossing: any pair of lines overlaps. */
export function areLinesCrossing(
  ops: BoxOps,
  lines: readonly StaffLineSegment[]
): boolean {
  return lines.some((line, i) =>
    lines.slice(i + 1).some((other) => segmentsOverlap(ops, line, other))
  );
}

/**
 * are_lines_parallel: no fragment wider than `2 * unitSize` is further than
 * MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL degrees from the mean angle of them all.
 * The mean is np.mean over every fragment of every line, 8 or more values on
 * 18 of the 120 Kesh calls, so numeric.mean's pairwise order is load-bearing.
 */
export function areLinesParallel(
  lines: readonly StaffLineSegment[],
  unitSize: number
): boolean {
  const fragments = lines.flatMap((line) => line.fragments);
  if (fragments.length === 0) {
    return false;
  }
  const average = mean(fragments.map((fragment) => fragment.rect.angle));
  return !fragments.some(
    (fragment) =>
      Math.abs(fragment.rect.angle - average) >
        MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL &&
      fragment.rect.w > isShortConnectedLine(unitSize)
  );
}

/**
 * begins_or_ends_on_one_staff_line. Despite the name: the symbol's centre is
 * within one unit of *any* of the lines at its x. True for almost every
 * centred bar line; reproduced as written.
 */
export function beginsOrEndsOnOneStaffLine(
  symbol: RotatedBox,
  lines: readonly StaffLineSegment[],
  unitSize: number
): boolean {
  return lines.some((line) => {
    const fragment = fragmentAt(line, symbol.rect.cx);
    return (
      fragment !== null &&
      Math.abs(
        getCenterExtrapolated(fragment, symbol.rect.cx) - symbol.rect.cy
      ) < unitSize
    );
  });
}

/**
 * Anchors: a clef or a bar line crossing exactly five parallel staff lines.
 * Port of staff_detection.StaffAnchor, find_staff_anchors and
 * filter_unusual_anchors.
 */

import { mean, roundHalfEven, std, truncToInt } from "../image/numeric.js";
import { isShortConnectedLine, LINES_PER_STAFF } from "../model/constants.js";
import type { BoxOps } from "./box-ops.js";
import {
  getCenterExtrapolated,
  makeBoxTaller,
  moveToXHorizontalBy,
} from "./box-transforms.js";
import type { RotatedBox } from "./boxes.js";
import {
  areLinesCrossing,
  areLinesParallel,
  asFiveLines,
  beginsOrEndsOnOneStaffLine,
  connectStaffLines,
  type FiveLines,
} from "./staff-lines.js";

/**
 * StaffAnchor's own `max_number_of_ledger_lines = 5`, a local in its
 * constructor. constants.py says 4 and Staff._y_tolerance uses that one.
 */
export const ANCHOR_ZONE_LEDGER_LINES = 5;

/** Python's `range(start, stop)`; `stop` is read *inclusively* by find_raw_staffs. */
export interface AnchorZone {
  readonly start: number;
  readonly stop: number;
}

export interface StaffAnchor {
  /** Mean of the four gaps between the lines at the symbol's x. A float; never the rounded estimate find_staff_anchors searched with. */
  readonly averageUnitSize: number;
  readonly lines: FiveLines;
  readonly maxY: number;
  readonly minY: number;
  /**
   * The **shifted copy** that found the lines, not the clef or bar line it was
   * copied from. Resampling starts at this x.
   */
  readonly symbol: RotatedBox;
  readonly zone: AnchorZone;
}

/**
 * StaffAnchor(staff_lines, symbol). The line ordinates come from each line's
 * *first* fragment extrapolated to the symbol's x. Zone bounds are int(), so
 * truncated toward zero, not floored.
 */
export function createStaffAnchor(
  lines: FiveLines,
  symbol: RotatedBox
): StaffAnchor {
  const ys = lines
    .map((line) => getCenterExtrapolated(line.fragments[0], symbol.rect.cx))
    .sort((a, b) => a - b);
  const gaps = ys.slice(1).map((y, i) => Math.abs(y - (ys[i] ?? y)));
  const averageUnitSize = mean(gaps);
  const maxY = Math.max(...lines.map((line) => line.maxY));
  const minY = Math.min(...lines.map((line) => line.minY));
  const reach = ANCHOR_ZONE_LEDGER_LINES * averageUnitSize;
  return {
    averageUnitSize,
    lines,
    maxY,
    minY,
    symbol,
    zone: { start: truncToInt(minY - reach), stop: truncToInt(maxY + reach) },
  };
}

/** Replaces find_staff_anchors' `are_clefs` flag. */
export type AnchorSymbolKind = "barLine" | "clef";

/**
 * The x shifts tried around each symbol, in homr's order. A clef interrupts
 * the lines, so they are searched for beside it and mostly to its right.
 */
export const ANCHOR_SHIFTS = {
  barLine: [-10, -5, 0, 5, 10],
  clef: [-10, 0, 10, 30, 60, 80],
} as const satisfies Readonly<Record<AnchorSymbolKind, readonly number[]>>;

function anchorAt(
  ops: BoxOps,
  fragments: readonly RotatedBox[],
  symbol: RotatedBox,
  kind: AnchorSymbolKind
): StaffAnchor | null {
  // staff_detection.py:363 is Python's round(), half to even, and `size[1] / 4` is k + 0.5 on ordinary pages.
  const unit = roundHalfEven(symbol.rect.h / (LINES_PER_STAFF - 1));
  const probe = makeBoxTaller(symbol, unit);
  let lines = connectStaffLines(
    fragments.filter((fragment) => ops.intersects(fragment, probe)),
    unit
  );
  if (lines.length > LINES_PER_STAFF) {
    lines = lines.filter(
      (line) => line.maxX - line.minX > isShortConnectedLine(unit)
    );
  }
  const five = asFiveLines(lines);
  if (
    five === null ||
    !areLinesParallel(five, unit) ||
    areLinesCrossing(ops, five) ||
    (kind === "barLine" && !beginsOrEndsOnOneStaffLine(symbol, five, unit))
  ) {
    return null;
  }
  return createStaffAnchor(five, symbol);
}

/**
 * find_staff_anchors. One symbol yields up to six anchors (clef) or five (bar
 * line); they are not deduplicated here. A shift of 0 is the symbol itself.
 */
export function findStaffAnchors(
  ops: BoxOps,
  fragments: readonly RotatedBox[],
  symbols: readonly RotatedBox[],
  kind: AnchorSymbolKind
): StaffAnchor[] {
  const result: StaffAnchor[] = [];
  for (const symbol of symbols) {
    for (const shift of ANCHOR_SHIFTS[kind]) {
      const anchor = anchorAt(
        ops,
        fragments,
        shift === 0 ? symbol : moveToXHorizontalBy(symbol, shift),
        kind
      );
      if (anchor !== null) {
        result.push(anchor);
      }
    }
  }
  return result;
}

/**
 * filter_unusual_anchors: drops anchors whose unit size is more than three
 * population standard deviations from the mean. A deviation of 0 or NaN keeps
 * every anchor. np.mean and np.std over one value per anchor, more than a
 * hundred on a page, so numeric's pairwise order is the one that matches.
 */
export function filterUnusualAnchors(
  anchors: readonly StaffAnchor[]
): StaffAnchor[] {
  const units = anchors.map((anchor) => anchor.averageUnitSize);
  const average = mean(units);
  const deviation = std(units);
  return anchors.filter(
    (anchor) => !(Math.abs(anchor.averageUnitSize - average) > 3 * deviation)
  );
}

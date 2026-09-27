/**
 * Port of the staff half of homr's model.py: StaffPoint, Staff, MultiStaff.
 * Fields only; the methods (get_at, is_on_staff_zone, merge,
 * transform_coordinates, extend_to_x_range, find_position_in_unit_sizes)
 * arrive in phase 5 as free functions in this file.
 *
 * homr computes several fields in constructors from other fields
 * (Staff.min_x from the grid, average_unit_size as a median). They are
 * kept as stored fields, filled by the factories below and never written
 * afterwards, because they are read in the hot loops of note and brace
 * detection and a median is not something to recompute per read. The
 * factories are the single writer; the golden decoder recomputes each one
 * and asserts it matches what Python stored, which is also the test that
 * the port's median and mean are numpy's.
 *
 * Coordinate space. A staff exists in two spaces during one page: page
 * pixels for detection, and the 1280 x 256 encoder canvas after phase 6's
 * dewarp (canvas-<n>-staff.json). The two are structurally identical, so
 * `space` is the field that tells them apart and lets a stage signature
 * refuse the wrong one.
 */

import type { RotatedBox } from "../geometry/boxes.js";
import { mean, median } from "../image/numeric.js";
import { LINES_PER_STAFF, MAX_LEDGER_LINES } from "./constants.js";
import type { SymbolOnStaff } from "./symbols.js";

export class ModelError extends Error {}

export const COORDINATE_SPACES = { canvas: "canvas", page: "page" } as const;
export type CoordinateSpace =
  (typeof COORDINATE_SPACES)[keyof typeof COORDINATE_SPACES];

/**
 * The y of each staff line at one x, top line first, ascending. Five for a
 * staff, ten for a grand staff merged from two (StaffPoint.merge sorts the
 * union). A union of two tuple types rather than number[] so that y[0]
 * through y[4], which every staff routine reads, are numbers under
 * noUncheckedIndexedAccess; the tenth line is reached through lastLineY.
 * (Python accepts any multiple of five; nothing in 0.7.0 makes more than
 * ten.)
 */
export type StaffLineYs =
  | readonly [number, number, number, number, number]
  | readonly [
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
      number,
    ];

export interface StaffPoint {
  /** Degrees; the mean angle of the line fragments this point was resampled from. */
  readonly angle: number;
  /** `np.mean(np.diff(y))`: the local line spacing. */
  readonly averageUnitSize: number;
  readonly x: number;
  readonly y: StaffLineYs;
}

/** StaffPoint(x, y, angle): throws unless y.length is 5 or 10; computes averageUnitSize. */
export function createStaffPoint(
  x: number,
  y: readonly number[],
  angle: number
): StaffPoint {
  if (y.length !== LINES_PER_STAFF && y.length !== 2 * LINES_PER_STAFF) {
    throw new ModelError(
      `a staff point needs 5 or 10 line ordinates, got ${y.length}`
    );
  }
  const diffs: number[] = [];
  for (let i = 1; i < y.length; i += 1) {
    diffs.push((y[i] ?? 0) - (y[i - 1] ?? 0));
  }
  return {
    angle,
    averageUnitSize: mean(diffs),
    x,
    y: [...y] as unknown as StaffLineYs,
  };
}

export function lineCount(point: StaffPoint): 5 | 10 {
  return point.y.length === LINES_PER_STAFF ? 5 : 10;
}

/** `point.y[-1]`. */
export function lastLineY(point: StaffPoint): number {
  return point.y.at(-1) ?? point.y[4];
}

export interface Staff {
  /** `np.median([p.average_unit_size for p in grid])`, numpy's even-count median. */
  readonly averageUnitSize: number;
  /**
   * Resampled points, roughly ascending in x: resample_staff_segment yields
   * a few out-of-order neighbours (the Kesh page has 700 after 701) and
   * only Staff.merge sorts. Never empty: Staff.__init__ reads grid[0] and
   * grid[-1] unconditionally, so an empty grid is not a Staff.
   */
  readonly grid: readonly [StaffPoint, ...StaffPoint[]];
  /** True only for the result of Staff.merge (a brace-joined pair). */
  readonly isGrandstaff: boolean;
  /** grid[-1].x. */
  readonly maxX: number;
  /** max over all points of max(point.y). */
  readonly maxY: number;
  /** grid[0].x, exactly (not the minimum over the grid). */
  readonly minX: number;
  /** min over all points of min(point.y). */
  readonly minY: number;
  readonly space: CoordinateSpace;
  /**
   * Detection order; add_symbol appends. The one mutable field on a staff:
   * notes are attached after the staff exists and homr never rebuilds the
   * staff to do it. Empty in staffs.json, filled in multistaffs.json.
   */
  readonly symbols: SymbolOnStaff[];
}

export interface StaffOptions {
  readonly isGrandstaff?: boolean;
  readonly space?: CoordinateSpace;
  readonly symbols?: SymbolOnStaff[];
}

/**
 * Staff(grid), plus the fields homr sets after construction. Throws on an
 * empty grid. `symbols` is adopted, not copied: the caller hands over the
 * array and the staff owns it from then on. Defaults to page space.
 */
export function createStaff(
  grid: readonly StaffPoint[],
  options: StaffOptions = {}
): Staff {
  const [first, ...rest] = grid;
  if (first === undefined) {
    throw new ModelError("a staff needs at least one grid point");
  }
  const last = rest.at(-1) ?? first;
  let minY = Number.POSITIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  for (const point of grid) {
    minY = Math.min(minY, ...point.y);
    maxY = Math.max(maxY, ...point.y);
  }
  return {
    averageUnitSize: median(grid.map((p) => p.averageUnitSize)),
    grid: [first, ...rest],
    isGrandstaff: options.isGrandstaff ?? false,
    maxX: last.x,
    maxY,
    minX: first.x,
    minY,
    space: options.space ?? "page",
    symbols: options.symbols ?? [],
  };
}

/**
 * homr's Staff._y_tolerance, MAX_LEDGER_LINES * averageUnitSize. A
 * constant times a stored field, so it is derived on demand rather than
 * stored; the golden decoder checks the stored Python value against it.
 */
export function yTolerance(staff: Staff): number {
  return MAX_LEDGER_LINES * staff.averageUnitSize;
}

/**
 * A grand staff, or a system of staffs playing together. `staffs` is
 * sorted by minY (MultiStaff.__init__ sorts) and never empty; a MultiStaff
 * of one staff is how a plain staff travels through parsing.
 */
export interface MultiStaff {
  /** The brace, bracket or bar-line boxes that joined the staffs; empty for a lone staff. */
  readonly connections: readonly RotatedBox[];
  readonly staffs: readonly [Staff, ...Staff[]];
}

export function createMultiStaff(
  staffs: readonly Staff[],
  connections: readonly RotatedBox[] = []
): MultiStaff {
  const sorted = [...staffs].sort((a, b) => a.minY - b.minY);
  const [first, ...rest] = sorted;
  if (first === undefined) {
    throw new ModelError("a multi staff needs at least one staff");
  }
  return { connections, staffs: [first, ...rest] };
}

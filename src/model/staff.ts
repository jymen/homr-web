/**
 * Port of the staff half of homr's model.py: StaffPoint, Staff, MultiStaff,
 * with their methods as free functions. transform_coordinates and
 * extend_to_x_range belong to the dewarp and are not here yet.
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

import {
  type AnyBox,
  type AxisBox,
  centerOf,
  createAxisBox,
  type Point,
  pointListFromPairs,
  type RotatedBox,
  sameRect,
} from "../geometry/boxes.js";
import {
  argmin,
  mean,
  median,
  roundHalfEven,
  truncToInt,
} from "../image/numeric.js";
import {
  LINES_PER_STAFF,
  MAX_LEDGER_LINES,
  STAFF_POSITION_TOLERANCE,
} from "./constants.js";
import { DetectionError } from "./pipeline.js";
import type { SymbolOnStaff } from "./symbols.js";

/** A staff or multi-staff that cannot exist. Named for what it is about, as PlaneError, GoldenError and VocabularyError are; phase 2's src/models/ owns the separate ModelError. */
export class StaffError extends Error {}

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
    throw new StaffError(
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
   * Detection order. Empty in staffs.json, filled in multistaffs.json. homr
   * appends to this list in place (add_symbol); the port builds a second
   * staff with withSymbols, so no field of a Staff is ever written twice.
   */
  readonly symbols: readonly SymbolOnStaff[];
}

export interface StaffOptions {
  readonly isGrandstaff?: boolean;
  readonly space?: CoordinateSpace;
  readonly symbols?: readonly SymbolOnStaff[];
}

/**
 * Staff(grid), plus the fields homr sets after construction. Throws a
 * StaffError on an empty grid, so a caller standing where Python's
 * Staff([]) raises IndexError checks first and throws its DetectionError.
 * Defaults to page space.
 */
export function createStaff(
  grid: readonly StaffPoint[],
  options: StaffOptions = {}
): Staff {
  const [first, ...rest] = grid;
  if (first === undefined) {
    throw new StaffError("a staff needs at least one grid point");
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
    throw new StaffError("a multi staff needs at least one staff");
  }
  return { connections, staffs: [first, ...rest] };
}

/**
 * Staff.get_at: the grid point nearest `x`, or null when even that one is
 * more than STAFF_POSITION_TOLERANCE away.
 *
 * The first minimum in grid order wins, and the grid is neither sorted nor
 * free of duplicate x values, so this is a scan with a strict `<` and must
 * never become a binary search.
 */
export function staffPointAt(staff: Staff, x: number): StaffPoint | null {
  let [closest] = staff.grid;
  let distance = Math.abs(closest.x - x);
  for (const point of staff.grid) {
    const candidate = Math.abs(point.x - x);
    if (candidate < distance) {
      closest = point;
      distance = candidate;
    }
  }
  return distance > STAFF_POSITION_TOLERANCE ? null : closest;
}

/** What Staff.y_distance_to answers where the staff has no point near x: "something large to mimic infinity". */
const NO_STAFF_AT_X = 1e10;

/** Staff.y_distance_to: the distance from `point` to the nearest line of the staff at its x. */
export function yDistanceTo(staff: Staff, point: Point): number {
  const staffPoint = staffPointAt(staff, point.x);
  if (staffPoint === null) {
    return NO_STAFF_AT_X;
  }
  return Math.min(...staffPoint.y.map((y) => Math.abs(y - point.y)));
}

/** Staff.is_on_staff_zone: within the ledger-line tolerance of the outer lines, both ends inclusive. */
export function isOnStaffZone(staff: Staff, item: AnyBox): boolean {
  const center = centerOf(item);
  const point = staffPointAt(staff, center.x);
  if (point === null) {
    return false;
  }
  const tolerance = yTolerance(staff);
  return !(
    center.y > lastLineY(point) + tolerance || center.y < point.y[0] - tolerance
  );
}

/**
 * StaffPoint.find_position_in_unit_sizes:
 * `2 * (len(y) - idx) + round(2 * distance / unit) - 1`, from the first
 * nearest line and with Python's round, so a centre midway between a line
 * and a space goes to the even side.
 */
export function findPositionInUnitSizes(
  point: StaffPoint,
  box: AnyBox
): number {
  const centerY = centerOf(box).y;
  const nearest = argmin(point.y.map((y) => Math.abs(y - centerY)));
  const distance = (point.y[nearest] ?? Number.NaN) - centerY;
  return (
    2 * (point.y.length - nearest) +
    roundHalfEven((2 * distance) / point.averageUnitSize) -
    1
  );
}

/** The debug id homr gives the box of a staff point. */
const STAFF_POINT_BOX_DEBUG_ID = -2;

/**
 * StaffPoint.to_bounding_box: a zero-width box at `int(x)` from `int(y[0])`
 * to `int(y[-1])`, with an empty contour.
 */
export function staffPointToAxisBox(point: StaffPoint): AxisBox {
  const x = truncToInt(point.x);
  return createAxisBox(
    x,
    truncToInt(point.y[0]),
    x,
    truncToInt(lastLineY(point)),
    pointListFromPairs([]),
    STAFF_POINT_BOX_DEBUG_ID
  );
}

/** How far apart in x StaffPoint.merge lets two points be. */
const MERGE_X_TOLERANCE = 1e-3;

/**
 * StaffPoint.merge: the lines of both at the first point's x, sorted, and the
 * mean of the two angles. Throws a StaffError where Python raises ValueError.
 */
export function mergeStaffPoints(a: StaffPoint, b: StaffPoint): StaffPoint {
  if (Math.abs(a.x - b.x) > MERGE_X_TOLERANCE) {
    throw new StaffError(
      `cannot merge staff points at different positions: x ${a.x} and ${b.x}`
    );
  }
  return createStaffPoint(
    a.x,
    [...a.y, ...b.y].sort((first, second) => first - second),
    (a.angle + b.angle) / 2
  );
}

/** `{int(round(p.x)): p for p in grid}`: a later point at the same key replaces the earlier one. */
function pointsByRoundedX(staff: Staff): Map<number, StaffPoint> {
  const points = new Map<number, StaffPoint>();
  for (const point of staff.grid) {
    points.set(truncToInt(roundHalfEven(point.x)), point);
  }
  return points;
}

/**
 * Staff.merge: the grand staff of two staffs, over the x positions both
 * have, ascending, with the symbols of `self` before those of `other`.
 *
 * Throws DetectionError `staff-without-points` when the two share no x,
 * where Python's Staff([]) raises IndexError.
 */
export function mergeStaffs(self: Staff, other: Staff): Staff {
  const others = pointsByRoundedX(other);
  const shared: [number, StaffPoint, StaffPoint][] = [];
  for (const [key, point] of pointsByRoundedX(self)) {
    const match = others.get(key);
    if (match !== undefined) {
      shared.push([key, point, match]);
    }
  }
  if (shared.length === 0) {
    throw new DetectionError("staff-without-points");
  }
  shared.sort(([first], [second]) => first - second);
  return createStaff(
    shared.map(([, point, match]) => mergeStaffPoints(point, match)),
    {
      isGrandstaff: true,
      space: self.space,
      symbols: [...self.symbols, ...other.symbols],
    }
  );
}

/**
 * The same staff carrying `symbols`, in place of homr's add_symbol. Not
 * createStaff: nothing derived changes.
 */
export function withSymbols(
  staff: Staff,
  symbols: readonly SymbolOnStaff[]
): Staff {
  return { ...staff, symbols };
}

/**
 * MultiStaff.merge: staffs deduplicated by identity (Staff has no __eq__),
 * connections by the value of their rect, first seen first, then the staffs
 * sorted by minY.
 */
export function mergeMultiStaffs(
  self: MultiStaff,
  other: MultiStaff
): MultiStaff {
  const staffs: Staff[] = [];
  for (const staff of [...self.staffs, ...other.staffs]) {
    if (!staffs.includes(staff)) {
      staffs.push(staff);
    }
  }
  const connections: RotatedBox[] = [];
  for (const connection of [...self.connections, ...other.connections]) {
    if (!connections.some((kept) => sameRect(kept.rect, connection.rect))) {
      connections.push(connection);
    }
  }
  return createMultiStaff(staffs, connections);
}

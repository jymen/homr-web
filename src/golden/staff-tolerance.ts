/**
 * What "agrees with Python" means from the staffs to the multi-staffs.
 * Functions returning reports, as box-tolerance.ts does for box lists, so a
 * test and the bench page print the same numbers.
 *
 * Structure is exact everywhere: counts, order, every grid x, every note
 * position, every stem direction. Only derived floats carry a bound, and
 * there are two bounds because there are two kinds of test.
 */

import { type AngledBox, type Point, sameRect } from "../geometry/boxes.js";
import type { RawStaff } from "../geometry/raw-staffs.js";
import type { StaffAnchor } from "../geometry/staff-anchors.js";
import type { FiveLines } from "../geometry/staff-lines.js";
import type { MultiStaff, Staff, StaffPoint } from "../model/staff.js";
import {
  type Note,
  type NoteheadWithStem,
  type Stem,
  symbolsOfKind,
} from "../model/symbols.js";
import {
  BOX_RECT_TOLERANCE,
  type BoxListComparison,
  compareBoxLists,
} from "./box-tolerance.js";

export interface StaffFloatTolerance {
  /** Degrees, on StaffPoint.angle. */
  readonly angle: number;
  /** Pixels, on every line ordinate and on every unit size derived from them. */
  readonly y: number;
}

/**
 * `fromPythonInputs`: the stage under test read Python's boxes, whose rects
 * are float32 values stored exactly, and everything after is float64 in both
 * languages. What is left is Math.tan against libm's tan, one ulp, times at
 * most the 100 px a fragment spans: below 1e-10 on a 2716 px page. Measured
 * 2026-10-02 from Python's fragments, clefs and bar lines to staffs.json: on
 * the grand-staff page the worst ordinate is 5.7e-14 off and the worst unit
 * size 1.4e-14, every angle is exact, and the Kesh page is exact throughout.
 *
 * `fromOwnBoxes`: the stage read boxes the port fitted itself, which differ
 * from Python's by up to 2.4e-4 in centre and 7.7e-7 degrees in angle
 * (BOX_RECT_TOLERANCE's measurement), and a line ordinate inherits both.
 */
export const STAFF_TOLERANCES = {
  fromOwnBoxes: { angle: 1e-3, y: 3e-4 },
  fromPythonInputs: { angle: 1e-9, y: 1e-9 },
} as const satisfies Readonly<Record<string, StaffFloatTolerance>>;

export class StaffGoldenMismatch extends Error {}

/** The largest delta of one quantity, where it is, and how many values broke the bound. */
interface Worst {
  at: string;
  delta: number;
  got: number;
  over: number;
  want: number;
}

const noWorst = (): Worst => ({ at: "", delta: 0, got: 0, over: 0, want: 0 });

function track(
  worst: Worst,
  at: string,
  got: number,
  want: number,
  bound: number
): void {
  const delta = Math.abs(got - want);
  // Written so that a NaN on either side counts as over the bound.
  const over = !(delta <= bound);
  if (over) {
    worst.over += 1;
  }
  if ((over && worst.over === 1) || delta > worst.delta) {
    worst.at = at;
    worst.delta = delta;
    worst.got = got;
    worst.want = want;
  }
}

/** One line for a whole staff's worth of one quantity, naming the worst value. */
function boundFailure(worst: Worst, bound: number): string[] {
  if (worst.over === 0) {
    return [];
  }
  const others =
    worst.over > 1
      ? `; ${worst.over - 1} more of this staff are outside it`
      : "";
  return [
    `${worst.at}: ${worst.got}, Python ${worst.want}, off by ${worst.delta.toExponential(2)} (bound ${bound})${others}`,
  ];
}

export interface StaffListComparison {
  readonly count: { readonly got: number; readonly want: number };
  /** One line per violation. Empty is the pass condition. */
  readonly failures: readonly string[];
  readonly label: string;
  /** Per staff, grid points got against want; equal on a pass. */
  readonly points: ReadonlyArray<{
    readonly got: number;
    readonly want: number;
  }>;
  readonly worstAngle: number;
  readonly worstUnit: number;
  readonly worstY: number;
  /** Grid points whose x differs. Zero on a pass; one failure line per staff, so a shifted grid reads as one fact. */
  readonly xMismatches: number;
}

interface StaffComparison {
  readonly angle: Worst;
  readonly failures: string[];
  /** The first grid point whose x differs, as a failure line. */
  firstX?: string;
  readonly unit: Worst;
  xMismatches: number;
  readonly y: Worst;
}

function comparePoint(
  at: string,
  got: StaffPoint,
  want: StaffPoint,
  tolerance: StaffFloatTolerance,
  into: StaffComparison
): void {
  if (got.x !== want.x) {
    into.xMismatches += 1;
    into.firstX ??= `${at}: x ${got.x}, Python ${want.x}`;
  }
  if (got.y.length !== want.y.length) {
    into.failures.push(
      `${at}.y: ${got.y.length} lines, Python ${want.y.length}`
    );
    return;
  }
  for (const [line, y] of want.y.entries()) {
    track(
      into.y,
      `${at}.y[${line}]`,
      got.y[line] ?? Number.NaN,
      y,
      tolerance.y
    );
  }
  track(into.angle, `${at}.angle`, got.angle, want.angle, tolerance.angle);
  track(
    into.unit,
    `${at}.averageUnitSize`,
    got.averageUnitSize,
    want.averageUnitSize,
    tolerance.y
  );
}

function compareStaff(
  at: string,
  got: Staff,
  want: Staff,
  tolerance: StaffFloatTolerance
): StaffComparison {
  const into: StaffComparison = {
    angle: noWorst(),
    failures: [],
    unit: noWorst(),
    xMismatches: 0,
    y: noWorst(),
  };
  for (const field of ["isGrandstaff", "space", "minX", "maxX"] as const) {
    if (got[field] !== want[field]) {
      into.failures.push(
        `${at}.${field}: ${got[field]}, Python ${want[field]}`
      );
    }
  }
  if (got.grid.length !== want.grid.length) {
    into.failures.push(
      `${at}.grid: ${got.grid.length} points, Python ${want.grid.length}`
    );
    return into;
  }
  for (const [i, point] of want.grid.entries()) {
    comparePoint(
      `${at}.grid[${i}]`,
      got.grid[i] ?? point,
      point,
      tolerance,
      into
    );
  }
  if (into.firstX !== undefined) {
    into.failures.push(
      `${into.firstX}; ${into.xMismatches} of ${want.grid.length} points of this staff differ in x`
    );
  }
  track(into.y, `${at}.minY`, got.minY, want.minY, tolerance.y);
  track(into.y, `${at}.maxY`, got.maxY, want.maxY, tolerance.y);
  track(
    into.unit,
    `${at}.averageUnitSize`,
    got.averageUnitSize,
    want.averageUnitSize,
    tolerance.y
  );
  into.failures.push(
    ...boundFailure(into.y, tolerance.y),
    ...boundFailure(into.angle, tolerance.angle),
    ...boundFailure(into.unit, tolerance.y)
  );
  return into;
}

/**
 * Staff by staff, point by point: x, line count, isGrandstaff, space, minX
 * and maxX exact; y, angle and unit size within `tolerance`. Symbols are not
 * compared here. Never throws.
 */
export function compareStaffLists(
  label: string,
  got: readonly Staff[],
  want: readonly Staff[],
  tolerance: StaffFloatTolerance
): StaffListComparison {
  const count = { got: got.length, want: want.length };
  if (got.length !== want.length) {
    return {
      count,
      failures: [`${label}: ${got.length} staffs, Python found ${want.length}`],
      label,
      points: [],
      worstAngle: 0,
      worstUnit: 0,
      worstY: 0,
      xMismatches: 0,
    };
  }
  const staffs = want.map((expected, i) =>
    compareStaff(`${label}[${i}]`, got[i] ?? expected, expected, tolerance)
  );
  return {
    count,
    failures: staffs.flatMap((one) => one.failures),
    label,
    points: want.map((expected, i) => ({
      got: got[i]?.grid.length ?? 0,
      want: expected.grid.length,
    })),
    worstAngle: Math.max(0, ...staffs.map((one) => one.angle.delta)),
    worstUnit: Math.max(0, ...staffs.map((one) => one.unit.delta)),
    worstY: Math.max(0, ...staffs.map((one) => one.y.delta)),
    xMismatches: staffs.reduce((n, one) => n + one.xMismatches, 0),
  };
}

/** One line for a test log or the bench page. */
export function describeStaffComparison(report: StaffListComparison): string {
  const points = report.points.map((one) => `${one.got}/${one.want}`).join(" ");
  return [
    `${report.label}: ${report.count.got}/${report.count.want}`,
    `points ${points}`,
    `x mismatches ${report.xMismatches}`,
    `y ${report.worstY.toExponential(1)}`,
    `angle ${report.worstAngle.toExponential(1)}`,
    `unit ${report.worstUnit.toExponential(1)}`,
  ].join(", ");
}

export interface NoteheadListComparison {
  /** The notehead ellipses, by the box rule of box-tolerance.ts. */
  readonly boxes: BoxListComparison;
  readonly count: { readonly got: number; readonly want: number };
  /** Entries whose stem is absent on both sides or points the same way on both. */
  readonly directionsExact: number;
  readonly failures: readonly string[];
  readonly label: string;
  /** The stems of the entries that have one on both sides. A stem on one side only is a failure, not an entry. */
  readonly stems: BoxListComparison;
}

export interface NoteListComparison extends NoteheadListComparison {
  readonly positionsExact: number;
}

interface Stemmed {
  readonly box: AngledBox;
  readonly stem: Stem | null;
}

function compareStemmed(
  label: string,
  got: readonly Stemmed[],
  want: readonly Stemmed[]
): NoteheadListComparison {
  const failures: string[] = [];
  const owners: number[] = [];
  let directionsExact = 0;
  for (const [i, expected] of want.entries()) {
    const actual = got[i]?.stem ?? null;
    if ((actual === null) !== (expected.stem === null)) {
      failures.push(
        `${label}[${i}].stem: ${actual === null ? "none" : "one"}, Python has ${expected.stem === null ? "none" : "one"}`
      );
      continue;
    }
    if (actual !== null && expected.stem !== null) {
      owners.push(i);
      if (actual.direction !== expected.stem.direction) {
        failures.push(
          `${label}[${i}].stem.direction: ${actual.direction}, Python ${expected.stem.direction}`
        );
        continue;
      }
    }
    directionsExact += 1;
  }
  const stemBoxes = (entries: readonly Stemmed[]): AngledBox[] =>
    owners.flatMap((i) => {
      const stem = entries[i]?.stem ?? null;
      return stem === null ? [] : [stem.box];
    });
  const boxes = compareBoxLists(
    `${label}.box`,
    got.map((one) => one.box),
    want.map((one) => one.box),
    (i) => `${label}[${i}].box`
  );
  const stems = compareBoxLists(
    `${label}.stem`,
    stemBoxes(got),
    stemBoxes(want),
    (k) => `${label}[${owners[k] ?? k}].stem`
  );
  return {
    boxes,
    count: { got: got.length, want: want.length },
    directionsExact,
    failures: [...boxes.failures, ...failures, ...stems.failures],
    label,
    stems,
  };
}

const emptyBoxComparison = (label: string): BoxListComparison =>
  compareBoxLists(label, [], []);

function countMismatch(
  label: string,
  got: number,
  want: number,
  what: string
): NoteheadListComparison {
  return {
    boxes: emptyBoxComparison(`${label}.box`),
    count: { got, want },
    directionsExact: 0,
    failures: [`${label}: ${got} ${what}, Python found ${want}`],
    label,
    stems: emptyBoxComparison(`${label}.stem`),
  };
}

const centerMoved = (got: Point, want: Point): boolean =>
  !(
    Math.abs(got.x - want.x) <= BOX_RECT_TOLERANCE.center &&
    Math.abs(got.y - want.y) <= BOX_RECT_TOLERANCE.center
  );

/** notes.json, and the notes of a staff's `symbols`: ellipse, centre, position, stem and direction. Never throws. */
export function compareNoteLists(
  label: string,
  got: readonly Note[],
  want: readonly Note[]
): NoteListComparison {
  if (got.length !== want.length) {
    return {
      ...countMismatch(label, got.length, want.length, "notes"),
      positionsExact: 0,
    };
  }
  const report = compareStemmed(label, got, want);
  const failures: string[] = [];
  let positionsExact = 0;
  for (const [i, expected] of want.entries()) {
    const actual = got[i] ?? expected;
    if (actual.position === expected.position) {
      positionsExact += 1;
    } else {
      failures.push(
        `${label}[${i}].position: ${actual.position}, Python ${expected.position}`
      );
    }
    if (centerMoved(actual.center, expected.center)) {
      failures.push(
        `${label}[${i}].center: (${actual.center.x}, ${actual.center.y}), Python (${expected.center.x}, ${expected.center.y})`
      );
    }
  }
  return {
    ...report,
    failures: [...report.failures, ...failures],
    positionsExact,
  };
}

/** noteheads-with-stems.json: the same checks without a position. Never throws. */
export function compareNoteheadLists(
  label: string,
  got: readonly NoteheadWithStem[],
  want: readonly NoteheadWithStem[]
): NoteheadListComparison {
  if (got.length !== want.length) {
    return countMismatch(label, got.length, want.length, "noteheads");
  }
  const stemmed = (entries: readonly NoteheadWithStem[]): Stemmed[] =>
    entries.map((one) => ({ box: one.notehead, stem: one.stem }));
  return compareStemmed(label, stemmed(got), stemmed(want));
}

export interface MultiStaffComparison {
  /** Per multi-staff. */
  readonly connections: readonly BoxListComparison[];
  readonly failures: readonly string[];
  /** Staffs per multi-staff, got against want: the grouping, which is the result that matters. */
  readonly grouping: {
    readonly got: readonly number[];
    readonly want: readonly number[];
  };
  readonly label: string;
  /** Per multi-staff, per staff: the notes among its symbols. */
  readonly notes: readonly (readonly NoteListComparison[])[];
  /** Per multi-staff. */
  readonly staffs: readonly StaffListComparison[];
}

function compareSymbols(
  at: string,
  got: Staff,
  want: Staff
): { readonly kinds: string[]; readonly notes: NoteListComparison } {
  const kindsOf = (staff: Staff): string =>
    staff.symbols.map((symbol) => symbol.kind).join(",");
  return {
    kinds:
      kindsOf(got) === kindsOf(want)
        ? []
        : [
            `${at}: ${got.symbols.length} symbols, not the kinds Python has in its ${want.symbols.length}`,
          ],
    notes: compareNoteLists(
      at,
      symbolsOfKind(got.symbols, "note"),
      symbolsOfKind(want.symbols, "note")
    ),
  };
}

/** multistaffs.json: the grouping, then each group's staffs, their symbols and the connections. Never throws. */
export function compareMultiStaffLists(
  label: string,
  got: readonly MultiStaff[],
  want: readonly MultiStaff[],
  tolerance: StaffFloatTolerance
): MultiStaffComparison {
  const grouping = {
    got: got.map((multi) => multi.staffs.length),
    want: want.map((multi) => multi.staffs.length),
  };
  if (got.length !== want.length) {
    return {
      connections: [],
      failures: [
        `${label}: ${got.length} multi staffs, Python found ${want.length}`,
      ],
      grouping,
      label,
      notes: [],
      staffs: [],
    };
  }
  const failures: string[] = [];
  const connections: BoxListComparison[] = [];
  const notes: NoteListComparison[][] = [];
  const staffs: StaffListComparison[] = [];
  for (const [i, expected] of want.entries()) {
    const actual = got[i] ?? expected;
    const at = `${label}[${i}]`;
    if (actual.staffs.length !== expected.staffs.length) {
      failures.push(
        `${at}: ${actual.staffs.length} staffs, Python grouped ${expected.staffs.length}`
      );
      continue;
    }
    const staffReport = compareStaffLists(
      `${at}.staffs`,
      actual.staffs,
      expected.staffs,
      tolerance
    );
    const symbolReports = expected.staffs.map((staff, s) =>
      compareSymbols(
        `${at}.staffs[${s}].symbols`,
        actual.staffs[s] ?? staff,
        staff
      )
    );
    const connectionReport = compareBoxLists(
      `${at}.connections`,
      actual.connections,
      expected.connections
    );
    staffs.push(staffReport);
    notes.push(symbolReports.map((one) => one.notes));
    connections.push(connectionReport);
    failures.push(
      ...staffReport.failures,
      ...symbolReports.flatMap((one) => [...one.kinds, ...one.notes.failures]),
      ...connectionReport.failures
    );
  }
  return { connections, failures, grouping, label, notes, staffs };
}

const sameLines = (got: FiveLines, want: FiveLines): boolean =>
  want.every((line, i) => {
    const fragments = got[i]?.fragments ?? [];
    return (
      fragments.length === line.fragments.length &&
      line.fragments.every((fragment, k) => {
        const other = fragments[k];
        return other !== undefined && sameRect(other.rect, fragment.rect);
      })
    );
  });

const sameAnchor = (got: StaffAnchor, want: StaffAnchor): boolean =>
  sameRect(got.symbol.rect, want.symbol.rect) &&
  sameLines(got.lines, want.lines);

export interface AnchorListComparison {
  readonly count: { readonly got: number; readonly want: number };
  readonly failures: readonly string[];
  readonly label: string;
  /** Anchors whose five lines hold exactly Python's fragments in Python's order. */
  readonly linesExact: number;
  /** The shifted symbol each anchor was found at, by the box rule of box-tolerance.ts. */
  readonly symbols: BoxListComparison;
  readonly worstUnit: number;
  /** Over minY and maxY. */
  readonly worstY: number;
  readonly zonesExact: number;
}

/**
 * staff-anchors.json's lists: per anchor the symbol, the fragments of each
 * line by rect value and in order, the zone exactly, and averageUnitSize,
 * minY and maxY within `tolerance.y`. Never throws.
 */
export function compareAnchorLists(
  label: string,
  got: readonly StaffAnchor[],
  want: readonly StaffAnchor[],
  tolerance: StaffFloatTolerance
): AnchorListComparison {
  const count = { got: got.length, want: want.length };
  if (got.length !== want.length) {
    return {
      count,
      failures: [
        `${label}: ${got.length} anchors, Python found ${want.length}`,
      ],
      label,
      linesExact: 0,
      symbols: compareBoxLists(`${label}.symbol`, [], []),
      worstUnit: 0,
      worstY: 0,
      zonesExact: 0,
    };
  }
  const symbols = compareBoxLists(
    `${label}.symbol`,
    got.map((anchor) => anchor.symbol),
    want.map((anchor) => anchor.symbol),
    (i) => `${label}[${i}].symbol`
  );
  const failures: string[] = [];
  const unit = noWorst();
  const y = noWorst();
  let linesExact = 0;
  let zonesExact = 0;
  for (const [i, expected] of want.entries()) {
    const actual = got[i] ?? expected;
    const at = `${label}[${i}]`;
    if (sameLines(actual.lines, expected.lines)) {
      linesExact += 1;
    } else {
      failures.push(`${at}.lines: not the fragments Python connected`);
    }
    if (
      actual.zone.start === expected.zone.start &&
      actual.zone.stop === expected.zone.stop
    ) {
      zonesExact += 1;
    } else {
      failures.push(
        `${at}.zone: [${actual.zone.start}, ${actual.zone.stop}], Python [${expected.zone.start}, ${expected.zone.stop}]`
      );
    }
    track(
      unit,
      `${at}.averageUnitSize`,
      actual.averageUnitSize,
      expected.averageUnitSize,
      tolerance.y
    );
    track(y, `${at}.minY`, actual.minY, expected.minY, tolerance.y);
    track(y, `${at}.maxY`, actual.maxY, expected.maxY, tolerance.y);
  }
  return {
    count,
    failures: [
      ...symbols.failures,
      ...failures,
      ...boundFailure(unit, tolerance.y),
      ...boundFailure(y, tolerance.y),
    ],
    label,
    linesExact,
    symbols,
    worstUnit: unit.delta,
    worstY: y.delta,
    zonesExact,
  };
}

export interface RawStaffListComparison {
  /** Per staff, how many anchors, got against want. */
  readonly anchorCounts: ReadonlyArray<{
    readonly got: number;
    readonly want: number;
  }>;
  /** Staffs whose anchors are Python's, in Python's order. */
  readonly anchorsExact: number;
  /** Each staff's box, by the box rule of box-tolerance.ts; its debugId is the staff id. */
  readonly boxes: BoxListComparison;
  readonly count: { readonly got: number; readonly want: number };
  readonly failures: readonly string[];
  readonly label: string;
  readonly linesExact: number;
}

/**
 * raw-staffs.json's lists, in list order, which is part of the result: the
 * box, the fragments of each line and the anchors. Never throws.
 */
export function compareRawStaffLists(
  label: string,
  got: readonly RawStaff[],
  want: readonly RawStaff[]
): RawStaffListComparison {
  const count = { got: got.length, want: want.length };
  if (got.length !== want.length) {
    return {
      anchorCounts: [],
      anchorsExact: 0,
      boxes: compareBoxLists(`${label}.box`, [], []),
      count,
      failures: [
        `${label}: ${got.length} raw staffs, Python found ${want.length}`,
      ],
      label,
      linesExact: 0,
    };
  }
  const boxes = compareBoxLists(
    `${label}.box`,
    got.map((staff) => staff.box),
    want.map((staff) => staff.box),
    (i) => `${label}[${i}].box`
  );
  const failures: string[] = [];
  let anchorsExact = 0;
  let linesExact = 0;
  for (const [i, expected] of want.entries()) {
    const actual = got[i] ?? expected;
    const at = `${label}[${i}]`;
    if (sameLines(actual.lines, expected.lines)) {
      linesExact += 1;
    } else {
      failures.push(`${at}.lines: not the fragments Python's staff holds`);
    }
    if (
      actual.anchors.length === expected.anchors.length &&
      expected.anchors.every((anchor, k) => {
        const other = actual.anchors[k];
        return other !== undefined && sameAnchor(other, anchor);
      })
    ) {
      anchorsExact += 1;
    } else {
      failures.push(
        `${at}.anchors: ${actual.anchors.length}, not Python's ${expected.anchors.length} in Python's order`
      );
    }
  }
  return {
    anchorCounts: want.map((expected, i) => ({
      got: got[i]?.anchors.length ?? 0,
      want: expected.anchors.length,
    })),
    anchorsExact,
    boxes,
    count,
    failures: [...boxes.failures, ...failures],
    label,
    linesExact,
  };
}

/** Anything with `failures`: one throw for every report shape above. */
export function assertGoldenMatches(report: {
  readonly failures: readonly string[];
}): void {
  if (report.failures.length > 0) {
    throw new StaffGoldenMismatch(report.failures.join("\n"));
  }
}

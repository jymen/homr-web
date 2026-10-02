/**
 * What "agrees with Python" means for a box list. A function rather than a set
 * of assertions, so a test and the bench overlay report the same numbers, the
 * way planeAgreement already does for masks.
 *
 * Four of the five checks are exact and only the rect has a tolerance. The
 * strongest of them is free: a group's concatenated contour is byte-identical or
 * it is not, so a wrong merge grouping, or the right group in the wrong order,
 * shows up there directly instead of as a drifted rect three steps later.
 */

import {
  type AngledBox,
  type PointList,
  pointCount,
  type RotatedRectParams,
} from "../geometry/boxes.js";

/**
 * Centre to 4e-4, size to 3e-4, angle to 1e-3.
 *
 * All are tighter than the plan's blanket 1e-3 and the causes are named,
 * which is what a bound has to have here. Python stores the rect as float32 and
 * the port recomputes in float64: worst size delta measured across 967 entries
 * is 2.4e-4 and worst angle delta 7.7e-7.
 *
 * The centre is minAreaRect's rotatingCalipers, which computes
 * `dx * a + dy * b` in float32. The arm64 cv2 build contracts it into a fused
 * multiply-add, one rounding, and WebAssembly has none, two roundings. So a
 * centre differs by whole float32 ulps of the coordinate: at most three over
 * the 899 broken fragments of the two public pages, 3.66e-4 on fragment 275
 * of the grand-staff page at x 1776, where an ulp is 1.22e-4. A calipers port
 * that emulates the fused form is bit-identical to Python on 2573 of 2573
 * golden rects, that fragment included (docs/design/phase-5-minarearect.md).
 * The bound admits three ulps on a page up to 2048 px wide and not four.
 */
export const BOX_RECT_TOLERANCE = {
  angle: 1e-3,
  center: 4e-4,
  size: 3e-4,
} as const;

/**
 * A polygon corner may differ by one pixel, and only when that same entry's
 * rect is not bit-exact.
 *
 * The rect differs by about 1e-4 between the two builds, which puts a corner
 * either side of an integer; 18 entries of 967 land there. Conditioning the
 * allowance on an independently visible cause rather than on a magnitude is what
 * keeps it falsifiable: a corner that moves while the rect is bit-identical has
 * no noise source left and is a real truncation or point-order bug. A blanket
 * epsilon would be vacuous anyway -- over 90 % of the 5072 corner coordinates
 * are exactly integers, so 1e-2 accepts 95 % of them for free.
 */
export const POLYGON_CORNER_SLACK = 1;

export interface BoxComparisonEntry {
  readonly contourExact: boolean;
  readonly debugIdExact: boolean;
  readonly index: number;
  /** Chebyshev distance of the furthest-moved corner; 0 when the polygon is exact. */
  readonly maxCornerDelta: number;
  readonly polygonExplainedByRectNoise: boolean;
  readonly rectExact: boolean;
  readonly worstAngle: number;
  readonly worstCenter: number;
  readonly worstSize: number;
}

export interface BoxListComparison {
  readonly contoursExact: number;
  readonly count: { readonly got: number; readonly want: number };
  readonly debugIdsExact: number;
  readonly entries: readonly BoxComparisonEntry[];
  /** One line per hard violation. Empty is the pass condition. */
  readonly failures: readonly string[];
  readonly label: string;
  readonly polygonsExact: number;
  /** How many entries needed the one-pixel allowance. Reported, never asserted. */
  readonly slackUsed: number;
  readonly worstAngle: number;
  readonly worstCenter: number;
  readonly worstSize: number;
}

export class BoxGoldenMismatch extends Error {}

const samePoints = (a: PointList, b: PointList): boolean => {
  if (a.length !== b.length) {
    return false;
  }
  for (const [i, value] of a.entries()) {
    if (value !== b[i]) {
      return false;
    }
  }
  return true;
};

const rectIsExact = (a: RotatedRectParams, b: RotatedRectParams): boolean =>
  a.cx === b.cx &&
  a.cy === b.cy &&
  a.w === b.w &&
  a.h === b.h &&
  a.angle === b.angle;

const maxCornerDelta = (got: PointList, want: PointList): number => {
  let worst = 0;
  for (const [i, value] of got.entries()) {
    worst = Math.max(worst, Math.abs(value - (want[i] ?? 0)));
  }
  return worst;
};

interface EntryComparison {
  readonly entry: BoxComparisonEntry;
  readonly failures: readonly string[];
}

function comparePolygons(
  actual: AngledBox,
  expected: AngledBox,
  rectExact: boolean
): { readonly delta: number; readonly explained: boolean } {
  if (actual.polygon.length !== expected.polygon.length) {
    return { delta: Number.NaN, explained: false };
  }
  const delta = maxCornerDelta(actual.polygon, expected.polygon);
  return {
    delta,
    explained: delta > 0 && delta <= POLYGON_CORNER_SLACK && !rectExact,
  };
}

function compareEntry(
  at: string,
  index: number,
  actual: AngledBox,
  expected: AngledBox
): EntryComparison {
  const failures: string[] = [];
  const worstCenter = Math.max(
    Math.abs(actual.rect.cx - expected.rect.cx),
    Math.abs(actual.rect.cy - expected.rect.cy)
  );
  const worstSize = Math.max(
    Math.abs(actual.rect.w - expected.rect.w),
    Math.abs(actual.rect.h - expected.rect.h)
  );
  const worstAngle = Math.abs(actual.rect.angle - expected.rect.angle);
  const rectExact = rectIsExact(actual.rect, expected.rect);
  const contourExact = samePoints(actual.contour, expected.contour);
  const debugIdExact = actual.debugId === expected.debugId;
  const { delta, explained } = comparePolygons(actual, expected, rectExact);
  if (actual.kind !== expected.kind) {
    failures.push(
      `${at}: a ${actual.kind} where Python has a ${expected.kind}`
    );
  }
  if (
    worstCenter > BOX_RECT_TOLERANCE.center ||
    worstSize > BOX_RECT_TOLERANCE.size ||
    worstAngle > BOX_RECT_TOLERANCE.angle
  ) {
    failures.push(
      `${at}: rect outside tolerance, centre ${worstCenter.toExponential(2)}, size ${worstSize.toExponential(2)}, angle ${worstAngle.toExponential(2)}`
    );
  }
  if (!contourExact) {
    failures.push(
      `${at}: contour of ${pointCount(actual.contour)} points differs from Python's ${pointCount(expected.contour)}`
    );
  }
  if (!debugIdExact) {
    failures.push(
      `${at}: debugId ${actual.debugId}, Python's ${expected.debugId}`
    );
  }
  if (Number.isNaN(delta)) {
    failures.push(
      `${at}: polygon of ${pointCount(actual.polygon)} points, Python's ${pointCount(expected.polygon)}`
    );
  } else if (delta > 0 && !explained) {
    failures.push(
      `${at}: a polygon corner moved ${delta} px with the rect ${rectExact ? "bit-exact" : "inexact"}, which nothing explains`
    );
  }
  return {
    entry: {
      contourExact,
      debugIdExact,
      index,
      maxCornerDelta: delta,
      polygonExplainedByRectNoise: explained,
      rectExact,
      worstAngle,
      worstCenter,
      worstSize,
    },
    failures,
  };
}

const countWhere = (
  entries: readonly BoxComparisonEntry[],
  holds: (entry: BoxComparisonEntry) => boolean
): number => entries.filter(holds).length;

/**
 * Pure and never throws, so the bench page can render it for a by-eye diff.
 * `pathOf` names entry `index` in a failure line, for a list whose boxes are
 * members of something else (the stem of note 7 is not `stems[5]`).
 */
export function compareBoxLists(
  label: string,
  got: readonly AngledBox[],
  want: readonly AngledBox[],
  pathOf: (index: number) => string = (index) => `${label}[${index}]`
): BoxListComparison {
  const count = { got: got.length, want: want.length };
  if (got.length !== want.length) {
    return {
      contoursExact: 0,
      count,
      debugIdsExact: 0,
      entries: [],
      failures: [`${label}: ${got.length} boxes, Python found ${want.length}`],
      label,
      polygonsExact: 0,
      slackUsed: 0,
      worstAngle: 0,
      worstCenter: 0,
      worstSize: 0,
    };
  }
  const compared = want.map((expected, index) =>
    compareEntry(pathOf(index), index, got[index] ?? expected, expected)
  );
  const entries = compared.map((one) => one.entry);
  return {
    contoursExact: countWhere(entries, (entry) => entry.contourExact),
    count,
    debugIdsExact: countWhere(entries, (entry) => entry.debugIdExact),
    entries,
    failures: compared.flatMap((one) => one.failures),
    label,
    polygonsExact: countWhere(entries, (entry) => entry.maxCornerDelta === 0),
    slackUsed: countWhere(
      entries,
      (entry) => entry.polygonExplainedByRectNoise
    ),
    worstAngle: Math.max(0, ...entries.map((entry) => entry.worstAngle)),
    worstCenter: Math.max(0, ...entries.map((entry) => entry.worstCenter)),
    worstSize: Math.max(0, ...entries.map((entry) => entry.worstSize)),
  };
}

/** One line for a test log or the bench page. */
export function describeBoxComparison(report: BoxListComparison): string {
  return [
    `${report.label}: ${report.count.got}/${report.count.want}`,
    `centre ${report.worstCenter.toExponential(1)}`,
    `size ${report.worstSize.toExponential(1)}`,
    `angle ${report.worstAngle.toExponential(1)}`,
    `contours ${report.contoursExact}`,
    `ids ${report.debugIdsExact}`,
    `polygons ${report.polygonsExact} exact + ${report.slackUsed} within ${POLYGON_CORNER_SLACK} px`,
  ].join(", ");
}

export function assertBoxListMatches(report: BoxListComparison): void {
  if (report.failures.length > 0) {
    throw new BoxGoldenMismatch(report.failures.join("\n"));
  }
}

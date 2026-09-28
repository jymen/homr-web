/**
 * Port of the data half of homr's bounding_boxes.py, plus the pure arithmetic
 * that decides what a rotated rect means: the angle-convention ladder and
 * homr's own normalisation. Everything that needs opencv.js to fit or redraw a
 * shape is in src/cv/box*.ts, the grouping is in src/geometry/boxMerge.ts, and
 * the DebugDrawable hierarchy and every draw method are not ported.
 *
 * The shapes are plain readonly objects so they cross the Worker boundary
 * by structured clone and compare structurally in tests. homr's __eq__ and
 * __hash__ on AngledBoundingBox (tuple equality on the rect) become
 * sameRect and rectKey; the Python `in` and set membership that depend on
 * them are ported through those two.
 */

import type { RotatedRect as CvNativeRect } from "@techstark/opencv-js";

// Points and point lists

export interface Point {
  readonly x: number;
  readonly y: number;
}

declare const pointListBrand: unique symbol;

/**
 * A flat list of integer points: x0, y0, x1, y1, ... in an Int32Array of
 * even length. This is the byte layout of a CV_32SC2 Mat's `data32S`, so a
 * contour comes out of `findContours` and goes into `pointPolygonTest`,
 * `minAreaRect` and `fitEllipse` without reshaping. The same type serves
 * homr's `contours` (the source pixels of a shape, cv2 shape (n, 1, 2)) and
 * its `polygon` (the sampled outline, shape (n, 2)); they differ in meaning,
 * not representation, and no code path confuses one for the other because
 * they sit in differently named fields.
 *
 * The brand is type-only; a PointList is still an Int32Array at runtime and
 * clones as one.
 */
export type PointList = Int32Array & { readonly [pointListBrand]: true };

export class GeometryError extends Error {}

/** Validates even length; no copy. The way opencv.js contour data enters. */
export function pointListFromInt32(data: Int32Array): PointList {
  if (data.length % 2 !== 0) {
    throw new GeometryError(
      `a point list needs an even length, got ${data.length}`
    );
  }
  return data as PointList;
}

/** From `[[x, y], ...]` pairs, the golden JSON form; values are truncated to int32. */
export function pointListFromPairs(
  pairs: ReadonlyArray<readonly [number, number]>
): PointList {
  const data = new Int32Array(pairs.length * 2);
  for (const [i, [x, y]] of pairs.entries()) {
    data[i * 2] = Math.trunc(x);
    data[i * 2 + 1] = Math.trunc(y);
  }
  return data as PointList;
}

export function pointCount(list: PointList): number {
  return list.length / 2;
}

/** Throws on an index out of range rather than returning undefined. */
export function pointAt(list: PointList, index: number): Point {
  const x = list[index * 2];
  const y = list[index * 2 + 1];
  if (x === undefined || y === undefined) {
    throw new GeometryError(
      `point ${index} is outside a list of ${pointCount(list)}`
    );
  }
  return { x, y };
}

/** `np.concatenate(contours)`: what merged groups carry as their contour. */
export function concatPointLists(lists: readonly PointList[]): PointList {
  let length = 0;
  for (const list of lists) {
    length += list.length;
  }
  const out = new Int32Array(length);
  let offset = 0;
  for (const list of lists) {
    out.set(list, offset);
    offset += list.length;
  }
  return out as PointList;
}

/** The list-comprehension filters of staff_detection.break_wide_fragments. */
export function filterPoints(
  list: PointList,
  keep: (x: number, y: number, index: number) => boolean
): PointList {
  const kept: number[] = [];
  for (let i = 0; i < pointCount(list); i += 1) {
    const { x, y } = pointAt(list, i);
    if (keep(x, y, i)) {
      kept.push(x, y);
    }
  }
  return Int32Array.from(kept) as PointList;
}

/** `sorted(contours, key=lambda c: c[0][0])`: stable, ascending x. */
export function sortPointsByX(list: PointList): PointList {
  const indices = Array.from({ length: pointCount(list) }, (_, i) => i);
  indices.sort((a, b) => (list[a * 2] ?? 0) - (list[b * 2] ?? 0));
  const out = new Int32Array(list.length);
  for (const [slot, i] of indices.entries()) {
    out[slot * 2] = list[i * 2] ?? 0;
    out[slot * 2 + 1] = list[i * 2 + 1] ?? 0;
  }
  return out as PointList;
}

// Rotated rectangles

/** cv2's RotatedRect triple flattened: centre, size, angle in degrees. */
export interface RotatedRectParams {
  readonly angle: number;
  readonly cx: number;
  readonly cy: number;
  readonly h: number;
  readonly w: number;
}

declare const rawMinAreaRectBrand: unique symbol;

/**
 * cv.minAreaRect's result, untouched. @techstark/opencv-js 4.12.0 reports the
 * angle in (0, 90] -- the post-4.5.1 convention -- where opencv-python 4.x
 * reports it in [-90, 0), and homr's geometry is written against the latter.
 * The only thing to do with one of these is toLegacyAngleConvention.
 */
export type RawMinAreaRect = RotatedRectParams & {
  readonly [rawMinAreaRectBrand]: true;
};

declare const rawFitEllipseBrand: unique symbol;

/**
 * cv.fitEllipse's result, untouched. Angle in [0, 180) in both builds:
 * measured 2026-09-28 on all 81 Kesh noteheads against opencv-python 4.14.0,
 * worst delta exactly 0 on centre, size and angle. So a fitEllipse rect must
 * *not* be converted, and that is why the two producers carry different brands
 * instead of sharing one "raw cv rect" type -- converting this one would
 * corrupt it as surely as leaving a minAreaRect one alone.
 */
export type RawFitEllipseRect = RotatedRectParams & {
  readonly [rawFitEllipseBrand]: true;
};

declare const legacyConventionBrand: unique symbol;

/**
 * A rect in the convention homr was written against. normalizeRotatedRect and
 * cv.boxPoints see this, never a raw producer's result, and the three ways to
 * obtain one are named below.
 */
export type LegacyConventionRect = RotatedRectParams & {
  readonly [legacyConventionBrand]: true;
};

export function rawMinAreaRectOf(native: CvNativeRect): RawMinAreaRect {
  return {
    angle: native.angle,
    cx: native.center.x,
    cy: native.center.y,
    h: native.size.height,
    w: native.size.width,
  } as RawMinAreaRect;
}

export function rawFitEllipseRectOf(native: CvNativeRect): RawFitEllipseRect {
  return {
    angle: native.angle,
    cx: native.center.x,
    cy: native.center.y,
    h: native.size.height,
    w: native.size.width,
  } as RawFitEllipseRect;
}

/**
 * opencv.js's minAreaRect angle convention to opencv-python's: subtract 90 and
 * swap w with h, while the angle is at or above zero.
 *
 * homr's own normalisation absorbs the difference almost everywhere, because a
 * rectangle at -90 and one at +90 normalise alike. It does not absorb it at
 * exactly +-45, where homr's comparisons are strict, so +45 stays +45 while
 * -45 stays -45 with the dimensions the other way round: 86 of 717 golden
 * rotated-box entries disagreed before this conversion existed and none after.
 *
 * A loop and not an `if`, because an angle of exactly 90 needs two iterations:
 * the two swaps cancel and the result is -90 with the size unchanged, which is
 * what opencv-python reports for an axis-aligned rect. The loop is also why no
 * range check guards it -- an angle already in [-90, 0) passes through
 * untouched, so a pre-4.5.1 build would still be handled correctly and a check
 * would only refuse it.
 */
export function toLegacyAngleConvention(
  raw: RawMinAreaRect
): LegacyConventionRect {
  let { angle, h, w } = raw;
  while (angle >= 0) {
    angle -= 90;
    [w, h] = [h, w];
  }
  return { angle, cx: raw.cx, cy: raw.cy, h, w } as LegacyConventionRect;
}

/**
 * A relabel, not a computation: fitEllipse's angle never changed across the
 * 4.5.1 split. It is a function rather than a cast at the call site so that a
 * future measurement saying otherwise has one body to change.
 */
export function legacyFromFitEllipse(
  raw: RawFitEllipseRect
): LegacyConventionRect {
  const { angle, cx, cy, h, w } = raw;
  return { angle, cx, cy, h, w } as LegacyConventionRect;
}

/**
 * For numbers that were already in opencv-python's convention when they were
 * written down: the hand-authored rects in the tests. Not a third producer of
 * measured values -- everything fitted here comes through
 * toLegacyAngleConvention or legacyFromFitEllipse.
 */
export function legacyConventionRectOf(
  params: RotatedRectParams
): LegacyConventionRect {
  return params as LegacyConventionRect;
}

declare const normalizedBrand: unique symbol;

/**
 * A rotated rectangle after homr's angle normalisation
 * (AngledBoundingBox.__init__), the form every stored box holds:
 *
 *   if angle >  135: angle -= 180
 *   elif angle < -135: angle += 180
 *   elif angle >  45: angle -= 90 and swap w, h
 *   elif angle < -45: angle += 90 and swap w, h
 *
 * so angle is in [-45, 45] (both ends occur in the golden data) and w is
 * the extent along the nearly horizontal axis. The rule is idempotent,
 * which is why homr can rebuild a box from an already-normalised one
 * (make_box_thicker) through the same constructor. The brand is type-only
 * and only two functions produce it: normalizeRotatedRect for a cv2
 * result, and the golden decoder after checking the angle range.
 */
export type RotatedRect = RotatedRectParams & {
  readonly [normalizedBrand]: true;
};

export function normalizeRotatedRect(
  raw: LegacyConventionRect | RotatedRect
): RotatedRect {
  const { cx, cy, w, h } = raw;
  let { angle } = raw;
  let width = w;
  let height = h;
  if (angle > 135) {
    angle -= 180;
  } else if (angle < -135) {
    angle += 180;
  } else if (angle > 45) {
    angle -= 90;
    width = h;
    height = w;
  } else if (angle < -45) {
    angle += 90;
    width = h;
    height = w;
  }
  return { angle, cx, cy, h: height, w: width } as RotatedRect;
}

/** The golden decoder's entry: accepts a triple homr already normalised, refusing one it did not. */
export function assertNormalizedRect(rect: RotatedRectParams): RotatedRect {
  if (!(rect.angle >= -45 && rect.angle <= 45)) {
    throw new GeometryError(
      `angle ${rect.angle} is outside homr's normalised range [-45, 45]`
    );
  }
  return rect as RotatedRect;
}

/**
 * homr's _has_box_valid_size. Applied to the raw fit before the convention
 * conversion, as Python does; the conversion only swaps the dimensions, so the
 * answer is the same either side of it.
 */
export function hasValidRectSize(rect: RotatedRectParams): boolean {
  return (
    !(Number.isNaN(rect.w) || Number.isNaN(rect.h)) && rect.w > 0 && rect.h > 0
  );
}

/**
 * homr's top_left, bottom_left, top_right, bottom_right: centre plus or
 * minus half the size, the angle ignored. This is an axis-aligned
 * approximation and homr uses it as one (to_bounding_box, notehead clump
 * splitting); the rotated corners are `polygon` on a RotatedBox. Pure
 * float64 arithmetic on float32-exact inputs, so the golden values match
 * exactly and the decoder asserts them.
 */
export interface Corners {
  readonly bottomLeft: Point;
  readonly bottomRight: Point;
  readonly topLeft: Point;
  readonly topRight: Point;
}

export function cornersOf(rect: RotatedRectParams): Corners {
  const halfW = rect.w / 2;
  const halfH = rect.h / 2;
  return {
    bottomLeft: { x: rect.cx - halfW, y: rect.cy + halfH },
    bottomRight: { x: rect.cx + halfW, y: rect.cy + halfH },
    topLeft: { x: rect.cx - halfW, y: rect.cy - halfH },
    topRight: { x: rect.cx + halfW, y: rect.cy - halfH },
  };
}

/** homr's AngledBoundingBox.__eq__: the rect triples are equal, component for component. */
export function sameRect(a: RotatedRectParams, b: RotatedRectParams): boolean {
  return (
    a.cx === b.cx &&
    a.cy === b.cy &&
    a.w === b.w &&
    a.h === b.h &&
    a.angle === b.angle
  );
}

/**
 * Stands in for __hash__ where Python uses a set or `in` on boxes
 * (used_stems, StaffLineSegment's frozenset, MultiStaff.merge). A string
 * of the five components; equal rects give equal keys and nothing else does.
 */
export function rectKey(rect: RotatedRectParams): string {
  return `${rect.cx},${rect.cy},${rect.w},${rect.h},${rect.angle}`;
}

// Boxes

export const BOX_KINDS = {
  axis: "axis",
  ellipse: "ellipse",
  rotated: "rotated",
} as const;

interface AngledShape<K extends "rotated" | "ellipse"> {
  /**
   * The source contour the shape was fitted to (findContours output, or
   * the concatenation of a merged group's contours, or a two-point Hough
   * line). Later stages refit boxes to subsets of it
   * (staff_detection.break_wide_fragments), so it is data, not debug.
   */
  readonly contour: PointList;
  /**
   * homr's debug_id: the contour index at creation. Unique on staff
   * fragments (phase 5 uses it as a staff line id) and always 0 on
   * merged stems and noteheads, so it is an identity only where the
   * creator made it one. 0 when unknown.
   */
  readonly debugId: number;
  readonly kind: K;
  /**
   * The sampled outline that the overlap tests walk: cv2.boxPoints for a
   * rotated box (4 points), cv2.ellipse2Poly for an ellipse (tens of
   * points), each truncated to int as homr's astype(np.int64) does.
   *
   * Stored, not derived, because homr computes it from the rect handed to
   * the constructor, which is the raw cv2 result for a freshly fitted shape
   * and the normalised one for a rebuilt shape; a swapped (w, h) with a
   * rotated angle draws the same outline but starts the point sequence
   * elsewhere, so recomputing from `rect` would not reproduce the golden
   * lists (measured: 19 of 377 on the Kesh page). Phase 4 computes it once
   * in createRotatedBox / createEllipse.
   */
  readonly polygon: PointList;
  /** homr's .box, .center, .size and .angle in one normalised triple. */
  readonly rect: RotatedRect;
}

/** homr's RotatedBoundingBox. */
export type RotatedBox = AngledShape<"rotated">;

/** homr's BoundingEllipse: same data, an ellipse2Poly outline. */
export type Ellipse = AngledShape<"ellipse">;

/**
 * homr's BoundingBox: integer, axis-aligned, (x1, y1, x2, y2). Used for
 * clefs, rests and accidentals and by the staff-position loader. Its
 * centre, size, rotated form (angle 0) and 4-point polygon are derived by
 * the accessors below and not stored; homr stores them but never reads a
 * value that could differ from the derivation.
 */
export interface AxisBox {
  readonly contour: PointList;
  readonly debugId: number;
  readonly kind: "axis";
  readonly x1: number;
  readonly x2: number;
  readonly y1: number;
  readonly y2: number;
}

/** homr's AngledBoundingBox: the two shapes that carry a rect and an outline. */
export type AngledBox = RotatedBox | Ellipse;

/** Anything homr's is_overlapping accepts on either side. */
export type AnyBox = AngledBox | AxisBox;

/** Assembles a rotated box from parts homr already computed (the golden decoder, and phase 4 after boxPoints). */
export function rotatedBoxFromParts(
  rect: RotatedRect,
  polygon: PointList,
  contour: PointList,
  debugId: number
): RotatedBox {
  return { contour, debugId, kind: "rotated", polygon, rect };
}

/** Assembles an ellipse from parts homr already computed (the golden decoder, and phase 4 after ellipse2Poly). */
export function ellipseFromParts(
  rect: RotatedRect,
  polygon: PointList,
  contour: PointList,
  debugId: number
): Ellipse {
  return { contour, debugId, kind: "ellipse", polygon, rect };
}

export function createAxisBox(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  contour: PointList,
  debugId = 0
): AxisBox {
  return { contour, debugId, kind: "axis", x1, x2, y1, y2 };
}

/** The box's rect: stored for angled shapes, (centre, size, 0) for an AxisBox (homr's rotated_box). */
export function rotatedRectOf(box: AnyBox): RotatedRectParams {
  if (box.kind === "axis") {
    return {
      angle: 0,
      cx: (box.x1 + box.x2) / 2,
      cy: (box.y1 + box.y2) / 2,
      h: box.y2 - box.y1,
      w: box.x2 - box.x1,
    };
  }
  return box.rect;
}

/** The outline for polygon tests: stored for angled shapes, the four corners for an AxisBox. */
export function polygonOf(box: AnyBox): PointList {
  if (box.kind === "axis") {
    return Int32Array.from([
      box.x1,
      box.y1,
      box.x2,
      box.y1,
      box.x2,
      box.y2,
      box.x1,
      box.y2,
    ]) as PointList;
  }
  return box.polygon;
}

/**
 * homr's _can_shapes_possibly_touch: the centres are no further apart than the
 * sum of the two longer sides.
 *
 * Not the circumscribing radius, so not a conservative bound either -- it can
 * reject a pair that does touch. It is what homr does, and the merge grouping
 * depends on the answer.
 *
 * Math.sqrt where Python writes `** 0.5`, which is libm's pow: for an exponent
 * of 0.5 both are the correctly rounded square root on every platform this runs
 * on, and JavaScript's `**` carries no such guarantee.
 */
export function canShapesPossiblyTouch(a: AnyBox, b: AnyBox): boolean {
  const first = rotatedRectOf(a);
  const second = rotatedRectOf(b);
  const dx = first.cx - second.cx;
  const dy = first.cy - second.cy;
  const distance = Math.sqrt(dx * dx + dy * dy);
  return !(
    distance >
    Math.max(first.w, first.h) + Math.max(second.w, second.h)
  );
}

export function centerOf(box: AnyBox): Point {
  const rect = rotatedRectOf(box);
  return { x: rect.cx, y: rect.cy };
}

export function sizeOf(box: AnyBox): {
  readonly w: number;
  readonly h: number;
} {
  const rect = rotatedRectOf(box);
  return { h: rect.h, w: rect.w };
}

/**
 * RotatedBoundingBox.to_bounding_box: the axis-aligned corners truncated to
 * int, contour and id carried over. Rotated boxes only -- BoundingEllipse does
 * not inherit this, the method is defined on RotatedBoundingBox and not on
 * AngledBoundingBox, so an Ellipse never reaches it in homr either.
 */
export function axisBoxOf(box: RotatedBox): AxisBox {
  const corners = cornersOf(box.rect);
  return createAxisBox(
    Math.trunc(corners.topLeft.x),
    Math.trunc(corners.topLeft.y),
    Math.trunc(corners.bottomRight.x),
    Math.trunc(corners.bottomRight.y),
    box.contour,
    box.debugId
  );
}

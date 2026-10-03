/**
 * The staff half of homr's staff_dewarping.py: where the control points go
 * and the transform built on them. The three warp_image_randomly* functions
 * are training augmenters and are not ported.
 */

import type { OpenCv } from "../cv/opencv.js";
import type { Point } from "../geometry/boxes.js";
import type { GrayImage } from "../image/plane.js";
import { type Staff, staffPointAt } from "../model/staff.js";
import { triangulate } from "./delaunay.js";
import {
  estimatePiecewiseAffine,
  type PiecewiseAffineTransform,
} from "./piecewise-affine.js";

/** Rows of control points: where the middle line is, and where it should be. */
export interface ControlPoints {
  readonly optimal: readonly (readonly Point[])[];
  readonly span: readonly (readonly Point[])[];
}

const NUMBER_OF_Y_INTERVALS = 6;
const X_STEP = 80;
const EDGE_MARGIN = 10;
const MINIMUM_POINTS_PER_ROW = 2;

const isOnImage = (p: Point, width: number, height: number): boolean =>
  p.x >= EDGE_MARGIN &&
  p.x <= width - EDGE_MARGIN &&
  p.y >= EDGE_MARGIN &&
  p.y <= height - EDGE_MARGIN;

/** pyRange(start, stop, step) for a positive step. */
function* pyRange(
  start: number,
  stop: number,
  step: number
): Generator<number> {
  for (let v = start; v < stop; v += step) {
    yield v;
  }
}

/**
 * calculate_span_and_optimal_points. Each row follows the staff's middle line
 * relative to the first middle-line y found, and its optimal row is that
 * row flattened to its mean y.
 */
export function spanAndOptimalPoints(
  staff: Staff,
  width: number,
  height: number
): ControlPoints {
  const span: Point[][] = [];
  const optimal: Point[][] = [];
  const yStep = Math.trunc(height / NUMBER_OF_Y_INTERVALS);
  if (yStep === 0) {
    return { optimal, span };
  }
  // Python tests `if not first_y_offset`, so a reference of exactly 0.0 is
  // replaced by the next middle-line y it sees (and a NaN is kept).
  let first: number | null = null;
  for (const y of pyRange(2, height - 2, yStep)) {
    const row: Point[] = [];
    for (const x of pyRange(2, width, X_STEP)) {
      const point = staffPointAt(staff, x);
      if (point === null) {
        continue;
      }
      const [, , offset] = point.y;
      let delta = 0;
      if (first === null || first === 0) {
        first = offset;
      } else {
        delta = Math.trunc(offset - first);
      }
      const candidate = { x, y: y + delta };
      if (isOnImage(candidate, width, height)) {
        row.push(candidate);
      }
    }
    if (row.length > MINIMUM_POINTS_PER_ROW) {
      const averageY = Math.trunc(
        row.reduce((total, p) => total + p.y, 0) / row.length
      );
      span.push(row);
      optimal.push(row.map((p) => ({ x: p.x, y: averageY })));
    }
  }
  return { optimal, span };
}

/**
 * calculate_dewarp_transformation's point lists: every row stretched to both
 * image edges, then a row along the top edge and one along the bottom.
 */
function framed(
  rows: readonly (readonly Point[])[],
  width: number,
  height: number
): Point[] {
  const out: Point[] = [
    { x: 0, y: 0 },
    { x: width, y: 0 },
  ];
  for (const row of rows) {
    const [first] = row;
    const last = row.at(-1);
    if (first === undefined || last === undefined) {
      continue;
    }
    out.push({ x: 0, y: first.y }, ...row, { x: width, y: last.y });
  }
  out.push({ x: 0, y: height }, { x: width, y: height });
  return out;
}

/** calculate_dewarp_transformation: a transform from the span rows to the optimal rows. */
export function dewarpTransformation(
  cv: OpenCv,
  width: number,
  height: number,
  points: ControlPoints
): PiecewiseAffineTransform {
  const src = framed(points.span, width, height);
  const dst = framed(points.optimal, width, height);
  return estimatePiecewiseAffine(cv, triangulate(src), dst);
}

/**
 * dewarp_staff_image without the debug drawing. homr catches any exception
 * here and dewarps nothing; the port answers null in that case, and the
 * caller keeps the image as it is.
 */
export function dewarpStaffImage(
  cv: OpenCv,
  image: GrayImage,
  staff: Staff
): PiecewiseAffineTransform | null {
  try {
    return dewarpTransformation(
      cv,
      image.width,
      image.height,
      spanAndOptimalPoints(staff, image.width, image.height)
    );
  } catch {
    return null;
  }
}

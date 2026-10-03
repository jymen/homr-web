/**
 * homr's PiecewiseAffineTransform (staff_dewarping.py): one affine map per
 * triangle of a mesh over the source points, applied to an image triangle by
 * triangle with getAffineTransform, warpAffine and a fillConvexPoly mask. The
 * OpenCV calls go through opencv.js so that the bilinear sampling, the
 * fixed-point matrix and the polygon rasterisation are OpenCV's own.
 */

import type { Mat } from "@techstark/opencv-js";
import { planeToMat } from "../cv/mat-plane.js";
import { type MatScope, type OpenCv, withMatScope } from "../cv/opencv.js";
import type { Point } from "../geometry/boxes.js";
import { pySliceBounds } from "../image/numeric.js";
import { createGray, type GrayImage } from "../image/plane.js";
import {
  findSimplex,
  float32Points,
  type Triangle,
  type TriangleMesh,
} from "./delaunay.js";

/** cv2's 2 by 3 float64 affine matrix, row-major. */
export type AffineMatrix = readonly [
  number,
  number,
  number,
  number,
  number,
  number,
];

/**
 * The estimated transform: the mesh over the source points, the destination
 * point of each source point, and per triangle its matrix, or null where homr
 * stores None (a degenerate source or destination triangle).
 */
export interface PiecewiseAffineTransform {
  readonly dst: readonly Point[];
  readonly matrices: readonly (AffineMatrix | null)[];
  readonly mesh: TriangleMesh;
}

interface CvRect {
  readonly height: number;
  readonly width: number;
  readonly x: number;
  readonly y: number;
}

type Corners = readonly [Point, Point, Point];

const corners = (points: readonly Point[], [i, j, k]: Triangle): Corners => {
  const a = points[i];
  const b = points[j];
  const c = points[k];
  if (a === undefined || b === undefined || c === undefined) {
    throw new RangeError(`triangle ${i}, ${j}, ${k} indexes past the points`);
  }
  return [a, b, c];
};

/** `_is_degenerate_triangle` on float32 corners: NumPy 2 keeps every step in float32. */
function isDegenerate([a, b, c]: Corners): boolean {
  const f = Math.fround;
  const cross = f(
    f(f(b.x - a.x) * f(c.y - a.y)) - f(f(b.y - a.y) * f(c.x - a.x))
  );
  return f(Math.abs(cross) / 2) < 1e-6;
}

const pointMat = (
  cv: OpenCv,
  scope: MatScope,
  type: number,
  values: readonly number[]
): Mat => scope.keep(cv.matFromArray(values.length / 2, 1, type, [...values]));

const flat = (points: Corners): number[] => points.flatMap((p) => [p.x, p.y]);

function boundingRect(cv: OpenCv, scope: MatScope, points: Corners): CvRect {
  return cv.boundingRect(pointMat(cv, scope, cv.CV_32FC2, flat(points)));
}

/** cv2.getAffineTransform on float32 corners; null where cv2 raises. */
function affineOf(
  cv: OpenCv,
  scope: MatScope,
  from: Corners,
  to: Corners
): AffineMatrix | null {
  try {
    const matrix = scope.keep(
      cv.getAffineTransform(
        pointMat(cv, scope, cv.CV_32FC2, flat(from)),
        pointMat(cv, scope, cv.CV_32FC2, flat(to))
      )
    );
    const m = matrix.data64F;
    return [m[0] ?? 0, m[1] ?? 0, m[2] ?? 0, m[3] ?? 0, m[4] ?? 0, m[5] ?? 0];
  } catch {
    return null;
  }
}

/** `PiecewiseAffineTransform.estimate`, on a mesh already built over `src`. */
export function estimatePiecewiseAffine(
  cv: OpenCv,
  mesh: TriangleMesh,
  dst: readonly Point[]
): PiecewiseAffineTransform {
  const dst32 = float32Points(dst);
  const matrices = withMatScope((scope) =>
    mesh.triangles.map((triangle) => {
      const from = corners(mesh.points, triangle);
      const to = corners(dst32, triangle);
      return isDegenerate(from) || isDegenerate(to)
        ? null
        : affineOf(cv, scope, from, to);
    })
  );
  return { dst: dst32, matrices, mesh };
}

/** `transform_point`: the matrix of the first triangle holding the point, or the point itself. */
export function transformPoint(
  transform: PiecewiseAffineTransform,
  point: Point
): Point {
  const m = transform.matrices[findSimplex(transform.mesh, point)];
  if (m === undefined || m === null) {
    return point;
  }
  const x = Math.fround(point.x);
  const y = Math.fround(point.y);
  return { x: m[0] * x + m[1] * y + m[2], y: m[3] * x + m[4] * y + m[5] };
}

const shifted = ([a, b, c]: Corners, by: CvRect): Corners => {
  const move = (p: Point): Point => ({
    x: Math.fround(p.x - by.x),
    y: Math.fround(p.y - by.y),
  });
  return [move(a), move(b), move(c)];
};

/** One triangle warped into its destination rect: the warped pixels and the triangle's mask. */
interface WarpedPiece {
  readonly mask: Uint8Array;
  readonly pixels: Uint8Array;
  readonly rect: CvRect;
}

const isEmpty = (rect: CvRect): boolean => rect.width <= 0 || rect.height <= 0;

/** The body of warp_image's loop up to the np.where, or null where homr skips the triangle. */
function warpTriangle(
  cv: OpenCv,
  scope: MatScope,
  source: Mat,
  from: Corners,
  to: Corners,
  fill: number
): WarpedPiece | null {
  const fromRect = boundingRect(cv, scope, from);
  const toRect = boundingRect(cv, scope, to);
  if (isEmpty(fromRect) || isEmpty(toRect)) {
    return null;
  }
  const fromCropped = shifted(from, fromRect);
  const toCropped = shifted(to, toRect);
  if (isDegenerate(fromCropped) || isDegenerate(toCropped)) {
    return null;
  }
  const matrix = affineOf(cv, scope, fromCropped, toCropped);
  const rows = pySliceBounds(
    fromRect.y,
    fromRect.y + fromRect.height,
    source.rows
  );
  const cols = pySliceBounds(
    fromRect.x,
    fromRect.x + fromRect.width,
    source.cols
  );
  if (matrix === null || rows.stop === rows.start || cols.stop === cols.start) {
    return null;
  }
  // A clone, because cv2 receives the numpy slice as a Mat with no parent,
  // and an opencv.js roi still knows the image around it.
  const view = scope.keep(
    source.roi(
      new cv.Rect(
        cols.start,
        rows.start,
        cols.stop - cols.start,
        rows.stop - rows.start
      )
    )
  );
  const warped = scope.keep(new cv.Mat());
  cv.warpAffine(
    scope.keep(view.clone()),
    warped,
    scope.keep(cv.matFromArray(2, 3, cv.CV_64F, [...matrix])),
    new cv.Size(toRect.width, toRect.height),
    cv.INTER_LINEAR,
    cv.BORDER_CONSTANT,
    new cv.Scalar(fill, fill, fill, fill)
  );
  const mask = scope.keep(
    cv.Mat.zeros(toRect.height, toRect.width, cv.CV_8UC1)
  );
  const vertices = flat(toCropped).map((v) => Math.trunc(v));
  cv.fillConvexPoly(
    mask,
    pointMat(cv, scope, cv.CV_32SC2, vertices),
    new cv.Scalar(255)
  );
  return {
    mask: Uint8Array.from(mask.data),
    pixels: Uint8Array.from(warped.data),
    rect: toRect,
  };
}

/** `np.where(mask > 0, warped, output)` over the part of the rect inside the output. */
function paint(output: GrayImage, { mask, pixels, rect }: WarpedPiece): void {
  const { width } = output;
  const y2 = Math.min(output.height, rect.y + rect.height);
  const x2 = Math.min(width, rect.x + rect.width);
  for (let y = Math.max(0, rect.y); y < y2; y += 1) {
    for (let x = Math.max(0, rect.x); x < x2; x += 1) {
      const local = (y - rect.y) * rect.width + (x - rect.x);
      if ((mask[local] ?? 0) > 0) {
        output.data[y * width + x] = pixels[local] ?? 0;
      }
    }
  }
}

/**
 * `warp_image(image, fill_color, order=1)`: starts from an image of `fill`
 * and paints each triangle in mesh order, so a later triangle wins on every
 * pixel two masks share.
 */
export function warpImage(
  cv: OpenCv,
  transform: PiecewiseAffineTransform,
  image: GrayImage,
  fill = 1
): GrayImage {
  const output = createGray(image.width, image.height, fill);
  withMatScope((outer) => {
    const source = planeToMat(cv, outer, image);
    transform.mesh.triangles.forEach((triangle, index) => {
      const from = corners(transform.mesh.points, triangle);
      const to = corners(transform.dst, triangle);
      if (
        transform.matrices[index] === null ||
        isDegenerate(from) ||
        isDegenerate(to)
      ) {
        return;
      }
      const piece = withMatScope((scope) =>
        warpTriangle(cv, scope, source, from, to, fill)
      );
      if (piece !== null) {
        paint(output, piece);
      }
    });
  });
  return output;
}

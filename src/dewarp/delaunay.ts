/**
 * homr's DelaunayTriangulation (staff_dewarping.py) over delaunator, since
 * opencv.js 4.12 has no Subdiv2D.
 *
 * Subdiv2D does not triangulate the points alone. It starts from three
 * virtual vertices at a finite distance and triangulates them with the
 * points, and getTriangleList then drops every triangle touching one. Near
 * the hull that is a different triangulation: a flat triangle along the edge
 * of the image has a circumcircle holding a virtual vertex, so Subdiv2D
 * leaves it out and warp_image never paints it. delaunator over the same
 * augmented set reproduces that; what is left is the choice of diagonal in
 * co-circular cells, which the decision log measures.
 */

import Delaunator from "delaunator";
import type { Point } from "../geometry/boxes.js";

export type Triangle = readonly [number, number, number];

/** Points as float32, as homr casts them, and triangles as indices into them. */
export interface TriangleMesh {
  readonly points: readonly Point[];
  readonly triangles: readonly Triangle[];
}

export class TriangulationError extends Error {}

/** `np.array(points, dtype=np.float32)`. */
export function float32Points(points: readonly Point[]): Point[] {
  return points.map((p) => ({ x: Math.fround(p.x), y: Math.fround(p.y) }));
}

/** DelaunayTriangulation(points), which raises under three points. */
export function triangulate(points: readonly Point[]): TriangleMesh {
  if (points.length < 3) {
    throw new TriangulationError("Need at least 3 points for triangulation");
  }
  const float32 = float32Points(points);
  const virtual = subdivVirtualVertices(float32);
  const { triangles } = Delaunator.from(
    [...virtual, ...float32],
    (p) => p.x,
    (p) => p.y
  );
  const out: Triangle[] = [];
  for (let i = 0; i + 2 < triangles.length; i += 3) {
    const a = (triangles[i] ?? 0) - virtual.length;
    const b = (triangles[i + 1] ?? 0) - virtual.length;
    const c = (triangles[i + 2] ?? 0) - virtual.length;
    if (a >= 0 && b >= 0 && c >= 0) {
      out.push([a, b, c]);
    }
  }
  return { points: float32, triangles: out };
}

/**
 * Subdiv2D::initDelaunay's three outer vertices for homr's rect, which is
 * cv2.boundingRect of the float points grown by 10 on every side. OpenCV
 * 4.14 places them 6 * max(width, height) away, as getVertex reads them back
 * on the oracle.
 */
function subdivVirtualVertices(points: readonly Point[]): Point[] {
  const xs = points.map((p) => Math.floor(p.x));
  const ys = points.map((p) => Math.floor(p.y));
  const left = Math.min(...xs) - 10;
  const top = Math.min(...ys) - 10;
  const width = Math.max(...xs) - Math.min(...xs) + 1 + 20;
  const height = Math.max(...ys) - Math.min(...ys) + 1 + 20;
  const big = Math.fround(6 * Math.max(width, height));
  return [
    { x: Math.fround(left + big), y: top },
    { x: left, y: Math.fround(top + big) },
    { x: Math.fround(left - big), y: Math.fround(top - big) },
  ];
}

/**
 * `_point_in_triangle` in float32, as numpy evaluates it on float32 scalars:
 * every product, sum and quotient rounds to float32.
 */
function inTriangle(point: Point, a: Point, b: Point, c: Point): boolean {
  const f = Math.fround;
  const x = f(point.x);
  const y = f(point.y);
  const denom = f(
    f(f(b.y - c.y) * f(a.x - c.x)) + f(f(c.x - b.x) * f(a.y - c.y))
  );
  if (Math.abs(denom) < 1e-10) {
    return false;
  }
  const u = f(
    f(f(f(b.y - c.y) * f(x - c.x)) + f(f(c.x - b.x) * f(y - c.y))) / denom
  );
  const v = f(
    f(f(f(c.y - a.y) * f(x - c.x)) + f(f(a.x - c.x) * f(y - c.y))) / denom
  );
  const w = f(f(1 - u) - v);
  return u >= -1e-10 && v >= -1e-10 && w >= -1e-10;
}

/** find_simplex for one point: the first triangle that holds it, or -1. */
export function findSimplex(mesh: TriangleMesh, point: Point): number {
  return mesh.triangles.findIndex(([i, j, k]) => {
    const a = mesh.points[i];
    const b = mesh.points[j];
    const c = mesh.points[k];
    return (
      a !== undefined &&
      b !== undefined &&
      c !== undefined &&
      inTriangle(point, a, b, c)
    );
  });
}

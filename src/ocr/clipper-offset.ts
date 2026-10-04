/**
 * The part of Clipper 6.4.2's ClipperOffset (Angus Johnson, Boost licence,
 * the copy pyclipper 1.4.0 bundles) that RapidOCR's DBPostProcess.unclip
 * reaches: one closed polygon, round joins, a positive delta, the default
 * MiterLimit 2 and ArcTolerance 0.25.
 *
 * Clipper's Execute ends with a union of the offset polygon, which only
 * removes self-intersections, duplicates and collinear points. The polygon
 * here is a rectangle grown outward, which has none to remove that change its
 * convex hull, and the only consumer is minAreaRect, which reads the hull. So
 * the union is not ported; test/ocr-unclip.test.ts compares the hull's
 * rectangle with pyclipper's on 240 vectors.
 */

export type IntPoint = readonly [number, number];

const ARC_TOLERANCE = 0.25;

/** Clipper's Round: half away from zero, then truncation to cInt. */
const clipperRound = (value: number): number =>
  value < 0 ? Math.trunc(value - 0.5) : Math.trunc(value + 0.5);

/** Clipper's Area, positive for its own orientation. */
function area(path: readonly IntPoint[]): number {
  if (path.length < 3) {
    return 0;
  }
  let sum = 0;
  let previous = path.at(-1) ?? [0, 0];
  for (const point of path) {
    sum += (previous[0] + point[0]) * (previous[1] - point[1]);
    previous = point;
  }
  return -sum * 0.5;
}

function unitNormal(from: IntPoint, to: IntPoint): IntPoint {
  if (from[0] === to[0] && from[1] === to[1]) {
    return [0, 0];
  }
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const f = 1 / Math.sqrt(dx * dx + dy * dy);
  return [dy * f, -dx * f];
}

/** AddPath's clean-up: trailing copies of the first point and consecutive duplicates go. */
function cleaned(path: readonly IntPoint[]): IntPoint[] {
  let high = path.length - 1;
  const [first] = path;
  if (first === undefined) {
    return [];
  }
  while (
    high > 0 &&
    path[high]?.[0] === first[0] &&
    path[high]?.[1] === first[1]
  ) {
    high -= 1;
  }
  const out: IntPoint[] = [first];
  for (let i = 1; i <= high; i += 1) {
    const point = path[i];
    const last = out.at(-1);
    if (
      point !== undefined &&
      last !== undefined &&
      (point[0] !== last[0] || point[1] !== last[1])
    ) {
      out.push(point);
    }
  }
  return out;
}

interface Arc {
  readonly cos: number;
  readonly delta: number;
  readonly sin: number;
  readonly stepsPerRad: number;
}

function arcFor(delta: number): Arc {
  const tolerance = Math.min(ARC_TOLERANCE, Math.abs(delta) * ARC_TOLERANCE);
  let steps = Math.PI / Math.acos(1 - tolerance / Math.abs(delta));
  if (steps > Math.abs(delta) * Math.PI) {
    steps = Math.abs(delta) * Math.PI;
  }
  return {
    cos: Math.cos((2 * Math.PI) / steps),
    delta,
    sin: Math.sin((2 * Math.PI) / steps),
    stepsPerRad: steps / (2 * Math.PI),
  };
}

const offsetBy = (
  point: IntPoint,
  normal: IntPoint,
  delta: number
): IntPoint => [
  clipperRound(point[0] + normal[0] * delta),
  clipperRound(point[1] + normal[1] * delta),
];

/** DoRound: the arc from one edge's normal to the next, stepped by the precomputed rotation. */
function roundJoin(
  out: IntPoint[],
  point: IntPoint,
  from: IntPoint,
  to: IntPoint,
  sinA: number,
  arc: Arc
): void {
  const angle = Math.atan2(sinA, from[0] * to[0] + from[1] * to[1]);
  const steps = Math.max(clipperRound(arc.stepsPerRad * Math.abs(angle)), 1);
  let [x, y] = from;
  for (let i = 0; i < steps; i += 1) {
    out.push(offsetBy(point, [x, y], arc.delta));
    const x2 = x;
    x = x * arc.cos - arc.sin * y;
    y = x2 * arc.sin + y * arc.cos;
  }
  out.push(offsetBy(point, to, arc.delta));
}

/**
 * `PyclipperOffset().AddPath(path, JT_ROUND, ET_CLOSEDPOLYGON)` then
 * `Execute(delta)` for delta > 0, before the union. pyclipper truncates float
 * coordinates to integers on the way in, which the caller does.
 */
export function offsetClosedPolygonRound(
  path: readonly IntPoint[],
  delta: number
): IntPoint[] {
  const source = cleaned(path);
  if (source.length < 3) {
    return [];
  }
  if (area(source) < 0) {
    source.reverse();
  }
  const arc = arcFor(delta);
  const count = source.length;
  const normals = source.map((point, j) =>
    unitNormal(point, source[(j + 1) % count] ?? point)
  );
  const out: IntPoint[] = [];
  let k = count - 1;
  for (const [j, point] of source.entries()) {
    const nk = normals[k] ?? [0, 0];
    const nj = normals[j] ?? [0, 0];
    const sinA = nk[0] * nj[1] - nj[0] * nk[1];
    const cosA = nk[0] * nj[0] + nj[1] * nk[1];
    if (Math.abs(sinA * delta) < 1 && cosA > 0) {
      // Clipper returns here without advancing k, so the next joint measures from the same edge.
      out.push(offsetBy(point, nk, delta));
      continue;
    }
    const clamped =
      Math.abs(sinA * delta) < 1 ? sinA : Math.min(Math.max(sinA, -1), 1);
    if (clamped * delta < 0) {
      out.push(offsetBy(point, nk, delta), point, offsetBy(point, nj, delta));
    } else {
      roundJoin(out, point, nk, nj, clamped, arc);
    }
    k = j;
  }
  return out;
}

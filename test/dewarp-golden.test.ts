/**
 * Each dewarp stage from Python's output of the stage before it: the warp
 * from dewarp-<n>-input.png and Python's control points, never from the
 * port's own crop.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { describe, expect, it } from "vitest";
import {
  findSimplex,
  type Triangle,
  triangulate,
} from "../src/dewarp/delaunay.js";
import {
  type AffineMatrix,
  estimatePiecewiseAffine,
  transformPoint,
  warpImage,
} from "../src/dewarp/piecewise-affine.js";
import {
  dewarpTransformation,
  spanAndOptimalPoints,
} from "../src/dewarp/staff.js";
import type { Point } from "../src/geometry/boxes.js";
import { decodeStaff } from "../src/golden/decode.js";
import { type GrayImage, planeFromBytes } from "../src/image/plane.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";
import { vectorSet } from "./support/vectors.js";

const meanAbsoluteDifference = (a: GrayImage, b: GrayImage): number => {
  if (a.width !== b.width || a.height !== b.height) {
    return Number.POSITIVE_INFINITY;
  }
  let total = 0;
  for (let i = 0; i < a.data.length; i += 1) {
    total += Math.abs((a.data[i] ?? 0) - (b.data[i] ?? 0));
  }
  return total / a.data.length;
};

const maxMatrixDifference = (
  a: readonly (AffineMatrix | null)[],
  b: readonly (AffineMatrix | null)[]
): number => {
  if (a.length !== b.length) {
    return Number.POSITIVE_INFINITY;
  }
  let worst = 0;
  a.forEach((m, i) => {
    const other = b[i];
    if ((m === null) !== (other === null || other === undefined)) {
      worst = Number.POSITIVE_INFINITY;
      return;
    }
    m?.forEach((v, j) => {
      worst = Math.max(worst, Math.abs(v - (other?.[j] ?? Number.NaN)));
    });
  });
  return worst;
};

/**
 * testing.md's ceiling for a typeset page. The warp on delaunator's mesh
 * differs from Subdiv2D's only where the two pick another diagonal of a
 * co-circular cell; the measured values are in the decision log.
 */
const TYPESET_CEILING = 1;

const affineOf = (value: unknown): (AffineMatrix | null)[] =>
  (value as (number[][] | null)[]).map((m) =>
    m === null ? null : (m.flat() as unknown as AffineMatrix)
  );

const differing = (a: GrayImage, b: GrayImage): number =>
  a.data.reduce((count, v, i) => count + (v === b.data[i] ? 0 : 1), 0);

describe.each(listGoldenFixtures())("$name", (fixture) => {
  const page = goldenPageOf(fixture);
  const canvases = Array.from({ length: page.staffCount() }, (_, i) => i);

  it.each(canvases)(
    "canvas %i: the warp on Python's mesh is exact",
    async (n) => {
      const cv = await testOpenCv();
      const dewarp = page.dewarp(n);
      const transform = estimatePiecewiseAffine(
        cv,
        { points: dewarp.src, triangles: dewarp.simplices },
        dewarp.dst
      );
      expect(
        maxMatrixDifference(transform.matrices, dewarp.affine)
      ).toBeLessThan(1e-9);
      const warped = warpImage(cv, transform, page.dewarpImage(n, "input"));
      expect(differing(warped, page.dewarpImage(n, "warped"))).toBe(0);
    }
  );

  it.each(canvases)("canvas %i: the warp on delaunator's mesh", async (n) => {
    const cv = await testOpenCv();
    const dewarp = page.dewarp(n);
    const transform = estimatePiecewiseAffine(
      cv,
      triangulate(dewarp.src),
      dewarp.dst
    );
    const warped = warpImage(cv, transform, page.dewarpImage(n, "input"));
    const golden = page.dewarpImage(n, "warped");
    expect(transform.mesh.triangles).toHaveLength(dewarp.simplices.length);
    expect(meanAbsoluteDifference(warped, golden)).toBeLessThanOrEqual(
      TYPESET_CEILING
    );
  });
});

const pointsOf = (value: unknown): Point[] =>
  (value as [number, number][]).map(([x, y]) => ({ x, y }));

const grayPng = (name: string): GrayImage => {
  const png = PNG.sync.read(
    readFileSync(join(import.meta.dirname, "golden", "vectors", name))
  );
  const data = new Uint8Array(png.width * png.height);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = png.data[i * 4] ?? 0;
  }
  return planeFromBytes("gray", png.width, png.height, data);
};

describe("spanAndOptimalPoints and dewarpTransformation, dewarp-points.json", () => {
  const { cases } = vectorSet("dewarp-points");
  it.each(cases.map((c) => [c.name, c] as const))("%s", async (_, one) => {
    const staff = decodeStaff(one.staff, {}, "staff");
    const width = one.width as number;
    const height = one.height as number;
    const points = spanAndOptimalPoints(staff, width, height);
    expect(points.span).toEqual((one.span as unknown[]).map(pointsOf));
    expect(points.optimal).toEqual((one.optimal as unknown[]).map(pointsOf));
    if (points.span.length === 0) {
      return;
    }
    const cv = await testOpenCv();
    const transform = dewarpTransformation(cv, width, height, points);
    expect(transform.mesh.points).toEqual(pointsOf(one.src));
    expect(transform.dst).toEqual(pointsOf(one.dst));
    const python = estimatePiecewiseAffine(
      cv,
      { points: pointsOf(one.src), triangles: one.simplices as Triangle[] },
      pointsOf(one.dst)
    );
    expect(
      maxMatrixDifference(python.matrices, affineOf(one.affine))
    ).toBeLessThan(1e-9);
  });
});

describe("the warp of a curved staff on delaunator's mesh, dewarp-warp.json", () => {
  const { cases } = vectorSet("dewarp-warp");
  it.each(cases.map((c) => [c.name, c] as const))("%s", async (_, one) => {
    const cv = await testOpenCv();
    const mesh = triangulate(pointsOf(one.src));
    const transform = estimatePiecewiseAffine(cv, mesh, pointsOf(one.dst));
    const warped = warpImage(cv, transform, grayPng(one.input as string));
    const golden = grayPng(one.warped as string);
    expect(mesh.triangles).toHaveLength((one.simplices as unknown[]).length);
    expect(meanAbsoluteDifference(warped, golden)).toBeLessThanOrEqual(
      TYPESET_CEILING
    );
    const python = estimatePiecewiseAffine(
      cv,
      { points: pointsOf(one.src), triangles: one.simplices as Triangle[] },
      pointsOf(one.dst)
    );
    expect(
      differing(warpImage(cv, python, grayPng(one.input as string)), golden)
    ).toBe(0);
    for (const [point, simplex, moved] of one.probes as [
      [number, number],
      number,
      [number, number],
    ][]) {
      const p = { x: point[0], y: point[1] };
      expect(findSimplex(python.mesh, p)).toBe(simplex);
      const q = transformPoint(python, p);
      expect(q.x).toBeCloseTo(moved[0], 9);
      expect(q.y).toBeCloseTo(moved[1], 9);
    }
  });
});

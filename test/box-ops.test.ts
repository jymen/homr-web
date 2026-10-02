/**
 * BoxOps on the real opencv.js, against the pairs Python answered for, and as
 * a literal: the reason the interface exists is that an ordering rule can be
 * tested with no opencv.js at all.
 */

import { describe, expect, it } from "vitest";
import {
  createCvBoxOps,
  isIntersecting,
  withBoxOps,
} from "../src/cv/box-ops.js";
import { type Deletable, MatScope, withMatScope } from "../src/cv/opencv.js";
import { type BoxOps, overlapsAny } from "../src/geometry/box-ops.js";
import { rotatedBoxFromRect } from "../src/geometry/box-transforms.js";
import {
  legacyConventionRectOf,
  pointListFromPairs,
  type RotatedBox,
} from "../src/geometry/boxes.js";
import { testOpenCv } from "./support/opencv.js";
import { type VectorCase, vectorSet } from "./support/vectors.js";

type VectorRect = readonly [
  readonly [number, number],
  readonly [number, number],
  number,
];

const boxOf = (value: unknown, debugId = 0): RotatedBox => {
  const [[cx, cy], [w, h], angle] = value as VectorRect;
  return rotatedBoxFromRect(
    legacyConventionRectOf({ angle, cx, cy, h, w }),
    pointListFromPairs([]),
    debugId
  );
};

class CountingScope extends MatScope {
  kept = 0;

  override keep<T extends Deletable>(value: T): T {
    this.kept += 1;
    return super.keep(value);
  }
}

describe("intersections.json", () => {
  const { cases } = vectorSet("intersections");
  const wrongOn = (
    answer: (one: VectorCase) => unknown,
    key: "code" | "intersecting"
  ): string[] =>
    cases.flatMap((one, i) => {
      const got = answer(one);
      return got === one[key]
        ? []
        : [`cases[${i}]: ${String(got)}, Python ${String(one[key])}`];
    });

  it("opencv.js's rotatedRectangleIntersection returns cv2's code on every pair", async () => {
    const cv = await testOpenCv();
    const wrong = withMatScope((scope) => {
      const region = scope.keep(new cv.Mat());
      const native = (value: unknown) => {
        const { rect } = boxOf(value);
        return {
          angle: rect.angle,
          center: { x: rect.cx, y: rect.cy },
          size: { height: rect.h, width: rect.w },
        };
      };
      return wrongOn(
        (one) =>
          cv.rotatedRectangleIntersection(native(one.a), native(one.b), region),
        "code"
      );
    });
    expect(wrong).toEqual([]);
  });

  it("isIntersecting is RotatedBoundingBox.is_intersecting on every pair, touching ones included", async () => {
    const cv = await testOpenCv();
    const wrong = withBoxOps(cv, (ops) =>
      wrongOn(
        (one) => ops.intersects(boxOf(one.a), boxOf(one.b)),
        "intersecting"
      )
    );
    expect(wrong).toEqual([]);
    expect(
      cases.filter((one) => one.intersecting === true).length
    ).toBeGreaterThan(50);
    expect(
      cases.filter((one) => one.intersecting === false).length
    ).toBeGreaterThan(50);
  });

  it("answers every intersects of a scope with one Mat", async () => {
    const cv = await testOpenCv();
    const scope = new CountingScope();
    try {
      const ops = createCvBoxOps(cv, scope);
      for (const one of cases) {
        ops.intersects(boxOf(one.a), boxOf(one.b));
      }
      expect(scope.kept).toBe(1);
    } finally {
      scope.release();
    }
  });

  it("does not ask opencv.js about boxes too far apart to touch", async () => {
    const cv = await testOpenCv();
    const far = boxOf([[500, 500], [20, 10], 0]);
    const near = boxOf([[50, 50], [20, 10], 0]);
    const unusable = null as unknown as Parameters<typeof isIntersecting>[1];
    expect(isIntersecting(cv, unusable, near, far)).toBe(false);
  });
});

describe("createCvBoxOps", () => {
  const upright = boxOf([[50, 50], [20, 10], 0], 7);

  it("keeps one outline Mat per box however often it is compared", async () => {
    const cv = await testOpenCv();
    const scope = new CountingScope();
    try {
      const ops = createCvBoxOps(cv, scope);
      const others = [
        boxOf([[60, 50], [20, 10], 0]),
        boxOf([[55, 52], [4, 4], 0]),
      ];
      const before = scope.kept;
      for (let i = 0; i < 20; i += 1) {
        expect(overlapsAny(ops, upright, others)).toBe(true);
      }
      expect(scope.kept - before).toBe(2);
    } finally {
      scope.release();
    }
  });

  it("thickens a rotated box in place of its centre and keeps its kind", async () => {
    const cv = await testOpenCv();
    const thick = withBoxOps(cv, (ops) => ops.thicker(upright, 6));
    expect(thick.kind).toBe("rotated");
    expect(thick.rect).toMatchObject({ cx: 50, cy: 50, h: 16, w: 26 });
    expect(thick.debugId).toBe(7);
  });

  it("fits a rotated box to a contour with no size check", async () => {
    const cv = await testOpenCv();
    const contour = pointListFromPairs([
      [10, 10],
      [30, 10],
      [30, 14],
      [10, 14],
    ]);
    const fitted = withBoxOps(cv, (ops) => ops.fitRotatedBox(contour, 3));
    expect(fitted.rect).toMatchObject({
      angle: 0,
      cx: 20,
      cy: 12,
      h: 4,
      w: 20,
    });
    expect(fitted.debugId).toBe(3);
    expect(fitted.contour).toBe(contour);
  });

  it("builds an ellipse from numbers, normalised, with an ellipse2Poly outline", async () => {
    const cv = await testOpenCv();
    const ellipse = withBoxOps(cv, (ops) =>
      ops.ellipseFromRect(
        legacyConventionRectOf({ angle: 90, cx: 40, cy: 30, h: 22, w: 18 }),
        pointListFromPairs([]),
        5
      )
    );
    expect(ellipse.kind).toBe("ellipse");
    expect(ellipse.rect).toMatchObject({
      angle: 0,
      cx: 40,
      cy: 30,
      h: 18,
      w: 22,
    });
    expect(ellipse.polygon.length).toBeGreaterThan(8);
  });
});

describe("overlapsAny", () => {
  it("asks a literal BoxOps in list order and stops at the first overlap", () => {
    const asked: number[] = [];
    const ops: BoxOps = {
      ellipseFromRect: () => {
        throw new Error("not asked");
      },
      fitRotatedBox: () => {
        throw new Error("not asked");
      },
      intersects: () => false,
      overlaps: (_box, other) => {
        asked.push(other.debugId);
        return other.debugId === 2;
      },
      thicker: (box) => box,
    };
    const others = [1, 2, 3].map((id) => boxOf([[0, 0], [1, 1], 0], id));
    expect(overlapsAny(ops, others[0] as RotatedBox, others)).toBe(true);
    expect(asked).toEqual([1, 2]);
    expect(overlapsAny(ops, others[0] as RotatedBox, [])).toBe(false);
  });
});

describe("rotatedBoxFromRect", () => {
  it("normalises the rect and draws the outline from the rect as handed over", () => {
    const box = rotatedBoxFromRect(
      legacyConventionRectOf({ angle: 90, cx: 20, cy: 15, h: 20, w: 10 }),
      pointListFromPairs([]),
      4
    );
    expect(box.rect).toMatchObject({ angle: 0, cx: 20, cy: 15, h: 10, w: 20 });
    expect(box.debugId).toBe(4);
    expect(box.polygon).toHaveLength(8);
  });
});

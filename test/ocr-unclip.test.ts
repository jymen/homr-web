import { describe, expect, it } from "vitest";
import { withMatScope } from "../src/cv/opencv.js";
import type { IntPoint } from "../src/ocr/clipper-offset.js";
import {
  miniBox,
  type Quad,
  unclip,
  unclipDistance,
} from "../src/ocr/db-postprocess.js";
import { testOpenCv } from "./support/opencv.js";
import { vectorSet } from "./support/vectors.js";

interface UnclipCase {
  readonly box: Quad;
  readonly distance: number;
  readonly expanded: readonly IntPoint[];
  readonly points: Quad;
  readonly sside: number;
}

const cases = vectorSet("ocr-unclip").cases as unknown as readonly UnclipCase[];
/** phase-5-minarearect.md: opencv.js and opencv-python's rotatingCalipers round apart, 460 of 2573 rects by up to 1e-3. */
const MINAREARECT_DRIFT = 1e-3;
const key = ([x, y]: IntPoint) => `${x},${y}`;

describe("DBPostProcess.unclip and get_mini_boxes", () => {
  it(`gives shapely's distance on ${cases.length} rectangles`, () => {
    for (const c of cases) {
      expect(unclipDistance(c.points)).toBe(c.distance);
    }
  });

  it("keeps every vertex pyclipper keeps after its union", () => {
    const missing = cases.flatMap((c, i) => {
      const mine = new Set(unclip(c.points).map(key));
      return c.expanded
        .filter((p) => !mine.has(key(p)))
        .map((p) => `case ${i}: ${key(p)}`);
    });
    expect(missing).toEqual([]);
  });

  it("gives the expanded rectangle Python gives, within minAreaRect's float32 drift", async () => {
    const cv = await testOpenCv();
    const drifts = withMatScope((scope) =>
      cases.map((c) => {
        const got = miniBox(cv, scope, unclip(c.points));
        return Math.max(
          Math.abs(got.sside - c.sside),
          ...got.box.flatMap((p, j) => [
            Math.abs(p[0] - (c.box[j]?.[0] ?? Number.NaN)),
            Math.abs(p[1] - (c.box[j]?.[1] ?? Number.NaN)),
          ])
        );
      })
    );
    const exact = drifts.filter((d) => d === 0).length;
    process.stdout.write(
      `unclip: ${exact} of ${cases.length} rectangles exact, worst ${Math.max(...drifts).toExponential(2)}\n`
    );
    expect(Math.max(...drifts)).toBeLessThanOrEqual(MINAREARECT_DRIFT);
  });
});

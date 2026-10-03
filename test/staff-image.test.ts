/**
 * prepare_staff_image stage by stage, each from Python's output of the stage
 * before it, then the whole of staffCanvases from multistaffs.json and
 * preprocessed.png against canvas-<n>.png.
 */

import { describe, expect, it } from "vitest";
import { spanAndOptimalPoints } from "../src/dewarp/staff.js";
import type { Point } from "../src/geometry/boxes.js";
import { decodeStaffs } from "../src/golden/decode.js";
import {
  cropPlaneAndReturnNewTop,
  type GrayImage,
} from "../src/image/plane.js";
import { createMultiStaff, type Staff } from "../src/model/staff.js";
import {
  centerImageOnCanvas,
  ensureSameNumberOfStaffs,
  removeBlackContoursAtEdges,
  resizeGray,
  staffCanvases,
  staffInCrop,
  staffRegion,
  staffRegions,
  startOfClosestStaffAbove,
  startOfClosestStaffBelow,
  trOmrCanvasSize,
} from "../src/pipeline/staff-image.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";
import { vectorSet } from "./support/vectors.js";

const differing = (a: GrayImage, b: GrayImage): number =>
  a.width === b.width && a.height === b.height
    ? a.data.reduce((count, v, i) => count + (v === b.data[i] ? 0 : 1), 0)
    : Number.POSITIVE_INFINITY;

const maxDifference = (a: GrayImage, b: GrayImage): number =>
  a.width === b.width && a.height === b.height
    ? a.data.reduce(
        (worst, v, i) => Math.max(worst, Math.abs(v - (b.data[i] ?? 0))),
        0
      )
    : Number.POSITIVE_INFINITY;

const meanAbsoluteDifference = (a: GrayImage, b: GrayImage): number => {
  let total = 0;
  for (let i = 0; i < a.data.length; i += 1) {
    total += Math.abs((a.data[i] ?? 0) - (b.data[i] ?? 0));
  }
  return total / a.data.length;
};

/** The largest difference between two staffs' grid ordinates, grid x and symbol centres. */
function staffDistance(a: Staff, b: Staff): number {
  if (
    a.grid.length !== b.grid.length ||
    a.symbols.length !== b.symbols.length
  ) {
    return Number.POSITIVE_INFINITY;
  }
  let worst = 0;
  a.grid.forEach((p, i) => {
    const q = b.grid[i];
    worst = Math.max(worst, Math.abs(p.x - (q?.x ?? Number.NaN)));
    p.y.forEach((y, j) => {
      worst = Math.max(worst, Math.abs(y - (q?.y[j] ?? Number.NaN)));
    });
  });
  a.symbols.forEach((s, i) => {
    const t = b.symbols[i];
    worst = Math.max(
      worst,
      Math.abs(s.center.x - (t?.center.x ?? Number.NaN)),
      Math.abs(s.center.y - (t?.center.y ?? Number.NaN))
    );
  });
  return worst;
}

/**
 * cv2.resize on the arm64 oracle runs OpenCV's KleidiCV HAL, whose
 * INTER_LINEAR lands up to 1.93 gray levels from the exact bilinear value;
 * opencv.js stays within 0.53 of it. The two differ by up to two levels on a
 * few percent of the pixels (measured in the decision log). Every resize is
 * checked to within two levels, and the stage after it starts from Python's
 * image again.
 */
const RESIZE_TOLERANCE = 2;

describe("regrouping, staff-regrouping.json", () => {
  const { cases } = vectorSet("staff-regrouping");
  for (const one of cases) {
    if ("sizes" in one) {
      it("trOmrCanvasSize", () => {
        for (const [h, w, [width, height]] of one.sizes as [
          number,
          number,
          [number, number],
        ][]) {
          expect(trOmrCanvasSize(h, w)).toEqual({ height, width });
        }
      });
      continue;
    }
    it(one.name as string, () => {
      const staffs = decodeStaffs(one.staffs, "staffs");
      const systems = (one.systems as number[][]).map((indices) =>
        createMultiStaff(indices.map((i) => staffs[i] as Staff))
      );
      const result = ensureSameNumberOfStaffs(systems, one.height as number);
      expect(
        result.map((ms) => ms.staffs.map((s) => staffs.indexOf(s)))
      ).toEqual(one.result);
      const regions = staffRegions(systems);
      for (const [y, above, below] of one.regions as [
        number,
        number,
        number,
      ][]) {
        expect([
          startOfClosestStaffAbove(regions, y),
          startOfClosestStaffBelow(regions, y),
        ]).toEqual([above, below]);
      }
    });
  }
});

describe.each(listGoldenFixtures())("$name", (fixture) => {
  const page = goldenPageOf(fixture);
  const canvases = Array.from({ length: page.staffCount() }, (_, i) => i);
  const systems = () =>
    ensureSameNumberOfStaffs(page.multiStaffs(), page.preprocessed().height);
  const staffOf = (n: number): Staff => {
    const all = systems();
    const voices = all[0]?.staffs.length ?? 1;
    const staff =
      all[n % all.length]?.staffs[Math.floor(n / all.length) % voices];
    if (staff === undefined) {
      throw new Error(`no staff for canvas ${n}`);
    }
    return staff;
  };

  it.each(canvases)("canvas %i: region, size and scale", (n) => {
    const dewarp = page.dewarp(n);
    const region = staffRegion(staffOf(n), staffRegions(systems()));
    expect(region).toEqual(dewarp.region);
    const size = trOmrCanvasSize(region[3] - region[1], region[2] - region[0]);
    expect([size.width, size.height]).toEqual(dewarp.imageDimensions);
    expect(size.height / (region[3] - region[1])).toBe(dewarp.scalingFactor);
  });

  it.each(canvases)(
    "canvas %i: the resize and the first crop, within two levels",
    async (n) => {
      const cv = await testOpenCv();
      const dewarp = page.dewarp(n);
      const [width, height] = dewarp.resizedSize;
      const resized = resizeGray(cv, page.preprocessed(), width, height);
      const crop = cropPlaneAndReturnNewTop(resized, ...dewarp.regionStep1);
      expect({ x: crop.left, y: crop.top }).toEqual(dewarp.topLeftStep1);
      expect(
        maxDifference(crop.plane, page.dewarpImage(n, "input"))
      ).toBeLessThanOrEqual(RESIZE_TOLERANCE);
    }
  );

  it.each(canvases)("canvas %i: the staff in the crop", (n) => {
    const dewarp = page.dewarp(n);
    const topLeft: Point = {
      x: dewarp.topLeftStep1.x / dewarp.scalingFactor,
      y: dewarp.topLeftStep1.y / dewarp.scalingFactor,
    };
    const moved = staffInCrop(staffOf(n), topLeft, dewarp.scalingFactor);
    expect(staffDistance(moved, page.canvasStaff(n))).toBeLessThan(1e-9);
  });

  it.each(canvases)("canvas %i: the control points are exact", (n) => {
    const dewarp = page.dewarp(n);
    const input = page.dewarpImage(n, "input");
    const points = spanAndOptimalPoints(
      page.canvasStaff(n),
      input.width,
      input.height
    );
    expect(points.span).toEqual(dewarp.spanPoints);
    expect(points.optimal).toEqual(dewarp.optimalPoints);
  });

  it.each(canvases)(
    "canvas %i: the second crop and the clean-up are exact",
    async (n) => {
      const cv = await testOpenCv();
      const dewarp = page.dewarp(n);
      const crop = cropPlaneAndReturnNewTop(
        page.dewarpImage(n, "warped"),
        ...dewarp.regionStep2
      );
      expect({ x: crop.left, y: crop.top }).toEqual(dewarp.topLeftStep2);
      const cleaned = removeBlackContoursAtEdges(
        cv,
        crop.plane,
        page.canvasStaff(n).averageUnitSize
      );
      expect(differing(cleaned, page.dewarpImage(n, "cleaned"))).toBe(0);
    }
  );

  it.each(canvases)("canvas %i: the centring, within two levels", async (n) => {
    const cv = await testOpenCv();
    const [width, height] = page.dewarp(n).imageDimensions;
    const canvas = centerImageOnCanvas(cv, page.dewarpImage(n, "cleaned"), {
      height,
      width,
    });
    expect(maxDifference(canvas, page.canvas(n))).toBeLessThanOrEqual(
      RESIZE_TOLERANCE
    );
  });

  it("staffCanvases from multistaffs.json and preprocessed.png", async () => {
    const cv = await testOpenCv();
    const result = staffCanvases(cv, page.multiStaffs(), page.preprocessed());
    expect(result).toHaveLength(page.staffCount());
    result.forEach((canvas, n) => {
      const golden = page.canvas(n);
      expect(meanAbsoluteDifference(canvas.image, golden)).toBeLessThanOrEqual(
        1
      );
      expect(staffDistance(canvas.staff, page.canvasStaff(n))).toBeLessThan(
        1e-9
      );
    });
  });
});

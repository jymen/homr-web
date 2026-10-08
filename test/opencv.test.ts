import type { Mat } from "@techstark/opencv-js";
import { describe, expect, it, vi } from "vitest";
import {
  CvError,
  loadOpenCv,
  MatScope,
  type OpenCvSource,
  withMatScope,
} from "../src/cv/opencv.js";
import { nodeOpenCvSource, testOpenCv } from "./support/opencv.js";

/**
 * loadOpenCv memoises per realm, so the two loads that must fail come first: a
 * successful load moved above them would be handed back to them instead of their
 * own source, and they would fail rather than quietly stop testing anything. No
 * test below needs an earlier one to have run, so each still passes when it is
 * the only one selected.
 */

const ACQUISITION_FAILED = new Error("the opencv.js bundle could not be read");
const SCOPE_BODY_FAILED = new Error("the operation inside the scope failed");
/** Pins the order REQUIRED_MEMBERS is written in, which is what makes a wrong-build message reproducible. */
const NAMES_EVERY_MISSING_MEMBER = /no boundingRect, calcHist, CLAHE/;
const PORT_MEMBERS = [
  "boundingRect",
  "calcHist",
  "CLAHE",
  "connectedComponentsWithStats",
  "contourArea",
  "cvtColor",
  "dilate",
  "ellipse2Poly",
  "erode",
  "findContours",
  "fillConvexPoly",
  "fitEllipse",
  "getAffineTransform",
  "getStructuringElement",
  "Mat",
  "matFromArray",
  "MatVector",
  "minAreaRect",
  "morphologyEx",
  "pointPolygonTest",
  "PointVector",
  "resize",
  "rotatedRectangleIntersection",
  "Size",
  "subtract",
  "threshold",
  "warpAffine",
];

const countingSource = (counter: { calls: number }): OpenCvSource => {
  const inner = nodeOpenCvSource();
  return () => {
    counter.calls += 1;
    return inner();
  };
};

/**
 * Complete, and thenable the way the real module is, but with a `then` that
 * settles with something else instead of recursing. A loader that stopped
 * clearing `then` hands back that sentinel, so the assertion goes red where the
 * real module would only hang.
 */
const thenableModule = (): Record<string, unknown> => {
  const module: Record<string, unknown> = {
    then: (settle: (value: unknown) => void) => settle("the then shim won"),
  };
  for (const name of PORT_MEMBERS) {
    module[name] = () => undefined;
  }
  return module;
};

describe("loadOpenCv", () => {
  it("refuses a module object that is not the pinned build", async () => {
    const refused = loadOpenCv(() => Promise.resolve({ module: {} }));
    await expect(refused).rejects.toBeInstanceOf(CvError);
    await expect(refused).rejects.toThrow(NAMES_EVERY_MISSING_MEMBER);
  });

  it.each([
    "erode",
    "getStructuringElement",
    "rotatedRectangleIntersection",
    "subtract",
    "warpAffine",
  ])("refuses a build with no %s, and names it", async (name) => {
    vi.resetModules();
    const fresh = await import("../src/cv/opencv.js");
    const { then: _then, [name]: _member, ...module } = thenableModule();
    module.Mat = () => undefined;
    await expect(
      fresh.loadOpenCv(() => Promise.resolve({ module }))
    ).rejects.toThrow(`no ${name}.`);
  });

  it("does not cache a rejected load, initializes once, then stops asking", async () => {
    await expect(
      loadOpenCv(() => Promise.reject(ACQUISITION_FAILED))
    ).rejects.toBe(ACQUISITION_FAILED);
    const shared = { calls: 0 };
    const source = countingSource(shared);
    const [first, second] = await Promise.all([
      loadOpenCv(source),
      loadOpenCv(source),
    ]);
    expect(shared.calls).toBe(1);
    expect(first).toBe(second);
    const later = { calls: 0 };
    await loadOpenCv(countingSource(later));
    expect(later.calls).toBe(0);
  });

  it("hands back the module itself, not what its `then` shim settles with", async () => {
    vi.resetModules();
    const fresh = await import("../src/cv/opencv.js");
    const module = thenableModule();
    const loaded = await fresh.loadOpenCv(() => Promise.resolve({ module }));
    expect(loaded).toBe(module);
  });

  it("loads the pinned 4.12.0 build", async () => {
    const cv = await testOpenCv();
    expect(String(cv.getBuildInformation())).toContain("4.12.0");
  });

  it("round-trips bytes through a Mat", async () => {
    const cv = await testOpenCv();
    const read = withMatScope((scope) => {
      const mat = scope.keep(cv.matFromArray(2, 2, cv.CV_8UC1, [1, 2, 3, 4]));
      return Array.from(mat.data);
    });
    expect(read).toEqual([1, 2, 3, 4]);
  });
});

describe("MatScope", () => {
  it("frees what it kept, and repeating that is a no-op", async () => {
    const cv = await testOpenCv();
    const scope = new MatScope();
    const mat = scope.keep(new cv.Mat());
    expect(mat.isDeleted()).toBe(false);
    scope.release();
    expect(mat.isDeleted()).toBe(true);
    scope.release();
    expect(mat.isDeleted()).toBe(true);
  });

  it("tolerates a value the caller freed itself", async () => {
    const cv = await testOpenCv();
    const scope = new MatScope();
    const mat = scope.keep(new cv.Mat());
    mat.delete();
    expect(() => scope.release()).not.toThrow();
  });

  it("releases even when the body throws", async () => {
    const cv = await testOpenCv();
    let kept: Mat | undefined;
    expect(() =>
      withMatScope((scope) => {
        kept = scope.keep(new cv.Mat());
        throw SCOPE_BODY_FAILED;
      })
    ).toThrow(SCOPE_BODY_FAILED);
    expect(kept?.isDeleted()).toBe(true);
  });
});

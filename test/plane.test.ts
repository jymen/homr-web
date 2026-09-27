import { describe, expect, it } from "vitest";
import {
  applyMask,
  argmaxPlanes,
  blitInPlace,
  colorImageFromRgba,
  createClassMap,
  createGray,
  createMask,
  cropPlane,
  cropPlaneAndReturnNewTop,
  fillRectInPlace,
  grayFromMask,
  maskOfClass,
  meanOfRegion,
  nonzeroRowBounds,
  PlaneError,
  planeAgreement,
  planeFromBytes,
  rowNonzeroCounts,
  sampleIndex,
} from "../src/image/plane.js";

const gray4x3 = () => {
  const g = createGray(4, 3);
  for (let i = 0; i < g.data.length; i += 1) {
    g.data[i] = i;
  }
  return g;
};

describe("factories and wrapping", () => {
  it("enforces the byte-length invariant and the 0/1 rule for masks", () => {
    expect(() => planeFromBytes("gray", 2, 2, new Uint8Array(3))).toThrow(
      PlaneError
    );
    expect(() =>
      planeFromBytes("mask", 2, 1, Uint8Array.from([0, 255]))
    ).toThrow(PlaneError);
    const mask = planeFromBytes("mask", 2, 1, Uint8Array.from([0, 1]));
    expect(mask.kind).toBe("mask");
    expect(mask.channels).toBe(1);
    expect(createMask(3, 2).data).toHaveLength(6);
    expect(createClassMap(3, 2).kind).toBe("classes");
  });
  it("converts RGBA to BGR", () => {
    const img = colorImageFromRgba(
      1,
      1,
      Uint8ClampedArray.from([10, 20, 30, 255])
    );
    expect(Array.from(img.data)).toEqual([30, 20, 10]);
    expect(sampleIndex(img, 0, 0, 2)).toBe(2);
  });
});

describe("crop (image_utils.crop_image)", () => {
  it("orders corners, clamps to size - 1, slices half-open", () => {
    const g = gray4x3();
    const { plane, left, top } = cropPlaneAndReturnNewTop(g, 3, 2.5, 1, -4);
    expect([left, top]).toEqual([1, 0]);
    expect([plane.width, plane.height]).toEqual([2, 2]);
    expect(Array.from(plane.data)).toEqual([1, 2, 5, 6]);
    expect(cropPlane(g, 0, 0, 100, 100).width).toBe(3);
  });
});

describe("in-place writes", () => {
  it("blits with clipping and fills a rectangle", () => {
    const dst = createGray(3, 3);
    const src = createGray(2, 2, 7);
    blitInPlace(dst, src, 2, 2);
    expect(Array.from(dst.data)).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 7]);
    fillRectInPlace(dst, 0, 0, 2, 1, 9);
    expect(Array.from(dst.data.subarray(0, 3))).toEqual([9, 9, 0]);
    expect(() => blitInPlace(dst, createMask(1, 1) as never, 0, 0)).toThrow(
      PlaneError
    );
  });
});

describe("reductions", () => {
  it("counts non-zero samples per row and finds the non-zero row span", () => {
    const g = createGray(3, 3);
    g.data[4] = 5;
    g.data[7] = 5;
    expect(Array.from(rowNonzeroCounts(g))).toEqual([0, 1, 1]);
    expect(nonzeroRowBounds(g, 0, 0, 3, 3)).toEqual({ maxY: 2, minY: 1 });
    expect(nonzeroRowBounds(g, 0, 0, 1, 3)).toBeNull();
    expect(meanOfRegion(g, 1, 1, 2, 3)).toBe(5);
    expect(meanOfRegion(g, 2, 2, 2, 2)).toBeNaN();
  });
  it("splits a class map, masks a plane and measures agreement", () => {
    const logits = Float32Array.from([0, 1, 5, 5, 1, 0, 9, 4]);
    const classes = argmaxPlanes(logits, 2, 2, 2);
    expect(Array.from(classes.data)).toEqual([1, 0, 1, 0]);
    const mask = maskOfClass(classes, 1);
    expect(Array.from(mask.data)).toEqual([1, 0, 1, 0]);
    const g = createGray(2, 2, 200);
    expect(Array.from(applyMask(g, mask).data)).toEqual([200, 0, 200, 0]);
    expect(Array.from(grayFromMask(mask).data)).toEqual([255, 0, 255, 0]);
    expect(planeAgreement(mask, maskOfClass(classes, 0))).toBe(0);
    expect(planeAgreement(mask, mask)).toBe(1);
  });
});

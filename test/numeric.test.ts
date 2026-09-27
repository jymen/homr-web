import { describe, expect, it } from "vitest";
import {
  argmax,
  argmin,
  clampToIndex,
  diff,
  floorDiv,
  formatPythonFloat,
  mean,
  median,
  roundHalfEven,
  std,
  truncToInt,
} from "../src/image/numeric.js";

describe("roundHalfEven (Python round)", () => {
  it.each([
    [0.5, 0],
    [1.5, 2],
    [2.5, 2],
    [3.5, 4],
    [-0.5, 0],
    [-1.5, -2],
    [-2.5, -2],
    [2.4999, 2],
    [2.5001, 3],
    [7, 7],
    [-7, -7],
  ])("round(%s) = %s", (x, expected) => {
    expect(roundHalfEven(x)).toBe(expected);
  });
});

describe("floorDiv and truncToInt", () => {
  it("floors toward minus infinity like Python //", () => {
    expect(floorDiv(7, 2)).toBe(3);
    expect(floorDiv(-7, 2)).toBe(-4);
    expect(floorDiv(-1, 20)).toBe(-1);
    expect(floorDiv(1, 20)).toBe(0);
  });
  it("truncates toward zero like int()", () => {
    expect(truncToInt(2.9)).toBe(2);
    expect(truncToInt(-2.9)).toBe(-2);
  });
});

describe("clampToIndex (image_utils._limit_x)", () => {
  it("rounds half to even, then clamps", () => {
    expect(clampToIndex(2.5, 100)).toBe(2);
    expect(clampToIndex(3.5, 100)).toBe(4);
    expect(clampToIndex(-3, 100)).toBe(0);
    expect(clampToIndex(250, 100)).toBe(99);
    expect(clampToIndex(99.5, 100)).toBe(99);
  });
});

describe("numpy reductions", () => {
  it("median of even and odd counts", () => {
    expect(median([3, 1, 2])).toBe(2);
    expect(median([4, 1, 3, 2])).toBe(2.5);
    expect(median([5])).toBe(5);
    expect(median([])).toBeNaN();
  });
  it("mean, std (population) and diff", () => {
    expect(mean([1, 2, 3, 4])).toBe(2.5);
    expect(mean([])).toBeNaN();
    expect(std([2, 4, 4, 4, 5, 5, 7, 9])).toBe(2);
    expect(Array.from(diff([1, 4, 9, 16]))).toEqual([3, 5, 7]);
    expect(diff([1])).toHaveLength(0);
  });
  it("argmin and argmax pick the first extreme and refuse empty input", () => {
    expect(argmin([3, 1, 1, 2])).toBe(1);
    expect(argmax([3, 5, 5, 2])).toBe(1);
    expect(() => argmin([])).toThrow(RangeError);
    expect(() => argmax([])).toThrow(RangeError);
  });
});

describe("formatPythonFloat (str(float))", () => {
  it.each([
    [1, "1.0"],
    [0, "0.0"],
    [-0, "-0.0"],
    [0.494_791_666_666_666_7, "0.4947916666666667"],
    [0.028_526_400_198_166_768, "0.028526400198166768"],
    [1e-5, "1e-05"],
    [0.000_012_34, "1.234e-05"],
    [1e16, "1e+16"],
    [123_456_789_012_345_680, "1.2345678901234568e+17"],
    [0.0001, "0.0001"],
    [Number.NaN, "nan"],
    [Number.POSITIVE_INFINITY, "inf"],
  ])("str(%s) = %s", (x, expected) => {
    expect(formatPythonFloat(x)).toBe(expected);
  });
});

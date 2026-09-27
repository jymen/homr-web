import { describe, expect, it } from "vitest";
import {
  decodeFloat16Array,
  encodeFloat16Array,
  float16FromFloat32,
  float32FromFloat16,
} from "../src/models/dtype.js";

const SUBNORMAL_UNIT = 2 ** -24;
const TOO_SHORT = /into holds 4 values, the input has 8/;

describe("float16FromFloat32 (round to nearest, ties to even)", () => {
  it.each([
    ["zero", 0, 0x0000],
    ["negative zero", -0, 0x8000],
    ["one", 1, 0x3c00],
    ["minus one", -1, 0xbc00],
    ["two", 2, 0x4000],
    ["minus two", -2, 0xc000],
    ["0.1, which no half represents exactly", 0.1, 0x2e66],
    ["the smallest positive subnormal", SUBNORMAL_UNIT, 0x0001],
    ["a mid subnormal", 512 * SUBNORMAL_UNIT, 0x0200],
    ["the largest subnormal", 1023 * SUBNORMAL_UNIT, 0x03ff],
    ["the smallest normal", 2 ** -14, 0x0400],
    ["the largest finite half", 65_504, 0x7bff],
    ["a float32 subnormal, far below the half grid", 1e-45, 0x0000],
    ["2 ** -26, below half of the smallest subnormal", 2 ** -26, 0x0000],
  ])("%s", (_label, value, expected) => {
    expect(float16FromFloat32(value)).toBe(expected);
  });

  it("keeps a NaN a NaN rather than letting a dropped payload read as Infinity", () => {
    const pattern = float16FromFloat32(Number.NaN);
    expect(pattern).toBe(0x7e00);
    expect(float32FromFloat16(pattern)).toBeNaN();
  });
});

describe("ties land on the even neighbour", () => {
  it.each([
    ["2049 is halfway between 2048 and 2050", 2049, 0x6800],
    ["2051 is halfway between 2050 and 2052", 2051, 0x6802],
    ["-2049", -2049, 0xe800],
    ["-2051", -2051, 0xe802],
    [
      "2 ** -25 is halfway between zero and the smallest subnormal",
      2 ** -25,
      0x0000,
    ],
    ["1.5 subnormal units", 1.5 * SUBNORMAL_UNIT, 0x0002],
    ["2.5 subnormal units", 2.5 * SUBNORMAL_UNIT, 0x0002],
    ["-1.5 subnormal units", -1.5 * SUBNORMAL_UNIT, 0x8002],
    ["-2.5 subnormal units", -2.5 * SUBNORMAL_UNIT, 0x8002],
  ])("%s", (_label, value, expected) => {
    expect(float16FromFloat32(value)).toBe(expected);
  });

  it("rounds the top subnormal up into the smallest normal", () => {
    expect(float16FromFloat32(Math.fround(1023.5 * SUBNORMAL_UNIT))).toBe(
      0x0400
    );
    expect(float32FromFloat16(0x0400)).toBe(2 ** -14);
  });
});

describe("overflow keeps the sign", () => {
  it.each([
    ["just under the midpoint stays finite", Math.fround(65_519.99), 0x7bff],
    ["the midpoint 65520 ties up, out of the range", 65_520, 0x7c00],
    ["-65520", -65_520, 0xfc00],
    ["65536", 65_536, 0x7c00],
    ["1e30", 1e30, 0x7c00],
    ["Infinity", Number.POSITIVE_INFINITY, 0x7c00],
    ["-Infinity", Number.NEGATIVE_INFINITY, 0xfc00],
  ])("%s", (_label, value, expected) => {
    expect(float16FromFloat32(value)).toBe(expected);
  });
});

describe("float32FromFloat16", () => {
  it.each([
    [0x0000, 0],
    [0x0001, SUBNORMAL_UNIT],
    [0x0200, 512 * SUBNORMAL_UNIT],
    [0x0400, 2 ** -14],
    [0x2e66, 0.099_975_585_937_5],
    [0x3c00, 1],
    [0x7bff, 65_504],
    [0x7c00, Number.POSITIVE_INFINITY],
    [0xc000, -2],
    [0xfc00, Number.NEGATIVE_INFINITY],
  ])("0x%s decodes to %s", (half, expected) => {
    expect(float32FromFloat16(half)).toBe(expected);
  });

  it("decodes 0x8000 to negative zero, not zero", () => {
    expect(Object.is(float32FromFloat16(0x8000), -0)).toBe(true);
  });

  it("decodes every exponent-31 pattern with a payload to NaN", () => {
    for (let mantissa = 1; mantissa < 0x400; mantissa += 1) {
      expect(float32FromFloat16(0x7c00 + mantissa)).toBeNaN();
      expect(float32FromFloat16(0xfc00 + mantissa)).toBeNaN();
    }
  });
});

describe("the whole 16-bit space", () => {
  it("round-trips every non-NaN half through a decode and an encode", () => {
    const broken: string[] = [];
    for (let half = 0; half < 0x1_0000; half += 1) {
      const value = float32FromFloat16(half);
      if (Number.isNaN(value)) {
        continue;
      }
      const again = float16FromFloat32(value);
      if (again !== half) {
        broken.push(
          `0x${half.toString(16)} -> ${value} -> 0x${again.toString(16)}`
        );
      }
    }
    expect(broken).toEqual([]);
  });

  it("decodes every half to a value float32 holds exactly", () => {
    const inexact: number[] = [];
    for (let half = 0; half < 0x1_0000; half += 1) {
      const value = float32FromFloat16(half);
      if (!Number.isNaN(value) && Math.fround(value) !== value) {
        inexact.push(half);
      }
    }
    expect(inexact).toEqual([]);
  });

  it("gives the lookup table the same answers as the scalar decode", () => {
    const halves = new Uint16Array(0x1_0000);
    for (let half = 0; half < 0x1_0000; half += 1) {
      halves[half] = half;
    }
    const table = decodeFloat16Array(halves);
    const disagreeing: number[] = [];
    for (let half = 0; half < 0x1_0000; half += 1) {
      const scalar = float32FromFloat16(half);
      const fromTable = table[half];
      const same = Number.isNaN(scalar)
        ? Number.isNaN(fromTable ?? 0)
        : Object.is(fromTable, scalar);
      if (!same) {
        disagreeing.push(half);
      }
    }
    expect(disagreeing).toEqual([]);
  });
});

describe("buffer round trip", () => {
  const halves = new Uint16Array([
    0x0000, 0x8000, 0x3c00, 0xbc00, 0x7bff, 0x0001, 0x0400, 0x2e66,
  ]);
  const values = [
    0,
    -0,
    1,
    -1,
    65_504,
    SUBNORMAL_UNIT,
    2 ** -14,
    0.099_975_585_937_5,
  ];

  it("decodes a buffer and encodes it back to the same patterns", () => {
    const decoded = decodeFloat16Array(halves);
    expect(Array.from(decoded)).toEqual(values);
    expect(Object.is(decoded[1], -0)).toBe(true);
    expect(Array.from(encodeFloat16Array(decoded))).toEqual(Array.from(halves));
  });

  it("writes into a supplied buffer of the same length and returns it", () => {
    const into = new Float32Array(halves.length);
    const decoded = decodeFloat16Array(halves, into);
    expect(decoded).toBe(into);
    expect(Array.from(into)).toEqual(values);

    const back = new Uint16Array(halves.length);
    expect(encodeFloat16Array(into, back)).toBe(back);
    expect(Array.from(back)).toEqual(Array.from(halves));
  });

  it("writes into the front of an oversized buffer and returns that prefix", () => {
    const pool = new Float32Array(32);
    const decoded = decodeFloat16Array(halves, pool);
    expect(decoded).toHaveLength(halves.length);
    expect(decoded.buffer).toBe(pool.buffer);
    expect(Array.from(pool.subarray(0, halves.length))).toEqual(values);
    expect(pool[halves.length]).toBe(0);

    const poolOut = new Uint16Array(32);
    expect(encodeFloat16Array(decoded, poolOut)).toHaveLength(halves.length);
    expect(Array.from(poolOut.subarray(0, halves.length))).toEqual(
      Array.from(halves)
    );
  });

  it("refuses a buffer too short to hold the result", () => {
    expect(() => decodeFloat16Array(halves, new Float32Array(4))).toThrow(
      TOO_SHORT
    );
    expect(() =>
      encodeFloat16Array(new Float32Array(8), new Uint16Array(4))
    ).toThrow(RangeError);
  });
});

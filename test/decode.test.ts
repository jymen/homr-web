/**
 * The symbol classes the public fixture never produces (Clef, Rest,
 * Accidental, BarLine on a staff) and the decoder's refusals, on synthetic
 * JSON in the dumper's shape.
 */

import { describe, expect, it } from "vitest";
import {
  decodeAxisBox,
  decodeRotatedBox,
  decodeStaff,
  decodeStaffPositions,
  decodeSymbolOnStaff,
  decodeTokens,
  GoldenError,
} from "../src/golden/decode.js";

const NORMALISED_RANGE = /normalised range/;
const UNKNOWN_CLASS = /unknown symbol class/;
const HAS_DOT = /has_dot/;
const STEM_PAIR = /both be null or both be set/;
const MAX_Y = /max_y/;
const Y_TOLERANCE = /_y_tolerance/;
const RHYTHM_VOCABULARY = /rhythm vocabulary/;

const rotated = (
  cx: number,
  cy: number,
  w: number,
  h: number,
  angle: number
) => ({
  __class__: "RotatedBoundingBox",
  angle,
  bottom_left: [cx - w / 2, cy + h / 2],
  bottom_right: [cx + w / 2, cy + h / 2],
  box: [[cx, cy], [w, h], angle],
  center: [cx, cy],
  contours: [[[0, 0]], [[1, 1]]],
  debug_id: 3,
  polygon: [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1],
  ],
  size: [w, h],
  top_left: [cx - w / 2, cy - h / 2],
  top_right: [cx + w / 2, cy - h / 2],
});

const axis = (x1: number, y1: number, x2: number, y2: number) => ({
  __class__: "BoundingBox",
  box: [x1, y1, x2, y2],
  center: [(x1 + x2) / 2, (y1 + y2) / 2],
  contours: [[[x1, y1]]],
  debug_id: 0,
  polygon: [
    [x1, y1],
    [x2, y1],
    [x2, y2],
    [x1, y2],
  ],
  rotated_box: [[(x1 + x2) / 2, (y1 + y2) / 2], [x2 - x1, y2 - y1], 0],
  size: [x2 - x1, y2 - y1],
});

describe("boxes", () => {
  it("decodes a rotated box and refuses a corner Python did not derive", () => {
    const box = decodeRotatedBox(rotated(10, 20, 4, 6, -45));
    expect(box.rect).toEqual({ angle: -45, cx: 10, cy: 20, h: 6, w: 4 });
    expect(box.debugId).toBe(3);
    expect(() =>
      decodeRotatedBox({ ...rotated(10, 20, 4, 6, 0), top_left: [0, 0] })
    ).toThrow(GoldenError);
    expect(() => decodeRotatedBox(rotated(10, 20, 4, 6, 90))).toThrow(
      NORMALISED_RANGE
    );
  });
  it("decodes an axis box", () => {
    expect(decodeAxisBox(axis(1, 2, 5, 8))).toMatchObject({
      kind: "axis",
      x1: 1,
      x2: 5,
      y1: 2,
      y2: 8,
    });
  });
});

describe("symbols on a staff", () => {
  it("decodes the four classes the Kesh page has none of", () => {
    const clef = decodeSymbolOnStaff({
      __class__: "Clef",
      box: axis(1, 2, 5, 8),
      center: [3, 5],
    });
    expect(clef).toMatchObject({ center: { x: 3, y: 5 }, kind: "clef" });
    const rest = decodeSymbolOnStaff({
      __class__: "Rest",
      box: axis(1, 2, 5, 8),
      center: [3, 5],
      has_dot: false,
    });
    expect(rest.kind).toBe("rest");
    const accidental = decodeSymbolOnStaff({
      __class__: "Accidental",
      box: axis(1, 2, 5, 8),
      center: [3, 5],
      position: 4,
    });
    expect(accidental).toMatchObject({ kind: "accidental", position: 4 });
    const bar = decodeSymbolOnStaff({
      __class__: "BarLine",
      box: rotated(10, 20, 2, 30, 0),
      center: [10, 20],
    });
    expect(bar).toMatchObject({ box: { kind: "rotated" }, kind: "barLine" });
    expect(() => decodeSymbolOnStaff({ __class__: "Slur" })).toThrow(
      UNKNOWN_CLASS
    );
  });
  it("trips when homr starts using a field the port dropped", () => {
    const note = {
      __class__: "Note",
      beams: [],
      box: {
        ...rotated(1, 2, 3, 3, 0),
        __class__: "BoundingEllipse",
        polygon: [
          [0, 0],
          [1, 0],
          [1, 1],
          [0, 1],
          [0, 0],
        ],
      },
      center: [1, 2],
      circle_of_fifth: 0,
      flags: [],
      has_dot: true,
      position: 3,
      stem: null,
      stem_direction: null,
    };
    expect(() => decodeSymbolOnStaff(note)).toThrow(HAS_DOT);
    expect(() =>
      decodeSymbolOnStaff({ ...note, has_dot: false, stem_direction: "UP" })
    ).toThrow(STEM_PAIR);
  });
});

describe("staffs, tokens and positions", () => {
  const point = (x: number, ys: number[]) => ({
    __class__: "StaffPoint",
    angle: 0,
    average_unit_size: (ys[4] ?? 0) - (ys[0] ?? 0) === 40 ? 10 : Number.NaN,
    x,
    y: ys,
  });
  it("refuses a staff whose stored bounds disagree with its grid", () => {
    const staff = {
      __class__: "Staff",
      _y_tolerance: 40,
      average_unit_size: 10,
      grid: [
        point(0, [100, 110, 120, 130, 140]),
        point(50, [100, 110, 120, 130, 140]),
      ],
      is_grandstaff: false,
      max_x: 50,
      max_y: 140,
      min_x: 0,
      min_y: 100,
      symbols: [],
    };
    expect(decodeStaff(staff, { space: "canvas" }).space).toBe("canvas");
    expect(() => decodeStaff({ ...staff, max_y: 141 })).toThrow(MAX_Y);
    expect(() => decodeStaff({ ...staff, _y_tolerance: 41 })).toThrow(
      Y_TOLERANCE
    );
  });
  it("refuses a token outside its vocabulary", () => {
    const token = {
      articulation: "_",
      coordinates: [1, 2],
      lift: "_",
      pitch: "_",
      position: "upper",
      rhythm: "clef_G2",
      slur: "_",
    };
    expect(decodeTokens([token])).toHaveLength(1);
    expect(() => decodeTokens([{ ...token, rhythm: "newline" }])).toThrow(
      RHYTHM_VOCABULARY
    );
  });
  it("parses staff positions and refuses a malformed line", () => {
    expect(decodeStaffPositions("1 0.5 0.25 0.8 0.03\n")).toEqual([
      { cx: 0.5, cy: 0.25, h: 0.03, isGrandstaff: true, w: 0.8 },
    ]);
    expect(() => decodeStaffPositions("2 0.5 0.25 0.8 0.03")).toThrow(
      GoldenError
    );
    expect(() => decodeStaffPositions("0 0.5")).toThrow(GoldenError);
  });
});

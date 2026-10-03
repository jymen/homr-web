/**
 * The boundary between the Python golden dumps and the domain types. Every
 * golden JSON value is a generic dump (`{"__class__": Name, ...fields}`,
 * numpy arrays as nested lists, enums as their names, Python snake_case)
 * and this is the one place that knows that shape: it renames, restructures
 * (stem + stem_direction into Stem, contours (n, 1, 2) into a PointList),
 * validates every invariant the types promise, and asserts the derived
 * fields Python stored (corners, min/max, averageUnitSize, _y_tolerance)
 * against the port's own derivation, so decoding a golden file is already
 * a test of cornersOf, median and mean.
 *
 * Input is `unknown` (the parsed JSON); a mismatch throws GoldenError
 * naming the JSON path, never returns a partial value. Nothing here is a
 * test dependency: the bench page uses the same decoders to overlay
 * homr's own output on a page next to the port's.
 */

import type { Triangle } from "../dewarp/delaunay.js";
import type { AffineMatrix } from "../dewarp/piecewise-affine.js";
import {
  type AxisBox,
  assertNormalizedRect,
  cornersOf,
  createAxisBox,
  type Ellipse,
  ellipseFromParts,
  type Point,
  type PointList,
  pointCount,
  pointListFromPairs,
  type RotatedBox,
  type RotatedRect,
  rotatedBoxFromParts,
} from "../geometry/boxes.js";
import {
  type RawStaff,
  rawStaffContour,
  rawStaffFromParts,
} from "../geometry/raw-staffs.js";
import {
  createStaffAnchor,
  type StaffAnchor,
} from "../geometry/staff-anchors.js";
import {
  asFiveLines,
  createStaffLineSegment,
  type FiveLines,
  type StaffLineSegment,
} from "../geometry/staff-lines.js";
import { diff, mean } from "../image/numeric.js";
import { LINES_PER_STAFF } from "../model/constants.js";
import {
  type CoordinateSpace,
  createMultiStaff,
  createStaff,
  createStaffPoint,
  type MultiStaff,
  type Staff,
  type StaffPoint,
  yTolerance,
} from "../model/staff.js";
import type {
  Accidental,
  BarLine,
  Clef,
  Note,
  NoteheadWithStem,
  Rest,
  Stem,
  SymbolOnStaff,
} from "../model/symbols.js";
import type { StaffPosition } from "../result.js";
import type { DecodedSymbol, EncodedSymbol } from "../transformer/symbol.js";
import { type Head, isToken, VOCABULARIES } from "../transformer/vocabulary.js";

export class GoldenError extends Error {
  /** JSON path of the offending value, e.g. "[3].stem.box[2]". */
  readonly path: string;

  constructor(
    path: string,
    message: string,
    options?: { readonly cause: unknown }
  ) {
    super(`${path}: ${message}`, options);
    this.path = path;
  }
}

const WHITESPACE = /\s+/;

/** A domain error caught at the boundary, re-thrown as a GoldenError at `path` with the original as its cause. */
function goldenErrorAt(path: string, error: unknown): GoldenError {
  const message = error instanceof Error ? error.message : String(error);
  return new GoldenError(path, message, { cause: error });
}

// Primitive readers. Each takes the value and its path and returns a narrowed value or throws.

type JsonObject = Record<string, unknown>;

function asObject(value: unknown, path: string): JsonObject {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new GoldenError(path, "expected an object");
  }
  return value as JsonObject;
}

function asArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new GoldenError(path, "expected an array");
  }
  return value;
}

function asNumber(value: unknown, path: string): number {
  if (typeof value !== "number") {
    throw new GoldenError(path, `expected a number, got ${typeof value}`);
  }
  return value;
}

function asString(value: unknown, path: string): string {
  if (typeof value !== "string") {
    throw new GoldenError(path, `expected a string, got ${typeof value}`);
  }
  return value;
}

function asBoolean(value: unknown, path: string): boolean {
  if (typeof value !== "boolean") {
    throw new GoldenError(path, `expected a boolean, got ${typeof value}`);
  }
  return value;
}

function field(object: JsonObject, name: string, path: string): unknown {
  if (!(name in object)) {
    throw new GoldenError(path, `missing field ${name}`);
  }
  return object[name];
}

function ofClass(value: unknown, className: string, path: string): JsonObject {
  const object = asObject(value, path);
  if (object.__class__ !== className) {
    throw new GoldenError(
      path,
      `expected __class__ ${className}, got ${String(object.__class__)}`
    );
  }
  return object;
}

function numbers(value: unknown, path: string): number[] {
  return asArray(value, path).map((v, i) => asNumber(v, `${path}[${i}]`));
}

function pair(value: unknown, path: string): readonly [number, number] {
  const list = numbers(value, path);
  const [a, b] = list;
  if (list.length !== 2 || a === undefined || b === undefined) {
    throw new GoldenError(path, `expected a pair, got ${list.length} values`);
  }
  return [a, b];
}

function pointOf(value: unknown, path: string): Point {
  const [x, y] = pair(value, path);
  return { x, y };
}

/** Exact for values Python derived with the same float64 arithmetic; a hair of slack for sums in another order. */
function assertClose(
  actual: number,
  expected: number,
  path: string,
  what: string
): void {
  const tolerance = 1e-9 * Math.max(1, Math.abs(expected));
  if (!(Math.abs(actual - expected) <= tolerance)) {
    throw new GoldenError(
      path,
      `${what}: the port derives ${actual}, Python stored ${expected}`
    );
  }
}

function assertPointClose(
  actual: Point,
  expected: Point,
  path: string,
  what: string
): void {
  assertClose(actual.x, expected.x, path, `${what}.x`);
  assertClose(actual.y, expected.y, path, `${what}.y`);
}

// Point lists

/** `polygon`: a list of [x, y] (boxPoints or ellipse2Poly, int). */
function polygonOf(value: unknown, path: string): PointList {
  return pointListFromPairs(
    asArray(value, path).map((p, i) => pair(p, `${path}[${i}]`))
  );
}

/** `contours`: cv2's (n, 1, 2) list of single-point lists. */
function contourOf(value: unknown, path: string): PointList {
  return pointListFromPairs(
    asArray(value, path).map((entry, i) => {
      const inner = asArray(entry, `${path}[${i}]`);
      if (inner.length !== 1) {
        throw new GoldenError(
          `${path}[${i}]`,
          `expected a (1, 2) contour entry, got ${inner.length}`
        );
      }
      return pair(inner[0], `${path}[${i}][0]`);
    })
  );
}

// Rotated rectangles and boxes

/** The `box: [[cx, cy], [w, h], angle]` triple, checked against the `center`, `size`, `angle` copies and the four corners Python stored. */
function rectOf(object: JsonObject, path: string): RotatedRect {
  const box = asArray(field(object, "box", path), `${path}.box`);
  if (box.length !== 3) {
    throw new GoldenError(
      `${path}.box`,
      `expected [[cx, cy], [w, h], angle], got ${box.length} parts`
    );
  }
  const [cx, cy] = pair(box[0], `${path}.box[0]`);
  const [w, h] = pair(box[1], `${path}.box[1]`);
  const angle = asNumber(box[2], `${path}.box[2]`);
  let rect: RotatedRect;
  try {
    rect = assertNormalizedRect({ angle, cx, cy, h, w });
  } catch (error) {
    throw goldenErrorAt(`${path}.box`, error);
  }
  const center = pair(field(object, "center", path), `${path}.center`);
  const size = pair(field(object, "size", path), `${path}.size`);
  if (center[0] !== cx || center[1] !== cy || size[0] !== w || size[1] !== h) {
    throw new GoldenError(path, "center/size are not copies of box");
  }
  if (asNumber(field(object, "angle", path), `${path}.angle`) !== angle) {
    throw new GoldenError(`${path}.angle`, "angle is not a copy of box[2]");
  }
  const corners = cornersOf(rect);
  assertPointClose(
    corners.topLeft,
    pointOf(field(object, "top_left", path), `${path}.top_left`),
    path,
    "top_left"
  );
  assertPointClose(
    corners.bottomLeft,
    pointOf(field(object, "bottom_left", path), `${path}.bottom_left`),
    path,
    "bottom_left"
  );
  assertPointClose(
    corners.topRight,
    pointOf(field(object, "top_right", path), `${path}.top_right`),
    path,
    "top_right"
  );
  assertPointClose(
    corners.bottomRight,
    pointOf(field(object, "bottom_right", path), `${path}.bottom_right`),
    path,
    "bottom_right"
  );
  return rect;
}

function angledParts(
  object: JsonObject,
  path: string
): {
  rect: RotatedRect;
  polygon: PointList;
  contour: PointList;
  debugId: number;
} {
  return {
    contour: contourOf(field(object, "contours", path), `${path}.contours`),
    debugId: asNumber(field(object, "debug_id", path), `${path}.debug_id`),
    polygon: polygonOf(field(object, "polygon", path), `${path}.polygon`),
    rect: rectOf(object, path),
  };
}

export function decodeRotatedBox(json: unknown, path = "$"): RotatedBox {
  const object = ofClass(json, "RotatedBoundingBox", path);
  const parts = angledParts(object, path);
  return rotatedBoxFromParts(
    parts.rect,
    parts.polygon,
    parts.contour,
    parts.debugId
  );
}

export function decodeEllipse(json: unknown, path = "$"): Ellipse {
  const object = ofClass(json, "BoundingEllipse", path);
  const parts = angledParts(object, path);
  return ellipseFromParts(
    parts.rect,
    parts.polygon,
    parts.contour,
    parts.debugId
  );
}

/** homr's BoundingBox: `box: [x1, y1, x2, y2]`, the rest derived and checked. */
export function decodeAxisBox(json: unknown, path = "$"): AxisBox {
  const object = ofClass(json, "BoundingBox", path);
  const box = numbers(field(object, "box", path), `${path}.box`);
  const [x1, y1, x2, y2] = box;
  if (
    box.length !== 4 ||
    x1 === undefined ||
    y1 === undefined ||
    x2 === undefined ||
    y2 === undefined
  ) {
    throw new GoldenError(
      `${path}.box`,
      `expected [x1, y1, x2, y2], got ${box.length} values`
    );
  }
  const axis = createAxisBox(
    x1,
    y1,
    x2,
    y2,
    contourOf(field(object, "contours", path), `${path}.contours`),
    asNumber(field(object, "debug_id", path), `${path}.debug_id`)
  );
  const center = pair(field(object, "center", path), `${path}.center`);
  assertClose((x1 + x2) / 2, center[0], path, "center.x");
  assertClose((y1 + y2) / 2, center[1], path, "center.y");
  return axis;
}

/** A list of `__class__: RotatedBoundingBox` objects: boxes-*.json except noteheads, barlines.barLines, MultiStaff.connections. */
export function decodeRotatedBoxes(json: unknown, path = "$"): RotatedBox[] {
  return asArray(json, path).map((v, i) =>
    decodeRotatedBox(v, `${path}[${i}]`)
  );
}

/** A list of `__class__: BoundingEllipse`: boxes-noteheads.json. */
export function decodeEllipses(json: unknown, path = "$"): Ellipse[] {
  return asArray(json, path).map((v, i) => decodeEllipse(v, `${path}[${i}]`));
}

// Stems and noteheads

function stemOf(object: JsonObject, path: string): Stem | null {
  const box = field(object, "stem", path);
  const direction = field(object, "stem_direction", path);
  if (box === null && direction === null) {
    return null;
  }
  if (box === null || direction === null) {
    throw new GoldenError(
      path,
      "stem and stem_direction must both be null or both be set"
    );
  }
  const name = asString(direction, `${path}.stem_direction`);
  if (name !== "UP" && name !== "DOWN") {
    throw new GoldenError(
      `${path}.stem_direction`,
      `expected UP or DOWN, got ${name}`
    );
  }
  return { box: decodeRotatedBox(box, `${path}.stem`), direction: name };
}

/** noteheads-with-stems.json; asserts stem and stem_direction are both null or both set. */
export function decodeNoteheadsWithStems(
  json: unknown,
  path = "$"
): NoteheadWithStem[] {
  return asArray(json, path).map((v, i) => {
    const object = ofClass(v, "NoteheadWithStem", `${path}[${i}]`);
    return {
      notehead: decodeEllipse(
        field(object, "notehead", `${path}[${i}]`),
        `${path}[${i}].notehead`
      ),
      stem: stemOf(object, `${path}[${i}]`),
    };
  });
}

/** barlines.json. */
export function decodeBarLines(
  json: unknown,
  path = "$"
): {
  readonly averageNoteHeadHeight: number;
  readonly barLines: RotatedBox[];
} {
  const object = asObject(json, path);
  return {
    averageNoteHeadHeight: asNumber(
      field(object, "averageNoteHeadHeight", path),
      `${path}.averageNoteHeadHeight`
    ),
    barLines: decodeRotatedBoxes(
      field(object, "barLines", path),
      `${path}.barLines`
    ),
  };
}

// Symbols on a staff

const VESTIGIAL_NOTE_FIELDS: ReadonlyArray<readonly [string, unknown]> = [
  ["has_dot", false],
  ["circle_of_fifth", 0],
];

function noteOf(object: JsonObject, path: string): Note {
  for (const [name, expected] of VESTIGIAL_NOTE_FIELDS) {
    if (field(object, name, path) !== expected) {
      throw new GoldenError(
        `${path}.${name}`,
        `homr started using ${name}; port it`
      );
    }
  }
  for (const name of ["beams", "flags"]) {
    if (asArray(field(object, name, path), `${path}.${name}`).length !== 0) {
      throw new GoldenError(
        `${path}.${name}`,
        `homr started using ${name}; port it`
      );
    }
  }
  return {
    box: decodeEllipse(field(object, "box", path), `${path}.box`),
    center: pointOf(field(object, "center", path), `${path}.center`),
    kind: "note",
    position: asNumber(field(object, "position", path), `${path}.position`),
    stem: stemOf(object, path),
  };
}

function barLineOf(object: JsonObject, path: string): BarLine {
  return {
    box: decodeRotatedBox(field(object, "box", path), `${path}.box`),
    center: pointOf(field(object, "center", path), `${path}.center`),
    kind: "barLine",
  };
}

function clefOf(object: JsonObject, path: string): Clef {
  return {
    box: decodeAxisBox(field(object, "box", path), `${path}.box`),
    center: pointOf(field(object, "center", path), `${path}.center`),
    kind: "clef",
  };
}

function restOf(object: JsonObject, path: string): Rest {
  if (field(object, "has_dot", path) !== false) {
    throw new GoldenError(
      `${path}.has_dot`,
      "homr started using has_dot on rests; port it"
    );
  }
  return {
    box: decodeAxisBox(field(object, "box", path), `${path}.box`),
    center: pointOf(field(object, "center", path), `${path}.center`),
    kind: "rest",
  };
}

function accidentalOf(object: JsonObject, path: string): Accidental {
  return {
    box: decodeAxisBox(field(object, "box", path), `${path}.box`),
    center: pointOf(field(object, "center", path), `${path}.center`),
    kind: "accidental",
    position: asNumber(field(object, "position", path), `${path}.position`),
  };
}

export function decodeSymbolOnStaff(json: unknown, path = "$"): SymbolOnStaff {
  const object = asObject(json, path);
  switch (object.__class__) {
    case "Note":
      return noteOf(object, path);
    case "BarLine":
      return barLineOf(object, path);
    case "Clef":
      return clefOf(object, path);
    case "Rest":
      return restOf(object, path);
    case "Accidental":
      return accidentalOf(object, path);
    default:
      throw new GoldenError(
        path,
        `unknown symbol class ${String(object.__class__)}`
      );
  }
}

/** notes.json. */
export function decodeNotes(json: unknown, path = "$"): Note[] {
  return asArray(json, path).map((v, i) =>
    noteOf(ofClass(v, "Note", `${path}[${i}]`), `${path}[${i}]`)
  );
}

// Staffs

function staffPointOf(json: unknown, path: string): StaffPoint {
  const object = ofClass(json, "StaffPoint", path);
  const point = createStaffPoint(
    asNumber(field(object, "x", path), `${path}.x`),
    numbers(field(object, "y", path), `${path}.y`),
    asNumber(field(object, "angle", path), `${path}.angle`)
  );
  assertClose(
    point.averageUnitSize,
    asNumber(
      field(object, "average_unit_size", path),
      `${path}.average_unit_size`
    ),
    path,
    "average_unit_size (mean of diff(y))"
  );
  assertClose(
    mean(diff(point.y)),
    point.averageUnitSize,
    path,
    "average_unit_size"
  );
  return point;
}

export interface StaffDecodeOptions {
  /** Page space unless the file is a canvas-<n>-staff.json. */
  readonly space?: CoordinateSpace;
}

/**
 * One `__class__: Staff`: canvas-<n>-staff.json, and the elements of
 * staffs.json and MultiStaff.staffs. Asserts min_x, max_x, min_y, max_y,
 * average_unit_size and _y_tolerance against the port's derivation and the
 * grid's x order; decodes symbols by `__class__` (Note, BarLine, Clef,
 * Rest, Accidental) and asserts Note's vestigial fields are at their
 * defaults. The grid's x order is not checked: homr's resampling yields a
 * few out-of-order neighbours (700 after 701 on the Kesh page) and
 * Staff.__init__ reads grid[0] and grid[-1] regardless.
 */
export function decodeStaff(
  json: unknown,
  options: StaffDecodeOptions = {},
  path = "$"
): Staff {
  const object = ofClass(json, "Staff", path);
  const grid = asArray(field(object, "grid", path), `${path}.grid`).map(
    (v, i) => staffPointOf(v, `${path}.grid[${i}]`)
  );
  const symbols = asArray(
    field(object, "symbols", path),
    `${path}.symbols`
  ).map((v, i) => decodeSymbolOnStaff(v, `${path}.symbols[${i}]`));
  let staff: Staff;
  try {
    staff = createStaff(grid, {
      isGrandstaff: asBoolean(
        field(object, "is_grandstaff", path),
        `${path}.is_grandstaff`
      ),
      symbols,
      ...(options.space === undefined ? {} : { space: options.space }),
    });
  } catch (error) {
    throw goldenErrorAt(path, error);
  }
  const stored = (name: string) =>
    asNumber(field(object, name, path), `${path}.${name}`);
  assertClose(staff.minX, stored("min_x"), path, "min_x");
  assertClose(staff.maxX, stored("max_x"), path, "max_x");
  assertClose(staff.minY, stored("min_y"), path, "min_y");
  assertClose(staff.maxY, stored("max_y"), path, "max_y");
  assertClose(
    staff.averageUnitSize,
    stored("average_unit_size"),
    path,
    "average_unit_size (median)"
  );
  assertClose(yTolerance(staff), stored("_y_tolerance"), path, "_y_tolerance");
  return staff;
}

/** staffs.json. */
export function decodeStaffs(json: unknown, path = "$"): Staff[] {
  return asArray(json, path).map((v, i) => decodeStaff(v, {}, `${path}[${i}]`));
}

/** multistaffs.json; asserts the staffs of each are ascending in minY. */
export function decodeMultiStaffs(json: unknown, path = "$"): MultiStaff[] {
  return asArray(json, path).map((v, i) => {
    const object = ofClass(v, "MultiStaff", `${path}[${i}]`);
    const staffs = asArray(
      field(object, "staffs", `${path}[${i}]`),
      `${path}[${i}].staffs`
    ).map((s, j) => decodeStaff(s, {}, `${path}[${i}].staffs[${j}]`));
    for (let j = 1; j < staffs.length; j += 1) {
      if ((staffs[j]?.minY ?? 0) < (staffs[j - 1]?.minY ?? 0)) {
        throw new GoldenError(
          `${path}[${i}].staffs[${j}]`,
          "staffs are not ascending in minY"
        );
      }
    }
    try {
      return createMultiStaff(
        staffs,
        decodeRotatedBoxes(
          field(object, "connections", `${path}[${i}]`),
          `${path}[${i}].connections`
        )
      );
    } catch (error) {
      throw goldenErrorAt(`${path}[${i}]`, error);
    }
  });
}

// Tokens

function coordinatesOf(value: unknown, path: string): Point | null {
  return value === null ? null : pointOf(value, path);
}

function tokenField<H extends Head>(
  object: JsonObject,
  head: H,
  path: string
): (typeof VOCABULARIES)[H][number] {
  const value = asString(field(object, head, path), `${path}.${head}`);
  if (!isToken(head, value)) {
    throw new GoldenError(
      `${path}.${head}`,
      `${JSON.stringify(value)} is not in the ${head} vocabulary`
    );
  }
  return value;
}

/**
 * tokens-<n>.json. Strict: every field must be in its head's table, which
 * is the phase 1 proof that the TypeScript vocabularies contain what the
 * pinned model emits. Index equality against Python is a separate check
 * (see decodeVocabulary).
 */
export function decodeTokens(json: unknown, path = "$"): DecodedSymbol[] {
  return asArray(json, path).map((v, i) => {
    const object = asObject(v, `${path}[${i}]`);
    const at = `${path}[${i}]`;
    return {
      articulation: tokenField(object, "articulation", at),
      coordinates: coordinatesOf(
        field(object, "coordinates", at),
        `${at}.coordinates`
      ),
      lift: tokenField(object, "lift", at),
      pitch: tokenField(object, "pitch", at),
      position: tokenField(object, "position", at),
      rhythm: tokenField(object, "rhythm", at),
      slur: tokenField(object, "slur", at),
    };
  });
}

/** voices.json: one open-vocabulary list per voice (rhythm, articulation and slur may be strings homr made). */
export function decodeVoices(json: unknown, path = "$"): EncodedSymbol[][] {
  return asArray(json, path).map((voice, v) =>
    asArray(voice, `${path}[${v}]`).map((s, i) => {
      const at = `${path}[${v}][${i}]`;
      const object = asObject(s, at);
      return {
        articulation: asString(
          field(object, "articulation", at),
          `${at}.articulation`
        ),
        coordinates: coordinatesOf(
          field(object, "coordinates", at),
          `${at}.coordinates`
        ),
        lift: tokenField(object, "lift", at),
        pitch: tokenField(object, "pitch", at),
        position: tokenField(object, "position", at),
        rhythm: asString(field(object, "rhythm", at), `${at}.rhythm`),
        slur: asString(field(object, "slur", at), `${at}.slur`),
      };
    })
  );
}

/** vocabulary.json (each head's `dict[str, int]`) as token lists in index order, comparable to VOCABULARIES with toEqual. */
export function decodeVocabulary(
  json: unknown,
  path = "$"
): Readonly<Record<Head, readonly string[]>> {
  const object = asObject(json, path);
  const heads = Object.keys(VOCABULARIES) as Head[];
  const out: Partial<Record<Head, readonly string[]>> = {};
  for (const head of heads) {
    const table = asObject(field(object, head, path), `${path}.${head}`);
    const entries = Object.entries(table).map(
      ([token, index]) =>
        [token, asNumber(index, `${path}.${head}.${token}`)] as const
    );
    entries.sort((a, b) => a[1] - b[1]);
    for (const [position, [token, index]] of entries.entries()) {
      if (index !== position) {
        throw new GoldenError(
          `${path}.${head}.${token}`,
          `index ${index} at position ${position}: the table has a gap`
        );
      }
    }
    out[head] = entries.map(([token]) => token);
  }
  return out as Record<Head, readonly string[]>;
}

/** staff-positions.txt: one StaffPosition per line, in file order ("<0|1> cx cy w h"). */
export function decodeStaffPositions(text: string): StaffPosition[] {
  return text
    .split("\n")
    .filter((line) => line.trim() !== "")
    .map((line, i) => {
      const parts = line.trim().split(WHITESPACE);
      const [flag, cx, cy, w, h] = parts.map(Number);
      if (
        parts.length !== 5 ||
        [flag, cx, cy, w, h].some((v) => v === undefined || Number.isNaN(v))
      ) {
        throw new GoldenError(
          `line ${i + 1}`,
          `expected "<0|1> cx cy w h", got ${JSON.stringify(line)}`
        );
      }
      if (flag !== 0 && flag !== 1) {
        throw new GoldenError(
          `line ${i + 1}`,
          `is_grandstaff flag must be 0 or 1, got ${flag}`
        );
      }
      return {
        cx: cx ?? 0,
        cy: cy ?? 0,
        h: h ?? 0,
        isGrandstaff: flag === 1,
        w: w ?? 0,
      };
    });
}

// Staff detection intermediates (phase 5). Anchors and raw staffs are built
// through their own factories, so decoding asserts the fields those derive.

function integerOf(value: unknown, path: string): number {
  const n = asNumber(value, path);
  if (!Number.isInteger(n)) {
    throw new GoldenError(path, `expected an integer, got ${n}`);
  }
  return n;
}

/** The element of `list` a golden index names; the dumper writes object identity as a position in an earlier list. */
function pick<T>(
  list: readonly T[],
  value: unknown,
  path: string,
  what: string
): T {
  const index = integerOf(value, path);
  const item = list[index];
  if (index < 0 || item === undefined) {
    throw new GoldenError(
      path,
      `index ${index} is outside ${what} (${list.length} entries)`
    );
  }
  return item;
}

function indicesInto(
  value: unknown,
  length: number,
  path: string,
  what: string
): number[] {
  return asArray(value, path).map((v, i) => {
    const index = integerOf(v, `${path}[${i}]`);
    if (index < 0 || index >= length) {
      throw new GoldenError(
        `${path}[${i}]`,
        `index ${index} is outside ${what} (${length} entries)`
      );
    }
    return index;
  });
}

function assertDistinct(list: readonly number[], path: string): void {
  if (new Set(list).size !== list.length) {
    throw new GoldenError(path, "an index appears twice");
  }
}

/** Five staff lines, each the fragments of one StaffLineSegment in the segment's own order. */
function fiveLinesOf(
  value: unknown,
  fragments: readonly RotatedBox[],
  path: string
): FiveLines {
  const lines = asArray(value, path).map((line, i) => {
    const at = `${path}[${i}]`;
    const picked = asArray(line, at).map((v, k) =>
      pick(fragments, v, `${at}[${k}]`, "the fragment list")
    );
    let segment: StaffLineSegment;
    try {
      segment = createStaffLineSegment(picked);
    } catch (error) {
      throw goldenErrorAt(at, error);
    }
    if (segment.fragments.some((fragment, k) => fragment !== picked[k])) {
      throw new GoldenError(at, "fragments are not ascending in centre x");
    }
    return segment;
  });
  const five = asFiveLines(lines);
  if (five === null) {
    throw new GoldenError(
      path,
      `expected ${LINES_PER_STAFF} lines, got ${lines.length}`
    );
  }
  return five;
}

/**
 * One StaffAnchor, rebuilt by createStaffAnchor from its lines and its symbol.
 * averageUnitSize, minY, maxY and the zone are asserted against what Python
 * stored, so decoding the file tests the constructor.
 */
function staffAnchorOf(
  json: unknown,
  fragments: readonly RotatedBox[],
  path: string
): StaffAnchor {
  const object = asObject(json, path);
  const anchor = createStaffAnchor(
    fiveLinesOf(field(object, "lines", path), fragments, `${path}.lines`),
    decodeRotatedBox(field(object, "symbol", path), `${path}.symbol`)
  );
  const stored = (name: string) =>
    asNumber(field(object, name, path), `${path}.${name}`);
  assertClose(
    anchor.averageUnitSize,
    stored("averageUnitSize"),
    path,
    "averageUnitSize (mean of the four gaps at the symbol's x)"
  );
  assertClose(anchor.minY, stored("minY"), path, "minY");
  assertClose(anchor.maxY, stored("maxY"), path, "maxY");
  const [start, stop] = pair(field(object, "zone", path), `${path}.zone`);
  if (start !== anchor.zone.start || stop !== anchor.zone.stop) {
    throw new GoldenError(
      `${path}.zone`,
      `the port derives [${anchor.zone.start}, ${anchor.zone.stop}], Python stored [${start}, ${stop}]`
    );
  }
  return anchor;
}

/** One column zone of predict_other_anchors_from_clefs. */
export interface GoldenClefZone {
  /** find_horizontal_lines over the zone's columns: the row of each line of every complete group of five. */
  readonly lines: readonly (readonly number[])[];
  /** Python's `range(start, stop)` of columns. */
  readonly start: number;
  readonly stop: number;
}

/**
 * staff-anchors.json. Lines arrive as indices into
 * boxes-staff_fragments-broken.json, so the decoded anchors hold the very box
 * objects `fragments` does, and a stage fed these anchors and that list sees
 * one set of fragments, as Python's stage did.
 */
export interface GoldenStaffAnchors {
  /** find_staff_anchors over the bar-line boxes. */
  readonly barLines: StaffAnchor[];
  /** find_staff_anchors over clefs_keys. */
  readonly clefs: StaffAnchor[];
  /** What filter_unusual_anchors kept of clefs + otherClefs + barLines: the same objects, in that order. */
  readonly kept: StaffAnchor[];
  /** predict_other_anchors_from_clefs. */
  readonly otherClefSymbols: RotatedBox[];
  /** find_staff_anchors over otherClefSymbols. */
  readonly otherClefs: StaffAnchor[];
  /** init_zone, and find_horizontal_lines over each zone's columns. Empty when no clef anchor was found. */
  readonly zones: GoldenClefZone[];
}

export function decodeStaffAnchors(
  json: unknown,
  fragments: readonly RotatedBox[],
  path = "$"
): GoldenStaffAnchors {
  const object = asObject(json, path);
  const anchorsOf = (name: string): StaffAnchor[] =>
    asArray(field(object, name, path), `${path}.${name}`).map((v, i) =>
      staffAnchorOf(v, fragments, `${path}.${name}[${i}]`)
    );
  const clefs = anchorsOf("clefs");
  const otherClefs = anchorsOf("otherClefs");
  const barLines = anchorsOf("barLines");
  const all = [...clefs, ...otherClefs, ...barLines];
  const keptIndices = indicesInto(
    field(object, "kept", path),
    all.length,
    `${path}.kept`,
    "clefs + otherClefs + barLines"
  );
  for (let i = 1; i < keptIndices.length; i += 1) {
    if ((keptIndices[i] ?? 0) <= (keptIndices[i - 1] ?? 0)) {
      throw new GoldenError(
        `${path}.kept[${i}]`,
        "filter_unusual_anchors keeps list order, so kept must ascend"
      );
    }
  }
  const zones = asArray(field(object, "zones", path), `${path}.zones`).map(
    (v, i): GoldenClefZone => {
      const at = `${path}.zones[${i}]`;
      const zone = asObject(v, at);
      const start = integerOf(field(zone, "start", at), `${at}.start`);
      const stop = integerOf(field(zone, "stop", at), `${at}.stop`);
      if (start < 0 || stop < start) {
        throw new GoldenError(at, `not a column range: [${start}, ${stop})`);
      }
      const lines = asArray(field(zone, "lines", at), `${at}.lines`).map(
        (group, g) => {
          const rows = numbers(group, `${at}.lines[${g}]`);
          if (rows.length !== LINES_PER_STAFF) {
            throw new GoldenError(
              `${at}.lines[${g}]`,
              `expected ${LINES_PER_STAFF} rows, got ${rows.length}`
            );
          }
          for (let r = 1; r < rows.length; r += 1) {
            if ((rows[r] ?? 0) <= (rows[r - 1] ?? 0)) {
              throw new GoldenError(
                `${at}.lines[${g}]`,
                "rows of a group must ascend"
              );
            }
          }
          return rows;
        }
      );
      return { lines, start, stop };
    }
  );
  if (zones.length > 0 && clefs.length === 0) {
    throw new GoldenError(`${path}.zones`, "zones without a clef anchor");
  }
  return {
    barLines,
    clefs,
    kept: keptIndices.map((index, i) =>
      pick(all, index, `${path}.kept[${i}]`, "clefs + otherClefs + barLines")
    ),
    otherClefSymbols: decodeRotatedBoxes(
      field(object, "otherClefSymbols", path),
      `${path}.otherClefSymbols`
    ),
    otherClefs,
    zones,
  };
}

/**
 * One RawStaff. The box is Python's stored rect and polygon, not a refit: the
 * decoder has no opencv.js. Its contour is rebuilt as RawStaff.__init__ builds
 * it, from the lines.
 */
function rawStaffOf(
  json: unknown,
  fragments: readonly RotatedBox[],
  keptAnchors: readonly StaffAnchor[],
  path: string
): RawStaff {
  const object = asObject(json, path);
  const box = asArray(field(object, "box", path), `${path}.box`);
  if (box.length !== 3) {
    throw new GoldenError(
      `${path}.box`,
      `expected [[cx, cy], [w, h], angle], got ${box.length} parts`
    );
  }
  const [cx, cy] = pair(box[0], `${path}.box[0]`);
  const [w, h] = pair(box[1], `${path}.box[1]`);
  let rect: RotatedRect;
  try {
    rect = assertNormalizedRect({
      angle: asNumber(box[2], `${path}.box[2]`),
      cx,
      cy,
      h,
      w,
    });
  } catch (error) {
    throw goldenErrorAt(`${path}.box`, error);
  }
  const polygon = polygonOf(field(object, "polygon", path), `${path}.polygon`);
  if (pointCount(polygon) !== 4) {
    throw new GoldenError(`${path}.polygon`, "expected four corners");
  }
  const [first, ...rest] = asArray(
    field(object, "anchors", path),
    `${path}.anchors`
  ).map((v, i) =>
    pick(keptAnchors, v, `${path}.anchors[${i}]`, "the kept anchors")
  );
  if (first === undefined) {
    throw new GoldenError(`${path}.anchors`, "a raw staff with no anchor");
  }
  const lines = fiveLinesOf(
    field(object, "lines", path),
    fragments,
    `${path}.lines`
  );
  return rawStaffFromParts(
    rotatedBoxFromParts(
      rect,
      polygon,
      rawStaffContour(lines),
      integerOf(field(object, "staffId", path), `${path}.staffId`)
    ),
    lines,
    [first, ...rest]
  );
}

/** raw-staffs.json. */
export interface GoldenRawStaffs {
  /** find_raw_staffs_by_connecting_line_fragments, in list order. */
  readonly connected: RawStaff[];
  /** remove_duplicate_staffs: objects of `connected`, in output order. resample_staffs returns one Staff per entry, in this order. */
  readonly deduplicated: RawStaff[];
  /** Indices into `deduplicated` whose resampled staff filter_edge_of_vision dropped. */
  readonly droppedAtEdge: readonly number[];
  /** For staffs.json[i], the index into `deduplicated` of the raw staff it was resampled from. */
  readonly resampledFrom: readonly number[];
}

/** Asserts that every deduplicated staff is either in staffs.json or dropped at the edge, and never both. */
export function decodeRawStaffs(
  json: unknown,
  fragments: readonly RotatedBox[],
  keptAnchors: readonly StaffAnchor[],
  path = "$"
): GoldenRawStaffs {
  const object = asObject(json, path);
  const connected = asArray(
    field(object, "connected", path),
    `${path}.connected`
  ).map((v, i) =>
    rawStaffOf(v, fragments, keptAnchors, `${path}.connected[${i}]`)
  );
  const deduplicatedIndices = indicesInto(
    field(object, "deduplicated", path),
    connected.length,
    `${path}.deduplicated`,
    "connected"
  );
  assertDistinct(deduplicatedIndices, `${path}.deduplicated`);
  const resampledFrom = indicesInto(
    field(object, "resampledFrom", path),
    deduplicatedIndices.length,
    `${path}.resampledFrom`,
    "deduplicated"
  );
  const droppedAtEdge = indicesInto(
    field(object, "droppedAtEdge", path),
    deduplicatedIndices.length,
    `${path}.droppedAtEdge`,
    "deduplicated"
  );
  const accounted = [...resampledFrom, ...droppedAtEdge];
  assertDistinct(accounted, `${path}.resampledFrom + droppedAtEdge`);
  if (accounted.length !== deduplicatedIndices.length) {
    throw new GoldenError(
      path,
      `${deduplicatedIndices.length} deduplicated staffs, of which ${accounted.length} are resampled or dropped`
    );
  }
  return {
    connected,
    deduplicated: deduplicatedIndices.map((index, i) =>
      pick(connected, index, `${path}.deduplicated[${i}]`, "connected")
    ),
    droppedAtEdge,
    resampledFrom,
  };
}

/** notehead-splits.json: one (staff, notehead) pair that split_clumps_of_noteheads cut into more than one piece. */
export interface GoldenNoteheadSplit {
  /** Index into noteheads-with-stems.json. */
  readonly notehead: number;
  /** The pieces in split order, before add_notes_to_staffs filters them by size. They share the clump's contour. */
  readonly pieces: Ellipse[];
  /** Index into staffs.json. */
  readonly staff: number;
}

/** Empty on a page with no clumped noteheads, as the Kesh page is. */
export function decodeNoteheadSplits(
  json: unknown,
  path = "$"
): GoldenNoteheadSplit[] {
  return asArray(json, path).map((v, i) => {
    const at = `${path}[${i}]`;
    const object = asObject(v, at);
    const pieces = decodeEllipses(field(object, "pieces", at), `${at}.pieces`);
    if (pieces.length < 2) {
      throw new GoldenError(
        `${at}.pieces`,
        `a split has at least two pieces, got ${pieces.length}`
      );
    }
    const index = (name: string): number => {
      const n = integerOf(field(object, name, at), `${at}.${name}`);
      if (n < 0) {
        throw new GoldenError(`${at}.${name}`, `negative index ${n}`);
      }
      return n;
    };
    return { notehead: index("notehead"), pieces, staff: index("staff") };
  });
}

/** One (staff, neighbour) pair of find_braces_brackets_and_grand_staff_lines with at least one connection. */
export interface GoldenBraceConnection {
  /** Index into staffs.json: staff - 1 or staff + 1. */
  readonly neighbour: number;
  /** Index into staffs.json. */
  readonly staff: number;
  /** _get_connections_between_staffs: indices into boxes-brace_dot.json, in result order. A box found by two of the three searches appears twice. */
  readonly symbols: readonly number[];
}

/** braces.json. Every number is an index: into boxes-brace_dot.json, or into the staffs of staffs.json. */
export interface GoldenBraces {
  /** In homr's loop order: by staff, the upper neighbour before the lower. */
  readonly connections: GoldenBraceConnection[];
  /** Staff indices per MultiStaff after _merge_multi_staff_if_they_share_a_staff, before grand staffs. */
  readonly merged: readonly (readonly number[])[];
  /** How many of notes.json's notes each staff of staffs.json received, in order. */
  readonly notesPerStaff: readonly number[];
  /** _filter_for_tall_elements, in boxes-brace_dot.json order. */
  readonly tall: readonly number[];
}

export function decodeBraces(json: unknown, path = "$"): GoldenBraces {
  const object = asObject(json, path);
  const notesPerStaff = asArray(
    field(object, "notesPerStaff", path),
    `${path}.notesPerStaff`
  ).map((v, i) => {
    const count = integerOf(v, `${path}.notesPerStaff[${i}]`);
    if (count < 0) {
      throw new GoldenError(`${path}.notesPerStaff[${i}]`, "negative count");
    }
    return count;
  });
  const staffCount = notesPerStaff.length;
  const tall = asArray(field(object, "tall", path), `${path}.tall`).map(
    (v, i) => integerOf(v, `${path}.tall[${i}]`)
  );
  for (let i = 0; i < tall.length; i += 1) {
    if ((tall[i] ?? 0) < 0 || (i > 0 && (tall[i] ?? 0) <= (tall[i - 1] ?? 0))) {
      throw new GoldenError(
        `${path}.tall[${i}]`,
        "tall must be ascending indices: the filter keeps list order"
      );
    }
  }
  const tallSet = new Set(tall);
  const connections = asArray(
    field(object, "connections", path),
    `${path}.connections`
  ).map((v, i): GoldenBraceConnection => {
    const at = `${path}.connections[${i}]`;
    const connection = asObject(v, at);
    const [staff] = indicesInto(
      [field(connection, "staff", at)],
      staffCount,
      `${at}.staff`,
      "the staffs"
    );
    const [neighbour] = indicesInto(
      [field(connection, "neighbour", at)],
      staffCount,
      `${at}.neighbour`,
      "the staffs"
    );
    if (
      staff === undefined ||
      neighbour === undefined ||
      Math.abs(staff - neighbour) !== 1
    ) {
      throw new GoldenError(at, "a neighbour is the staff above or below");
    }
    const symbols = asArray(
      field(connection, "symbols", at),
      `${at}.symbols`
    ).map((s, k) => {
      const symbol = integerOf(s, `${at}.symbols[${k}]`);
      if (!tallSet.has(symbol)) {
        throw new GoldenError(
          `${at}.symbols[${k}]`,
          `${symbol} is not one of the tall elements`
        );
      }
      return symbol;
    });
    if (symbols.length === 0) {
      throw new GoldenError(`${at}.symbols`, "a connection with no symbol");
    }
    return { neighbour, staff, symbols };
  });
  const merged = asArray(field(object, "merged", path), `${path}.merged`).map(
    (v, i) => {
      const staffs = indicesInto(
        v,
        staffCount,
        `${path}.merged[${i}]`,
        "the staffs"
      );
      if (staffs.length === 0) {
        throw new GoldenError(`${path}.merged[${i}]`, "an empty multi staff");
      }
      assertDistinct(staffs, `${path}.merged[${i}]`);
      return staffs;
    }
  );
  const placed = new Set(merged.flat());
  if (placed.size !== staffCount) {
    throw new GoldenError(
      `${path}.merged`,
      `${placed.size} of ${staffCount} staffs are in a multi staff`
    );
  }
  return { connections, merged, notesPerStaff, tall };
}

export const NOISE_OUTCOMES = ["clean", "masked", "skipped"] as const;

/**
 * What filter_predictions did: `clean` found no noisy tile, `masked` blanked
 * the noisy tiles in every mask and in the page, `skipped` found more than
 * half the tiles noisy and left everything alone.
 */
export type GoldenNoiseOutcome = (typeof NOISE_OUTCOMES)[number];

/** noise.json: create_grid, apply_noise_filter and handle_filter_results on the page's staff mask. */
export interface GoldenNoise {
  /** Tiles above the noise limit with a neighbour above it. */
  readonly filtered: number;
  /** One row per tile row: the uint8 numpy stored, which wraps a noise estimate above 255. */
  readonly grid: readonly (readonly number[])[];
  readonly outcome: GoldenNoiseOutcome;
  /** Tile size in pixels: the page's height and width floor-divided by 20. */
  readonly tile: { readonly height: number; readonly width: number };
  readonly total: number;
}

export function decodeNoise(json: unknown, path = "$"): GoldenNoise {
  const object = asObject(json, path);
  const [tileHeight, tileWidth] = pair(
    field(object, "tile", path),
    `${path}.tile`
  );
  if (
    !(Number.isInteger(tileHeight) && Number.isInteger(tileWidth)) ||
    tileHeight < 1 ||
    tileWidth < 1
  ) {
    throw new GoldenError(`${path}.tile`, "expected two positive integers");
  }
  const grid = asArray(field(object, "grid", path), `${path}.grid`).map(
    (row, i) =>
      numbers(row, `${path}.grid[${i}]`).map((value, j) => {
        if (!Number.isInteger(value) || value < 0 || value > 255) {
          throw new GoldenError(
            `${path}.grid[${i}][${j}]`,
            `expected a uint8, got ${value}`
          );
        }
        return value;
      })
  );
  const columns = grid[0]?.length ?? 0;
  if (columns === 0 || grid.some((row) => row.length !== columns)) {
    throw new GoldenError(`${path}.grid`, "expected a non-empty rectangle");
  }
  const filtered = integerOf(
    field(object, "filtered", path),
    `${path}.filtered`
  );
  const total = integerOf(field(object, "total", path), `${path}.total`);
  if (total !== grid.length * columns) {
    throw new GoldenError(
      `${path}.total`,
      `${total} tiles counted in a ${grid.length} by ${columns} grid`
    );
  }
  if (filtered < 0 || filtered > total) {
    throw new GoldenError(`${path}.filtered`, `${filtered} of ${total} tiles`);
  }
  const name = asString(field(object, "outcome", path), `${path}.outcome`);
  const outcome = NOISE_OUTCOMES.find((known) => known === name);
  if (outcome === undefined) {
    throw new GoldenError(`${path}.outcome`, `unknown outcome ${name}`);
  }
  let expected: GoldenNoiseOutcome = "masked";
  if (filtered === 0) {
    expected = "clean";
  } else if (filtered / total > 0.5) {
    expected = "skipped";
  }
  if (outcome !== expected) {
    throw new GoldenError(
      `${path}.outcome`,
      `${outcome}, where ${filtered} of ${total} filtered tiles means ${expected}`
    );
  }
  return {
    filtered,
    grid,
    outcome,
    tile: { height: tileHeight, width: tileWidth },
    total,
  };
}

// dewarp-<n>.json (phase 6)

/** The four corners homr writes as [x1, y1, x2, y2]. */
export type GoldenCorners = readonly [number, number, number, number];

/** What tools/dump-golden.py's unrolled prepare_staff_image saved for one canvas. */
export interface GoldenDewarp {
  readonly affine: readonly (AffineMatrix | null)[];
  readonly dst: readonly Point[];
  /** get_tr_omr_canvas_size: [width, height]. */
  readonly imageDimensions: readonly [number, number];
  readonly optimalPoints: readonly (readonly Point[])[];
  readonly region: GoldenCorners;
  readonly regionStep1: GoldenCorners;
  readonly regionStep2: GoldenCorners;
  /** (width, height) handed to cv2.resize. */
  readonly resizedSize: readonly [number, number];
  readonly scaledRegion: GoldenCorners;
  readonly scalingFactor: number;
  readonly simplices: readonly Triangle[];
  readonly spanPoints: readonly (readonly Point[])[];
  readonly src: readonly Point[];
  readonly topLeftStep1: Point;
  readonly topLeftStep2: Point;
}

function regionCornersOf(value: unknown, path: string): GoldenCorners {
  const [x1, y1, x2, y2, ...rest] = numbers(value, path);
  if (
    x1 === undefined ||
    y1 === undefined ||
    x2 === undefined ||
    y2 === undefined ||
    rest.length > 0
  ) {
    throw new GoldenError(path, "expected four numbers");
  }
  return [x1, y1, x2, y2];
}

const pointsOf = (value: unknown, path: string): Point[] =>
  asArray(value, path).map((p, i) => pointOf(p, `${path}[${i}]`));

export function decodeDewarp(json: unknown, path = "$"): GoldenDewarp {
  const object = asObject(json, path);
  const at = (name: string): unknown => field(object, name, path);
  const rows = (name: string): Point[][] =>
    asArray(at(name), `${path}.${name}`).map((row, i) =>
      pointsOf(row, `${path}.${name}[${i}]`)
    );
  const src = pointsOf(at("src"), `${path}.src`);
  const simplices = asArray(at("simplices"), `${path}.simplices`).map(
    (value, i): Triangle => {
      const [a, b, c, ...rest] = numbers(value, `${path}.simplices[${i}]`);
      if (
        a === undefined ||
        b === undefined ||
        c === undefined ||
        rest.length > 0 ||
        ![a, b, c].every((v) => Number.isInteger(v) && v >= 0 && v < src.length)
      ) {
        throw new GoldenError(
          `${path}.simplices[${i}]`,
          "not three point indices"
        );
      }
      return [a, b, c];
    }
  );
  const affine = asArray(at("affine"), `${path}.affine`).map(
    (value, i): AffineMatrix | null => {
      if (value === null) {
        return null;
      }
      const flat = asArray(value, `${path}.affine[${i}]`).flatMap((row, r) =>
        numbers(row, `${path}.affine[${i}][${r}]`)
      );
      const [m0, m1, m2, m3, m4, m5, ...rest] = flat;
      if (
        m0 === undefined ||
        m1 === undefined ||
        m2 === undefined ||
        m3 === undefined ||
        m4 === undefined ||
        m5 === undefined ||
        rest.length > 0
      ) {
        throw new GoldenError(`${path}.affine[${i}]`, "not a 2 by 3 matrix");
      }
      return [m0, m1, m2, m3, m4, m5];
    }
  );
  if (affine.length !== simplices.length) {
    throw new GoldenError(`${path}.affine`, "one matrix per simplex");
  }
  return {
    affine,
    dst: pointsOf(at("dst"), `${path}.dst`),
    imageDimensions: pair(at("imageDimensions"), `${path}.imageDimensions`),
    optimalPoints: rows("optimalPoints"),
    region: regionCornersOf(at("region"), `${path}.region`),
    regionStep1: regionCornersOf(at("regionStep1"), `${path}.regionStep1`),
    regionStep2: regionCornersOf(at("regionStep2"), `${path}.regionStep2`),
    resizedSize: pair(at("resizedSize"), `${path}.resizedSize`),
    scaledRegion: regionCornersOf(at("scaledRegion"), `${path}.scaledRegion`),
    scalingFactor: asNumber(at("scalingFactor"), `${path}.scalingFactor`),
    simplices,
    spanPoints: rows("spanPoints"),
    src,
    topLeftStep1: pointOf(at("topLeftStep1"), `${path}.topLeftStep1`),
    topLeftStep2: pointOf(at("topLeftStep2"), `${path}.topLeftStep2`),
  };
}

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

import {
  type AxisBox,
  assertNormalizedRect,
  cornersOf,
  createAxisBox,
  type Ellipse,
  ellipseFromParts,
  type Point,
  type PointList,
  pointListFromPairs,
  type RotatedBox,
  type RotatedRect,
  rotatedBoxFromParts,
} from "../geometry/boxes.js";
import { diff, mean } from "../image/numeric.js";
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

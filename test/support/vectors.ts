/**
 * test/golden/vectors/<name>.json: small inputs run through the pinned Python
 * by tools/dump-vectors.py, for the branches no page reaches. Read by tests
 * only, never by a stage, which is why this is not in src/golden/.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  decodeBraces,
  decodeMultiStaffs,
  decodeNoise,
  decodeNoteheadSplits,
  decodeNoteheadsWithStems,
  decodeNotes,
  decodeRotatedBoxes,
  decodeStaff,
  decodeStaffs,
} from "../../src/golden/decode.js";
import { type Mask, planeFromBytes } from "../../src/image/plane.js";

const vectorRoot = join(import.meta.dirname, "..", "golden", "vectors");

export const VECTOR_FILES = [
  "argsort",
  "bbox-split",
  "braces",
  "braces-units",
  "connect-lines",
  "connect-lines-cleanup",
  "dewarp-points",
  "dewarp-warp",
  "edge-of-vision",
  "find-anchors",
  "find-peaks",
  "floor-div",
  "grand-staffs",
  "intersections",
  "line-groups",
  "line-peak-groups",
  "multi-staff-merge",
  "noise",
  "notehead-clumps",
  "pairwise",
  "raw-staff-merge",
  "resample",
  "staff-merge",
  "staff-regrouping",
] as const;

export type VectorFile = (typeof VECTOR_FILES)[number];

export class VectorError extends Error {}

/** Parsed and untyped; each test narrows its own file. */
export function readVectors(name: VectorFile): unknown {
  return JSON.parse(readFileSync(join(vectorRoot, `${name}.json`), "utf8"));
}

export type VectorCase = Readonly<Record<string, unknown>>;

/** What every vector file is: the oracle that wrote it, and its cases. */
export interface VectorSet {
  readonly cases: readonly VectorCase[];
  readonly meta: Readonly<Record<string, string>>;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

/** The file's envelope, checked; the cases stay untyped. */
export function vectorSet(name: VectorFile): VectorSet {
  const file = readVectors(name);
  if (!(isRecord(file) && isRecord(file.meta) && Array.isArray(file.cases))) {
    throw new VectorError(
      `${name}.json is not {"meta": {...}, "cases": [...]}`
    );
  }
  const meta: Record<string, string> = {};
  for (const [key, value] of Object.entries(file.meta)) {
    if (typeof value !== "string") {
      throw new VectorError(`${name}.json: meta.${key} is not a string`);
    }
    meta[key] = value;
  }
  const cases = file.cases.map((entry: unknown, i) => {
    if (!isRecord(entry)) {
      throw new VectorError(`${name}.json: cases[${i}] is not an object`);
    }
    return entry;
  });
  return { cases, meta };
}

function need(condition: boolean, what: string): asserts condition {
  if (!condition) {
    throw new VectorError(what);
  }
}

const isNumber = (value: unknown): value is number => typeof value === "number";
const isInteger = (value: unknown): value is number => Number.isInteger(value);

function list(value: unknown, at: string): unknown[] {
  need(Array.isArray(value), `${at}: expected a list`);
  return value;
}

function numberList(value: unknown, at: string, length?: number): number[] {
  const values = list(value, at);
  need(values.every(isNumber), `${at}: expected numbers`);
  need(
    length === undefined || values.length === length,
    `${at}: expected ${length} numbers, got ${values.length}`
  );
  return values;
}

const MASK_ROW = /^[01]+$/;

/** The 0/1 mask a vector file writes one string per row. */
export function maskOfRows(value: unknown, at: string): Mask {
  const [width, height] = maskSize(value, at);
  const data = Uint8Array.from((value as string[]).join(""), Number);
  return planeFromBytes("mask", width, height, data);
}

/** A 0/1 mask written one string per row, all of one width. Returns [width, height]. */
function maskSize(value: unknown, at: string): readonly [number, number] {
  const rows = list(value, at);
  const width = typeof rows[0] === "string" ? rows[0].length : 0;
  need(
    rows.every(
      (row) =>
        typeof row === "string" && row.length === width && MASK_ROW.test(row)
    ),
    `${at}: expected rows of 0 and 1 of one width`
  );
  return [width, rows.length];
}

/** A `[[cx, cy], [w, h], angle]` rect whose every number float32 holds exactly, as opencv.js will hold it. */
function needRect(value: unknown, at: string): void {
  const parts = list(value, at);
  need(parts.length === 3 && isNumber(parts[2]), `${at}: expected a rect`);
  const numbers = [
    ...numberList(parts[0], at, 2),
    ...numberList(parts[1], at, 2),
    parts[2],
  ];
  need(
    numbers.every((n) => Math.fround(n) === n),
    `${at}: a rect value is not a float32`
  );
}

/** A list of rects. Returns how many, which is what an index into it must stay under. */
function needRects(value: unknown, at: string): number {
  const rects = list(value, at);
  for (const [i, rect] of rects.entries()) {
    needRect(rect, `${at}[${i}]`);
  }
  return rects.length;
}

/** Five staff lines, each a non-empty list of indices into `count` fragments. */
function needFiveLines(value: unknown, count: number, at: string): void {
  const lines = list(value, at);
  need(lines.length === 5, `${at}: expected five lines`);
  for (const [i, line] of lines.entries()) {
    const members = numberList(line, `${at}[${i}]`);
    need(
      members.length > 0 &&
        members.every((k) => isInteger(k) && k >= 0 && k < count),
      `${at}[${i}]: expected indices into fragments`
    );
  }
}

/** A StaffAnchor as the staff-chain files write it: symbol rect, five lines, unit size. */
function needAnchor(value: unknown, count: number, at: string): void {
  need(isRecord(value), `${at}: expected an object`);
  needRect(value.symbol, `${at}.symbol`);
  needFiveLines(value.lines, count, `${at}.lines`);
  need(isNumber(value.averageUnitSize), `${at}.averageUnitSize`);
}

/** A case holds either the function's result or the name of the exception Python raised. */
function needOutcome(one: VectorCase, result: string, at: string): void {
  need(
    "error" in one !== result in one,
    `${at}: expected ${result} or error, not both`
  );
  need(
    !("error" in one) || typeof one.error === "string",
    `${at}.error: expected an exception name`
  );
}

interface VectorCheck {
  readonly check: (one: VectorCase, at: string) => void;
  /** Present in every case of the file. */
  readonly keys: readonly string[];
}

/** connect-lines.json and connect-lines-cleanup.json hold the same case shape. */
const CONNECT_LINES_CHECK: VectorCheck = {
  check: (one, at) => {
    const fragments = list(one.fragments, `${at}.fragments`);
    for (const [i, fragment] of fragments.entries()) {
      needRect(fragment, `${at}.fragments[${i}]`);
    }
    for (const [i, line] of list(one.lines, `${at}.lines`).entries()) {
      const members = numberList(line, `${at}.lines[${i}]`);
      need(
        members.length > 0 &&
          members.every((k) => isInteger(k) && k >= 0 && k < fragments.length),
        `${at}.lines[${i}]: expected indices into fragments`
      );
    }
    need(typeof one.parallel === "boolean", `${at}.parallel`);
    for (const [i, probe] of list(one.probes, `${at}.probes`).entries()) {
      need(
        typeof probe === "object" && probe !== null,
        `${at}.probes[${i}]: expected an object`
      );
      const { line, onOneLine } = probe as VectorCase;
      needRect(line, `${at}.probes[${i}].line`);
      need(typeof onOneLine === "boolean", `${at}.probes[${i}].onOneLine`);
    }
  },
  keys: ["name", "fragments", "unitSize", "lines", "parallel", "probes"],
};

/** braces.json and braces-units.json: one shape, the second with staffs of unlike units. */
const BRACES_CHECK: VectorCheck = {
  check: (one, at) => {
    const staffs = decodeStaffs(one.staffs, `${at}.staffs`);
    const braceDot = decodeRotatedBoxes(one.braceDot, `${at}.braceDot`);
    needOutcome(one, "multiStaffs", at);
    if ("error" in one) {
      return;
    }
    const braces = decodeBraces(one, at);
    need(
      braces.notesPerStaff.length === staffs.length,
      `${at}.notesPerStaff: one count per staff`
    );
    need(
      braces.tall.every((index) => index < braceDot.length),
      `${at}.tall: an index outside braceDot`
    );
    const multiStaffs = decodeMultiStaffs(one.multiStaffs, `${at}.multiStaffs`);
    need(
      multiStaffs.length === braces.merged.length,
      `${at}: ${multiStaffs.length} multi staffs for ${braces.merged.length} merged groups`
    );
  },
  keys: ["name", "staffs", "braceDot"],
};

const VECTOR_CHECKS: Record<VectorFile, VectorCheck> = {
  argsort: {
    check: (one, at) => {
      const values = numberList(one.values, `${at}.values`);
      const order = numberList(one.order, `${at}.order`, values.length);
      need(
        new Set(order).size === order.length &&
          order.every((i) => isInteger(i) && i >= 0 && i < values.length),
        `${at}.order: not a permutation`
      );
      for (let i = 1; i < order.length; i += 1) {
        const before = values[order[i - 1] ?? 0] ?? Number.NaN;
        const after = values[order[i] ?? 0] ?? Number.NaN;
        need(before <= after, `${at}.order: does not sort the values at ${i}`);
      }
      need(isInteger(one.heapsorted), `${at}.heapsorted: expected a count`);
    },
    keys: ["values", "order", "heapsorted"],
  },
  "bbox-split": {
    check: (one, at) => {
      maskSize(one.mask, `${at}.mask`);
      numberList(one.bbox, `${at}.bbox`, 4);
      numberList(one.adjusted, `${at}.adjusted`, 4);
      need(isNumber(one.unitSize), `${at}.unitSize: expected a number`);
      needOutcome(one, "boxes", at);
      for (const box of "boxes" in one ? list(one.boxes, `${at}.boxes`) : []) {
        numberList(box, `${at}.boxes`, 4);
      }
    },
    keys: ["name", "mask", "bbox", "unitSize", "adjusted"],
  },
  braces: BRACES_CHECK,
  "braces-units": BRACES_CHECK,
  "connect-lines": CONNECT_LINES_CHECK,
  "connect-lines-cleanup": CONNECT_LINES_CHECK,
  "dewarp-points": {
    check: (one, at) => {
      decodeStaff(one.staff, {}, `${at}.staff`);
      need(
        isInteger(one.width) && isInteger(one.height),
        `${at}: width and height`
      );
      const rows = list(one.span, `${at}.span`);
      need(
        list(one.optimal, `${at}.optimal`).length === rows.length,
        `${at}.optimal: one row per span row`
      );
      if (rows.length > 0) {
        const count = list(one.src, `${at}.src`).length;
        need(
          list(one.dst, `${at}.dst`).length === count,
          `${at}.dst: one point per src point`
        );
        need(
          list(one.affine, `${at}.affine`).length ===
            list(one.simplices, `${at}.simplices`).length,
          `${at}.affine: one matrix or null per simplex`
        );
      }
    },
    keys: ["name", "staff", "width", "height", "span", "optimal"],
  },
  "dewarp-warp": {
    check: (one, at) => {
      need(
        typeof one.input === "string" && typeof one.warped === "string",
        `${at}: input and warped name PNG files`
      );
      for (const probe of list(one.probes, `${at}.probes`)) {
        need(
          Array.isArray(probe) && probe.length === 3,
          `${at}.probes: [point, simplex, transformed]`
        );
      }
    },
    keys: ["name", "input", "warped", "src", "dst", "simplices", "probes"],
  },
  "edge-of-vision": {
    check: (one, at) => {
      const staffs = decodeStaffs(one.staffs, `${at}.staffs`);
      need(
        isInteger(one.height) && isInteger(one.width),
        `${at}: expected a size`
      );
      need(
        list(one.labels, `${at}.labels`).length === staffs.length,
        `${at}.labels: one per staff`
      );
      const kept = numberList(one.kept, `${at}.kept`);
      need(
        kept.every(
          (k, i) => isInteger(k) && k > (kept[i - 1] ?? -1) && k < staffs.length
        ),
        `${at}.kept: expected ascending indices into staffs`
      );
      need(isNumber(one.usualWidth), `${at}.usualWidth: expected a number`);
    },
    keys: ["name", "height", "width", "labels", "staffs", "kept", "usualWidth"],
  },
  "find-anchors": {
    check: (one, at) => {
      const count = needRects(one.fragments, `${at}.fragments`);
      needRect(one.symbol, `${at}.symbol`);
      need(one.kind === "clef" || one.kind === "barLine", `${at}.kind`);
      for (const [i, anchor] of list(one.anchors, `${at}.anchors`).entries()) {
        needAnchor(anchor, count, `${at}.anchors[${i}]`);
      }
    },
    keys: ["name", "fragments", "symbol", "kind", "anchors"],
  },
  "find-peaks": {
    check: (one, at) => {
      const x = numberList(one.x, `${at}.x`);
      const peaks = numberList(one.peaks, `${at}.peaks`);
      for (const [i, peak] of peaks.entries()) {
        need(
          peak > (i === 0 ? 0 : (peaks[i - 1] ?? 0)) && peak < x.length - 1,
          `${at}.peaks[${i}]: not an ascending interior index`
        );
      }
      for (const option of ["height", "distance", "prominence"]) {
        need(
          one[option] === null || isNumber(one[option]),
          `${at}.${option}: expected a number or null`
        );
      }
    },
    keys: ["x", "height", "distance", "prominence", "peaks"],
  },
  "floor-div": {
    check: (one, at) => {
      need(isNumber(one.a) && isNumber(one.b), `${at}: expected numbers`);
      need(isInteger(one.q), `${at}.q: expected an integral float`);
    },
    keys: ["a", "b", "q"],
  },
  "grand-staffs": {
    check: (one, at) => {
      const staffs = decodeStaffs(one.staffs, `${at}.staffs`);
      const braces = decodeRotatedBoxes(one.braces, `${at}.braces`);
      const scores = list(one.scores, `${at}.scores`);
      need(
        scores.length === staffs.length - 1,
        `${at}.scores: one row per adjacent pair`
      );
      for (const [i, row] of scores.entries()) {
        numberList(row, `${at}.scores[${i}]`, braces.length);
      }
      decodeMultiStaffs([one.multiStaff], `${at}.multiStaff`);
    },
    keys: ["name", "staffs", "braces", "scores", "multiStaff"],
  },
  intersections: {
    check: (one, at) => {
      needRect(one.a, `${at}.a`);
      needRect(one.b, `${at}.b`);
      need(one.code === 0 || one.code === 1 || one.code === 2, `${at}.code`);
      need(
        one.intersecting === (one.code !== 0),
        `${at}.intersecting: disagrees with code`
      );
    },
    keys: ["a", "b", "code", "intersecting"],
  },
  "line-groups": {
    check: (one, at) => {
      const { height, width } = one;
      need(isInteger(height) && isInteger(width), `${at}: expected a size`);
      for (const entry of list(one.rowCounts, `${at}.rowCounts`)) {
        const [y, count] = numberList(entry, `${at}.rowCounts`, 2);
        need(
          y !== undefined &&
            count !== undefined &&
            y >= 0 &&
            y < height &&
            count <= width,
          `${at}.rowCounts: a row or a count outside the image`
        );
      }
      needOutcome(one, "groups", at);
      for (const group of "groups" in one
        ? list(one.groups, `${at}.groups`)
        : []) {
        numberList(group, `${at}.groups`, 5);
      }
    },
    keys: ["name", "height", "width", "rowCounts", "unitSize"],
  },
  "line-peak-groups": {
    check: (one, at) => {
      const peaks = numberList(one.peaks, `${at}.peaks`);
      numberList(one.groups, `${at}.groups`, peaks.length);
    },
    keys: ["name", "peaks", "groups"],
  },
  "multi-staff-merge": {
    check: (one, at) => {
      for (const key of ["given", "merged"]) {
        for (const [i, entry] of list(one[key], `${at}.${key}`).entries()) {
          need(isRecord(entry), `${at}.${key}[${i}]: expected an object`);
          const staffs = numberList(entry.staffs, `${at}.${key}[${i}].staffs`);
          const links = numberList(
            entry.connections,
            `${at}.${key}[${i}].connections`
          );
          need(
            staffs.length > 0 &&
              staffs.every((k) => isInteger(k) && k >= 0 && k < 5) &&
              links.every((k) => isInteger(k) && k >= 0 && k < 4),
            `${at}.${key}[${i}]: expected staffs 0 to 4 and connections 0 to 3`
          );
        }
      }
    },
    keys: ["name", "given", "merged"],
  },
  noise: {
    check: (one, at) => {
      const [width, height] = maskSize(one.staff, `${at}.staff`);
      need(
        width === one.width && height === one.height,
        `${at}.staff: not width by height`
      );
      const noise = decodeNoise(one, at);
      need(
        noise.tile.height === Math.floor(height / 20) &&
          noise.tile.width === Math.floor(width / 20) &&
          noise.grid.length === Math.ceil(height / noise.tile.height),
        `${at}: the grid is not the staff mask cut in twentieths`
      );
      need(
        (one.mask === null) === (noise.outcome !== "masked"),
        `${at}.mask: present exactly when the outcome is masked`
      );
      if (one.mask !== null) {
        const [maskWidth, maskHeight] = maskSize(one.mask, `${at}.mask`);
        need(
          maskWidth === width && maskHeight === height,
          `${at}.mask: not the staff mask's size`
        );
      }
    },
    keys: [
      "name",
      "height",
      "width",
      "staff",
      "tile",
      "grid",
      "filtered",
      "total",
      "outcome",
      "mask",
    ],
  },
  "notehead-clumps": {
    check: (one, at) => {
      maskSize(one.mask, `${at}.mask`);
      const staffs = decodeStaffs(one.staffs, `${at}.staffs`);
      const noteheads = decodeNoteheadsWithStems(
        one.noteheads,
        `${at}.noteheads`
      );
      const notes = decodeNotes(one.notes, `${at}.notes`);
      const perStaff = numberList(
        one.notesPerStaff,
        `${at}.notesPerStaff`,
        staffs.length
      );
      need(
        perStaff.reduce((total, n) => total + n, 0) === notes.length,
        `${at}.notesPerStaff: does not add up to the notes`
      );
      need(
        staffs.every((staff) => staff.symbols.length === 0),
        `${at}.staffs: expected the staffs before any note`
      );
      for (const split of decodeNoteheadSplits(one.splits, `${at}.splits`)) {
        need(
          split.staff < staffs.length && split.notehead < noteheads.length,
          `${at}.splits: an index outside staffs or noteheads`
        );
      }
    },
    keys: [
      "name",
      "mask",
      "staffs",
      "noteheads",
      "splits",
      "notes",
      "notesPerStaff",
    ],
  },
  pairwise: {
    check: (one, at) => {
      const values = numberList(one.values, `${at}.values`);
      need(isNumber(one.sum), `${at}.sum: expected a number`);
      for (const name of ["mean", "std"]) {
        need(
          values.length === 0 ? one[name] === null : isNumber(one[name]),
          `${at}.${name}: a number, or null for no values`
        );
      }
    },
    keys: ["values", "sum", "mean", "std"],
  },
  "raw-staff-merge": {
    check: (one, at) => {
      const count = needRects(one.fragments, `${at}.fragments`);
      needAnchor(one.anchor, count, `${at}.anchor`);
      for (const name of ["self", "other", "merged"]) {
        need(isRecord(one[name]), `${at}.${name}: expected an object`);
        const staff = one[name] as VectorCase;
        need(isInteger(staff.staffId), `${at}.${name}.staffId`);
        needFiveLines(staff.lines, count, `${at}.${name}.lines`);
      }
    },
    keys: ["name", "fragments", "anchor", "self", "other", "merged"],
  },
  resample: {
    check: (one, at) => {
      const count = needRects(one.fragments, `${at}.fragments`);
      needFiveLines(one.lines, count, `${at}.lines`);
      needAnchor(one.anchor, count, `${at}.anchor`);
      needRect(one.box, `${at}.box`);
      decodeStaff(one.staff, {}, `${at}.staff`);
    },
    keys: ["name", "fragments", "lines", "anchor", "box", "staff"],
  },
  "staff-merge": {
    check: (one, at) => {
      decodeStaff(one.a, {}, `${at}.a`);
      decodeStaff(one.b, {}, `${at}.b`);
      needOutcome(one, "merged", at);
      if ("merged" in one) {
        const merged = decodeStaff(one.merged, {}, `${at}.merged`);
        need(
          merged.isGrandstaff && merged.grid[0].y.length === 10,
          `${at}.merged: expected a ten-line grand staff`
        );
      }
    },
    keys: ["name", "a", "b"],
  },
  "staff-regrouping": {
    check: (one, at) => {
      if ("sizes" in one) {
        for (const size of list(one.sizes, `${at}.sizes`)) {
          need(
            Array.isArray(size) && size.length === 3,
            `${at}.sizes: [height, width, [w, h]]`
          );
        }
        return;
      }
      const staffs = decodeStaffs(one.staffs, `${at}.staffs`);
      for (const key of ["systems", "result"]) {
        for (const system of list(one[key], `${at}.${key}`)) {
          need(
            numberList(system, `${at}.${key}`).every(
              (i) => isInteger(i) && i < staffs.length
            ),
            `${at}.${key}: an index outside staffs`
          );
        }
      }
      for (const probe of list(one.regions, `${at}.regions`)) {
        numberList(probe, `${at}.regions`, 3);
      }
    },
    keys: ["name"],
  },
};

/** Throws unless the case has the keys and the value shapes its file promises. */
export function checkVectorCase(
  name: VectorFile,
  one: VectorCase,
  at: string
): void {
  const { check, keys } = VECTOR_CHECKS[name];
  for (const key of keys) {
    need(key in one, `${at}: missing ${key}`);
  }
  check(one, at);
}

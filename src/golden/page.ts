/**
 * One golden fixture behind one object. `page.staffs()` replaces composing
 * a filename, reading a file, JSON.parse and a decoder call, so the
 * dumper's naming lives in exactly one place. The reader is injected: the
 * vitest helper passes node:fs and pngjs, a browser bench page passes fetch
 * and a canvas, and the library itself depends on neither.
 *
 * Golden data is the *input* of every stage test from phase 4 on (each
 * stage consumes the Python output of the stage before it), which is why
 * this lives in src/ and not in test/.
 */

import type { Ellipse, RotatedBox } from "../geometry/boxes.js";
import { type GrayImage, type Mask, planeFromBytes } from "../image/plane.js";
import {
  ENCODER_CANVAS,
  MASK_CLASSES,
  type MaskClass,
} from "../model/pipeline.js";
import type { MultiStaff, Staff } from "../model/staff.js";
import type { Note, NoteheadWithStem } from "../model/symbols.js";
import type { StaffPosition } from "../result.js";
import type { DecodedSymbol, EncodedSymbol } from "../transformer/symbol.js";
import {
  decodeBarLines,
  decodeEllipses,
  decodeMultiStaffs,
  decodeNoteheadsWithStems,
  decodeNotes,
  decodeRotatedBoxes,
  decodeStaff,
  decodeStaffPositions,
  decodeStaffs,
  decodeTokens,
  decodeVoices,
  GoldenError,
} from "./decode.js";

/** A decoded PNG as one byte per pixel (the golden PNGs are 1-bit masks and 8-bit gray). */
export interface GoldenPng {
  readonly gray: Uint8Array;
  readonly height: number;
  readonly width: number;
}

export interface GoldenReader {
  readonly png: (name: string) => GoldenPng;
  /** Contents of test/golden/<fixture>/<name> as text; throws when absent. */
  readonly text: (name: string) => string;
}

export interface GoldenMeta {
  readonly fixture: string;
  readonly homrVersion: string;
  readonly imageSha256: string;
  readonly models: Readonly<Record<"segnet" | "encoder" | "decoder", string>>;
  readonly stages: readonly string[];
}

export const GOLDEN_BOX_FILES = {
  barLines: "boxes-bar_lines.json",
  braceDot: "boxes-brace_dot.json",
  clefsKeys: "boxes-clefs_keys.json",
  staffFragments: "boxes-staff_fragments.json",
  staffFragmentsBroken: "boxes-staff_fragments-broken.json",
  stemsRest: "boxes-stems_rest.json",
} as const;

export type GoldenBoxKind = keyof typeof GOLDEN_BOX_FILES;

export interface GoldenPage {
  readonly barLines: () => {
    readonly averageNoteHeadHeight: number;
    readonly barLines: RotatedBox[];
  };
  readonly boxes: (kind: GoldenBoxKind) => RotatedBox[];
  readonly canvas: (index: number) => GrayImage;
  /** canvas-<n>-staff.json: the staff in canvas space. */
  readonly canvasStaff: (index: number) => Staff;
  /** mask-<name>.png, or mask-filtered-<name>.png (after noise filtering and make_lines_stronger). */
  readonly mask: (name: MaskClass, filtered?: boolean) => Mask;
  readonly meta: () => GoldenMeta;
  readonly multiStaffs: () => MultiStaff[];
  readonly musicXml: () => string;
  readonly noteheads: () => Ellipse[];
  readonly noteheadsWithStems: () => NoteheadWithStem[];
  readonly notes: () => Note[];
  readonly preprocessed: () => GrayImage;
  /** How many canvas-<n>.png files the page has: the staffs parsed, in parse order. */
  readonly staffCount: () => number;
  readonly staffPositions: () => StaffPosition[];
  readonly staffs: () => Staff[];
  readonly tokens: (index: number) => DecodedSymbol[];
  readonly voices: () => EncodedSymbol[][];
}

const CANVAS_FILE = /^canvas-\d+\.png$/;

function memo<T>(compute: () => T): () => T {
  let value: { readonly v: T } | undefined;
  return () => {
    if (value === undefined) {
      value = { v: compute() };
    }
    return value.v;
  };
}

function memoBy<K, T>(compute: (key: K) => T): (key: K) => T {
  const cache = new Map<K, T>();
  return (key) => {
    const hit = cache.get(key);
    if (hit !== undefined) {
      return hit;
    }
    const value = compute(key);
    cache.set(key, value);
    return value;
  };
}

function isMeta(value: unknown): value is GoldenMeta {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const v = value as Record<string, unknown>;
  return (
    typeof v.fixture === "string" &&
    typeof v.imageSha256 === "string" &&
    typeof v.homrVersion === "string" &&
    Array.isArray(v.stages) &&
    typeof v.models === "object"
  );
}

export function createGoldenPage(reader: GoldenReader): GoldenPage {
  const json = (name: string): unknown => JSON.parse(reader.text(name));
  const gray = (name: string): GrayImage => {
    const png = reader.png(name);
    return planeFromBytes("gray", png.width, png.height, png.gray);
  };
  const meta = memo(() => {
    const value = json("meta.json");
    if (!isMeta(value)) {
      throw new GoldenError("meta.json", "not a golden meta file");
    }
    return value;
  });
  const staffCount = memo(
    () => meta().stages.filter((name) => CANVAS_FILE.test(name)).length
  );
  return {
    barLines: memo(() =>
      decodeBarLines(json("barlines.json"), "barlines.json")
    ),
    boxes: memoBy((kind: GoldenBoxKind) =>
      decodeRotatedBoxes(json(GOLDEN_BOX_FILES[kind]), GOLDEN_BOX_FILES[kind])
    ),
    canvas: memoBy((index: number) => {
      const image = gray(`canvas-${index}.png`);
      if (
        image.width !== ENCODER_CANVAS.width ||
        image.height !== ENCODER_CANVAS.height
      ) {
        throw new GoldenError(
          `canvas-${index}.png`,
          `expected ${ENCODER_CANVAS.width}x${ENCODER_CANVAS.height}, got ${image.width}x${image.height}`
        );
      }
      return image;
    }),
    canvasStaff: memoBy((index: number) =>
      decodeStaff(
        json(`canvas-${index}-staff.json`),
        { space: "canvas" },
        `canvas-${index}-staff.json`
      )
    ),
    mask: memoBy((key: string) => {
      const [name, filtered] = key.split("|") as [MaskClass, string];
      const png = reader.png(
        `mask-${filtered === "1" ? "filtered-" : ""}${MASK_CLASSES[name].golden}.png`
      );
      const data = new Uint8Array(png.gray.length);
      for (let i = 0; i < data.length; i += 1) {
        data[i] = (png.gray[i] ?? 0) > 0 ? 1 : 0;
      }
      return planeFromBytes("mask", png.width, png.height, data);
    }).bind(null) as never as GoldenPage["mask"],
    meta,
    multiStaffs: memo(() =>
      decodeMultiStaffs(json("multistaffs.json"), "multistaffs.json")
    ),
    musicXml: memo(() => reader.text("page.musicxml")),
    noteheads: memo(() =>
      decodeEllipses(json("boxes-noteheads.json"), "boxes-noteheads.json")
    ),
    noteheadsWithStems: memo(() =>
      decodeNoteheadsWithStems(
        json("noteheads-with-stems.json"),
        "noteheads-with-stems.json"
      )
    ),
    notes: memo(() => decodeNotes(json("notes.json"), "notes.json")),
    preprocessed: memo(() => gray("preprocessed.png")),
    staffCount,
    staffPositions: memo(() =>
      decodeStaffPositions(reader.text("staff-positions.txt"))
    ),
    staffs: memo(() => decodeStaffs(json("staffs.json"), "staffs.json")),
    tokens: memoBy((index: number) =>
      decodeTokens(json(`tokens-${index}.json`), `tokens-${index}.json`)
    ),
    voices: memo(() => decodeVoices(json("voices.json"), "voices.json")),
  };
}

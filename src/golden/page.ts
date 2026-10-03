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
import { NOISE_GRID_DIVISIONS } from "../geometry/noise-filter.js";
import { floorDiv } from "../image/numeric.js";
import {
  type ColorImage,
  colorImageFromRgba,
  type GrayImage,
  type Mask,
  planeFromBytes,
} from "../image/plane.js";
import {
  ENCODER_CANVAS,
  MASK_CLASSES,
  type MaskClass,
} from "../model/pipeline.js";
import { createStaff, type MultiStaff, type Staff } from "../model/staff.js";
import type { Note, NoteheadWithStem } from "../model/symbols.js";
import type { StaffPosition } from "../result.js";
import type { DecodedSymbol, EncodedSymbol } from "../transformer/symbol.js";
import {
  decodeBarLines,
  decodeBraces,
  decodeDewarp,
  decodeEllipses,
  decodeMultiStaffs,
  decodeNoise,
  decodeNoteheadSplits,
  decodeNoteheadsWithStems,
  decodeNotes,
  decodeRawStaffs,
  decodeRotatedBoxes,
  decodeStaff,
  decodeStaffAnchors,
  decodeStaffPositions,
  decodeStaffs,
  decodeTokens,
  decodeVoices,
  type GoldenBraces,
  type GoldenDewarp,
  GoldenError,
  type GoldenNoise,
  type GoldenNoteheadSplit,
  type GoldenRawStaffs,
  type GoldenStaffAnchors,
} from "./decode.js";

/**
 * A decoded PNG as RGBA, four bytes per pixel. That is what both real
 * readers already hold (pngjs on Node, a canvas in a browser) whatever the
 * file's own depth, so it is the one representation at the boundary; gray
 * and BGR are derived from it inside createGoldenPage.
 */
export interface GoldenPng {
  readonly height: number;
  readonly rgba: Uint8Array;
  readonly width: number;
}

export type GoldenDewarpStage = "input" | "warped" | "cleaned";

export interface GoldenReader {
  readonly png: (name: string) => GoldenPng;
  /** Contents of test/golden/<fixture>/<name> as text; throws when absent. */
  readonly text: (name: string) => string;
}

/**
 * What produced a dump. `machine` is recorded because np.argsort's tie order
 * and the float-to-uint8 cast in the noise grid are the oracle machine's.
 */
export interface GoldenOracle {
  /** platform.machine(), "arm64" on the machine the public fixtures were dumped on. */
  readonly machine: string;
  readonly numpy: string;
  readonly opencv: string;
  readonly python: string;
}

export interface GoldenMeta {
  readonly fixture: string;
  readonly homrVersion: string;
  readonly imageSha256: string;
  readonly models: Readonly<Record<"segnet" | "encoder" | "decoder", string>>;
  /** Absent from a meta.json written before phase 5. */
  readonly oracle?: GoldenOracle;
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
  /** autocropped.png: the page after autocrop, in BGR. */
  readonly autocropped: () => ColorImage;
  readonly barLines: () => {
    readonly averageNoteHeadHeight: number;
    readonly barLines: RotatedBox[];
  };
  readonly boxes: (kind: GoldenBoxKind) => RotatedBox[];
  /** mask-brace_dot.png: prepare_brace_dot_image, the mask boxes-brace_dot.json was fitted to. */
  readonly braceDotMask: () => Mask;
  /** braces.json, checked against staffs.json, notes.json and boxes-brace_dot.json. */
  readonly braces: () => GoldenBraces;
  readonly canvas: (index: number) => GrayImage;
  /** canvas-<n>-staff.json: the staff in canvas space. */
  readonly canvasStaff: (index: number) => Staff;
  /** mask-denoised-staff.png: the staff mask after filter_predictions, before make_lines_stronger. */
  readonly denoisedStaffMask: () => Mask;
  /** dewarp-<n>.json: prepare_staff_image's intermediates for canvas n. */
  readonly dewarp: (index: number) => GoldenDewarp;
  /** dewarp-<n>-<stage>.png: the first crop, the warped crop, the cleaned second crop. */
  readonly dewarpImage: (index: number, stage: GoldenDewarpStage) => GrayImage;
  /** mask-<name>.png, or mask-filtered-<name>.png (after noise filtering and make_lines_stronger). */
  readonly mask: (name: MaskClass, filtered?: boolean) => Mask;
  readonly meta: () => GoldenMeta;
  readonly multiStaffs: () => MultiStaff[];
  readonly musicXml: () => string;
  /** noise.json, checked against the staff mask's size. */
  readonly noise: () => GoldenNoise;
  /** mask-noise.png: the tiles filter_predictions kept, or null unless the outcome is `masked` (the file exists only then). */
  readonly noiseMask: () => Mask | null;
  readonly noteheadSplits: () => GoldenNoteheadSplit[];
  readonly noteheads: () => Ellipse[];
  readonly noteheadsWithStems: () => NoteheadWithStem[];
  readonly notes: () => Note[];
  /** other-clefs.json: the boxes predict_other_anchors_from_clefs builds, before it drops the ones overlapping a clef anchor's symbol. */
  readonly otherClefCandidates: () => RotatedBox[];
  readonly preprocessed: () => GrayImage;
  /** raw-staffs.json, resolved against staffAnchors().kept and boxes("staffFragmentsBroken"). */
  readonly rawStaffs: () => GoldenRawStaffs;
  /** resized.png: the autocropped page resized to width 1920, in BGR. */
  readonly resized: () => ColorImage;
  /** How many canvas-<n>.png files the page has: the staffs parsed, in parse order. */
  readonly staffAnchors: () => GoldenStaffAnchors;
  readonly staffCount: () => number;
  readonly staffPositions: () => StaffPosition[];
  /** staff-positions.txt as written, for a byte comparison; staffPositions() is the parsed form. */
  readonly staffPositionsText: () => string;
  readonly staffs: () => Staff[];
  /**
   * staffs.json with notes.json dealt out by braces.json's notesPerStaff:
   * Python's staffs as find_braces_brackets_and_grand_staff_lines received
   * them. A new list of new staffs on every call, since that stage tells
   * staffs apart by identity.
   */
  readonly staffsWithNotes: () => Staff[];
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
  /** Channel 0 is the pixel: an 8-bit grayscale PNG decodes to R = G = B. */
  const gray = (name: string): GrayImage => {
    const png = reader.png(name);
    const data = new Uint8Array(png.width * png.height);
    for (let i = 0; i < data.length; i += 1) {
      data[i] = png.rgba[i * 4] ?? 0;
    }
    return planeFromBytes("gray", png.width, png.height, data);
  };
  const color = (name: string): ColorImage => {
    const png = reader.png(name);
    return colorImageFromRgba(png.width, png.height, png.rgba);
  };
  /**
   * Keyed on the filename rather than on the class, because the raw and the
   * filtered mask of one class are two different files and a key that cannot
   * tell them apart is how `mask(name, true)` used to return the raw mask.
   */
  const maskOf = memoBy((file: string) => {
    const png = reader.png(file);
    const data = new Uint8Array(png.width * png.height);
    for (let i = 0; i < data.length; i += 1) {
      data[i] = (png.rgba[i * 4] ?? 0) > 0 ? 1 : 0;
    }
    return planeFromBytes("mask", png.width, png.height, data);
  });
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
  const boxes = memoBy((kind: GoldenBoxKind) =>
    decodeRotatedBoxes(json(GOLDEN_BOX_FILES[kind]), GOLDEN_BOX_FILES[kind])
  );
  const staffs = memo(() => decodeStaffs(json("staffs.json"), "staffs.json"));
  const notes = memo(() => decodeNotes(json("notes.json"), "notes.json"));
  const noteheadsWithStems = memo(() =>
    decodeNoteheadsWithStems(
      json("noteheads-with-stems.json"),
      "noteheads-with-stems.json"
    )
  );
  const staffAnchors = memo(() =>
    decodeStaffAnchors(
      json("staff-anchors.json"),
      boxes("staffFragmentsBroken"),
      "staff-anchors.json"
    )
  );
  const rawStaffs = memo(() => {
    const raw = decodeRawStaffs(
      json("raw-staffs.json"),
      boxes("staffFragmentsBroken"),
      staffAnchors().kept,
      "raw-staffs.json"
    );
    if (raw.resampledFrom.length !== staffs().length) {
      throw new GoldenError(
        "raw-staffs.json.resampledFrom",
        `${raw.resampledFrom.length} entries for the ${staffs().length} staffs of staffs.json`
      );
    }
    return raw;
  });
  const braces = memo(() => {
    const decoded = decodeBraces(json("braces.json"), "braces.json");
    if (decoded.notesPerStaff.length !== staffs().length) {
      throw new GoldenError(
        "braces.json.notesPerStaff",
        `${decoded.notesPerStaff.length} counts for the ${staffs().length} staffs of staffs.json`
      );
    }
    const dealt = decoded.notesPerStaff.reduce((n, count) => n + count, 0);
    if (dealt !== notes().length) {
      throw new GoldenError(
        "braces.json.notesPerStaff",
        `${dealt} notes dealt out, notes.json has ${notes().length}`
      );
    }
    const braceDotCount = boxes("braceDot").length;
    const beyond = decoded.tall.find((index) => index >= braceDotCount);
    if (beyond !== undefined) {
      throw new GoldenError(
        "braces.json.tall",
        `index ${beyond} is outside boxes-brace_dot.json (${braceDotCount} entries)`
      );
    }
    return decoded;
  });
  const noise = memo(() => {
    const decoded = decodeNoise(json("noise.json"), "noise.json");
    const staff = maskOf("mask-denoised-staff.png");
    const tile = {
      height: floorDiv(staff.height, NOISE_GRID_DIVISIONS),
      width: floorDiv(staff.width, NOISE_GRID_DIVISIONS),
    };
    const rows = Math.ceil(staff.height / tile.height);
    const columns = Math.ceil(staff.width / tile.width);
    if (
      decoded.tile.height !== tile.height ||
      decoded.tile.width !== tile.width ||
      decoded.grid.length !== rows ||
      decoded.grid[0]?.length !== columns
    ) {
      throw new GoldenError(
        "noise.json",
        `a ${staff.width}x${staff.height} mask has ${columns}x${rows} tiles of ${tile.width}x${tile.height}`
      );
    }
    return decoded;
  });
  return {
    autocropped: memo(() => color("autocropped.png")),
    barLines: memo(() =>
      decodeBarLines(json("barlines.json"), "barlines.json")
    ),
    boxes,
    braceDotMask: () => maskOf("mask-brace_dot.png"),
    braces,
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
    denoisedStaffMask: () => maskOf("mask-denoised-staff.png"),
    dewarp: memoBy((index: number) =>
      decodeDewarp(json(`dewarp-${index}.json`), `dewarp-${index}.json`)
    ),
    dewarpImage: (index: number, stage: GoldenDewarpStage) =>
      gray(`dewarp-${index}-${stage}.png`),
    mask: (name: MaskClass, filtered = false) =>
      maskOf(
        `mask-${filtered ? "filtered-" : ""}${MASK_CLASSES[name].golden}.png`
      ),
    meta,
    multiStaffs: memo(() =>
      decodeMultiStaffs(json("multistaffs.json"), "multistaffs.json")
    ),
    musicXml: memo(() => reader.text("page.musicxml")),
    noise,
    noiseMask: () =>
      noise().outcome === "masked" ? maskOf("mask-noise.png") : null,
    noteheadSplits: memo(() => {
      const splits = decodeNoteheadSplits(
        json("notehead-splits.json"),
        "notehead-splits.json"
      );
      for (const [i, split] of splits.entries()) {
        if (
          split.staff >= staffs().length ||
          split.notehead >= noteheadsWithStems().length
        ) {
          throw new GoldenError(
            `notehead-splits.json[${i}]`,
            `staff ${split.staff} of ${staffs().length}, notehead ${split.notehead} of ${noteheadsWithStems().length}`
          );
        }
      }
      return splits;
    }),
    noteheads: memo(() =>
      decodeEllipses(json("boxes-noteheads.json"), "boxes-noteheads.json")
    ),
    noteheadsWithStems,
    notes,
    otherClefCandidates: memo(() =>
      decodeRotatedBoxes(json("other-clefs.json"), "other-clefs.json")
    ),
    preprocessed: memo(() => gray("preprocessed.png")),
    rawStaffs,
    resized: memo(() => color("resized.png")),
    staffAnchors,
    staffCount,
    staffPositions: memo(() =>
      decodeStaffPositions(reader.text("staff-positions.txt"))
    ),
    staffPositionsText: memo(() => reader.text("staff-positions.txt")),
    staffs,
    staffsWithNotes: () => {
      let next = 0;
      return staffs().map((staff, i) => {
        const count = braces().notesPerStaff[i] ?? 0;
        const symbols = notes().slice(next, next + count);
        next += count;
        return createStaff(staff.grid, {
          isGrandstaff: staff.isGrandstaff,
          space: staff.space,
          symbols,
        });
      });
    },
    tokens: memoBy((index: number) =>
      decodeTokens(json(`tokens-${index}.json`), `tokens-${index}.json`)
    ),
    voices: memo(() => decodeVoices(json("voices.json"), "voices.json")),
  };
}

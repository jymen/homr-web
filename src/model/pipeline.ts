/**
 * The shapes that travel between phases: the segnet output, homr's
 * InputPredictions and main.py's PredictedSymbols, the detection result,
 * and the staff canvas the transformer reads. Each is a plain readonly
 * object; a stage that changes one member (filter_predictions,
 * make_lines_stronger, break_wide_fragments) builds a new object with the
 * rest shared.
 */

import type { Ellipse, RotatedBox } from "../geometry/boxes.js";
import type { ClassMap, ColorImage, GrayImage, Mask } from "../image/plane.js";
import { maskOfClass, PlaneError } from "../image/plane.js";
import type { MultiStaff, Staff } from "./staff.js";
import type { Note } from "./symbols.js";

/**
 * The five masks homr keeps, by the segnet output channel that produces
 * each (inference_segnet.inference: `merged == 1` is stems_rests, and so
 * on; channel 0 is background and discarded) and by the name the golden
 * dumper gives its PNG. One table, so the argmax split, the golden loader
 * and the tests read the same mapping.
 */
export const MASK_CLASSES = {
  clefsKeys: { channel: 3, golden: "clefs_keys" },
  notehead: { channel: 2, golden: "notehead" },
  staff: { channel: 4, golden: "staff" },
  stemsRest: { channel: 1, golden: "stems_rest" },
  symbols: { channel: 5, golden: "symbols" },
} as const;

export type MaskClass = keyof typeof MASK_CLASSES;

/** The mask names in homr's InputPredictions field order, which is also the dumper's. */
export const MASK_CLASS_NAMES = [
  "staff",
  "symbols",
  "stemsRest",
  "notehead",
  "clefsKeys",
] as const satisfies readonly MaskClass[];

export const SEGNET_INPUT = {
  /** The gray page is fed as three identical channels (cv2.COLOR_GRAY2BGR). */
  channels: 3,
  classes: 6,
  /**
   * main.py passes step_size = window, so interior tiles do not overlap;
   * but the last row and column are pulled back inside the page, and
   * merge_patches averages the overlap before the class comparison. Phase
   * 3 reproduces that averaging; it must not overwrite.
   */
  step: 320,
  /** Square tile side the model was trained on (win_size). */
  window: 320,
} as const;

/** Five masks of one page, all of the same width and height. */
export type SegmentationMasks = Readonly<Record<MaskClass, Mask>>;

/** Phase 3's output: the class map and the masks split from it. Both are kept because filtering later works on the masks, and the class map is what the bench page paints. */
export interface SegmentationResult {
  readonly classes: ClassMap;
  readonly height: number;
  readonly masks: SegmentationMasks;
  readonly width: number;
}

/** Split a class map into the five masks with maskOfClass. */
export function createSegmentationResult(
  classes: ClassMap
): SegmentationResult {
  const masks = {
    clefsKeys: maskOfClass(classes, MASK_CLASSES.clefsKeys.channel),
    notehead: maskOfClass(classes, MASK_CLASSES.notehead.channel),
    staff: maskOfClass(classes, MASK_CLASSES.staff.channel),
    stemsRest: maskOfClass(classes, MASK_CLASSES.stemsRest.channel),
    symbols: maskOfClass(classes, MASK_CLASSES.symbols.channel),
  };
  return { classes, height: classes.height, masks, width: classes.width };
}

function assertSameSize(
  name: string,
  a: { width: number; height: number },
  b: { width: number; height: number }
): void {
  if (a.width !== b.width || a.height !== b.height) {
    throw new PlaneError(
      `${name} is ${a.width}x${a.height}, expected ${b.width}x${b.height}`
    );
  }
}

/**
 * homr's InputPredictions: the page in colour and in gray, both resized to
 * the mask dimensions, and the masks. Every plane shares width and height;
 * the factory asserts it.
 */
export interface InputPredictions {
  readonly masks: SegmentationMasks;
  readonly original: ColorImage;
  readonly preprocessed: GrayImage;
}

export function createInputPredictions(
  original: ColorImage,
  preprocessed: GrayImage,
  masks: SegmentationMasks
): InputPredictions {
  assertSameSize("original", original, preprocessed);
  for (const name of MASK_CLASS_NAMES) {
    assertSameSize(`mask ${name}`, masks[name], preprocessed);
  }
  return { masks, original, preprocessed };
}

/** main.py's PredictedSymbols: phase 4's output, one list per golden boxes-<kind>.json. */
export interface PredictedSymbols {
  readonly barLines: readonly RotatedBox[];
  readonly clefsKeys: readonly RotatedBox[];
  readonly noteheads: readonly Ellipse[];
  readonly staffFragments: readonly RotatedBox[];
  readonly stemsRest: readonly RotatedBox[];
}

/** Phase 5's output: the detect_staffs_in_image return without the title future and the debug object. */
export interface PageDetection {
  readonly multiStaffs: readonly MultiStaff[];
  /** Every note found, also present in its staff's `symbols`; kept flat because the bench overlay and the xml writer read it that way. */
  readonly notes: readonly Note[];
  readonly preprocessed: GrayImage;
}

/** The transformer's fixed input size (Config.max_height, max_width): canvas-<n>.png is 1280 x 256. */
export const ENCODER_CANVAS = { height: 256, width: 1280 } as const;

/**
 * Phase 6's output per staff: the dewarped, centred gray canvas and the
 * same staff with its grid and symbol centres mapped into canvas
 * coordinates (canvas-<n>-staff.json, `space: "canvas"`). The image is
 * always ENCODER_CANVAS-sized; the factory asserts both.
 */
export interface StaffCanvas {
  readonly image: GrayImage;
  readonly staff: Staff;
}

export function createStaffCanvas(image: GrayImage, staff: Staff): StaffCanvas {
  assertSameSize("staff canvas", image, ENCODER_CANVAS);
  if (staff.space !== "canvas") {
    throw new PlaneError(
      `a staff canvas needs a staff in canvas space, got ${staff.space}`
    );
  }
  return { image, staff };
}

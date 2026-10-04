/**
 * main.py's detect_staffs_in_image from filter_predictions on: what segnet
 * saw, in; the staffs with their notes, grouped, out. Synchronous, no model.
 *
 * Not ported: the title future (OCR), the Debug object and every eprint.
 */

import { withBoxOps } from "../cv/box-ops.js";
import { createRotatedBoundingBoxes } from "../cv/create-boxes.js";
import {
  makeLinesStronger,
  prepareBraceDotImage,
} from "../cv/mask-morphology.js";
import type { OpenCv } from "../cv/opencv.js";
import { barLineCandidates, detectBarLines } from "../geometry/barlines.js";
import type { RotatedBox } from "../geometry/boxes.js";
import { findBracesBracketsAndGrandStaffLines } from "../geometry/braces.js";
import { breakWideFragments } from "../geometry/break-fragments.js";
import { filterPredictions } from "../geometry/noise-filter.js";
import {
  addNotesToStaffs,
  averageNoteheadHeight,
  combineNoteheadsWithStems,
} from "../geometry/notes.js";
import {
  DetectionError,
  type InputPredictions,
  type PageDetection,
  type SegmentationMasks,
} from "../model/pipeline.js";
import { detectStaff } from "./detect-staff.js";
import { predictSymbols } from "./predict-symbols.js";

/**
 * main.py:278-280, the sixth call shape of the box factories: no merge, at
 * most 100 wide, any height. `masks.staff` is the mask after
 * makeLinesStronger.
 */
export function predictBraceDot(
  cv: OpenCv,
  masks: Pick<SegmentationMasks, "staff" | "symbols">
): RotatedBox[] {
  return createRotatedBoundingBoxes(
    cv,
    prepareBraceDotImage(cv, masks.symbols, masks.staff),
    { maxSize: { h: -1, w: 100 }, skipMerging: true }
  );
}

/**
 * `predictions` is segnet's output as segmentation hands it over: the five
 * raw masks, the resized colour page and the CLAHE gray, unfiltered. Nothing
 * in it is modified.
 *
 * Throws DetectionError for a page with no noteheads or no staffs, which is
 * what a page that is not sheet music produces, and at the two places homr
 * itself crashes. Any other error is a defect.
 *
 * Everything from the fragments on shares one Mat scope, because the overlap
 * memo is keyed on box identity and the same boxes are asked about from the
 * noteheads to the braces.
 */
export function detectStaffsInImage(
  cv: OpenCv,
  predictions: InputPredictions
): PageDetection {
  const filtered = filterPredictions(predictions);
  const masks = {
    ...filtered.predictions.masks,
    staff: makeLinesStronger(cv, filtered.predictions.masks.staff),
  };
  const symbols = predictSymbols(cv, masks);

  return withBoxOps(cv, (ops) => {
    const fragments = breakWideFragments(ops, symbols.staffFragments);
    const noteheads = combineNoteheadsWithStems(
      ops,
      symbols.noteheads,
      symbols.stemsRest
    );
    if (noteheads.length === 0) {
      throw new DetectionError("no-noteheads");
    }
    const barLines = detectBarLines(
      barLineCandidates(ops, symbols.barLines, noteheads),
      averageNoteheadHeight(noteheads)
    );
    const found = detectStaff(
      ops,
      masks.staff,
      fragments,
      symbols.clefsKeys,
      barLines
    );
    const noted = addNotesToStaffs(ops, found, noteheads, masks.notehead);
    const [top, ...below] = noted.staffs;
    if (top === undefined) {
      throw new DetectionError("no-staffs");
    }
    const braceDot = predictBraceDot(cv, masks);
    return {
      multiStaffs: findBracesBracketsAndGrandStaffLines(
        ops,
        [top, ...below],
        braceDot
      ),
      noise: filtered.outcome,
      notes: noted.notes,
      preprocessed: filtered.predictions.preprocessed,
      topStaff: top,
    };
  });
}

/**
 * main.py's predict_symbols. The six call shapes in the whole pipeline are the
 * five here and brace_dot's, which is phase 5's because its mask comes from
 * prepare_brace_dot_image after detect_staff; there are no others, so these
 * parameters are the entire configuration surface of the box factories.
 *
 * The masks are the *filtered* five, after filter_predictions and
 * make_lines_stronger(staff, (1, 2)). Both of those are already baked into the
 * golden mask-filtered-*.png files, so this phase ports neither.
 */

import {
  createBoundingEllipses,
  createRotatedBoundingBoxes,
} from "../cv/create-boxes.js";
import type { OpenCv } from "../cv/opencv.js";
import { prepareBarLineImage } from "../geometry/barlines.js";
import type { PredictedSymbols, SegmentationMasks } from "../model/pipeline.js";

export function predictSymbols(
  cv: OpenCv,
  masks: SegmentationMasks
): PredictedSymbols {
  const noteheads = createBoundingEllipses(cv, masks.notehead, {
    minSize: { h: 4, w: 4 },
  });
  const staffFragments = createRotatedBoundingBoxes(cv, masks.staff, {
    maxSize: { h: 100, w: 10_000 },
    minSize: { h: 1, w: 5 },
    skipMerging: true,
  });
  const clefsKeys = createRotatedBoundingBoxes(cv, masks.clefsKeys, {
    maxSize: { h: 1000, w: 1000 },
    minSize: { h: 40, w: 20 },
  });
  const stemsRest = createRotatedBoundingBoxes(cv, masks.stemsRest);
  const barLines = createRotatedBoundingBoxes(
    cv,
    prepareBarLineImage(cv, masks.stemsRest),
    { minSize: { h: 5, w: 1 }, skipMerging: true }
  );
  return { barLines, clefsKeys, noteheads, staffFragments, stemsRest };
}

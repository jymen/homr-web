/**
 * main.py's predict_symbols. The six call shapes in the whole pipeline are the
 * five here and brace_dot's, whose mask is prepareBraceDotImage's and which
 * detection fits itself; there are no others, so these parameters are the
 * entire configuration surface of the box factories.
 *
 * The masks are the *filtered* five, after filter_predictions and
 * make_lines_stronger(staff, (1, 2)).
 */

import {
  createBoundingEllipses,
  createRotatedBoundingBoxes,
} from "../cv/create-boxes.js";
import { prepareBarLineImage } from "../cv/mask-morphology.js";
import type { OpenCv } from "../cv/opencv.js";
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

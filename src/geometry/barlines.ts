/**
 * bar_line_detection.py's contribution to predict_symbols. The rest of that
 * module is phase 5's; this one call has to come along because it sits inside
 * predict_symbols and produces the mask the bar-line boxes are fitted to.
 */

import { maskFromMat, planeToMat } from "../cv/mat-plane.js";
import { type OpenCv, withMatScope } from "../cv/opencv.js";
import type { Mask } from "../image/plane.js";

/** `np.ones((5, 3), np.uint8)`: five rows, three columns, all ones. */
const BAR_LINE_KERNEL = { cols: 3, rows: 5 } as const;

/**
 * prepare_bar_line_image: one dilate, one iteration. `Mat.ones` and not
 * `getStructuringElement`, which is a full rectangle only for MORPH_RECT and
 * something else for MORPH_ELLIPSE -- reaching for the helper would be right by
 * accident at best.
 */
export function prepareBarLineImage(cv: OpenCv, stemsRest: Mask): Mask {
  return withMatScope((scope) => {
    const source = planeToMat(cv, scope, stemsRest);
    const dilated = scope.keep(new cv.Mat());
    const kernel = scope.keep(
      cv.Mat.ones(BAR_LINE_KERNEL.rows, BAR_LINE_KERNEL.cols, cv.CV_8U)
    );
    cv.dilate(source, dilated, kernel);
    return maskFromMat(dilated);
  });
}

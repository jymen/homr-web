/**
 * The places homr reshapes a mask with a structuring element before it fits
 * boxes to it.
 */

import type { Mask } from "../image/plane.js";
import { maskFromMat, planeToMat } from "./mat-plane.js";
import { type MatScope, type OpenCv, withMatScope } from "./opencv.js";

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

/**
 * `cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height))`. Narrow
 * ellipses are not what the name suggests: (1, 2) and (1, 5) are full columns,
 * and (5, 35) is a column five wide that tapers to one pixel over its first
 * and last six rows.
 */
function ellipseKernel(
  cv: OpenCv,
  scope: MatScope,
  width: number,
  height: number
) {
  return scope.keep(
    cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(width, height))
  );
}

/**
 * make_lines_stronger(staff, (1, 2)): the only kernel homr ever passes, so it
 * is not a parameter. The kernel's anchor is its lower row, so each set pixel
 * also sets the pixel below it and a line grows downward only.
 */
export function makeLinesStronger(cv: OpenCv, staff: Mask): Mask {
  return withMatScope((scope) => {
    const source = planeToMat(cv, scope, staff);
    const dilated = scope.keep(new cv.Mat());
    cv.dilate(source, dilated, ellipseKernel(cv, scope, 1, 2));
    return maskFromMat(dilated);
  });
}

/**
 * prepare_brace_dot_image: `symbols - staff` saturating at zero, eroded by the
 * (1, 5) column to drop anything under five pixels tall, then dilated by the
 * (5, 35) kernel.
 *
 * `strongStaff` is the mask after makeLinesStronger; the raw staff mask leaves
 * a row of every line inside the brace image.
 */
export function prepareBraceDotImage(
  cv: OpenCv,
  symbols: Mask,
  strongStaff: Mask
): Mask {
  return withMatScope((scope) => {
    const braceDot = scope.keep(new cv.Mat());
    cv.subtract(
      planeToMat(cv, scope, symbols),
      planeToMat(cv, scope, strongStaff),
      braceDot
    );
    const eroded = scope.keep(new cv.Mat());
    cv.erode(braceDot, eroded, ellipseKernel(cv, scope, 1, 5));
    const dilated = scope.keep(new cv.Mat());
    cv.dilate(eroded, dilated, ellipseKernel(cv, scope, 5, 35));
    return maskFromMat(dilated);
  });
}

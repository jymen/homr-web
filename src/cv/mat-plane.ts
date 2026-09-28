/**
 * Where a Plane and a Mat meet. Two things bite here once each: `mat.data` is a
 * fresh view on the wasm heap for every read, so it is taken immediately after
 * the allocation that could have grown the heap and moved it; and a Mat that
 * comes back out carries 0 and 255 where a Mask carries 0 and 1, so the
 * threshold is part of the crossing rather than the caller's problem.
 */

import type { Mat } from "@techstark/opencv-js";
import { type Mask, type Plane, planeFromBytes } from "../image/plane.js";
import type { MatScope, OpenCv } from "./opencv.js";

/**
 * A plane's bytes in the wasm heap, at cv2's own layout: row-major, one byte
 * per sample, `channels` samples per pixel. `np.ones`-style kernels and the
 * segnet masks are single-channel; a page is BGR.
 */
export function planeToMat(cv: OpenCv, scope: MatScope, plane: Plane): Mat {
  const mat = scope.keep(
    new cv.Mat(
      plane.height,
      plane.width,
      plane.channels === 3 ? cv.CV_8UC3 : cv.CV_8UC1
    )
  );
  mat.data.set(plane.data);
  return mat;
}

/**
 * A single-channel Mat back out as a 0/1 Mask. Every non-zero byte becomes 1,
 * because cv2 writes 255 where it means "set" and homr's masks are 0/1
 * throughout; `planeFromBytes` refuses anything above 1, so an unthresholded
 * copy would fail loudly rather than silently feed 255s to findContours.
 */
export function maskFromMat(mat: Mat): Mask {
  const bytes = mat.data;
  const data = new Uint8Array(mat.rows * mat.cols);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = (bytes[i] ?? 0) > 0 ? 1 : 0;
  }
  return planeFromBytes("mask", mat.cols, mat.rows, data);
}

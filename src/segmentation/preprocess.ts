/**
 * autocrop.py and color_adjust.py, and the three lines of main.py's
 * `load_and_preprocess_predictions` that run them either side of resize.py:
 * `autocrop` -> `resize_image` -> `apply_clahe`. The page is BGR throughout,
 * as `cv2.imread` hands it over, and the gray the segnet is fed is the CLAHE
 * output.
 *
 * Both stages are cv2 calls, so both go through opencv.js, and the two
 * consequences of that are the shape of this module. Every Mat is owned by a
 * `MatScope`, because a leaked one is wasm heap that never comes back. And
 * `withMatScope` is synchronous, so the opencv handle is a parameter of the two
 * stages and only `preprocessPage` is async: an `await` inside a scope would
 * free its Mats at the first suspension.
 *
 * Three places where the JS build is not the Python and the difference is
 * silent rather than loud:
 *
 * `cv2.createCLAHE` does not exist here, only the `CLAHE` constructor, which is
 * an embind instance and therefore needs freeing like a Mat.
 *
 * `np.ones((7, 7), np.uint8)` is `cv.Mat.ones(7, 7, cv.CV_8U)`, a full
 * rectangle of ones. `getStructuringElement(MORPH_RECT, ...)` produces the same
 * rectangle and `MORPH_ELLIPSE` does not, so reaching for the structuring
 * element helper is right by accident at best.
 *
 * A Mat is filled by writing `mat.data`, a live view of the wasm heap, rather
 * than through `matFromArray`, which takes a JS array: the page is 26 MB and
 * `Array.from` of it is 8.7 million boxed numbers for no gain.
 */

import type { Mat } from "@techstark/opencv-js";
import {
  loadOpenCv,
  type MatScope,
  type OpenCv,
  withMatScope,
} from "../cv/opencv.js";
import {
  type ColorImage,
  type GrayImage,
  planeFromBytes,
} from "../image/plane.js";
import { resizeToTargetWidth } from "./resize.js";

/** `cv2.calcHist`'s `[256], [0, 256]`: one bin per 8-bit value. */
const HISTOGRAM_BINS = 256;
/** How far below the dominant value autocrop's threshold sits. */
const PAPER_THRESHOLD_OFFSET = 30;
const CLOSE_KERNEL_SIZE = 7;
const ERODE_KERNEL_SIZE = 9;
/**
 * `is_full_page_view`: a bounding rect starting within a quarter of the page of
 * either edge is read as the page itself rather than as paper inside a photo.
 */
const FULL_PAGE_VIEW_FRACTION = 0.25;
const CLAHE_CLIP_LIMIT = 1;
/** `tileGridSize=(8, 8)`: tiles per row and column, not tile pixels. */
const CLAHE_TILES = 8;

/**
 * What autocrop decided, separately from acting on it: the bounding rect of the
 * largest contour and whether the guard read it as a full-page view.
 *
 * Exported because on a page with no margin this is the only observable part of
 * autocrop. Such a page comes back unchanged, so a byte-equality check against
 * the Python's `autocropped.png` passes for an implementation that did nothing
 * at all; the rect and the flag are what pin the decision.
 */
export interface PaperRect {
  readonly height: number;
  readonly isFullPageView: boolean;
  readonly width: number;
  readonly x: number;
  readonly y: number;
}

/**
 * A BGR page's bytes in the wasm heap. `mat.data` is a fresh view on each read,
 * so it is taken immediately after the allocation that could have grown the heap
 * and moved it.
 */
function matOfPlane(cv: OpenCv, scope: MatScope, image: ColorImage): Mat {
  const mat = scope.keep(new cv.Mat(image.height, image.width, cv.CV_8UC3));
  mat.data.set(image.data);
  return mat;
}

/**
 * `cv2.calcHist([img], [0], None, [256], [0, 256])` then
 * `int(np.argmax(hist.flatten()))`: the most frequent value of **channel 0**,
 * which in a BGR page is blue and not the grayscale the threshold is then
 * applied to. Strictly-greater, so a tie goes to the lowest value as numpy's
 * argmax does.
 *
 * cv2's own histogram rather than a counting loop, because cv2 accumulates into
 * float32: past 2 ** 24 pixels of one value the counter stops advancing, and an
 * exact integer count would pick a different winner on a page that large.
 */
function dominantBlueValue(cv: OpenCv, scope: MatScope, src: Mat): number {
  const images = scope.keep(new cv.MatVector());
  images.push_back(src);
  const noMask = scope.keep(new cv.Mat());
  const histogram = scope.keep(new cv.Mat());
  cv.calcHist(
    images,
    [0],
    noMask,
    histogram,
    [HISTOGRAM_BINS],
    [0, HISTOGRAM_BINS]
  );
  const counts = histogram.data32F;
  let dominant = 0;
  let best = counts[0] ?? 0;
  for (let value = 1; value < HISTOGRAM_BINS; value += 1) {
    const count = counts[value] ?? 0;
    if (count > best) {
      best = count;
      dominant = value;
    }
  }
  return dominant;
}

/**
 * autocrop.py up to its `boundingRect`: threshold the gray at the dominant
 * value less 30, close then erode, and take the largest external contour.
 * `null` when there is no contour at all, which homr reads as a full page view
 * too — reachable on a chromatic page, where a blue-dominated histogram can put
 * the threshold above every gray value and leave the mask empty.
 *
 * Erode's default border value is +inf, so it never eats the image edge: paper
 * that runs off the photo keeps a bounding rect that reaches it, which is why
 * the crop below cannot clamp its bounds.
 */
export function findPaperRect(image: ColorImage, cv: OpenCv): PaperRect | null {
  return withMatScope((scope) => {
    const src = matOfPlane(cv, scope, image);
    const dominant = dominantBlueValue(cv, scope, src);
    const gray = scope.keep(new cv.Mat());
    cv.cvtColor(src, gray, cv.COLOR_BGR2GRAY);
    const mask = scope.keep(new cv.Mat());
    cv.threshold(
      gray,
      mask,
      dominant - PAPER_THRESHOLD_OFFSET,
      255,
      cv.THRESH_BINARY
    );
    const closed = scope.keep(new cv.Mat());
    cv.morphologyEx(
      mask,
      closed,
      cv.MORPH_CLOSE,
      scope.keep(cv.Mat.ones(CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE, cv.CV_8U))
    );
    const eroded = scope.keep(new cv.Mat());
    cv.morphologyEx(
      closed,
      eroded,
      cv.MORPH_ERODE,
      scope.keep(cv.Mat.ones(ERODE_KERNEL_SIZE, ERODE_KERNEL_SIZE, cv.CV_8U))
    );
    const contours = scope.keep(new cv.MatVector());
    cv.findContours(
      eroded,
      contours,
      scope.keep(new cv.Mat()),
      cv.RETR_EXTERNAL,
      cv.CHAIN_APPROX_SIMPLE
    );
    // `area > area_thresh` from a start of 0.0, so the first of equal areas
    // wins and a contour of zero area never does.
    let largestArea = 0;
    let largest: Mat | undefined;
    for (let index = 0; index < contours.size(); index += 1) {
      const contour = scope.keep(contours.get(index));
      const area = cv.contourArea(contour);
      if (area > largestArea) {
        largestArea = area;
        largest = contour;
      }
    }
    if (largest === undefined) {
      return null;
    }
    const rect = cv.boundingRect(largest);
    return {
      height: rect.height,
      isFullPageView:
        rect.x < image.width * FULL_PAGE_VIEW_FRACTION ||
        rect.y < image.height * FULL_PAGE_VIEW_FRACTION,
      width: rect.width,
      x: rect.x,
      y: rect.y,
    };
  });
}

/**
 * `img[y : y + h, x : x + w]`.
 *
 * Not `cropPlane`, which is the port of homr's own `crop_image`: its `_limit_x`
 * clamps each bound to size - 1, so it can never return a slice that includes
 * the last column or row. autocrop's crop is a plain numpy slice, and the rect
 * it is given does reach the edge whenever the paper does.
 */
function sliceRect(image: ColorImage, rect: PaperRect): ColorImage {
  const { channels } = image;
  const rowBytes = rect.width * channels;
  const data = new Uint8Array(rowBytes * rect.height);
  for (let row = 0; row < rect.height; row += 1) {
    const from = ((rect.y + row) * image.width + rect.x) * channels;
    data.set(image.data.subarray(from, from + rowBytes), row * rowBytes);
  }
  return planeFromBytes("bgr", rect.width, rect.height, data);
}

/**
 * `autocrop`: the sheet of paper cut out of a photograph of it, or the page
 * itself when the largest contour starts too near an edge to be paper inside a
 * larger image.
 *
 * Returns the plane it was given, not a copy, when it does not crop: a caller
 * can compare identity to tell, and a 26 MB page is not copied for nothing.
 */
export function autocrop(image: ColorImage, cv: OpenCv): ColorImage {
  const rect = findPaperRect(image, cv);
  if (rect === null || rect.isFullPageView) {
    return image;
  }
  return sliceRect(image, rect);
}

/**
 * `color_adjust.apply_clahe`: the BGR page as gray, contrast-equalized per
 * 8-by-8 tile grid at a clip limit of 1.0. This is segnet's input.
 */
export function applyClahe(image: ColorImage, cv: OpenCv): GrayImage {
  return withMatScope((scope) => {
    const src = matOfPlane(cv, scope, image);
    const gray = scope.keep(new cv.Mat());
    cv.cvtColor(src, gray, cv.COLOR_BGR2GRAY);
    // cv.Size is a plain JS object here, not an embind instance, so it is the
    // one thing in this module with no owner.
    const clahe = scope.keep(
      new cv.CLAHE(CLAHE_CLIP_LIMIT, new cv.Size(CLAHE_TILES, CLAHE_TILES))
    );
    const equalized = scope.keep(new cv.Mat());
    clahe.apply(gray, equalized);
    // Copied out of the heap before the scope frees it; `mat.data` is a view.
    return planeFromBytes(
      "gray",
      equalized.cols,
      equalized.rows,
      equalized.data.slice()
    );
  });
}

export interface PreprocessedPage {
  /** The gray page segnet is fed: `preprocessed.png`. */
  readonly preprocessed: GrayImage;
  /**
   * The page after autocrop and resize, in BGR: `resized.png`. homr keeps it as
   * `InputPredictions.original` for its debug overlays.
   */
  readonly resized: ColorImage;
}

/**
 * `load_and_preprocess_predictions`'s first three lines: autocrop, resize to
 * 1920 wide, CLAHE.
 *
 * Measured on the Kesh page (2481x3509), Node with one wasm thread: autocrop
 * 245 ms, resize 400 ms, CLAHE 95 ms in a bare script. test/preprocess.test.ts
 * and test/resize.test.ts print two to four times those numbers, which says more
 * about vitest's parallel workers than about the code. Either way the three
 * stages are about a second against the 28 s the whole page takes including
 * segnet, so preprocess is not on the critical path.
 *
 * The colour page it returns has no reader in this port yet. homr uses
 * `InputPredictions.original` only for debug overlays and feeds segnet the CLAHE
 * gray, so the resize's three bands are currently paid for the golden test
 * alone. The shortcut, if a profile ever asks for it: resize one band and skip
 * the colour page, which is correct exactly while the page is achromatic, because `cvtColor(BGR2GRAY)` of a pixel with B = G = R returns
 * that value unchanged (cv2's fixed-point weights sum to 1 << 14, so
 * `(v * 16384 + 8192) >> 14` is `v`; test/preprocess.test.ts checks all 256
 * values against opencv.js rather than trusting the arithmetic). A scan for
 * achromacy plus a branch is not worth 260 ms of 28,000, which is why it is
 * written down here instead of implemented.
 */
export async function preprocessPage(
  page: ColorImage,
  cv?: OpenCv
): Promise<PreprocessedPage> {
  const handle = cv ?? (await loadOpenCv());
  const resized = resizeToTargetWidth(autocrop(page, handle));
  return { preprocessed: applyClahe(resized, handle), resized };
}

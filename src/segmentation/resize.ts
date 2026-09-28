/**
 * resize.py: the autocropped page is taken to a fixed width of 1920 before
 * segmentation, because the segnet was trained on pages of that width.
 *
 * homr resizes with `PIL.Image.resize` and no `resample` argument, which since
 * Pillow 10 means bicubic. Reproducing the segnet's input byte for byte
 * therefore means reproducing PIL's resample and not merely calling some other
 * bicubic: PIL builds its kernel in 22-bit fixed point and accumulates it in a
 * C `int`, so a float kernel with a rounded result is a different function and
 * disagrees on a page's worth of pixels. What follows is Pillow's Resample.c
 * (`precompute_coeffs`, `normalize_coeffs_8bpc`,
 * `ImagingResampleHorizontal_8bpc`, `ImagingResampleVertical_8bpc`)
 * transcribed; every step that looks replaceable by cleaner floating point is
 * load-bearing and says so.
 *
 * This is the second module in src/ that is bit-level arithmetic (dtype.ts is
 * the other), and biome.jsonc names it for the same reason.
 */

import { roundHalfEven } from "../image/numeric.js";
import {
  type ColorImage,
  type GrayImage,
  planeFromBytes,
} from "../image/plane.js";

/** resize.py's `target_width`. */
export const RESIZE_TARGET_WIDTH = 1920;

/**
 * `calc_target_image_size`: the target width, and the height that preserves
 * the aspect ratio; the page unchanged when it is already that wide.
 *
 * The ratio is formed first, as the Python forms it, because
 * `height * (1920 / width)` and `(height * 1920) / width` are not the same
 * float64 and this rounding is one of the sites the golden pins. Python's
 * round() is half to even, so Math.round would take a page whose ratio lands
 * on .5 one row past the height the segnet was fed.
 */
export function targetImageSize(
  width: number,
  height: number
): { readonly height: number; readonly width: number } {
  if (width === RESIZE_TARGET_WIDTH) {
    return { height, width };
  }
  const ratio = RESIZE_TARGET_WIDTH / width;
  return { height: roundHalfEven(height * ratio), width: RESIZE_TARGET_WIDTH };
}

/** Pillow's `PRECISION_BITS` for 8-bit images: 32 - 8 - 2. */
const PRECISION_BITS = 22;
/** The half of a fixed-point unit each accumulator starts at, so the closing shift rounds to nearest instead of truncating. */
const ACCUMULATOR_BIAS = 1 << (PRECISION_BITS - 1);
const FIXED_POINT_ONE = 1 << PRECISION_BITS;
/** `BICUBIC`'s filter: support 2.0, and the `a` of the Keys family Pillow hardcodes. */
const BICUBIC_A = -0.5;
const BICUBIC_SUPPORT = 2;

/** Pillow's `bicubic_filter`, with its own parenthesisation: rearranging it would change the last bits of a weight. */
function bicubic(value: number): number {
  const x = Math.abs(value);
  if (x < 1) {
    return ((BICUBIC_A + 2) * x - (BICUBIC_A + 3)) * x * x + 1;
  }
  if (x < 2) {
    return (((x - 5) * x + 8) * x - 4) * BICUBIC_A;
  }
  return 0;
}

/**
 * `normalize_coeffs_8bpc`: the 0.5 is the round-to-nearest that C's cast to
 * int does not do on its own, and it has to be applied away from zero on both
 * signs because the bicubic kernel's outer lobes are negative.
 */
function fixedPoint(weight: number): number {
  const scaled = weight * FIXED_POINT_ONE;
  return Math.trunc(weight < 0 ? -0.5 + scaled : 0.5 + scaled);
}

/**
 * One axis' kernel. `bounds` holds (first input index, tap count) per output
 * pixel — PIL's `xmin` and its `xmax` after the subtraction that turns it into
 * a count — and `coefficients` holds `ksize` fixed-point taps per output
 * pixel, the slots past the count left at zero and never read.
 */
interface AxisTaps {
  readonly bounds: Int32Array;
  readonly coefficients: Int32Array;
  readonly ksize: number;
}

/** `precompute_coeffs` followed by `normalize_coeffs_8bpc`, which always run as a pair. */
function precomputeTaps(inSize: number, outSize: number): AxisTaps {
  // Two ratios, not one. `scale` is unclamped and places an output pixel's
  // centre in input space; `filterscale` is clamped up to 1 and only ever
  // widens the kernel, so an upscale keeps the filter's natural support.
  // Collapsing them into one variable moves every centre on an upscale.
  const scale = inSize / outSize;
  const filterscale = scale < 1 ? 1 : scale;
  const support = BICUBIC_SUPPORT * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const inverseFilterscale = 1 / filterscale;
  const bounds = new Int32Array(outSize * 2);
  const coefficients = new Int32Array(outSize * ksize);
  const weights = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx += 1) {
    const center = (xx + 0.5) * scale;
    // The +0.5 inside the truncation is PIL's, and the clamps are what let an
    // edge pixel keep a short kernel rather than reading outside the image.
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) {
      xmin = 0;
    }
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) {
      xmax = inSize;
    }
    xmax -= xmin;
    let total = 0;
    for (let x = 0; x < xmax; x += 1) {
      const weight = bicubic((x + xmin - center + 0.5) * inverseFilterscale);
      weights[x] = weight;
      total += weight;
    }
    const base = xx * ksize;
    for (let x = 0; x < xmax; x += 1) {
      const weight = weights[x] ?? 0;
      // A zero sum is left undivided rather than guarded against: PIL does the
      // same, and the case only arises for a kernel whose taps all fell outside
      // the image, where every weight is already zero.
      coefficients[base + x] = fixedPoint(
        total === 0 ? weight : weight / total
      );
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, coefficients, ksize };
}

/**
 * The close of one output sample: `_clip8(ss >> PRECISION_BITS)`. The shift is
 * PIL's arithmetic on a C `int` and a division by 2 ** 22 is a different
 * function — it rounds where this truncates, which is exactly the byte the
 * accumulator's bias was added to settle.
 */
function clipFixedPoint(accumulator: number): number {
  const value = accumulator >> PRECISION_BITS;
  if (value < 0) {
    return 0;
  }
  return value > 255 ? 255 : value;
}

/**
 * `ImagingResampleHorizontal_8bpc` on the plane layout: row-major, `bands`
 * samples per pixel, no padding. It reads `rows` input rows starting at
 * `firstRow` and writes them at the output width.
 */
function resampleHorizontal(
  src: Uint8Array,
  inWidth: number,
  bands: number,
  outWidth: number,
  firstRow: number,
  rows: number,
  taps: AxisTaps
): Uint8Array {
  const out = new Uint8Array(outWidth * rows * bands);
  for (let yy = 0; yy < rows; yy += 1) {
    const rowIn = (yy + firstRow) * inWidth * bands;
    const rowOut = yy * outWidth * bands;
    for (let xx = 0; xx < outWidth; xx += 1) {
      const xmin = taps.bounds[xx * 2] ?? 0;
      const count = taps.bounds[xx * 2 + 1] ?? 0;
      const base = xx * taps.ksize;
      for (let b = 0; b < bands; b += 1) {
        let acc = ACCUMULATOR_BIAS;
        const from = rowIn + xmin * bands + b;
        for (let x = 0; x < count; x += 1) {
          acc +=
            (src[from + x * bands] ?? 0) * (taps.coefficients[base + x] ?? 0);
        }
        out[rowOut + xx * bands + b] = clipFixedPoint(acc);
      }
    }
  }
  return out;
}

/**
 * `ImagingResampleVertical_8bpc`, reading the temp the horizontal pass wrote.
 * `firstRow` is the input row that temp row 0 holds, and every `ymin` is
 * rebased onto it.
 */
function resampleVertical(
  temp: Uint8Array,
  outWidth: number,
  outHeight: number,
  bands: number,
  firstRow: number,
  taps: AxisTaps
): Uint8Array {
  const out = new Uint8Array(outWidth * outHeight * bands);
  const rowBytes = outWidth * bands;
  for (let yy = 0; yy < outHeight; yy += 1) {
    const ymin = (taps.bounds[yy * 2] ?? 0) - firstRow;
    const count = taps.bounds[yy * 2 + 1] ?? 0;
    const base = yy * taps.ksize;
    const rowOut = yy * rowBytes;
    for (let xx = 0; xx < outWidth; xx += 1) {
      for (let b = 0; b < bands; b += 1) {
        let acc = ACCUMULATOR_BIAS;
        const from = ymin * rowBytes + xx * bands + b;
        for (let y = 0; y < count; y += 1) {
          acc +=
            (temp[from + y * rowBytes] ?? 0) *
            (taps.coefficients[base + y] ?? 0);
        }
        out[rowOut + xx * bands + b] = clipFixedPoint(acc);
      }
    }
  }
  return out;
}

/** `ImagingResample`: the two passes, and the window between them. */
function resampleBytes(
  src: Uint8Array,
  inWidth: number,
  inHeight: number,
  bands: number,
  outWidth: number,
  outHeight: number
): Uint8Array {
  const vertical = precomputeTaps(inHeight, outHeight);
  // The temp the horizontal pass writes holds only the input rows the vertical
  // pass will read: from the first output row's ymin to the last output row's
  // ymin plus its tap count. This is not an optimisation — a temp of the full
  // input height would put the vertical taps on different rows and quietly
  // change the result, since the vertical ymins are rebased onto the window.
  const firstRow = vertical.bounds[0] ?? 0;
  const lastRow =
    (vertical.bounds[outHeight * 2 - 2] ?? 0) +
    (vertical.bounds[outHeight * 2 - 1] ?? 0);
  const horizontal = precomputeTaps(inWidth, outWidth);
  const temp = resampleHorizontal(
    src,
    inWidth,
    bands,
    outWidth,
    firstRow,
    lastRow - firstRow,
    horizontal
  );
  return resampleVertical(temp, outWidth, outHeight, bands, firstRow, vertical);
}

/**
 * `resize_image`: the page at `RESIZE_TARGET_WIDTH`, or the plane it was given
 * when it is already that size. Generic over the two plane kinds a page can be
 * (the autocropped page is BGR, later stages are gray) so the band count comes
 * from `channels` and the fixed-point loop exists once.
 */
export function resizeToTargetWidth<P extends GrayImage | ColorImage>(
  image: P
): P {
  const target = targetImageSize(image.width, image.height);
  if (target.width === image.width && target.height === image.height) {
    return image;
  }
  const data = resampleBytes(
    image.data,
    image.width,
    image.height,
    image.channels,
    target.width,
    target.height
  );
  // planeFromBytes is generic in the kind, so for the union P is constrained to
  // it answers that union rather than P. The assertion re-ties them, as
  // cropPlane's does; it cannot lie, because the kind and the byte count both
  // come from `image`.
  return planeFromBytes(image.kind, target.width, target.height, data) as P;
}

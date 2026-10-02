/**
 * Planes: the array model of the port.
 *
 * homr hands numpy 2-D uint8 arrays (H, W) and BGR (H, W, 3) arrays to cv2.
 * The port keeps exactly that memory layout: row-major, one byte per
 * sample, `channels` samples per pixel, no stride, no padding. It is what
 * `cv.matFromArray(rows, cols, CV_8UC1 | CV_8UC3, data)` reads and what
 * `cv.Mat.data` exposes, so the only copies are the ones opencv.js forces
 * (into and out of the WASM heap), and they happen at that boundary (phase
 * 4), nowhere else.
 *
 * `kind` is a runtime discriminant, not a phantom brand. A Mask (0/1) and a
 * GrayImage (0..255) share one layout and would otherwise be interchangeable
 * at the type level; passing a gray image where a mask is expected is a
 * silent geometry bug (findContours on 0..255 data "works" and yields
 * nonsense), and a runtime tag lets the boundary decoders assert what they
 * produced and a debug page say what it is looking at.
 *
 * Every value here is a plain object over a typed array: it survives
 * `postMessage` (structured clone) and can be transferred by its buffer,
 * which is how phase 9 moves results out of the Worker.
 */

export const PLANE_KINDS = {
  /** B, G, R interleaved, cv2's channel order. */
  bgr: "bgr",
  /** segnet class index 0..5 per pixel (see SEGNET_CLASSES in model/pipeline.ts). */
  classes: "classes",
  /** 0..255 per pixel: preprocessed page, staff canvas. */
  gray: "gray",
  /** 0 or 1 per pixel: a segnet class mask or a morphology result. */
  mask: "mask",
} as const;
export type PlaneKind = (typeof PLANE_KINDS)[keyof typeof PLANE_KINDS];

interface PlaneBase<K extends PlaneKind, C extends 1 | 3> {
  readonly channels: C;
  /**
   * Invariant: data.length === width * height * channels.
   * Sample (x, y, c) lives at (y * width + x) * channels + c.
   * The reference is fixed; the bytes are mutable, because homr writes
   * regions in place (`gray[y:y+h, x:x+w] = 255`). Helpers that write say
   * "in place" in their name or doc; the rest return a new plane.
   */
  readonly data: Uint8Array;
  readonly height: number;
  readonly kind: K;
  readonly width: number;
}

export type Mask = PlaneBase<"mask", 1>;
export type GrayImage = PlaneBase<"gray", 1>;
export type ClassMap = PlaneBase<"classes", 1>;
export type ColorImage = PlaneBase<"bgr", 3>;

/** The single-channel planes; every 2-D helper below is written for these. */
export type Plane1 = Mask | GrayImage | ClassMap;
export type Plane = Plane1 | ColorImage;

export type PlaneOfKind<K extends PlaneKind> = Extract<Plane, { kind: K }>;

/** Fails with the plane's dimensions in the message, so a bad crop names itself. */
export class PlaneError extends Error {}

const CHANNELS: Readonly<Record<PlaneKind, 1 | 3>> = {
  bgr: 3,
  classes: 1,
  gray: 1,
  mask: 1,
};

export function channelsOf(kind: PlaneKind): 1 | 3 {
  return CHANNELS[kind];
}

function build<K extends PlaneKind>(
  kind: K,
  width: number,
  height: number,
  data: Uint8Array
): PlaneOfKind<K> {
  const plane = { channels: CHANNELS[kind], data, height, kind, width };
  return plane as unknown as PlaneOfKind<K>;
}

function checkDimensions(kind: PlaneKind, width: number, height: number): void {
  if (
    !(
      Number.isInteger(width) &&
      Number.isInteger(height) &&
      width >= 0 &&
      height >= 0
    )
  ) {
    throw new PlaneError(
      `${kind} plane needs non-negative integer dimensions, got ${width}x${height}`
    );
  }
}

// Factories. The only way to obtain a plane; they enforce the length invariant.

export function createMask(width: number, height: number): Mask {
  checkDimensions("mask", width, height);
  return build("mask", width, height, new Uint8Array(width * height));
}

export function createGray(width: number, height: number, fill = 0): GrayImage {
  checkDimensions("gray", width, height);
  return build(
    "gray",
    width,
    height,
    new Uint8Array(width * height).fill(fill)
  );
}

export function createClassMap(width: number, height: number): ClassMap {
  checkDimensions("classes", width, height);
  return build("classes", width, height, new Uint8Array(width * height));
}

export function createColor(
  width: number,
  height: number,
  fill = 255
): ColorImage {
  checkDimensions("bgr", width, height);
  return build(
    "bgr",
    width,
    height,
    new Uint8Array(width * height * 3).fill(fill)
  );
}

/**
 * Wraps existing bytes without copying (the WASM copy-out, the PNG decoder
 * and the golden loader all land here). Throws PlaneError unless
 * data.length === width * height * channels(kind). A Mask additionally
 * requires every byte to be 0 or 1: golden PNGs are 0/255 and must be
 * thresholded by the caller first, on purpose, so the two encodings never
 * meet inside the library.
 */
export function planeFromBytes<K extends PlaneKind>(
  kind: K,
  width: number,
  height: number,
  data: Uint8Array
): PlaneOfKind<K> {
  checkDimensions(kind, width, height);
  const expected = width * height * CHANNELS[kind];
  if (data.length !== expected) {
    throw new PlaneError(
      `${kind} plane ${width}x${height} needs ${expected} bytes, got ${data.length}`
    );
  }
  if (kind === "mask") {
    for (const value of data) {
      if (value > 1) {
        throw new PlaneError(
          `mask plane holds ${value}; threshold to 0/1 before wrapping`
        );
      }
    }
  }
  return build(kind, width, height, data);
}

/**
 * Browser boundary: an ImageData's RGBA bytes to cv2's BGR order, alpha
 * dropped. The one place the port meets the canvas element's pixel layout.
 */
export function colorImageFromRgba(
  width: number,
  height: number,
  rgba: Uint8ClampedArray | Uint8Array
): ColorImage {
  const image = createColor(width, height);
  const pixels = width * height;
  if (rgba.length !== pixels * 4) {
    throw new PlaneError(
      `rgba for ${width}x${height} needs ${pixels * 4} bytes, got ${rgba.length}`
    );
  }
  const out = image.data;
  for (let i = 0; i < pixels; i += 1) {
    out[i * 3] = rgba[i * 4 + 2] ?? 0;
    out[i * 3 + 1] = rgba[i * 4 + 1] ?? 0;
    out[i * 3 + 2] = rgba[i * 4] ?? 0;
  }
  return image;
}

/**
 * Browser boundary, the other way: a page as the RGBA bytes of an ImageData,
 * opaque. Gray is repeated on the three channels, BGR is reordered, and a 0/1
 * mask becomes black and white.
 */
export function rgbaFromPlane(
  plane: GrayImage | Mask | ColorImage
): Uint8ClampedArray {
  const pixels = plane.width * plane.height;
  const rgba = new Uint8ClampedArray(pixels * 4).fill(255);
  const { data } = plane;
  for (let i = 0; i < pixels; i += 1) {
    if (plane.kind === "bgr") {
      rgba[i * 4] = data[i * 3 + 2] ?? 0;
      rgba[i * 4 + 1] = data[i * 3 + 1] ?? 0;
      rgba[i * 4 + 2] = data[i * 3] ?? 0;
    } else {
      const level =
        plane.kind === "mask" ? (data[i] ?? 0) * 255 : (data[i] ?? 0);
      rgba.fill(level, i * 4, i * 4 + 3);
    }
  }
  return rgba;
}

// The numpy usage homr actually has on 2-D arrays. Each helper names the
// numpy or cv2 expression it stands in for; a reader of the Python can find
// the port line by line. Coordinates are (x, y) with y down, as in cv2.

/** Index of sample (x, y, c) in `plane.data`. */
export function sampleIndex(
  plane: Plane,
  x: number,
  y: number,
  channel = 0
): number {
  return (y * plane.width + x) * plane.channels + channel;
}

/**
 * One row of a single-channel plane as a view (no copy), for hot scans:
 * `for (const v of rowOf(p, y))` yields numbers, which is how a loop escapes
 * `noUncheckedIndexedAccess` without a non-null assertion.
 */
export function rowOf(plane: Plane1, y: number): Uint8Array {
  const start = y * plane.width;
  return plane.data.subarray(start, start + plane.width);
}

interface Region {
  readonly x1: number;
  readonly x2: number;
  readonly y1: number;
  readonly y2: number;
}

/** Half-open [x1, x2) x [y1, y2) clipped to the plane, integer bounds. */
function clipRegion(
  plane: Plane,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): Region {
  return {
    x1: Math.max(0, Math.min(plane.width, Math.trunc(x1))),
    x2: Math.max(0, Math.min(plane.width, Math.trunc(x2))),
    y1: Math.max(0, Math.min(plane.height, Math.trunc(y1))),
    y2: Math.max(0, Math.min(plane.height, Math.trunc(y2))),
  };
}

/** Copies half-open [x1, x2) x [y1, y2) into a new plane of the same kind; empty when inverted. */
function sliceRegion<P extends Plane>(plane: P, region: Region): P {
  const width = Math.max(0, region.x2 - region.x1);
  const height = Math.max(0, region.y2 - region.y1);
  const { channels } = plane;
  const out = new Uint8Array(width * height * channels);
  const rowBytes = width * channels;
  for (let row = 0; row < height; row += 1) {
    const src = sampleIndex(plane, region.x1, region.y1 + row);
    out.set(plane.data.subarray(src, src + rowBytes), row * rowBytes);
  }
  return build(plane.kind, width, height, out) as P;
}

/**
 * homr's image_utils.crop_image: orders the corners, clamps each to
 * [0, size - 1] after Python round() (half to even), then slices half-open
 * [y1:y2, x1:x2]. Returns a copy of the same kind; numpy returns a view but
 * every homr caller either reads it or writes it into a fresh array.
 */
export function cropPlane<P extends Plane>(
  plane: P,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): P {
  return cropPlaneAndReturnNewTop(plane, x1, y1, x2, y2).plane;
}

/** crop_image_and_return_new_top: the crop plus the clamped top-left it was cut at. */
export function cropPlaneAndReturnNewTop<P extends Plane>(
  plane: P,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): { readonly plane: P; readonly left: number; readonly top: number } {
  const left = clampToIndexOf(Math.min(x1, x2), plane.width);
  const top = clampToIndexOf(Math.min(y1, y2), plane.height);
  const right = clampToIndexOf(Math.max(x1, x2), plane.width);
  const bottom = clampToIndexOf(Math.max(y1, y2), plane.height);
  return {
    left,
    plane: sliceRegion(plane, { x1: left, x2: right, y1: top, y2: bottom }),
    top,
  };
}

/** `_limit_x`: max(0, min(size - 1, int(round(v)))). Local copy of numeric.clampToIndex to keep this module dependency-free. */
function clampToIndexOf(value: number, size: number): number {
  const fraction = Math.abs(value % 1);
  const rounded =
    fraction === 0.5 ? 2 * Math.round(value / 2) : Math.round(value);
  return Math.max(0, Math.min(Math.trunc(size - 1), Math.trunc(rounded)));
}

/** `dst[y0:y0+h, x0:x0+w] = src`, in place, clipped to dst. Both planes must share kind. */
export function blitInPlace<P extends Plane>(
  dst: P,
  src: P,
  x0: number,
  y0: number
): void {
  if (dst.kind !== src.kind) {
    throw new PlaneError(
      `cannot blit a ${src.kind} plane into a ${dst.kind} plane`
    );
  }
  const { channels } = dst;
  for (let row = 0; row < src.height; row += 1) {
    const y = y0 + row;
    if (y < 0 || y >= dst.height) {
      continue;
    }
    const xStart = Math.max(0, x0);
    const xEnd = Math.min(dst.width, x0 + src.width);
    if (xEnd <= xStart) {
      continue;
    }
    const srcIndex = sampleIndex(src, xStart - x0, row);
    const length = (xEnd - xStart) * channels;
    dst.data.set(
      src.data.subarray(srcIndex, srcIndex + length),
      sampleIndex(dst, xStart, y)
    );
  }
}

/** `plane[y1:y2, x1:x2] = value`, in place, half-open, clipped to the plane. */
export function fillRectInPlace(
  plane: Plane1,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  value: number
): void {
  const region = clipRegion(plane, x1, y1, x2, y2);
  for (let y = region.y1; y < region.y2; y += 1) {
    plane.data.fill(
      value,
      sampleIndex(plane, region.x1, y),
      sampleIndex(plane, region.x2, y)
    );
  }
}

/**
 * The `np.where(image > 0)` then `count[y] += 1` loop of
 * staff_detection.find_horizontal_lines: how many non-zero samples each row
 * holds. Length === plane.height.
 */
export function rowNonzeroCounts(plane: Plane1): Uint32Array {
  const counts = new Uint32Array(plane.height);
  for (let y = 0; y < plane.height; y += 1) {
    let n = 0;
    for (const v of rowOf(plane, y)) {
      if (v > 0) {
        n += 1;
      }
    }
    counts[y] = n;
  }
  return counts;
}

/**
 * note_detection.adjust_bbox: the min and max row holding a non-zero sample
 * inside the half-open region, in plane coordinates; null when the region is
 * blank (homr then keeps the box and lets a zero height eliminate it).
 */
export function nonzeroRowBounds(
  plane: Plane1,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): { readonly minY: number; readonly maxY: number } | null {
  const region = clipRegion(plane, x1, y1, x2, y2);
  let minY = -1;
  let maxY = -1;
  for (let y = region.y1; y < region.y2; y += 1) {
    const row = plane.data.subarray(
      sampleIndex(plane, region.x1, y),
      sampleIndex(plane, region.x2, y)
    );
    if (row.some((v) => v > 0)) {
      if (minY < 0) {
        minY = y;
      }
      maxY = y;
    }
  }
  return minY < 0 ? null : { maxY, minY };
}

/** `np.mean(plane[y1:y2, x1:x2])` as float64; NaN on an empty region, like numpy. */
export function meanOfRegion(
  plane: Plane1,
  x1: number,
  y1: number,
  x2: number,
  y2: number
): number {
  const region = clipRegion(plane, x1, y1, x2, y2);
  let total = 0;
  let count = 0;
  for (let y = region.y1; y < region.y2; y += 1) {
    for (const v of plane.data.subarray(
      sampleIndex(plane, region.x1, y),
      sampleIndex(plane, region.x2, y)
    )) {
      total += v;
      count += 1;
    }
  }
  return count === 0 ? Number.NaN : total / count;
}

function sameSize(a: Plane, b: Plane): boolean {
  return a.width === b.width && a.height === b.height;
}

/**
 * `cv2.bitwise_and(img, img, mask=mask)`: keep samples where mask != 0,
 * zero elsewhere. Returns a new plane of the same kind. Pure TypeScript on
 * purpose; noise_filtering is the only caller and it needs no Mat for this.
 */
export function applyMask<P extends Plane>(plane: P, mask: Mask): P {
  if (!sameSize(plane, mask)) {
    throw new PlaneError(
      `mask ${mask.width}x${mask.height} does not fit plane ${plane.width}x${plane.height}`
    );
  }
  const out = new Uint8Array(plane.data.length);
  const { channels } = plane;
  for (let i = 0; i < mask.data.length; i += 1) {
    if ((mask.data[i] ?? 0) !== 0) {
      const at = i * channels;
      for (let c = 0; c < channels; c += 1) {
        out[at + c] = plane.data[at + c] ?? 0;
      }
    }
  }
  return build(plane.kind, plane.width, plane.height, out) as P;
}

/** `255 * mask` as a GrayImage; what noise_filtering feeds its noise estimate. */
export function grayFromMask(mask: Mask): GrayImage {
  const out = new Uint8Array(mask.data.length);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = (mask.data[i] ?? 0) === 0 ? 0 : 255;
  }
  return build("gray", mask.width, mask.height, out);
}

/** `(merged == classIndex).astype(np.uint8)`: one binary mask out of a class map. */
export function maskOfClass(classes: ClassMap, classIndex: number): Mask {
  const out = new Uint8Array(classes.data.length);
  for (let i = 0; i < out.length; i += 1) {
    out[i] = classes.data[i] === classIndex ? 1 : 0;
  }
  return build("mask", classes.width, classes.height, out);
}

/**
 * `np.argmax(out, axis=0)` over a (C, H, W) float32 block, the segnet output
 * for one tile. First index wins a tie, as numpy. `logits.length` must be
 * classes * height * width, planar (all of class 0, then class 1, ...), which
 * is the ONNX NCHW layout for one batch item.
 */
export function argmaxPlanes(
  logits: Float32Array,
  classes: number,
  width: number,
  height: number
): ClassMap {
  const pixels = width * height;
  if (logits.length !== classes * pixels) {
    throw new PlaneError(
      `argmax over ${classes} classes of ${width}x${height} needs ${classes * pixels} logits, got ${logits.length}`
    );
  }
  const out = createClassMap(width, height);
  for (let i = 0; i < pixels; i += 1) {
    let best = 0;
    let bestValue = logits[i] ?? Number.NEGATIVE_INFINITY;
    for (let c = 1; c < classes; c += 1) {
      const v = logits[c * pixels + i] ?? Number.NEGATIVE_INFINITY;
      if (v > bestValue) {
        bestValue = v;
        best = c;
      }
    }
    out.data[i] = best;
  }
  return out;
}

/**
 * Fraction of pixels equal between two planes of the same size, in [0, 1].
 * testing.md's mask criterion (agreement >= 0.999 per class). Lives here
 * rather than in test/ because the bench page shows the same number.
 */
export function planeAgreement(a: Plane1, b: Plane1): number {
  if (!sameSize(a, b)) {
    throw new PlaneError(
      `cannot compare ${a.width}x${a.height} with ${b.width}x${b.height}`
    );
  }
  if (a.data.length === 0) {
    return 1;
  }
  let same = 0;
  for (let i = 0; i < a.data.length; i += 1) {
    if (a.data[i] === b.data[i]) {
      same += 1;
    }
  }
  return same / a.data.length;
}

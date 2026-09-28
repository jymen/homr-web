/**
 * The segnet tiling: `extract_patch`, the two loops of `inference` and
 * `merge_patches` from homr's `segmentation/inference_segnet.py`.
 *
 * homr feeds the segnet 320x320 windows of the preprocessed page and stitches
 * the per-window argmax back together. Everything here is integer arithmetic on
 * the grid of window origins, so it is testable without a model: phase 3's
 * session only has to run the tiles this module hands it and hand the class
 * maps back.
 *
 * One list of origins serves the extract pass, the merge pass and the batch
 * count, because in the Python they are three copies of the same two loops and
 * a disagreement between them silently mis-stitches the page rather than
 * failing. The Python's two ranges are not literally identical: `inference`
 * walks `range(0, max(h, win_size), step_size)` and `merge_patches` walks
 * `range(0, image_shape[0], step_size)`. They yield the same iterations for
 * every non-empty page as long as `step >= window`, which is what main.py
 * passes (step_size = win_size = 320); the port keeps the `inference` form,
 * since that one decides how many tiles the model is actually given.
 */

import type { ClassMap, GrayImage } from "../image/plane.js";
import {
  createClassMap,
  createGray,
  PlaneError,
  sampleIndex,
} from "../image/plane.js";
import { SEGNET_INPUT } from "../model/pipeline.js";
import { float16FromFloat32 } from "../models/dtype.js";

/**
 * One tile's top-left corner in page coordinates. Negative when the page is
 * smaller than the window: `min(loop, size - win_size)` pulls the last tile
 * back inside the page, and on a page narrower or shorter than one tile there
 * is nothing to pull back to.
 */
export interface TileOrigin {
  readonly x: number;
  readonly y: number;
}

/** The part of the page one tile covers: `extract_patch`'s y0/x0/y1/x1 as a half-open rectangle, whose size is the `ph`/`pw` that `merge_patches` reads back out of the patch. */
interface TileRegion {
  readonly height: number;
  readonly width: number;
  readonly x: number;
  readonly y: number;
}

/**
 * A step of 0 spins the loops forever and a fractional one drifts off the pixel
 * grid, so both are refused rather than clamped.
 *
 * The lower bound on the step is what keeps `tileCoverage`'s weight and
 * `mergeTileClasses`' sum inside a byte. `inference` is only ever given
 * `step_size = win_size` (main.py) or its own `win_size // 2` default, and at
 * half a window a pixel is covered at most four times, so a weight fits in a
 * byte and a sum of class indices reaches 20. A smaller step is not a slower
 * variant of homr, it is outside what homr does, and accepting it would wrap
 * both accumulators and hand back a silently wrong page instead of an error.
 */
function checkTiling(
  width: number,
  height: number,
  window: number,
  step: number
): void {
  if (
    !(
      Number.isInteger(width) &&
      Number.isInteger(height) &&
      width >= 0 &&
      height >= 0 &&
      Number.isInteger(window) &&
      window > 0 &&
      Number.isInteger(step) &&
      step > 0
    )
  ) {
    throw new PlaneError(
      `cannot tile a ${width}x${height} page with a ${window}px window every ${step}px: the page needs non-negative integer dimensions and the window and step positive integers`
    );
  }
  if (step * 2 < window) {
    throw new PlaneError(
      `a ${step}px step under a ${window}px window covers a pixel more than four times; homr tiles at the window or at half of it, and the byte accumulators of tileCoverage and mergeTileClasses are sized for that`
    );
  }
}

/**
 * The tile grid of one page in inference order, row-major: the port of
 * `inference`'s two loops.
 *
 * `min(loop, size - win_size)` is what makes the last row and column overlap
 * their predecessor instead of hanging off the page, and it is also why the
 * same origin can appear twice: with a step below the window, every clamped
 * iteration lands on `size - win_size`. homr's own grid does that, and
 * `merge_patches` then counts the duplicate in its weights, so the port keeps
 * it.
 */
export function tileGrid(
  width: number,
  height: number,
  window: number = SEGNET_INPUT.window,
  step: number = SEGNET_INPUT.step
): readonly TileOrigin[] {
  checkTiling(width, height, window, step);
  const origins: TileOrigin[] = [];
  const rows = Math.max(height, window);
  const columns = Math.max(width, window);
  for (let yLoop = 0; yLoop < rows; yLoop += step) {
    const y = Math.min(yLoop, height - window);
    for (let xLoop = 0; xLoop < columns; xLoop += step) {
      origins.push({ x: Math.min(xLoop, width - window), y });
    }
  }
  return origins;
}

function coveredRegion(
  origin: TileOrigin,
  width: number,
  height: number,
  window: number
): TileRegion {
  const x = Math.max(origin.x, 0);
  const y = Math.max(origin.y, 0);
  return {
    height: Math.min(origin.y + window, height) - y,
    width: Math.min(origin.x + window, width) - x,
    x,
    y,
  };
}

/**
 * `extract_patch`, in gray rather than in (3, win_size, win_size): the Python
 * patch's three channels are identical, since `inference` builds them with
 * `cv2.cvtColor(image, COLOR_GRAY2BGR)`, so the port carries one channel and
 * `writeTileInto` triples it at the tensor boundary.
 *
 * The padding is 255 because a tile hanging off the page must read as empty
 * paper: fill it with 0 and the segnet finds staff lines and note heads in the
 * margin, which then survive into the masks as symbols nobody drew.
 *
 * The copy lands at the tile's top-left corner whatever the clamping did —
 * `extract_patch`'s destination is `py0 = px0 = 0` — so a tile pulled back
 * inside the page is not shifted, while a negative origin pins the page to the
 * corner and the padding lands on the far side.
 *
 * The rectangle is copied row by row rather than with `cropPlane` and
 * `blitInPlace`: `cropPlane` is `image_utils.crop_image`, which clamps both
 * corners to `size - 1` and so can never reach the page's last row or column,
 * where `extract_patch` takes a half-open numpy slice; and a whole-page
 * `blitInPlace` clipped to the tile would copy past `x1`/`y1` for an origin
 * left of `width - window`, which the grid never produces but a caller could
 * hand over.
 */
export function extractTile(
  page: GrayImage,
  origin: TileOrigin,
  window: number = SEGNET_INPUT.window
): GrayImage {
  checkTiling(page.width, page.height, window, window);
  const tile = createGray(window, window, 255);
  const region = coveredRegion(origin, page.width, page.height, window);
  for (let row = 0; row < region.height; row += 1) {
    const from = sampleIndex(page, region.x, region.y + row);
    tile.data.set(page.data.subarray(from, from + region.width), row * window);
  }
  return tile;
}

/**
 * The 256 gray levels as binary16 bit patterns. `inference` only does
 * `astype(np.float32)`, so pixel values reach the model unnormalised as 0..255
 * and a gray page holds no other value: the fp16 branch below is therefore a
 * table lookup and not an encode per pixel. Built by calling
 * `float16FromFloat32` rather than by shifting, so it cannot drift from the
 * codec every other fp16 boundary in the library uses.
 */
function buildHalfOfByte(): Uint16Array {
  const table = new Uint16Array(256);
  for (let value = 0; value < table.length; value += 1) {
    table[value] = float16FromFloat32(value);
  }
  return table;
}
const HALF_OF_BYTE = buildHalfOfByte();

/**
 * One batch item of the segnet input tensor, NCHW: three identical planes of
 * the tile's gray values, written at `offset` elements into `target`. The
 * caller strides by `3 * window * window` per item, which is how
 * `np.stack(batch, axis=0)` lays a batch out.
 *
 * A `Uint16Array` target means the session wants fp16 (onnxruntime-web exposes
 * no Float16Array, so a half tensor is raw bit patterns); the branch comes from
 * the target's own type rather than from a flag, because the tensor the caller
 * allocated is the only authority on what the model was given.
 */
export function writeTileInto(
  target: Float32Array | Uint16Array,
  offset: number,
  tile: GrayImage,
  window: number = SEGNET_INPUT.window
): void {
  if (tile.width !== window || tile.height !== window) {
    throw new PlaneError(
      `a segnet tile is ${window}x${window}, got ${tile.width}x${tile.height}`
    );
  }
  const pixels = window * window;
  const needed = SEGNET_INPUT.channels * pixels;
  if (
    !Number.isInteger(offset) ||
    offset < 0 ||
    target.length - offset < needed
  ) {
    throw new RangeError(
      `writeTileInto needs ${needed} elements from ${offset}, the target holds ${target.length}`
    );
  }
  if (target instanceof Uint16Array) {
    for (let i = 0; i < pixels; i += 1) {
      target[offset + i] = HALF_OF_BYTE[tile.data[i] ?? 0] ?? 0;
    }
  } else {
    target.set(tile.data, offset);
  }
  // COLOR_GRAY2BGR duplicates the one channel it is given, so the other two
  // planes are a copy of the first and not a second conversion.
  for (let channel = 1; channel < SEGNET_INPUT.channels; channel += 1) {
    target.copyWithin(offset + channel * pixels, offset, offset + pixels);
  }
}

/**
 * `merge_patches`' `weight`: how many tiles cover each pixel, row-major. It is
 * the whole subtlety of the merge — the bands where two tiles vote on the same
 * pixel — so it is exported rather than hidden inside `mergeTileClasses`,
 * which reads it as its divisor.
 *
 * A byte per pixel is enough: with homr's step of one window no pixel is
 * covered more than four times (one pull-back band per axis), and even the
 * `step_size = win_size // 2` branch of `inference` stays in single figures.
 */
export function tileCoverage(
  width: number,
  height: number,
  window: number = SEGNET_INPUT.window,
  step: number = SEGNET_INPUT.step
): Uint8Array {
  const coverage = new Uint8Array(width * height);
  for (const origin of tileGrid(width, height, window, step)) {
    const region = coveredRegion(origin, width, height, window);
    for (let row = 0; row < region.height; row += 1) {
      const start = (region.y + row) * width + region.x;
      for (let column = 0; column < region.width; column += 1) {
        const at = start + column;
        coverage[at] = (coverage[at] ?? 0) + 1;
      }
    }
  }
  return coverage;
}

/**
 * `merge_patches`: stitch the per-tile argmax class maps back into one page.
 *
 * What it accumulates is the class indices themselves, not logits — `inference`
 * has already taken `np.argmax` per tile — so two overlapping tiles that
 * disagree produce the truncated mean of the two class numbers, which can name
 * a class neither tile chose: 4 and 5 give 4, 0 and 5 give 2. That is homr's
 * behaviour on every page it has ever processed, and reproducing it is the
 * point of this function; turning it into a vote would move the masks.
 *
 * The sum fits in a byte for the same reason the coverage does: a class index
 * is at most 5 and a pixel is covered at most four times.
 */
export function mergeTileClasses(
  tiles: readonly ClassMap[],
  width: number,
  height: number,
  window: number = SEGNET_INPUT.window,
  step: number = SEGNET_INPUT.step
): ClassMap {
  const grid = tileGrid(width, height, window, step);
  if (tiles.length !== grid.length) {
    throw new PlaneError(
      `merging a ${width}x${height} page in ${window}px tiles every ${step}px needs ${grid.length} class maps, got ${tiles.length}`
    );
  }
  const sum = new Uint8Array(width * height);
  for (const [index, origin] of grid.entries()) {
    const tile = tiles[index];
    if (tile === undefined) {
      throw new PlaneError(`class map ${index} of ${grid.length} is missing`);
    }
    if (tile.width !== window || tile.height !== window) {
      throw new PlaneError(
        `class map ${index} is ${tile.width}x${tile.height}, expected ${window}x${window}`
      );
    }
    const region = coveredRegion(origin, width, height, window);
    for (let row = 0; row < region.height; row += 1) {
      const into = (region.y + row) * width + region.x;
      // `patch[:ph, :pw]` reads from the patch's top-left corner, matching
      // extract_patch's destination.
      const from = row * window;
      for (let column = 0; column < region.width; column += 1) {
        const at = into + column;
        sum[at] = (sum[at] ?? 0) + (tile.data[from + column] ?? 0);
      }
    }
  }
  const coverage = tileCoverage(width, height, window, step);
  const merged = createClassMap(width, height);
  for (let i = 0; i < merged.data.length; i += 1) {
    const covered = coverage[i] ?? 0;
    // `weight[weight == 0] = 1` keeps a pixel no tile reached at 0 rather than
    // NaN. Math.trunc stands for the numpy float32-to-int cast in
    // `reconstructed.astype(patches[0].dtype)`, which truncates toward zero,
    // and not for Python's `//`, which floors; the two agree here because
    // neither a class index nor a weight is ever negative.
    merged.data[i] = Math.trunc((sum[i] ?? 0) / (covered === 0 ? 1 : covered));
  }
  return merged;
}

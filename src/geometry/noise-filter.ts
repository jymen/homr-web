/**
 * Port of noise_filtering.py: where the staff mask is noise rather than
 * lines, black out the page there, in every plane.
 *
 * Plain TypeScript, no opencv.js. The one cv2 call is a 3 x 3 filter2D over a
 * 0/255 tile with an integer kernel, so every intermediate is an integer below
 * 2^53 and any summation order gives the same float; and the tiles are filtered
 * in isolation, reflecting at their own edges, which a whole-image filter2D
 * would get wrong.
 */

import { floorDiv, truncToInt } from "../image/numeric.js";
import {
  applyMask,
  createMask,
  fillRectInPlace,
  type GrayImage,
  grayFromMask,
  type Mask,
  PlaneError,
} from "../image/plane.js";
import { IMAGE_NOISE_LIMIT } from "../model/constants.js";
import {
  createInputPredictions,
  type InputPredictions,
  type NoiseOutcome,
} from "../model/pipeline.js";

/** homr cuts the page into `height // 20` by `width // 20` tiles. */
export const NOISE_GRID_DIVISIONS = 20;

/**
 * create_grid's result. `rows` is `ceil(height / tileHeight)`, 21 for the Kesh
 * page and not 20: the last row of tiles is short.
 */
export interface NoiseGrid {
  readonly cols: number;
  readonly rows: number;
  readonly tileHeight: number;
  readonly tileWidth: number;
  /**
   * Row-major. uint8 as numpy stores it: the float estimate truncated, then
   * **wrapped modulo 256**. A dense tile reaches 2040 and is stored as 248; the
   * threshold reads the stored value. Measured on the arm64 oracle, where the
   * float-to-uint8 cast is the platform's.
   */
  readonly values: Uint8Array;
}

interface Tile {
  readonly height: number;
  readonly left: number;
  readonly top: number;
  readonly width: number;
}

/** cv2's BORDER_REFLECT_101 for an offset of one: -1 reads 1, n reads n - 2, and a single sample reads itself. */
function reflect101(index: number, length: number): number {
  if (length === 1) {
    return 0;
  }
  if (index < 0) {
    return -index;
  }
  return index >= length ? 2 * length - 2 - index : index;
}

/** The kernel `[[1, -2, 1], [-2, 4, -2], [1, -2, 1]]` is `[1, -2, 1]` down the rows of `[1, -2, 1]` along them. */
function noiseOfTile(gray: GrayImage, tile: Tile): number {
  const at = (y: number, x: number): number =>
    gray.data[(tile.top + y) * gray.width + tile.left + x] ?? 0;
  let total = 0;
  for (let y = 0; y < tile.height; y += 1) {
    const above = reflect101(y - 1, tile.height);
    const below = reflect101(y + 1, tile.height);
    for (let x = 0; x < tile.width; x += 1) {
      const left = reflect101(x - 1, tile.width);
      const right = reflect101(x + 1, tile.width);
      const response =
        at(above, left) -
        2 * at(above, x) +
        at(above, right) -
        2 * (at(y, left) - 2 * at(y, x) + at(y, right)) +
        (at(below, left) - 2 * at(below, x) + at(below, right));
      total += Math.abs(response);
    }
  }
  return total / (tile.height * tile.width);
}

/**
 * estimate_noise: the mean absolute response of
 * `[[1, -2, 1], [-2, 4, -2], [1, -2, 1]]` over the tile, BORDER_REFLECT_101
 * at the tile's own edges, divided by the tile's own area.
 */
export function estimateNoise(tile: GrayImage): number {
  return noiseOfTile(tile, {
    height: tile.height,
    left: 0,
    top: 0,
    width: tile.width,
  });
}

/** create_grid over `255 * staff`. Throws PlaneError on a page under 20 px either way, where Python divides by zero. */
export function createNoiseGrid(gray: GrayImage): NoiseGrid {
  const tileHeight = floorDiv(gray.height, NOISE_GRID_DIVISIONS);
  const tileWidth = floorDiv(gray.width, NOISE_GRID_DIVISIONS);
  if (tileHeight === 0 || tileWidth === 0) {
    throw new PlaneError(
      `a ${gray.width}x${gray.height} page is too small for a noise grid`
    );
  }
  const rows = Math.ceil(gray.height / tileHeight);
  const cols = Math.ceil(gray.width / tileWidth);
  const values = new Uint8Array(rows * cols);
  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      const top = i * tileHeight;
      const left = j * tileWidth;
      const noise = noiseOfTile(gray, {
        height: Math.min(tileHeight, gray.height - top),
        left,
        top,
        width: Math.min(tileWidth, gray.width - left),
      });
      // noise_filtering.py:44 `grid[i, j] = noise` stores a float64 in a uint8 cell: truncated, then modulo 256, which Uint8Array's own conversion is.
      values[i * cols + j] = truncToInt(noise);
    }
  }
  return { cols, rows, tileHeight, tileWidth, values };
}

/** get_neighbors: up, left, down, right, the ones that exist. */
function hasNoisyNeighbour(grid: NoiseGrid, i: number, j: number): boolean {
  const at = (row: number, column: number): number =>
    row < 0 || column < 0 || row >= grid.rows || column >= grid.cols
      ? 0
      : (grid.values[row * grid.cols + column] ?? 0);
  return [at(i - 1, j), at(i, j - 1), at(i + 1, j), at(i, j + 1)].some(
    (noise) => noise > IMAGE_NOISE_LIMIT
  );
}

/**
 * apply_noise_filter and handle_filter_results: a tile is filtered when it and
 * at least one of its up, left, down, right neighbours are above
 * IMAGE_NOISE_LIMIT. None filtered is "clean"; more than half is "skipped".
 */
export function noiseOutcomeOf(
  grid: NoiseGrid,
  image: { readonly height: number; readonly width: number }
): NoiseOutcome {
  const keep = createMask(image.width, image.height);
  const totalTiles = grid.rows * grid.cols;
  let filteredTiles = 0;
  for (let i = 0; i < grid.rows; i += 1) {
    for (let j = 0; j < grid.cols; j += 1) {
      const noise = grid.values[i * grid.cols + j] ?? 0;
      if (noise > IMAGE_NOISE_LIMIT && hasNoisyNeighbour(grid, i, j)) {
        filteredTiles += 1;
      } else {
        fillRectInPlace(
          keep,
          j * grid.tileWidth,
          i * grid.tileHeight,
          (j + 1) * grid.tileWidth,
          (i + 1) * grid.tileHeight,
          1
        );
      }
    }
  }
  if (filteredTiles / totalTiles > 0.5) {
    return { filteredTiles, kind: "skipped", totalTiles };
  }
  if (filteredTiles > 0) {
    return { filteredTiles, keep, kind: "masked", totalTiles };
  }
  return { kind: "clean", totalTiles };
}

/**
 * filter_predictions. On "masked" every plane is a new plane with the masked
 * tiles zeroed, `original` and `preprocessed` included; otherwise
 * `predictions` is the argument itself.
 */
export function filterPredictions(input: InputPredictions): {
  readonly outcome: NoiseOutcome;
  readonly predictions: InputPredictions;
} {
  const outcome = noiseOutcomeOf(
    createNoiseGrid(grayFromMask(input.masks.staff)),
    input.masks.staff
  );
  if (outcome.kind !== "masked") {
    return { outcome, predictions: input };
  }
  const masked = (plane: Mask): Mask => applyMask(plane, outcome.keep);
  return {
    outcome,
    predictions: createInputPredictions(
      applyMask(input.original, outcome.keep),
      applyMask(input.preprocessed, outcome.keep),
      {
        clefsKeys: masked(input.masks.clefsKeys),
        notehead: masked(input.masks.notehead),
        staff: masked(input.masks.staff),
        stemsRest: masked(input.masks.stemsRest),
        symbols: masked(input.masks.symbols),
      }
    ),
  };
}

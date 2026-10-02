/**
 * Port of homr's find_peaks.py, its own scipy-free peak finder, and not
 * scipy's semantics: a plateau's peak is the floor midpoint, a flat step
 * after a descent is a peak when a descent follows it (`3, 1, 1, 0` peaks at
 * index 2), prominence scans outward until a strictly higher sample with the
 * peak's own value seeding both minima, and `distance` is a float compared
 * with `>=`.
 */

import { npArgsort } from "./argsort.js";
import { floorDiv } from "./numeric.js";

/** Each is one of find_peaks's filters; left out, as Python's None, the filter does not run. */
export interface FindPeaksOptions {
  readonly distance?: number;
  readonly height?: number;
  readonly prominence?: number;
}

const at = (x: Float64Array, i: number): number => x[i] ?? Number.NaN;

function plateauPeaks(x: Float64Array): number[] {
  const last = x.length - 1;
  const peaks: number[] = [];
  let i = 1;
  while (i < last) {
    // find_peaks.py walks a rise and a flat continuation with the same code,
    // so a flat run needs no rise before it.
    if (at(x, i) >= at(x, i - 1)) {
      let j = i;
      while (j < last && at(x, j) === at(x, j + 1)) {
        j += 1;
      }
      if (j < last && at(x, j) > at(x, j + 1)) {
        peaks.push(floorDiv(i + j, 2));
      }
      i = j + 1;
    } else {
      i += 1;
    }
  }
  return peaks;
}

/** The lowest sample from `peak` in one direction, up to the first strictly higher one. */
function lowestBeside(x: Float64Array, peak: number, step: -1 | 1): number {
  const top = at(x, peak);
  let lowest = top;
  for (let k = peak + step; k >= 0 && k < x.length; k += step) {
    const value = at(x, k);
    if (value > top) {
      break;
    }
    if (value < lowest) {
      lowest = value;
    }
  }
  return lowest;
}

function prominenceOf(x: Float64Array, peak: number): number {
  const left = lowestBeside(x, peak, -1);
  const right = lowestBeside(x, peak, 1);
  return at(x, peak) - (right > left ? right : left);
}

/** Highest first in np.argsort's tie order; a peak is kept when every kept one is `distance` away. */
function spacedPeaks(
  x: Float64Array,
  peaks: readonly number[],
  distance: number
): number[] {
  const byHeight = npArgsort(peaks.map((peak) => at(x, peak))).reverse();
  const kept: number[] = [];
  for (const index of byHeight) {
    const peak = peaks[index] ?? 0;
    if (kept.every((other) => Math.abs(other - peak) >= distance)) {
      kept.push(peak);
    }
  }
  return kept.sort((a, b) => a - b);
}

/** Peak indices, ascending. Empty for fewer than three samples. */
export function findPeaks(
  x: Float64Array,
  options: FindPeaksOptions = {}
): Int32Array {
  const { distance, height, prominence } = options;
  let peaks = x.length < 3 ? [] : plateauPeaks(x);
  if (height !== undefined) {
    peaks = peaks.filter((peak) => at(x, peak) >= height);
  }
  if (prominence !== undefined) {
    peaks = peaks.filter((peak) => prominenceOf(x, peak) >= prominence);
  }
  if (distance !== undefined && peaks.length > 1) {
    peaks = spacedPeaks(x, peaks, distance);
  }
  return Int32Array.from(peaks);
}

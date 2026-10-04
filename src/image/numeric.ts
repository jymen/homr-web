/**
 * The 1-D numpy and Python arithmetic homr relies on, with the semantics
 * the golden tests need bit for bit. Every function here has a determinism
 * trap named in the plan's testing.md; the doc comment states the rule and
 * test/numeric.test.ts pins the boundary cases.
 *
 * Rule for the rest of the port: never call Math.round, Math.trunc or `| 0`
 * where the Python calls round(), int() or `//`; call these instead.
 */

/** Below this many values numpy sums left to right; from it up, in eight lanes. */
const PAIRWISE_LANES = 8;
/** Above this many values numpy halves the range and recurses. */
const PAIRWISE_BLOCK = 128;

/**
 * `np.sum` on float64, which is numpy's pairwise summation and not a left
 * fold: the two differ from 8 values up. Python's builtin sum() is the left
 * fold; a site that needs it writes its own loop.
 */
export function sum(values: ArrayLike<number>): number {
  return pairwiseSum(values, 0, values.length);
}

function pairwiseSum(
  values: ArrayLike<number>,
  from: number,
  count: number
): number {
  if (count < PAIRWISE_LANES) {
    let acc = 0;
    for (let i = 0; i < count; i += 1) {
      acc += values[from + i] ?? 0;
    }
    return acc;
  }
  if (count > PAIRWISE_BLOCK) {
    let half = Math.floor(count / 2);
    half -= half % PAIRWISE_LANES;
    return (
      pairwiseSum(values, from, half) +
      pairwiseSum(values, from + half, count - half)
    );
  }
  const lanes = Float64Array.from({ length: PAIRWISE_LANES }, (_, lane) =>
    Number(values[from + lane])
  );
  const laned = count - (count % PAIRWISE_LANES);
  for (let i = PAIRWISE_LANES; i < laned; i += 1) {
    const lane = i % PAIRWISE_LANES;
    lanes[lane] = (lanes[lane] ?? 0) + (values[from + i] ?? 0);
  }
  const [r0 = 0, r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0, r6 = 0, r7 = 0] =
    lanes;
  let acc = r0 + r1 + (r2 + r3) + (r4 + r5 + (r6 + r7));
  for (let i = laned; i < count; i += 1) {
    acc += values[from + i] ?? 0;
  }
  return acc;
}

/** `np.mean`, and `np.average` without weights; NaN on an empty input, as numpy (which also warns). */
export function mean(values: ArrayLike<number>): number {
  if (values.length === 0) {
    return Number.NaN;
  }
  return sum(values) / values.length;
}

/**
 * `np.median`: sort ascending, middle element for odd n, mean of the two
 * middle elements for even n. Staff.average_unit_size and the average
 * notehead height are medians, so a "middle element" shortcut would shift
 * every unit-size threshold on pages with an even grid count.
 */
export function median(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) {
    return Number.NaN;
  }
  const sorted = Float64Array.from(values).sort();
  const mid = Math.floor(n / 2);
  const upper = sorted[mid] ?? Number.NaN;
  if (n % 2 === 1) {
    return upper;
  }
  const lower = sorted[mid - 1] ?? Number.NaN;
  return (lower + upper) / 2;
}

/**
 * `np.std` with ddof = 0 (population), numpy's default; not the sample std.
 * The squares are summed pairwise too, and the root is Math.sqrt: `** 0.5`
 * is a different last bit on some inputs.
 */
export function std(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) {
    return Number.NaN;
  }
  const m = mean(values);
  const squares = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    const d = (values[i] ?? 0) - m;
    squares[i] = d * d;
  }
  return Math.sqrt(sum(squares) / n);
}

/** `np.diff`: values[i + 1] - values[i]; length n - 1 (0 for n < 2). */
export function diff(values: ArrayLike<number>): Float64Array {
  const n = Math.max(values.length - 1, 0);
  const out = new Float64Array(n);
  for (let i = 0; i < n; i += 1) {
    out[i] = (values[i + 1] ?? 0) - (values[i] ?? 0);
  }
  return out;
}

/** `np.argmin`: index of the first minimum. Throws on empty input, as numpy. */
export function argmin(values: ArrayLike<number>): number {
  if (values.length === 0) {
    throw new RangeError("argmin of an empty sequence");
  }
  let best = 0;
  for (let i = 1; i < values.length; i += 1) {
    if ((values[i] ?? Number.NaN) < (values[best] ?? Number.NaN)) {
      best = i;
    }
  }
  return best;
}

/** `np.argmax`: index of the first maximum. Throws on empty input, as numpy. */
export function argmax(values: ArrayLike<number>): number {
  if (values.length === 0) {
    throw new RangeError("argmax of an empty sequence");
  }
  let best = 0;
  for (let i = 1; i < values.length; i += 1) {
    if ((values[i] ?? Number.NaN) > (values[best] ?? Number.NaN)) {
      best = i;
    }
  }
  return best;
}

/**
 * Python's round(): half to even. Math.round is half up, so 2.5 differs.
 * Pinned on 0.5, 1.5, 2.5, -0.5, -1.5 and on values already integral.
 */
export function roundHalfEven(x: number): number {
  const fraction = Math.abs(x % 1);
  const rounded = fraction === 0.5 ? 2 * Math.round(x / 2) : Math.round(x);
  // Python's round(-0.5) is -0.0; as an index or a count that is 0, and a
  // signed zero would only leak into string formatting.
  return rounded === 0 ? 0 : rounded;
}

/** toFixed's ceiling: enough for the exact expansion of any double above 1e-14. */
const EXACT_DIGITS = 100;
const NEGLIGIBLE = 1e-14;
const ALL_ZEROS = /^0*$/;

/**
 * Python's round(x, ndigits): the decimal nearest the double's exact value,
 * ties to even. toFixed(100) is that exact value, so the decision is made on
 * its digits; toFixed(ndigits) alone breaks an exact tie upwards.
 */
export function pyRound(x: number, ndigits: number): number {
  if (Math.abs(x) < NEGLIGIBLE) {
    return x * 0;
  }
  const exact = Math.abs(x).toFixed(EXACT_DIGITS);
  const point = exact.indexOf(".");
  const kept = exact.slice(0, point + 1 + ndigits);
  const next = exact.charAt(point + 1 + ndigits);
  const rest = exact.slice(point + 2 + ndigits);
  const last = kept.at(-1);
  const tie = next === "5" && ALL_ZEROS.test(rest);
  const up =
    next > "5" || (next === "5" && !tie) || (tie && Number(last) % 2 === 1);
  const magnitude = Number(kept) + (up ? 10 ** -ndigits : 0);
  return Math.sign(x) * Number(magnitude.toFixed(ndigits));
}

/** numpy's ndarray.round(decimals) on a float64: scale, rint (ties to even), unscale. */
export function npRound(x: number, decimals: number): number {
  const scale = 10 ** decimals;
  return roundHalfEven(x * scale) / scale;
}

/**
 * Python's `//` on floats, as CPython's float_floor_div computes it: from
 * fmod, not from the quotient. `Math.floor(a / b)` is one too high when
 * `a / b` rounds up to an integer the true quotient is below
 * (-10.000000000000002 // 10 is -2, and the division gives -1 exactly).
 */
export function floorDiv(a: number, b: number): number {
  const mod = a % b;
  let div = (a - mod) / b;
  if (mod !== 0 && b < 0 !== mod < 0) {
    div -= 1;
  }
  if (div === 0) {
    return 0;
  }
  const floored = Math.floor(div);
  return div - floored > 0.5 ? floored + 1 : floored;
}

/** Python's int() on a float: truncation toward zero (also `astype(np.int64)`). */
export function truncToInt(x: number): number {
  return Math.trunc(x);
}

/**
 * homr's image_utils._limit_x / _limit_y:
 * `max(0, min(int(length - 1), int(round(x))))`. Banker's rounding, then
 * truncation, then the clamp; `Math.max(0, Math.min(length - 1,
 * Math.round(x)))` is wrong on the half values.
 */
export function clampToIndex(value: number, length: number): number {
  return Math.max(
    0,
    Math.min(truncToInt(length - 1), truncToInt(roundHalfEven(value)))
  );
}

/**
 * The float32 a cv2 result carries (minAreaRect, fitEllipse, boxPoints).
 * Use it only where the Python compares a threshold against a value cv2
 * produced; sprinkling it elsewhere makes the port less exact than the
 * oracle, not more.
 */
export function toFloat32(value: number): number {
  return Math.fround(value);
}

/**
 * Python's str(float), which homr's save_staff_positions writes and whose
 * output phase 9 must reproduce byte for byte. JavaScript's String(number)
 * agrees on the digits (both print the shortest round-tripping decimal) but
 * not on the form: Python prints "1.0" where JS prints "1", switches to
 * exponent notation below 1e-4 and at or above 1e16 where JS switches below
 * 1e-6 and at 1e21, and spells the exponent "1e-05" where JS spells "1e-5".
 */
const SINGLE_DIGIT_EXPONENT = /e([+-])(\d)$/;

export function formatPythonFloat(x: number): string {
  if (Number.isNaN(x)) {
    return "nan";
  }
  if (!Number.isFinite(x)) {
    return x > 0 ? "inf" : "-inf";
  }
  if (Object.is(x, -0)) {
    return "-0.0";
  }
  const magnitude = Math.abs(x);
  if (magnitude !== 0 && (magnitude < 1e-4 || magnitude >= 1e16)) {
    return x.toExponential().replace(SINGLE_DIGIT_EXPONENT, "e$10$2");
  }
  if (Number.isInteger(x)) {
    return `${x}.0`;
  }
  return String(x);
}

/**
 * The half-open range a Python slice `[start:stop]` selects on an axis of
 * `length`: a negative index counts from the end, an index past either end
 * clamps, and a stop at or before the start selects nothing. An empty slice
 * comes back as `stop === start`.
 */
export function pySliceBounds(
  start: number,
  stop: number,
  length: number
): { readonly start: number; readonly stop: number } {
  const clamp = (index: number): number =>
    Math.min(Math.max(index < 0 ? index + length : index, 0), length);
  const from = clamp(start);
  return { start: from, stop: Math.max(from, clamp(stop)) };
}

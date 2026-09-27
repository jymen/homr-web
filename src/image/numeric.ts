/**
 * The 1-D numpy and Python arithmetic homr relies on, with the semantics
 * the golden tests need bit for bit. Every function here has a determinism
 * trap named in the plan's testing.md; the doc comment states the rule and
 * test/numeric.test.ts pins the boundary cases.
 *
 * Rule for the rest of the port: never call Math.round, Math.trunc or `| 0`
 * where the Python calls round(), int() or `//`; call these instead.
 */

/** `np.mean`; NaN on an empty input, as numpy (which also warns). */
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

/** `np.std` with ddof = 0 (population), numpy's default; not the sample std. */
export function std(values: ArrayLike<number>): number {
  const n = values.length;
  if (n === 0) {
    return Number.NaN;
  }
  const m = mean(values);
  let acc = 0;
  for (let i = 0; i < n; i += 1) {
    const d = (values[i] ?? 0) - m;
    acc += d * d;
  }
  return Math.sqrt(acc / n);
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

/** `np.sum` accumulated left to right in float64. */
export function sum(values: ArrayLike<number>): number {
  let acc = 0;
  for (const value of Array.from(values)) {
    acc += value;
  }
  return acc;
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

/** Python's `//`: floor division toward minus infinity, also for negatives. */
export function floorDiv(a: number, b: number): number {
  return Math.floor(a / b);
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

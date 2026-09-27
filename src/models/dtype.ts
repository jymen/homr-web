/**
 * IEEE 754 binary16, the wire format onnxruntime-web 1.30.0 gives no
 * Float16Array for: an fp16 tensor is a Uint16Array of raw half bit patterns
 * on both sides of a run, so every fp16 input and output crosses this module.
 * docs/design/phase-2-models.md's "The half codec, and why phase 2 owns it"
 * argues why the codec lives here while the casts do not.
 *
 * This is the only module in the library holding bit-level arithmetic, and it
 * imports nothing.
 */

const HALF_SIGN = 0x8000;
const HALF_INFINITY = 0x7c00;
/** Set on a NaN result so a dropped payload cannot read back as an Infinity. */
const HALF_QUIET_BIT = 0x0200;
/** float32's exponent bias minus half's: 127 - 15. */
const EXPONENT_BIAS_DELTA = 112;
/** float32 keeps 23 mantissa bits, half keeps 10; 13 are rounded away. */
const MANTISSA_DROP = 13;
const DROPPED_MASK = 0x1fff;
const DROPPED_HALFWAY = 0x1000;
/** Below 2 ** -25 nothing survives, and that is a shift of 25 on the input. */
const SUBNORMAL_MAX_SHIFT = 24;
/** A half subnormal's mantissa unit. Written as a power so no decimal literal has to be trusted; float32 holds it exactly. */
const SUBNORMAL_UNIT = 2 ** -24;

// The float32 bits are read through views aliased onto one buffer rather than
// a DataView: both views address it in the platform's own byte order, so the
// endianness cancels and the read is a store and a load.
const scratch = new ArrayBuffer(4);
const asFloat32 = new Float32Array(scratch);
const asUint32 = new Uint32Array(scratch);

/** Ties go to the neighbour whose low bit is already 0, which is what makes this round-to-nearest-even and not round-half-up. */
function roundingIncrement(
  dropped: number,
  halfway: number,
  kept: number
): number {
  if (dropped > halfway) {
    return 1;
  }
  if (dropped < halfway) {
    return 0;
  }
  return kept & 1;
}

/**
 * IEEE 754 binary16, round-to-nearest-even, which is what ONNX and numpy do. The
 * result is the raw 16-bit pattern as a number in 0..65535, not a value.
 *
 * `value` is taken as a float32, since it is stored through a Float32Array on
 * the way in. That keeps a scalar call and encodeFloat16Array bit-identical on
 * the same number, and it is the representation every caller actually holds: a
 * float64 sitting on a half's midpoint only after that store rounds once, from
 * float32.
 */
export function float16FromFloat32(value: number): number {
  asFloat32[0] = value;
  const bits = asUint32[0] ?? 0;
  const sign = (bits >>> 16) & HALF_SIGN;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x007f_ffff;

  if (exponent === 0xff) {
    // A NaN must not become an Infinity. Only 10 of the 23 payload bits could
    // survive and all 10 can be zero, so set the quiet bit instead of carrying
    // a truncated payload across.
    return sign | HALF_INFINITY | (mantissa === 0 ? 0 : HALF_QUIET_BIT);
  }

  const halfExponent = exponent - EXPONENT_BIAS_DELTA;
  if (halfExponent >= 0x1f) {
    return sign | HALF_INFINITY;
  }
  if (halfExponent > 0) {
    const kept = (halfExponent << 10) | (mantissa >>> MANTISSA_DROP);
    // A carry out of the mantissa lands in the exponent field, and a carry out
    // of exponent 30 lands exactly on 0x7c00. Overflow to Infinity therefore
    // comes for free. That is what 65520 has to do: it is the midpoint above
    // the largest finite half, so its tie resolves upwards, out of the range.
    return (
      sign +
      kept +
      roundingIncrement(mantissa & DROPPED_MASK, DROPPED_HALFWAY, kept)
    );
  }
  if (exponent === 0) {
    // Every float32 subnormal is orders of magnitude below half of the
    // smallest half subnormal, and -0 stays -0 rather than turning positive.
    return sign;
  }

  const shift = EXPONENT_BIAS_DELTA + 14 - exponent;
  if (shift > SUBNORMAL_MAX_SHIFT) {
    return sign;
  }
  const significand = 0x0080_0000 | mantissa;
  const kept = significand >>> shift;
  // A round up from 1023 to 1024 spills into the exponent field and yields the
  // smallest normal half, which is the correct answer rather than an overflow.
  return (
    sign +
    kept +
    roundingIncrement(significand & ((1 << shift) - 1), 1 << (shift - 1), kept)
  );
}

/**
 * Every half is exactly representable in float32, so this is exact arithmetic
 * and not an approximation. The lookup table below is built by calling this
 * 65536 times rather than repeating the arithmetic, so the two can never
 * disagree.
 */
export function float32FromFloat16(half: number): number {
  const bits = half & 0xffff;
  const sign = (bits & HALF_SIGN) === 0 ? 1 : -1;
  const exponent = (bits >>> 10) & 0x1f;
  const mantissa = bits & 0x03ff;
  if (exponent === 0) {
    // sign * 0 is -0 for a negative sign, which is what 0x8000 asks for.
    return sign * mantissa * SUBNORMAL_UNIT;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Number.POSITIVE_INFINITY : Number.NaN;
  }
  return sign * (0x400 + mantissa) * 2 ** (exponent - 25);
}

/**
 * 256 KB, so it is built on the first decode and never paid for by a caller
 * that only encodes or only opens a session. Measured 3.1 ms cold on Node
 * 23.5 for the 65536 calls, which is once per realm.
 */
let decodeTable: Float32Array | null = null;

function decodeTableOf(): Float32Array {
  if (decodeTable === null) {
    const table = new Float32Array(0x1_0000);
    for (let half = 0; half < 0x1_0000; half += 1) {
      table[half] = float32FromFloat16(half);
    }
    decodeTable = table;
  }
  return decodeTable;
}

/**
 * Buffer-wide, through a memoised 65536-entry Float32Array lookup table: a
 * half decode becomes an array index, so a per-pixel decode in a hot loop
 * costs no arithmetic. Measured 1.4 ms for segnet's 6x320x320 output.
 *
 * `into` is an optional output buffer, reused when it can hold the input. One
 * too short throws a RangeError naming both lengths instead of allocating: the
 * reason to pass a buffer at all is to avoid an allocation per tile, so a
 * silent fallback would hide the mis-sizing until a profile found the garbage.
 * The result always has the input's length, so an oversized pool can be
 * iterated as it comes back.
 */
export function decodeFloat16Array(
  halves: Uint16Array,
  into?: Float32Array
): Float32Array {
  const count = halves.length;
  let out: Float32Array;
  if (into === undefined) {
    out = new Float32Array(count);
  } else if (into.length < count) {
    throw new RangeError(
      `decodeFloat16Array: into holds ${into.length} values, the input has ${count}`
    );
  } else {
    out = into.length === count ? into : into.subarray(0, count);
  }
  const table = decodeTableOf();
  for (let i = 0; i < count; i += 1) {
    out[i] = table[halves[i] ?? 0] ?? 0;
  }
  return out;
}

/** The encoding direction of decodeFloat16Array, with the same `into` contract; no table, since 65536 entries would not cover the 2 ** 32 inputs. */
export function encodeFloat16Array(
  values: Float32Array,
  into?: Uint16Array
): Uint16Array {
  const count = values.length;
  let out: Uint16Array;
  if (into === undefined) {
    out = new Uint16Array(count);
  } else if (into.length < count) {
    throw new RangeError(
      `encodeFloat16Array: into holds ${into.length} values, the input has ${count}`
    );
  } else {
    out = into.length === count ? into : into.subarray(0, count);
  }
  for (let i = 0; i < count; i += 1) {
    out[i] = float16FromFloat32(values[i] ?? Number.NaN);
  }
  return out;
}

/**
 * The duration half of homr's vocabulary.py: SymbolDuration, kern parsing and
 * the exact rationals Python's Fraction gives it. Everything here is small
 * integers (kern values up to 128, a few dots), so a numerator and denominator
 * in safe integers are exact.
 */

export interface Ratio {
  readonly den: number;
  readonly num: number;
}

const gcd = (a: number, b: number): number => {
  let x = Math.abs(a);
  let y = Math.abs(b);
  while (y !== 0) {
    [x, y] = [y, x % y];
  }
  return x;
};

export function ratio(num: number, den = 1): Ratio {
  const divisor = gcd(num, den) || 1;
  const sign = den < 0 ? -1 : 1;
  return { den: (sign * den) / divisor, num: (sign * num) / divisor };
}

export const ZERO: Ratio = ratio(0);

export const addRatio = (a: Ratio, b: Ratio): Ratio =>
  ratio(a.num * b.den + b.num * a.den, a.den * b.den);

export const mulRatio = (a: Ratio, b: Ratio): Ratio =>
  ratio(a.num * b.num, a.den * b.den);

/** Negative, zero or positive as a is less than, equal to or greater than b. */
export const compareRatio = (a: Ratio, b: Ratio): number =>
  a.num * b.den - b.num * a.den;

export interface SymbolDuration {
  readonly actualNotes: number;
  readonly baseDuration: Ratio;
  readonly dots: number;
  /** Relative to a whole note. */
  readonly fraction: Ratio;
  readonly kern: number;
  readonly normalNotes: number;
}

export function symbolDuration(
  baseDuration: Ratio,
  dots: number,
  actual: number,
  normal: number,
  kern: number
): SymbolDuration {
  const tuplet = ratio(actual, normal);
  let duration = baseDuration;
  let add = mulRatio(duration, ratio(1, 2));
  for (let i = 0; i < dots; i += 1) {
    duration = addRatio(duration, add);
    add = mulRatio(add, ratio(1, 2));
  }
  if (tuplet.num !== tuplet.den) {
    duration = mulRatio(duration, ratio(tuplet.den, tuplet.num));
  }
  return {
    actualNotes: tuplet.num,
    baseDuration,
    dots,
    fraction: duration,
    kern,
    normalNotes: tuplet.den,
  };
}

/** prior_power_of_two: the largest power of two not above n, and 1 below 1. */
export function priorPowerOfTwo(n: number): number {
  if (n < 1) {
    return 1;
  }
  let power = 1;
  while (power * 2 <= n) {
    power *= 2;
  }
  return power;
}

const isDigit = (c: string | undefined): boolean =>
  c !== undefined && c >= "0" && c <= "9";

/**
 * kern_to_symbol_duration. homr's multirest branch builds a value and drops
 * it, so "2m" falls through and parses as base 2; that is kept.
 */
export function kernToSymbolDuration(kern: string): SymbolDuration {
  let digits = 0;
  while (isDigit(kern[digits])) {
    digits += 1;
  }
  const baseText = kern.slice(0, digits);
  const rest = kern.slice(digits);
  const base = baseText === "" ? 4 : Number.parseInt(baseText, 10);
  const dots = rest.split(".").length - 1;
  if (kern.includes("G")) {
    return symbolDuration(ZERO, dots, 1, 1, base);
  }
  if (base === 0) {
    return symbolDuration(ratio(1), dots, 1, 1, base);
  }
  if (priorPowerOfTwo(base) === base) {
    return symbolDuration(ratio(1, base), dots, 1, 1, base);
  }
  const normal = priorPowerOfTwo(base);
  return symbolDuration(ratio(1, normal), dots, base, normal, normal);
}

/** EncodedSymbol.get_duration, for any rhythm: a non-note, non-rest is 0. */
export function durationOfRhythm(rhythm: string): SymbolDuration {
  if (!(rhythm.startsWith("note") || rhythm.startsWith("rest"))) {
    return symbolDuration(ZERO, 0, 1, 1, 1);
  }
  return kernToSymbolDuration(rhythm.split("_")[1] ?? "");
}

/**
 * vocabulary.py's remove_duplicated_symbols and the six helpers it runs, in
 * homr's order. Two homr bugs are kept because the oracle has them: a longer
 * duplicate pitch is stored under the wrong key and so never replaces the
 * first, and the multirest duration falls through (duration.ts).
 */

import {
  addRatio,
  compareRatio,
  durationOfRhythm,
  type Ratio,
  ZERO,
} from "./duration.js";
import {
  createEncodedSymbol,
  type EncodedSymbol,
  removeTuplet,
  toUpperPosition,
} from "./symbol.js";

type Chord = readonly EncodedSymbol[];
type Measure = readonly Chord[];

const isNoteOrRest = (rhythm: string): boolean =>
  rhythm.startsWith("note") || rhythm.startsWith("rest");

/** A chord token joins the next symbol to the last group. A leading chord token is swallowed and still joins the symbol after next. */
export function groupIntoChords(symbols: readonly EncodedSymbol[]): Chord[] {
  const chords: EncodedSymbol[][] = [];
  let isInChord = false;
  for (const symbol of symbols) {
    const last = chords.at(-1);
    if (symbol.rhythm === "chord") {
      isInChord = true;
    } else if (isInChord && last !== undefined) {
      last.push(symbol);
      isInChord = false;
    } else {
      chords.push([symbol]);
    }
  }
  return chords;
}

function flattenChords(chords: readonly Chord[]): EncodedSymbol[] {
  const result: EncodedSymbol[] = [];
  for (const chord of chords) {
    for (const [i, symbol] of chord.entries()) {
      if (i > 0) {
        result.push(createEncodedSymbol("chord"));
      }
      result.push(symbol);
    }
  }
  return result;
}

function groupIntoMeasures(chords: readonly Chord[]): Measure[] {
  const measures: Measure[] = [];
  let current: Chord[] = [];
  for (const chord of chords) {
    current.push(chord);
    const rhythm = chord[0]?.rhythm;
    if (
      rhythm !== undefined &&
      (rhythm.includes("barline") || rhythm.includes("repeat"))
    ) {
      measures.push(current);
      current = [];
    }
  }
  if (current.length > 0) {
    measures.push(current);
  }
  return measures;
}

/** Per chord, the shortest positive note or rest duration, summed over the measure. */
export function durationOfMeasure(measure: Measure): Ratio {
  let total = ZERO;
  for (const chord of measure) {
    let shortest = ZERO;
    for (const symbol of chord) {
      if (isNoteOrRest(symbol.rhythm)) {
        const { fraction } = durationOfRhythm(symbol.rhythm);
        if (
          compareRatio(fraction, ZERO) > 0 &&
          (compareRatio(fraction, shortest) < 0 ||
            compareRatio(shortest, ZERO) === 0)
        ) {
          shortest = fraction;
        }
      }
    }
    total = addRatio(total, shortest);
  }
  return total;
}

function typicalDurationOfMeasures(measures: readonly Measure[]): Ratio {
  const sorted = measures.map(durationOfMeasure).sort(compareRatio);
  return sorted[Math.floor(sorted.length / 2)] ?? ZERO;
}

function fixOverEagerTuplets(chords: readonly Chord[]): Chord[] {
  const measures = groupIntoMeasures(chords);
  const typical = typicalDurationOfMeasures(measures);
  return measures.flatMap((measure) =>
    compareRatio(durationOfMeasure(measure), typical) < 0
      ? measure.map((chord) => chord.map(removeTuplet))
      : measure
  );
}

function onlyKeepLowerStaffIfThereIsAClef(chords: readonly Chord[]): Chord[] {
  let hasLowerClef = false;
  return chords.map((chord, i) =>
    chord.map((symbol) => {
      if (hasLowerClef) {
        return symbol;
      }
      if (
        i < 5 &&
        symbol.rhythm.startsWith("clef") &&
        symbol.position === "lower"
      ) {
        hasLowerClef = true;
        return symbol;
      }
      return toUpperPosition(symbol);
    })
  );
}

function removeDuplicatedPitches(chord: Chord): Chord {
  const [first] = chord;
  if (chord.length <= 1 || first === undefined || !isNoteOrRest(first.rhythm)) {
    return chord;
  }
  const byPitch = new Map<string, EncodedSymbol>();
  const order: string[] = [];
  for (const symbol of chord) {
    const key = `${symbol.pitch} ${symbol.position}`;
    const kept = byPitch.get(key);
    if (kept === undefined) {
      byPitch.set(key, symbol);
      order.push(key);
    } else if (
      compareRatio(
        durationOfRhythm(symbol.rhythm).fraction,
        durationOfRhythm(kept.rhythm).fraction
      ) > 0
    ) {
      byPitch.set(symbol.pitch, symbol);
    }
  }
  return order.flatMap((key) => byPitch.get(key) ?? []);
}

function removeRedundantClefsKeysAndTimeSignatures(
  chords: readonly Chord[]
): Chord[] {
  let clefUpper = "";
  let clefLower = "";
  let key = "";
  let time = "";
  return chords.map((chord) =>
    chord.filter(({ position, rhythm }) => {
      if (rhythm.startsWith("clef")) {
        if (position === "upper") {
          const changed = rhythm !== clefUpper;
          clefUpper = rhythm;
          return changed;
        }
        const changed = rhythm !== clefLower;
        clefLower = rhythm;
        return changed;
      }
      if (rhythm.startsWith("keySignature")) {
        const changed = rhythm !== key;
        key = rhythm;
        return changed;
      }
      if (rhythm.startsWith("timeSignature")) {
        const changed = rhythm !== time;
        time = rhythm;
        return changed;
      }
      return true;
    })
  );
}

export function removeDuplicatedSymbols(
  symbols: readonly EncodedSymbol[],
  cleanupTuplets = true
): EncodedSymbol[] {
  let chords = groupIntoChords(symbols);
  if (cleanupTuplets) {
    chords = onlyKeepLowerStaffIfThereIsAClef(fixOverEagerTuplets(chords));
  }
  return flattenChords(
    removeRedundantClefsKeysAndTimeSignatures(
      chords.map(removeDuplicatedPitches)
    )
  );
}

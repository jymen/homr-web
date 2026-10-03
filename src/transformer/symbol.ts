/**
 * homr's EncodedSymbol: one musical symbol split across the six decoder
 * heads, plus the attention-derived coordinates. Two types share the
 * fields: DecodedSymbol, what the transformer emits, is closed over the
 * vocabularies; EncodedSymbol, what the post-processing and the MusicXML
 * writer handle, is open in three fields because homr rewrites them into
 * strings outside the tables ("newline" from parse_staffs, note_14 from
 * remove_tuplet, re-joined articulation and slur lists). A DecodedSymbol is
 * assignable to an EncodedSymbol; the reverse needs the guards below.
 */

import type { Point } from "../geometry/boxes.js";
import {
  type ArticulationToken,
  isToken,
  type LiftToken,
  NONOTE,
  type PitchToken,
  type PositionToken,
  type RhythmToken,
  type SlurToken,
} from "./vocabulary.js";

interface SymbolFields<R extends string, A extends string, S extends string> {
  readonly articulation: A;
  /**
   * Where the decoder was looking when it emitted the symbol, in encoder
   * canvas pixels (1280 x 256). Attention, not localisation: monotonic in
   * raster order when trustworthy, and null for symbols the port creates
   * itself (newline, chord markers).
   */
  readonly coordinates: Point | null;
  readonly lift: LiftToken;
  readonly pitch: PitchToken;
  readonly position: PositionToken;
  readonly rhythm: R;
  readonly slur: S;
}

/** Straight out of the decoder loop: every field is a table token. tokens-<n>.json decodes to this. */
export type DecodedSymbol = SymbolFields<
  RhythmToken,
  ArticulationToken,
  SlurToken
>;

/** After parse_staffs and remove_duplicated_symbols: voices.json decodes to this. */
export type EncodedSymbol = SymbolFields<string, string, string>;

/** The rhythm parse_staffs appends between staffs of a voice; not in RHYTHM_TOKENS. */
export const NEWLINE = "newline";

const CONTROL_RHYTHMS: ReadonlySet<string> = new Set(["BOS", "EOS", "PAD"]);

/**
 * EncodedSymbol(rhythm, ...) with homr's defaults: every other head
 * NONOTE and no coordinates.
 */
export function createEncodedSymbol(
  rhythm: string,
  fields: Partial<Omit<EncodedSymbol, "rhythm">> = {}
): EncodedSymbol {
  return {
    articulation: fields.articulation ?? NONOTE,
    coordinates: fields.coordinates ?? null,
    lift: fields.lift ?? NONOTE,
    pitch: fields.pitch ?? NONOTE,
    position: fields.position ?? NONOTE,
    rhythm,
    slur: fields.slur ?? NONOTE,
  };
}

/** EncodedSymbol.is_control_symbol: rhythm is BOS, EOS or PAD. */
export function isControlSymbol(symbol: EncodedSymbol): boolean {
  return CONTROL_RHYTHMS.has(symbol.rhythm);
}

/** vocabulary.has_rhythm_symbol_a_position: rhythm starts with note, rest or clef. */
export function hasPosition(rhythm: string): boolean {
  return (
    rhythm.startsWith("note") ||
    rhythm.startsWith("rest") ||
    rhythm.startsWith("clef")
  );
}

/** True when every open field is still a table token; narrows to DecodedSymbol. */
export function isDecodedSymbol(
  symbol: EncodedSymbol
): symbol is DecodedSymbol {
  return (
    isToken("rhythm", symbol.rhythm) &&
    isToken("articulation", symbol.articulation) &&
    isToken("slur", symbol.slur)
  );
}

/** EncodedSymbol.__eq__: the six heads equal, coordinates ignored. */
export function sameSymbol(a: EncodedSymbol, b: EncodedSymbol): boolean {
  return (
    a.rhythm === b.rhythm &&
    a.pitch === b.pitch &&
    a.lift === b.lift &&
    a.articulation === b.articulation &&
    a.slur === b.slur &&
    a.position === b.position
  );
}

const TUPLET_RHYTHM = /^(note|rest)_(\d+)(.*)/;

/**
 * EncodedSymbol.remove_tuplet: a duration divisible by 3, 5 or 7 (tested in
 * that order) is scaled back to the plain value it subdivides.
 */
export function removeTuplet(symbol: EncodedSymbol): EncodedSymbol {
  const match = symbol.rhythm.match(TUPLET_RHYTHM);
  if (!match) {
    return symbol;
  }
  const [, kind, digits, tail] = match;
  let duration = Number.parseInt(digits ?? "", 10);
  if (duration % 3 === 0) {
    duration = Math.floor(duration / 3) * 2;
  } else if (duration % 5 === 0) {
    duration = Math.floor(duration / 5) * 4;
  } else if (duration % 7 === 0) {
    duration = Math.floor(duration / 7) * 4;
  } else {
    return symbol;
  }
  return { ...symbol, rhythm: `${kind}_${duration}${tail}` };
}

/** EncodedSymbol.to_upper_position. */
export const toUpperPosition = (symbol: EncodedSymbol): EncodedSymbol =>
  symbol.position === "lower" ? { ...symbol, position: "upper" } : symbol;

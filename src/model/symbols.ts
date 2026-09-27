/**
 * Port of the symbol half of homr's model.py (SymbolOnStaff and its five
 * subclasses) and of note_detection.NoteheadWithStem. A discriminated
 * union replaces the class hierarchy: Staff.get_notes / get_bar_lines /
 * get_clefs become one filter on `kind`, and the values clone across the
 * Worker boundary.
 *
 * Deliberately dropped from Note: has_dot, circle_of_fifth, beams, flags.
 * In homr 0.7.0 they are set to their defaults in the constructor and only
 * read by draw_onto_image and __str__ (checked with grep over the installed
 * package); the golden files carry them as constants and the decoder
 * asserts they still are, so the day upstream starts using one the test
 * says so. NoteHeadType is likewise unused outside its own __str__ and is
 * not ported.
 */

import type { AxisBox, Ellipse, Point, RotatedBox } from "../geometry/boxes.js";
import { centerOf } from "../geometry/boxes.js";

export const STEM_DIRECTIONS = { down: "DOWN", up: "UP" } as const;
/** The enum's member names, which is how the golden JSON spells them. */
export type StemDirection =
  (typeof STEM_DIRECTIONS)[keyof typeof STEM_DIRECTIONS];

/**
 * homr carries `stem` and `stem_direction` as two nullable fields that are
 * always both null or both set (combine_noteheads_with_stems is the only
 * producer). One nullable object makes the pairing a type fact instead of a
 * convention every consumer re-checks.
 */
export interface Stem {
  readonly box: RotatedBox;
  /** UP when the stem's centre is above the notehead's, else DOWN. */
  readonly direction: StemDirection;
}

/** note_detection.NoteheadWithStem: a notehead ellipse and the stem it claimed, if any. */
export interface NoteheadWithStem {
  readonly notehead: Ellipse;
  readonly stem: Stem | null;
}

export const SYMBOL_KINDS = {
  accidental: "accidental",
  barLine: "barLine",
  clef: "clef",
  note: "note",
  rest: "rest",
} as const;

interface SymbolBase<K extends keyof typeof SYMBOL_KINDS> {
  /**
   * homr's SymbolOnStaff.center. Equal to the box centre at creation and
   * the only coordinate transform_coordinates moves: after dewarping, a
   * symbol's `center` is in canvas space while its `box` stays in page
   * space (see canvas-<n>-staff.json). So it is stored, not derived.
   */
  readonly center: Point;
  readonly kind: K;
}

export interface Note extends SymbolBase<"note"> {
  readonly box: Ellipse;
  /**
   * StaffPoint.find_position_in_unit_sizes: 0 at the bottom line's ledger
   * position below, 2 per line, odd numbers on spaces; 10 is the top line's
   * neighbour on a five-line staff.
   */
  readonly position: number;
  readonly stem: Stem | null;
}

export interface BarLine extends SymbolBase<"barLine"> {
  readonly box: RotatedBox;
}

export interface Clef extends SymbolBase<"clef"> {
  readonly box: AxisBox;
}

export interface Rest extends SymbolBase<"rest"> {
  readonly box: AxisBox;
}

export interface Accidental extends SymbolBase<"accidental"> {
  readonly box: AxisBox;
  readonly position: number;
}

export type SymbolOnStaff = Note | BarLine | Clef | Rest | Accidental;
export type SymbolKind = SymbolOnStaff["kind"];

/** Note(box, position, stem, stem_direction): center = box centre. */
export function createNote(
  box: Ellipse,
  position: number,
  stem: Stem | null
): Note {
  return { box, center: centerOf(box), kind: "note", position, stem };
}

export function createBarLine(box: RotatedBox): BarLine {
  return { box, center: centerOf(box), kind: "barLine" };
}

export function createClef(box: AxisBox): Clef {
  return { box, center: centerOf(box), kind: "clef" };
}

export function createRest(box: AxisBox): Rest {
  return { box, center: centerOf(box), kind: "rest" };
}

export function createAccidental(box: AxisBox, position: number): Accidental {
  return { box, center: centerOf(box), kind: "accidental", position };
}

/**
 * SymbolOnStaff.transform_coordinates: a copy with `center` mapped and
 * everything else, the box included, shared with the original. Phase 6's
 * dewarp is the caller.
 */
export function transformSymbol<S extends SymbolOnStaff>(
  symbol: S,
  map: (p: Point) => Point
): S {
  return { ...symbol, center: map(symbol.center) };
}

/** Staff.get_notes, get_bar_lines, get_clefs, get_all_except_notes as one typed filter. */
export function symbolsOfKind<K extends SymbolKind>(
  symbols: readonly SymbolOnStaff[],
  kind: K
): Extract<SymbolOnStaff, { kind: K }>[] {
  const out: Extract<SymbolOnStaff, { kind: K }>[] = [];
  for (const symbol of symbols) {
    if (symbol.kind === kind) {
      out.push(symbol as Extract<SymbolOnStaff, { kind: K }>);
    }
  }
  return out;
}

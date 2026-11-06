/**
 * The port's own, not homr's: which notehead on the page each decoded note
 * is. The decoder's attention point lags the note it emits by up to several
 * note spacings, so it cannot place a note by itself; segnet's noteheads sit
 * exactly where they are printed but carry no rhythm. An order-preserving
 * alignment (Needleman-Wunsch) of the staff's note events to its notehead
 * columns, scored mainly on the decoded pitch against the notehead's staff
 * position with the attention as a weak tie-break, ties each note to its
 * column. Measured in docs/design/button-tab-findings.md ("Alignment
 * instead of snapping", "H4: note columns").
 */

import type { Staff } from "../model/staff.js";
import { symbolsOfKind } from "../model/symbols.js";
import type { EncodedSymbol } from "../transformer/symbol.js";
import { type CanvasToPageX, canvasXToPage } from "./staff-image.js";

/** One notehead, `x` in page pixels, `position` StaffPoint.find_position_in_unit_sizes's (bottom line 1). */
export interface Notehead {
  readonly position: number;
  readonly x: number;
}

/** Noteheads printed one above the other: a chord, or a second voice. */
export interface NoteheadColumn {
  readonly positions: readonly number[];
  /** The mean of its heads' x. */
  readonly x: number;
}

/**
 * One decoded note, or the notes a chord token joined: what the staff
 * prints at one x. `positions` are the decoded pitches as staff positions,
 * null where the pitch is not one; `x` is the attention point in the same
 * space as the columns' x, null where the decoder gave none.
 */
export interface NoteEvent {
  readonly positions: readonly (number | null)[];
  readonly x: number | null;
}

/** Heads closer than this, in unit sizes, share a column. */
export const COLUMN_GAP_UNITS = 0.8;

export interface AlignmentCosts {
  /** The attention distance, in note spacings, stops counting here. */
  readonly attentionCap: number;
  /** The weight of the attention distance against the pitch cost. */
  readonly attentionWeight: number;
  /** Leaving an event or a column unpaired. */
  readonly gap: number;
}

/** The costs the findings measured: every note right on both pages, 99% with a tenth of the heads missing. */
export const ALIGNMENT_COSTS: AlignmentCosts = {
  attentionCap: 3,
  attentionWeight: 0.25,
  gap: 1,
};

/** Heads sorted by x and cut into columns wherever the next head is at least COLUMN_GAP_UNITS unit sizes past the last. */
export function groupNoteheadColumns(
  heads: readonly Notehead[],
  unitSize: number
): NoteheadColumn[] {
  const groups: Notehead[][] = [];
  for (const head of [...heads].sort((a, b) => a.x - b.x)) {
    const group = groups.at(-1);
    const lastX = group?.at(-1)?.x;
    if (
      group !== undefined &&
      lastX !== undefined &&
      head.x - lastX < COLUMN_GAP_UNITS * unitSize
    ) {
      group.push(head);
    } else {
      groups.push([head]);
    }
  }
  return groups.map((group) => ({
    positions: group.map((head) => head.position),
    x: group.reduce((sum, head) => sum + head.x, 0) / group.length,
  }));
}

/** 0 when a decoded pitch sits on a head of the column, 0.5 one step off, 1 otherwise. */
export function pitchCost(event: NoteEvent, column: NoteheadColumn): number {
  let best = Number.POSITIVE_INFINITY;
  for (const a of event.positions) {
    if (a !== null) {
      for (const b of column.positions) {
        best = Math.min(best, Math.abs(a - b));
      }
    }
  }
  if (best === 0) {
    return 0;
  }
  return best === 1 ? 0.5 : 1;
}

/** The smaller gap to a neighbouring column; infinite for a lone column. */
function spacingAt(columns: readonly NoteheadColumn[], j: number): number {
  const x = columns[j]?.x ?? 0;
  const before = columns[j - 1];
  const after = columns[j + 1];
  return Math.min(
    before === undefined ? Number.POSITIVE_INFINITY : x - before.x,
    after === undefined ? Number.POSITIVE_INFINITY : after.x - x
  );
}

const DIAGONAL = 0;
const SKIP_EVENT = 1;
const SKIP_COLUMN = 2;

/** The cost of pairing an event with a column: pitch, plus the weighted attention distance in spacings. */
function pairCost(
  event: NoteEvent,
  column: NoteheadColumn,
  spacing: number,
  costs: AlignmentCosts
): number {
  const distance =
    event.x === null ? 0 : Math.abs(event.x - column.x) / spacing;
  return (
    pitchCost(event, column) +
    costs.attentionWeight * Math.min(distance, costs.attentionCap)
  );
}

/** Needleman-Wunsch's table, row-major over (events + 1) by (columns + 1): the move that reached each cell. */
function alignmentMoves(
  events: readonly NoteEvent[],
  columns: readonly NoteheadColumn[],
  costs: AlignmentCosts
): Uint8Array {
  const width = columns.length + 1;
  const cells = (events.length + 1) * width;
  const total = new Float64Array(cells);
  const move = new Uint8Array(cells);
  for (let i = 1; i <= events.length; i += 1) {
    total[i * width] = i * costs.gap;
    move[i * width] = SKIP_EVENT;
  }
  for (let j = 1; j < width; j += 1) {
    total[j] = j * costs.gap;
    move[j] = SKIP_COLUMN;
  }
  const spacings = columns.map((_, k) => spacingAt(columns, k));
  for (const [e, event] of events.entries()) {
    for (const [c, column] of columns.entries()) {
      const cell = (e + 1) * width + c + 1;
      const options = [
        (total[cell - width - 1] ?? 0) +
          pairCost(event, column, spacings[c] ?? 0, costs),
        (total[cell - width] ?? 0) + costs.gap,
        (total[cell - 1] ?? 0) + costs.gap,
      ];
      const best = Math.min(...options);
      total[cell] = best;
      move[cell] = options.indexOf(best);
    }
  }
  return move;
}

/**
 * For each event, the index of the column it is paired with, or null. The
 * pairs never cross; ties prefer a pair, then leaving the event unpaired.
 */
export function alignNoteEvents(
  events: readonly NoteEvent[],
  columns: readonly NoteheadColumn[],
  costs: AlignmentCosts = ALIGNMENT_COSTS
): (number | null)[] {
  const move = alignmentMoves(events, columns, costs);
  const width = columns.length + 1;
  const paired: (number | null)[] = events.map(() => null);
  let i = events.length;
  let j = columns.length;
  while (i > 0 || j > 0) {
    const step = move[i * width + j];
    if (i > 0 && j > 0 && step === DIAGONAL) {
      paired[i - 1] = j - 1;
      i -= 1;
      j -= 1;
    } else if (i > 0 && (j === 0 || step === SKIP_EVENT)) {
      i -= 1;
    } else {
      j -= 1;
    }
  }
  return paired;
}

const CLEF = /^clef_([CFG])(\d)$/;
/** The diatonic step of the note each clef names: C4, F3, G4. */
const CLEF_NOTE_STEP: Readonly<Record<string, number>> = {
  C: 28,
  F: 24,
  G: 32,
};
const PITCH = /^([A-G])(\d)$/;
const STEPS = "CDEFGAB";

/** octave * 7 + the letter's index from C, or null for anything that is not a pitch. A lift does not move the head. */
export function diatonicStep(pitch: string): number | null {
  const match = pitch.match(PITCH);
  if (!match) {
    return null;
  }
  const [, letter = "", octave = ""] = match;
  return Number.parseInt(octave, 10) * 7 + STEPS.indexOf(letter);
}

/**
 * What to subtract from a diatonic step to get a staff position under this
 * clef: the clef's note sits on its line, line n at position 2n - 1. 29 for
 * G2, 17 for F4. Undefined for a token that is not a clef.
 */
export function clefOffset(rhythm: string): number | undefined {
  const match = rhythm.match(CLEF);
  if (!match) {
    return undefined;
  }
  const [, letter = "", line = ""] = match;
  const step = CLEF_NOTE_STEP[letter];
  return step === undefined ? undefined : step - (2 * Number(line) - 1);
}

/** A staff without a decoded clef is read as treble, as homr's own default. */
const TREBLE_OFFSET = 29;

interface StaffEvents {
  readonly events: NoteEvent[];
  /** Per event, the indices of its notes in the staff's symbols. */
  readonly members: number[][];
}

interface PendingEvent {
  readonly members: number[];
  readonly positions: (number | null)[];
  x: number | null;
}

const minOrEither = (a: number | null, b: number | null): number | null =>
  a === null || b === null ? (a ?? b) : Math.min(a, b);

/** The note's staff position under `offset`, and its attention x on the page. */
function placeNote(
  symbol: EncodedSymbol,
  offset: number,
  toPageX: CanvasToPageX
): { readonly position: number | null; readonly x: number | null } {
  const step = diatonicStep(symbol.pitch);
  return {
    position: step === null ? null : step - offset,
    x:
      symbol.coordinates === null
        ? null
        : canvasXToPage(toPageX, symbol.coordinates.x),
  };
}

/**
 * The staff's notes as events, a chord token joining the next note to the
 * event before it; a rest is no event. Positions follow the clef in force,
 * x is the leftmost attention point of the event, mapped to page x.
 */
export function noteEventsOfStaff(
  symbols: readonly EncodedSymbol[],
  toPageX: CanvasToPageX
): StaffEvents {
  const pending: PendingEvent[] = [];
  let offset = TREBLE_OFFSET;
  let joining = false;
  let current: PendingEvent | undefined;
  for (const [index, symbol] of symbols.entries()) {
    if (symbol.rhythm === "chord") {
      joining = true;
      continue;
    }
    offset = clefOffset(symbol.rhythm) ?? offset;
    if (symbol.rhythm.startsWith("note")) {
      const { position, x } = placeNote(symbol, offset, toPageX);
      if (joining && current !== undefined) {
        current.members.push(index);
        current.positions.push(position);
        current.x = minOrEither(current.x, x);
      } else {
        current = { members: [index], positions: [position], x };
        pending.push(current);
      }
    } else if (!joining) {
      current = undefined;
    }
    joining = false;
  }
  return {
    events: pending.map(({ positions, x }) => ({ positions, x })),
    members: pending.map(({ members }) => members),
  };
}

/**
 * The staff's decoded symbols with `column` set on every note the alignment
 * paired with one of the staff's notehead columns, as page x over
 * `pageWidth`. `staff` is the page-space staff, with its symbols. A grand
 * staff is returned as it is: its positions count across both staffs and
 * the pitch cost was never measured there.
 */
export function withNoteColumns<S extends EncodedSymbol>(
  symbols: readonly S[],
  staff: Staff,
  toPageX: CanvasToPageX,
  pageWidth: number
): S[] {
  const out = [...symbols];
  if (staff.isGrandstaff) {
    return out;
  }
  const columns = groupNoteheadColumns(
    symbolsOfKind(staff.symbols, "note").map((note) => ({
      position: note.position,
      x: note.center.x,
    })),
    staff.averageUnitSize
  );
  const { events, members } = noteEventsOfStaff(symbols, toPageX);
  for (const [k, j] of alignNoteEvents(events, columns).entries()) {
    const column = j === null ? undefined : columns[j];
    if (column !== undefined) {
      for (const index of members[k] ?? []) {
        const symbol = out[index];
        if (symbol !== undefined) {
          out[index] = { ...symbol, column: column.x / pageWidth };
        }
      }
    }
  }
  return out;
}

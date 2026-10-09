/**
 * Pitch from a tab reading, given what the page prints beside it: the open
 * strings' tuning and the capo. Pure arithmetic: open string + capo + fret.
 *
 * A capo raises every string, the banjo's short fifth string included: tab
 * software numbers that string's frets from the nut, and players capo it with
 * a spike at the same fret, so the fifth string (the bottom line in banjo tab,
 * g4 in open G) takes the capo like the rest. The prototype's pitches matched
 * the staff above the tab on that rule (.scratch/tab-sketch, Cripple Creek,
 * capo 2).
 */

import type { TabEvent, TabNote, TabReading } from "./read.js";

/** `strings` are MIDI numbers of the open strings, top tab line first; `capo` is a fret, 0 for none. */
export interface Tuning {
  readonly capo: number;
  readonly strings: readonly number[];
}

export interface PitchedNote extends TabNote {
  readonly midi: number;
}

export interface PitchedEvent {
  readonly notes: readonly [PitchedNote, ...PitchedNote[]];
  readonly x: number;
}

const STEPS: Readonly<Record<string, number>> = {
  A: 9,
  B: 11,
  C: 0,
  D: 2,
  E: 4,
  F: 5,
  G: 7,
};
const PITCH = /^([A-G])([#b]?)(-?\d)$/;

/** MIDI number of a scientific pitch name ("G4", "C#4", "Bb3"; C4 is 60). Throws a RangeError on anything else. */
export function midiOfPitch(name: string): number {
  const match = name.trim().match(PITCH);
  if (!match) {
    throw new RangeError(`not a pitch name: ${JSON.stringify(name)}`);
  }
  const [, letter, accidental, octave] = match;
  const alter = { "#": 1, b: -1 }[accidental ?? ""] ?? 0;
  return 12 * (Number(octave) + 1) + (STEPS[letter ?? ""] ?? 0) + alter;
}

/**
 * Every event of `reading` with each fret's MIDI pitch. Throws a RangeError
 * when the tuning names a string count other than the system's lines, or the
 * capo is not a fret from 0 to 24.
 */
export function pitchTab(
  reading: Pick<TabReading, "events" | "lines">,
  tuning: Tuning
): PitchedEvent[] {
  if (tuning.strings.length !== reading.lines) {
    throw new RangeError(
      `${tuning.strings.length} tuned strings for a ${reading.lines}-line tab`
    );
  }
  if (!Number.isInteger(tuning.capo) || tuning.capo < 0 || tuning.capo > 24) {
    throw new RangeError(`capo ${tuning.capo} is not a fret from 0 to 24`);
  }
  const pitch = (note: TabNote): PitchedNote => ({
    ...note,
    midi: (tuning.strings[note.string - 1] as number) + tuning.capo + note.fret,
  });
  return reading.events.map(
    (event: TabEvent): PitchedEvent => ({
      notes: event.notes.map(pitch) as [PitchedNote, ...PitchedNote[]],
      x: event.x,
    })
  );
}

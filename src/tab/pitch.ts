/**
 * Pitch from a tab reading, given what the page prints beside it: the open
 * strings' tuning and the capo, read off the page (src/tab/text.ts) or
 * passed by the caller. Pure arithmetic: open string + capo + fret.
 *
 * A capo raises every string, the banjo's short fifth string included: tab
 * software numbers that string's frets from the nut, and players capo it with
 * a spike at the same fret, so the fifth string (the bottom line in banjo tab,
 * g4 in open G) takes the capo like the rest. The prototype's pitches matched
 * the staff above the tab on that rule (.scratch/tab-sketch, Cripple Creek,
 * capo 2).
 */

import type { TabLineCount } from "./detect.js";
import type { TabEvent, TabNote, TabReading } from "./read.js";
import { midiOfPitch, standardStrings } from "./tuning.js";

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

/**
 * The tuning and capo the page printed for this system, in `pitchTab`'s
 * terms, or undefined when no tuning that fits it was read. No capo printed
 * is capo 0: a page names its capo when there is one.
 */
export function tuningOf(
  reading: Pick<TabReading, "capo" | "tuning">
): Tuning | undefined {
  const { capo, tuning } = reading;
  return tuning?.status === "read"
    ? { capo: capo?.fret ?? 0, strings: tuning.strings.map(midiOfPitch) }
    : undefined;
}

/**
 * The standard tuning for a line count, no capo: mandolin GDAE, five-string
 * banjo gDGBD, guitar EADGBE. Never applied by the library; a caller offers
 * it when the page names no tuning.
 */
export const standardTuning = (lines: TabLineCount): Tuning => ({
  capo: 0,
  strings: standardStrings(lines).map(midiOfPitch),
});

/**
 * Every event of `reading` with each fret's MIDI pitch, in `tuning`, or in
 * the tuning and capo read off the page when `tuning` is omitted. Throws a
 * RangeError when the tuning names a string count other than the system's
 * lines, the capo is not a fret from 0 to 24, or no tuning is given and none
 * that fits was read.
 */
export function pitchTab(
  reading: Pick<TabReading, "events" | "lines"> &
    Partial<Pick<TabReading, "capo" | "tuning">>,
  tuning: Tuning | undefined = tuningOf(reading)
): PitchedEvent[] {
  if (tuning === undefined) {
    throw new RangeError(
      `no tuning read for this ${reading.lines}-line tab${reading.tuning === undefined ? "" : ` (${reading.tuning.status}: ${JSON.stringify(reading.tuning.text)})`}; pass one, e.g. standardTuning(${reading.lines})`
    );
  }
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

/**
 * From tab readings to a tune, through what processing a tab needs to know
 * first. By design, not as a fallback: a tab gives positions and an order,
 * and five facts turn them into music.
 *
 * To turn positions into pitches:
 *   1. instrument: its string count, a short string (a five-string banjo's
 *      fifth) and whether strings are doubled in courses;
 *   2. tuning: one pitch per string;
 *   3. capo: 0 when the page prints none, reported as assumed.
 * To turn the order into a tune:
 *   4. meter;
 *   5. note durations, from the stems and beams; a page without rhythm
 *      marks needs them from the musician.
 *
 * Each is read from the page (with its source and confidence), assumed (with
 * the default, only the capo), or missing (with what was read and the
 * candidates). `tabPrerequisites` says which; `resolveTab` builds the tune
 * only when none is missing, and otherwise answers the needs, so the app can
 * ask the musician for exactly those and call `resolveTab` again with the
 * answers, on the same readings: the page is never read twice.
 *
 * Pure: no image, no model.
 */

import type { TabLineCount } from "./detect.js";
import type { TabNote, TabReading, TabSlurKind } from "./read.js";
import type { TabBarline, TabStem } from "./rhythm.js";
import {
  midiOfPitch,
  type TabInstrumentKind,
  type TabInstrumentName,
  tuningCandidates,
} from "./tuning.js";

// ---------------------------------------------------------------- values

export const NOTE_VALUES = [
  "whole",
  "half",
  "quarter",
  "eighth",
  "16th",
  "32nd",
] as const;
export type NoteValue = (typeof NOTE_VALUES)[number];

/** A written duration: a value and its dots. */
export interface Duration {
  readonly dots: 0 | 1 | 2;
  readonly value: NoteValue;
}

export interface TimeSignature {
  readonly den: number;
  readonly num: number;
}

/**
 * What the instrument means for the frets: `strings` is the line count,
 * `shortStrings` the strings (1 the top line) that start above the nut, at
 * `startFret` (a five-string banjo's fifth starts at the fifth fret), and
 * `courses` 2 when each string is a doubled pair (mandolin).
 */
export interface TabInstrument {
  readonly courses: 1 | 2;
  readonly kind: TabInstrumentKind;
  readonly shortStrings: readonly {
    readonly startFret: number;
    readonly string: number;
  }[];
  readonly strings: TabLineCount;
}

export const TAB_INSTRUMENTS: Readonly<
  Record<TabInstrumentKind, TabInstrument>
> = {
  banjo5: {
    courses: 1,
    kind: "banjo5",
    shortStrings: [{ startFret: 5, string: 5 }],
    strings: 5,
  },
  guitar: { courses: 1, kind: "guitar", shortStrings: [], strings: 6 },
  mandolin: { courses: 2, kind: "mandolin", shortStrings: [], strings: 4 },
  tenorBanjo: {
    courses: 1,
    kind: "tenorBanjo",
    shortStrings: [],
    strings: 4,
  },
};

// ---------------------------------------------------------------- prerequisites

/**
 * Where a value came from: the page's text read by OCR, the page's drawing
 * (line count, stems, beams, the time signature's digits), the PDF's own text
 * layer, or the musician's answer.
 */
export type PrerequisiteSource =
  | "page_text"
  | "page_marks"
  | "text_layer"
  | "musician";

/**
 * One fact the tune needs. `evidence` says, in words, what decided it.
 * `missing` keeps what was read (`read`, null when nothing; `confidence` 0
 * then) and the values to offer, most likely first; `candidates` may be
 * empty when nothing can be offered.
 */
export type Prerequisite<T> =
  | {
      readonly confidence: number;
      readonly evidence: string;
      readonly source: PrerequisiteSource;
      readonly status: "read";
      readonly value: T;
    }
  | {
      readonly evidence: string;
      readonly status: "assumed";
      readonly value: T;
    }
  | {
      readonly candidates: readonly T[];
      readonly confidence: number;
      readonly evidence: string;
      readonly read: string | null;
      readonly status: "missing";
    };

/** A bar's length against the meter, in quarter notes; `open` counts its notes with no stem, which share what the bar has left. */
export interface TabBarCheck {
  readonly bar: number;
  readonly expected: number;
  readonly fits: boolean;
  readonly open: number;
  /**
   * A first bar no longer than half the meter, all its notes stemmed: an
   * upbeat. The last bar may then be short by as much.
   */
  readonly pickup: boolean;
  readonly read: number;
  readonly system: number;
}

/** An event's time: a written duration, or `grace`, a grace note taking no time before the next event. */
export type EventDuration = Duration | "grace";

/** Every event's time, per system then per event, in the readings' order. */
export type TabDurations = readonly (readonly EventDuration[])[];

/** The five prerequisites, and the bar checks behind `durations`. */
export interface TabPrerequisites {
  readonly bars: readonly TabBarCheck[];
  readonly capo: Prerequisite<number>;
  readonly durations: Prerequisite<TabDurations>;
  readonly instrument: Prerequisite<TabInstrument>;
  readonly meter: Prerequisite<TimeSignature>;
  /** Open strings as pitch names ("G4"), top line first. */
  readonly tuning: Prerequisite<readonly string[]>;
}

export type TabPrerequisiteField = Exclude<keyof TabPrerequisites, "bars">;

/** A missing prerequisite, as the app asks the musician for it. */
export type TabNeed = {
  readonly [F in TabPrerequisiteField]: { readonly field: F } & Extract<
    TabPrerequisites[F],
    { readonly status: "missing" }
  >;
}[TabPrerequisiteField];

/**
 * The musician's answers, any subset. An answer replaces what was read.
 * `durations` either gives one value to every note (a page with no rhythm
 * marks) or corrects single notes, by system and event index.
 */
export interface TabAnswers {
  readonly capo?: number;
  readonly durations?:
    | { readonly uniform: Duration }
    | {
        readonly events: readonly {
          readonly duration: Duration;
          readonly event: number;
          readonly system: number;
        }[];
      };
  readonly instrument?: TabInstrumentKind;
  readonly meter?: TimeSignature;
  /** Pitch names, top line first ("D4", "B3", "G3", "D3", "G4"). */
  readonly tuning?: readonly string[];
}

// ---------------------------------------------------------------- the tune

/**
 * A note of the tune. `techniques` sit on the note a connection starts from
 * and lead to the next note on the string (MusicXML's convention); a tie
 * shows as `tieToNext` on the first note and `tieFromPrevious` on the
 * second, which is not plucked again.
 */
export interface TabTuneNote {
  readonly fret: number;
  readonly midi: number;
  readonly string: number;
  readonly techniques: readonly Exclude<TabSlurKind, "tie">[];
  readonly tieFromPrevious: boolean;
  readonly tieToNext: boolean;
}

export interface TabTuneEvent {
  readonly duration: Duration;
  /** Where it was read: system and event index in the readings. */
  readonly event: number;
  /** Grace notes struck before this event, taking none of its time, each group struck together. */
  readonly graces: readonly (readonly [TabTuneNote, ...TabTuneNote[]])[];
  readonly notes: readonly [TabTuneNote, ...TabTuneNote[]];
  readonly system: number;
}

export interface TabTuneBar {
  readonly events: readonly TabTuneEvent[];
  readonly pickup: boolean;
  readonly repeatEnd: boolean;
  readonly repeatStart: boolean;
  /** The bar opens a new line of the page. */
  readonly systemBreakBefore: boolean;
}

/** A playable tune: every prerequisite settled, every bar the meter's length, but an upbeat of at most half a bar and a last bar that completes it. */
export interface TabTune {
  readonly bars: readonly TabTuneBar[];
  readonly capo: number;
  readonly instrument: TabInstrument;
  readonly meter: TimeSignature;
  /** Open strings, top line first, as names and MIDI numbers, before the capo. */
  readonly tuning: {
    readonly midi: readonly number[];
    readonly names: readonly string[];
  };
}

export type TabResolution =
  | {
      readonly ok: true;
      readonly prerequisites: TabPrerequisites;
      readonly tune: TabTune;
    }
  | {
      readonly needs: readonly [TabNeed, ...TabNeed[]];
      readonly ok: false;
      readonly prerequisites: TabPrerequisites;
    };

// ---------------------------------------------------------------- durations

/** Ticks in a whole note: down to a double-dotted 32nd. */
const WHOLE = 3840;

const VALUE_TICKS: Readonly<Record<NoteValue, number>> = {
  "16th": WHOLE / 16,
  "32nd": WHOLE / 32,
  eighth: WHOLE / 8,
  half: WHOLE / 2,
  quarter: WHOLE / 4,
  whole: WHOLE,
};

export const ticksOf = (d: Duration): number =>
  VALUE_TICKS[d.value] * (2 - 2 ** -d.dots);

const barTicks = (meter: TimeSignature) => (WHOLE * meter.num) / meter.den;
const quarters = (ticks: number) => ticks / (WHOLE / 4);

/** Beams under a stem to a value: none a quarter, one an eighth, two a sixteenth, three a 32nd. */
const BEAMED: readonly NoteValue[] = ["quarter", "eighth", "16th", "32nd"];

/** What a stem says, or undefined for no stem: a half note or longer, the bar decides. */
export function durationOfStem(stem: TabStem): Duration | undefined {
  if (stem.kind === "none") {
    return;
  }
  const value = BEAMED[stem.beams];
  return value === undefined
    ? undefined
    : { dots: Math.min(2, stem.dots) as 0 | 1 | 2, value };
}

const lengthOf = (d: EventDuration) => (d === "grace" ? 0 : ticksOf(d));

/** Long values an unstemmed note may take, longest first. */
const LONG: readonly Duration[] = [
  { dots: 0, value: "whole" },
  { dots: 1, value: "half" },
  { dots: 0, value: "half" },
];

// ---------------------------------------------------------------- bars

interface RawBar {
  readonly events: readonly number[];
  readonly repeatEnd: boolean;
  readonly repeatStart: boolean;
  readonly system: number;
  readonly systemBreakBefore: boolean;
}

/** The events of each system cut at its barlines; bars holding no event are dropped. */
export function barsOf(
  systems: readonly Pick<TabReading, "barlines" | "events">[]
): RawBar[] {
  const bars: RawBar[] = [];
  for (const [system, reading] of systems.entries()) {
    const lines = [...reading.barlines].sort((a, b) => a.x - b.x);
    const edges: (TabBarline | undefined)[] = [undefined, ...lines, undefined];
    let first = true;
    for (let k = 0; k + 1 < edges.length; k += 1) {
      const left = edges[k];
      const right = edges[k + 1];
      const events = reading.events.flatMap((event, index) =>
        (left === undefined || event.x > left.x) &&
        (right === undefined || event.x < right.x)
          ? [index]
          : []
      );
      if (events.length === 0) {
        continue;
      }
      bars.push({
        events,
        repeatEnd: right?.repeatEnd ?? false,
        repeatStart: left?.repeatStart ?? false,
        system,
        systemBreakBefore: first && system > 0,
      });
      first = false;
    }
  }
  return bars;
}

interface Fitted {
  readonly checks: TabBarCheck[];
  readonly durations: EventDuration[][];
}

/** Read values: a duration, a grace, or undefined for a note with no stem. */
type ReadDurations = readonly (readonly (EventDuration | undefined)[])[];

/**
 * Each bar measured against the meter, and its unstemmed notes given what the
 * bar has left when that is one long value shared evenly. `read` holds the
 * stemmed values, undefined for no stem.
 */
function fitBars(
  bars: readonly RawBar[],
  read: ReadDurations,
  meter: TimeSignature
): Fitted {
  const durations = read.map((system) =>
    system.map((d): EventDuration => d ?? { dots: 0, value: "half" })
  );
  const full = barTicks(meter);
  const lengthOfBar = (bar: RawBar) =>
    bar.events.reduce((sum, e) => {
      const d = read[bar.system]?.[e];
      return sum + (d === undefined ? Number.NaN : lengthOf(d));
    }, 0);
  const first = bars[0] === undefined ? Number.NaN : lengthOfBar(bars[0]);
  const upbeat = first > 0 && first <= full / 2 ? first : undefined;
  const checks = bars.map((bar, index): TabBarCheck => {
    const values = bar.events.map((e) => read[bar.system]?.[e]);
    const known = values.reduce((sum, d) => sum + (d ? lengthOf(d) : 0), 0);
    const open = values.filter((d) => d === undefined).length;
    const share = open === 0 ? 0 : (full - known) / open;
    const long = LONG.find((d) => ticksOf(d) === share);
    if (long !== undefined) {
      for (const [k, e] of bar.events.entries()) {
        if (values[k] === undefined) {
          (durations[bar.system] as EventDuration[])[e] = long;
        }
      }
    }
    const pickup = index === 0 && upbeat !== undefined && bars.length > 1;
    const closing =
      index === bars.length - 1 &&
      index > 0 &&
      open === 0 &&
      upbeat !== undefined &&
      known + upbeat === full;
    return {
      bar: index,
      expected: quarters(full),
      fits:
        open === 0 ? known === full || pickup || closing : long !== undefined,
      open,
      pickup,
      read: quarters(known + (long === undefined ? 0 : open * ticksOf(long))),
      system: bar.system,
    };
  });
  return { checks, durations };
}

// ---------------------------------------------------------------- deciding

const METERS: readonly TimeSignature[] = [
  { den: 4, num: 4 },
  { den: 4, num: 2 },
  { den: 4, num: 3 },
  { den: 8, num: 6 },
  { den: 2, num: 2 },
  { den: 8, num: 9 },
  { den: 8, num: 12 },
];

/** Stems found on less than this share of the events: the page has no rhythm marks. */
const MIN_STEMMED = 0.25;

const meterText = (m: TimeSignature) => `${m.num}/${m.den}`;

/** Where the page's text was read: the PDF's text layer when the frets were, the recogniser otherwise. */
const textSource = (systems: readonly TabReading[]): PrerequisiteSource =>
  systems.some((s) => s.readFrom === "text_layer") ? "text_layer" : "page_text";

function instrumentOf(
  systems: readonly TabReading[]
): Prerequisite<TabInstrument> {
  const lines = systems[0]?.lines ?? 5;
  const named = systems.find((s) => s.instrumentName !== undefined)
    ?.instrumentName as TabInstrumentName | undefined;
  const byCount = Object.values(TAB_INSTRUMENTS).filter(
    (i) => i.strings === lines
  );
  if (named !== undefined && TAB_INSTRUMENTS[named.kind].strings === lines) {
    return {
      confidence: named.confidence,
      evidence: `"${named.text}" over ${lines}-line tab`,
      source: textSource(systems),
      status: "read",
      value: TAB_INSTRUMENTS[named.kind],
    };
  }
  const tuning = systems.find((s) => s.tuning?.status === "read")?.tuning;
  if (lines === 5 && tuning?.status === "read") {
    const [top, , , fourth, fifth] = tuning.strings.map(midiOfPitch);
    if (
      fifth !== undefined &&
      fourth !== undefined &&
      top !== undefined &&
      fifth > fourth
    ) {
      return {
        confidence: tuning.confidence,
        evidence: `5 lines and a tuning whose bottom line, ${tuning.strings[4]}, is above the line over it ("${tuning.text}"): a five-string banjo's short string`,
        source: textSource(systems),
        status: "read",
        value: TAB_INSTRUMENTS.banjo5,
      };
    }
  }
  return {
    candidates: byCount,
    confidence: named?.confidence ?? 0,
    evidence:
      named === undefined
        ? `${lines} lines and no instrument named on the page`
        : `"${named.text}" names a ${TAB_INSTRUMENTS[named.kind].strings}-string instrument over a ${lines}-line tab`,
    read: named?.text ?? null,
    status: "missing",
  };
}

function tuningEvidence(read: TabReading["tuning"], lines: number): string {
  if (read === undefined) {
    return "no tuning printed";
  }
  return read.status === "unknown_name"
    ? `"${read.text}" names a tuning the reader does not know`
    : `"${read.text}" is a ${read.strings.length}-string tuning over a ${lines}-line tab`;
}

function tuningOfSystems(
  systems: readonly TabReading[],
  instrument: TabInstrument | undefined
): Prerequisite<readonly string[]> {
  const lines = systems[0]?.lines ?? 5;
  const candidates = tuningCandidates(instrument?.kind ?? lines);
  const read = systems.map((s) => s.tuning).find((t) => t !== undefined);
  if (read?.status === "read") {
    return {
      confidence: read.confidence,
      evidence: `"${read.text}"${read.source === "named" ? ", a named tuning" : ""}`,
      source: textSource(systems),
      status: "read",
      value: read.strings,
    };
  }
  return {
    candidates,
    confidence: read?.confidence ?? 0,
    evidence: tuningEvidence(read, lines),
    read: read?.text ?? null,
    status: "missing",
  };
}

function capoOfSystems(systems: readonly TabReading[]): Prerequisite<number> {
  const capo = systems.map((s) => s.capo).find((c) => c !== undefined);
  return capo === undefined
    ? { evidence: "no capo printed", status: "assumed", value: 0 }
    : {
        confidence: capo.confidence,
        evidence: `"${capo.text}"`,
        source: textSource(systems),
        status: "read",
        value: capo.fret,
      };
}

function meterOfSystems(
  systems: readonly TabReading[],
  bars: readonly RawBar[],
  read: ReadDurations
): Prerequisite<TimeSignature> {
  const printed = systems.find(
    (s) => s.timeSignature !== undefined
  )?.timeSignature;
  if (printed !== undefined) {
    return {
      confidence: 1,
      evidence: `${printed.text} printed at the start of a system`,
      source:
        textSource(systems) === "text_layer" ? "text_layer" : "page_marks",
      status: "read",
      value: { den: printed.den, num: printed.num },
    };
  }
  const ranked = METERS.map((meter) => ({
    fit: fitBars(bars, read, meter).checks.filter((c) => c.fits).length,
    meter,
  })).sort((a, b) => b.fit - a.fit);
  return {
    candidates: ranked.map((r) => r.meter),
    confidence: 0,
    evidence: `no time signature printed; bars that fit: ${ranked
      .map((r) => `${meterText(r.meter)} ${r.fit}/${bars.length}`)
      .join(", ")}`,
    read: null,
    status: "missing",
  };
}

function durationsOf(
  systems: readonly TabReading[],
  bars: readonly RawBar[],
  read: ReadDurations,
  meter: TimeSignature | undefined,
  answered: boolean
): {
  readonly checks: TabBarCheck[];
  readonly prerequisite: Prerequisite<TabDurations>;
} {
  const events = systems.reduce((n, s) => n + s.events.length, 0);
  const stemmed = read.flat().filter((d) => d !== undefined).length;
  const offer: Duration[] = [
    { dots: 0, value: "eighth" },
    { dots: 0, value: "quarter" },
  ];
  if (!answered && stemmed < MIN_STEMMED * events) {
    const uniform = (d: Duration): TabDurations =>
      read.map((row) => row.map((e) => (e === "grace" ? e : d)));
    return {
      checks: [],
      prerequisite: {
        candidates: offer.map(uniform),
        confidence: 0,
        evidence: `stems under ${stemmed} of ${events} notes: the page draws no rhythm; the candidates give every note ${offer.map((d) => `an ${d.value}`).join(", or ")}`,
        read: null,
        status: "missing",
      },
    };
  }
  if (meter === undefined) {
    return {
      checks: [],
      prerequisite: {
        candidates: [],
        confidence: 0,
        evidence: "durations are checked against the meter, which is missing",
        read: `stems under ${stemmed} of ${events} notes`,
        status: "missing",
      },
    };
  }
  const { checks, durations } = fitBars(bars, read, meter);
  const fitting = checks.filter((c) => c.fits).length;
  const unfit = checks.filter((c) => !c.fits);
  const evidence = `${fitting} of ${checks.length} bars sum to ${meterText(meter)}`;
  return {
    checks,
    prerequisite:
      unfit.length === 0
        ? {
            confidence: 1,
            evidence,
            source: answered ? "musician" : "page_marks",
            status: "read",
            value: durations,
          }
        : {
            candidates: [durations],
            confidence: checks.length === 0 ? 0 : fitting / checks.length,
            evidence: `${evidence}; not: ${unfit
              .map(
                (c) =>
                  `bar ${c.bar + 1} (system ${c.system + 1}) holds ${c.read} of ${c.expected} quarters`
              )
              .join(", ")}`,
            read: `stems under ${stemmed} of ${events} notes`,
            status: "missing",
          },
  };
}

const settled = <T>(p: Prerequisite<T>): T | undefined =>
  p.status === "missing" ? undefined : p.value;

const answeredWith = <T>(value: T): Prerequisite<T> => ({
  confidence: 1,
  evidence: "the musician's answer",
  source: "musician",
  status: "read",
  value,
});

/** The stemmed values, with the musician's durations laid over them. */
function readDurations(
  systems: readonly TabReading[],
  answer: TabAnswers["durations"]
): (EventDuration | undefined)[][] {
  const read = systems.map((s) =>
    s.events.map((e): EventDuration | undefined =>
      e.grace ? "grace" : durationOfStem(e.stem)
    )
  );
  if (answer === undefined) {
    return read;
  }
  if ("uniform" in answer) {
    return read.map((system) =>
      system.map((d) => (d === "grace" ? d : answer.uniform))
    );
  }
  for (const { duration, event, system } of answer.events) {
    const row = read[system];
    if (row !== undefined && event >= 0 && event < row.length) {
      row[event] = duration;
    }
  }
  return read;
}

/**
 * What the readings say about the five prerequisites, with the musician's
 * answers laid over them. An answer that does not fit the tab (a tuning for
 * another string count, an instrument with another, a capo off the neck) is
 * not taken: the prerequisite stays missing and its evidence says why.
 */
export function tabPrerequisites(
  systems: readonly TabReading[],
  answers: TabAnswers = {}
): TabPrerequisites {
  const lines = systems[0]?.lines ?? 5;
  const instrument =
    answers.instrument !== undefined &&
    TAB_INSTRUMENTS[answers.instrument].strings === lines
      ? answeredWith(TAB_INSTRUMENTS[answers.instrument])
      : refused(instrumentOf(systems), answers.instrument, "instrument");
  const tuning =
    answers.tuning !== undefined &&
    answers.tuning.length === lines &&
    answers.tuning.every(isPitch)
      ? answeredWith(answers.tuning)
      : refused(
          tuningOfSystems(systems, settled(instrument)),
          answers.tuning,
          "tuning"
        );
  const capo =
    answers.capo !== undefined &&
    Number.isInteger(answers.capo) &&
    answers.capo >= 0 &&
    answers.capo <= 12
      ? answeredWith(answers.capo)
      : capoOfSystems(systems);
  const bars = barsOf(systems);
  const read = readDurations(systems, answers.durations);
  const meter =
    answers.meter !== undefined && barTicks(answers.meter) > 0
      ? answeredWith(answers.meter)
      : meterOfSystems(systems, bars, read);
  const durations = durationsOf(
    systems,
    bars,
    read,
    settled(meter),
    answers.durations !== undefined
  );
  return {
    bars: durations.checks,
    capo,
    durations: durations.prerequisite,
    instrument,
    meter,
    tuning,
  };
}

const isPitch = (name: string) => {
  try {
    midiOfPitch(name);
    return true;
  } catch {
    return false;
  }
};

/** A prerequisite as read, its evidence noting an answer that was not taken. */
function refused<T>(
  p: Prerequisite<T>,
  answer: unknown,
  field: string
): Prerequisite<T> {
  return answer === undefined || p.status !== "missing"
    ? p
    : {
        ...p,
        evidence: `${p.evidence}; the answer ${JSON.stringify(answer)} does not fit this ${field}`,
      };
}

const FIELDS: readonly TabPrerequisiteField[] = [
  "instrument",
  "tuning",
  "capo",
  "meter",
  "durations",
];

/** The prerequisites still missing, in the order to ask them: instrument first, durations last. */
export function needsOf(prerequisites: TabPrerequisites): TabNeed[] {
  return FIELDS.flatMap((field) => {
    const p = prerequisites[field];
    return p.status === "missing" ? [{ field, ...p } as TabNeed] : [];
  });
}

/**
 * The tune the readings make with the musician's answers, or the
 * prerequisites still missing. Call it with no answers first; ask for each
 * need; call it again with the answers on the same readings.
 */
export function resolveTab(
  systems: readonly TabReading[],
  answers: TabAnswers = {}
): TabResolution {
  const prerequisites = tabPrerequisites(systems, answers);
  const needs = needsOf(prerequisites);
  const [first, ...rest] = needs;
  if (first !== undefined) {
    return { needs: [first, ...rest], ok: false, prerequisites };
  }
  const { capo, durations, instrument, meter, tuning } = prerequisites;
  if (
    instrument.status === "missing" ||
    tuning.status === "missing" ||
    capo.status === "missing" ||
    meter.status === "missing" ||
    durations.status === "missing"
  ) {
    throw new Error("unreachable: a missing prerequisite with no need");
  }
  return {
    ok: true,
    prerequisites,
    tune: tuneOf(systems, {
      capo: capo.value,
      durations: durations.value,
      instrument: instrument.value,
      meter: meter.value,
      tuning: tuning.value,
    }),
  };
}

interface Settled {
  readonly capo: number;
  readonly durations: TabDurations;
  readonly instrument: TabInstrument;
  readonly meter: TimeSignature;
  readonly tuning: readonly string[];
}

function tuneOf(systems: readonly TabReading[], known: Settled): TabTune {
  const open = known.tuning.map(midiOfPitch);
  const techniques = new Map<string, Exclude<TabSlurKind, "tie">[]>();
  const tiedTo = new Set<string>();
  const tiedFrom = new Set<string>();
  const key = (system: number, event: number, string: number) =>
    `${system}:${event}:${string}`;
  for (const [system, reading] of systems.entries()) {
    for (const slur of reading.slurs) {
      if (slur.kind === "tie") {
        tiedFrom.add(key(system, slur.from, slur.string));
        tiedTo.add(key(system, slur.to, slur.string));
        continue;
      }
      const at = key(system, slur.from, slur.string);
      techniques.set(at, [...(techniques.get(at) ?? []), slur.kind]);
    }
  }
  const raw = barsOf(systems);
  const { checks } = fitBars(
    raw,
    known.durations.map((row) => [...row]),
    known.meter
  );
  const noteOf =
    (system: number, event: number) =>
    (note: TabNote): TabTuneNote => ({
      fret: note.fret,
      midi: (open[note.string - 1] as number) + known.capo + note.fret,
      string: note.string,
      techniques: techniques.get(key(system, event, note.string)) ?? [],
      tieFromPrevious: tiedTo.has(key(system, event, note.string)),
      tieToNext: tiedFrom.has(key(system, event, note.string)),
    });
  const bars = raw.map((bar, index): TabTuneBar => {
    const events: TabTuneEvent[] = [];
    let graces: [TabTuneNote, ...TabTuneNote[]][] = [];
    for (const event of bar.events) {
      const read = (systems[bar.system] as TabReading).events[event]?.notes;
      if (read === undefined) {
        continue;
      }
      const [first, ...rest] = read.map(noteOf(bar.system, event));
      if (first === undefined) {
        continue;
      }
      const notes: [TabTuneNote, ...TabTuneNote[]] = [first, ...rest];
      const duration = known.durations[bar.system]?.[event];
      if (duration === "grace") {
        graces.push(notes);
        continue;
      }
      events.push({
        duration: duration ?? { dots: 0, value: "quarter" },
        event,
        graces,
        notes,
        system: bar.system,
      });
      graces = [];
    }
    return {
      events,
      pickup: checks[index]?.pickup ?? false,
      repeatEnd: bar.repeatEnd,
      repeatStart: bar.repeatStart,
      systemBreakBefore: bar.systemBreakBefore,
    };
  });
  return {
    bars,
    capo: known.capo,
    instrument: known.instrument,
    meter: known.meter,
    tuning: { midi: open, names: known.tuning },
  };
}

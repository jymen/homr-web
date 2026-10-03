/**
 * Port of homr's music_xml_generator.py with the default
 * XmlGeneratorArguments, which is how `homr --write-staff-positions` runs
 * on the server: no large page, no metronome, no tempo. The rounding and
 * ordering traps are in docs/design/phase-9-findings.md; the bugs ported as
 * they are (a repeatStart on the right barline, `<duration>` of a
 * non-note given in time-signature beats) are listed there.
 */

import {
  addRatio,
  compareRatio,
  durationOfRhythm,
  mulRatio,
  type Ratio,
  ratio,
  type SymbolDuration,
  ZERO,
} from "../transformer/duration.js";
import type { EncodedSymbol } from "../transformer/symbol.js";
import { EMPTY, NONOTE } from "../transformer/vocabulary.js";
import { leaf, writeXmlDocument, XmlElement } from "./xml.js";

/** Where homr calls `eprint`, the port calls this with the same line. */
export type LogLine = (line: string) => void;

/** What homr raises inside generate_xml: an unsupported token, a duration name it has no entry for, a zero `<duration>`. */
export class MusicXmlError extends Error {
  override name = "MusicXmlError";
}

type TupletMark = "" | "start" | "stop";

interface SymbolChord {
  readonly symbols: readonly EncodedSymbol[];
  tupletMark: TupletMark;
}

/** str(EncodedSymbol). */
export const symbolText = (s: EncodedSymbol): string =>
  `${s.rhythm} ${s.pitch} ${s.lift} ${s.articulation} ${s.slur} ${s.position}`;

const isNoteOrRest = (rhythm: string): boolean =>
  rhythm.startsWith("note") || rhythm.startsWith("rest");

const subRatio = (a: Ratio, b: Ratio): Ratio =>
  ratio(a.num * b.den - b.num * a.den, a.den * b.den);

const sameRatio = (a: Ratio, b: Ratio): boolean => compareRatio(a, b) === 0;

/** Python's int() of a Fraction: truncation toward zero. */
const truncRatio = (r: Ratio): number => Math.trunc(r.num / r.den);

const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));

/** EncodedSymbol.get_duration, with the warning it prints for a symbol that is not a note or a rest. */
function durationOf(symbol: EncodedSymbol, log: LogLine): SymbolDuration {
  if (!isNoteOrRest(symbol.rhythm)) {
    log(
      "Warning, invalid symbol in group: Only notes and rests have durations"
    );
  }
  return durationOfRhythm(symbol.rhythm);
}

/**
 * vocabulary.sort_token_chords: a "chord" token joins the next symbol to the
 * previous one, and each chord is sorted by EncodedSymbol.__lt__, which is
 * str(self) > str(other), so descending by text. Array.sort is stable, as
 * Python's sorted is.
 */
export function sortTokenChords(
  symbols: readonly EncodedSymbol[]
): EncodedSymbol[][] {
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
  return chords.map((chord) =>
    [...chord].sort((a, b) => {
      const x = symbolText(a);
      const y = symbolText(b);
      if (x > y) {
        return -1;
      }
      return x < y ? 1 : 0;
    })
  );
}

function isBarline(chord: SymbolChord): boolean {
  const first = chord.symbols[0]?.rhythm;
  return (
    first !== undefined &&
    (first.includes("barline") || first.includes("repeat"))
  );
}

/** SymbolChord.get_duration: the shortest note or rest. */
function chordDuration(chord: SymbolChord): Ratio {
  let shortest: Ratio | undefined;
  for (const symbol of chord.symbols) {
    if (isNoteOrRest(symbol.rhythm)) {
      const { fraction } = durationOfRhythm(symbol.rhythm);
      if (shortest === undefined || compareRatio(fraction, shortest) < 0) {
        shortest = fraction;
      }
    }
  }
  return shortest ?? ZERO;
}

/** SymbolChord.into_positions: upper first, unless the lower holds nothing but rests. */
function intoPositions(chord: SymbolChord): SymbolChord[] {
  const upper: EncodedSymbol[] = [];
  const lower: EncodedSymbol[] = [];
  let lowerIsOnlyRest = true;
  for (const symbol of chord.symbols) {
    if (symbol.position === "upper") {
      upper.push(symbol);
    } else {
      lower.push(symbol);
      lowerIsOnlyRest = lowerIsOnlyRest && symbol.rhythm.startsWith("rest");
    }
  }
  const pair = [upper, lower];
  if (lowerIsOnlyRest) {
    pair.reverse();
  }
  return pair
    .filter((symbols) => symbols.length > 0)
    .map((symbols) => ({ symbols, tupletMark: chord.tupletMark }));
}

/** np.median of Fractions: the middle one, or the exact mean of the two middle ones. */
function medianRatio(values: readonly Ratio[]): Ratio {
  const sorted = [...values].sort(compareRatio);
  const middle = Math.floor(sorted.length / 2);
  const upper = sorted[middle] ?? ZERO;
  if (sorted.length % 2 === 1) {
    return upper;
  }
  return mulRatio(addRatio(sorted[middle - 1] ?? ZERO, upper), ratio(1, 2));
}

/** find_division_and_time_signature_nominator. */
function divisionAndNominator(groups: readonly SymbolChord[]): [number, Ratio] {
  const durations: Ratio[] = [ratio(1, 4)];
  const measures: Ratio[] = [];
  let inMeasure = ZERO;
  for (const chord of groups) {
    if (isBarline(chord) && compareRatio(inMeasure, ZERO) > 0) {
      measures.push(inMeasure);
      inMeasure = ZERO;
    } else {
      const duration = chordDuration(chord);
      if (compareRatio(duration, ZERO) > 0) {
        durations.push(duration);
        inMeasure = addRatio(inMeasure, duration);
      }
    }
  }
  if (compareRatio(inMeasure, ZERO) > 0) {
    measures.push(inMeasure);
  }
  let division = 1;
  const denominators = durations
    .filter((d) => compareRatio(d, ZERO) > 0)
    .map((d) => d.den);
  if (denominators.length > 0) {
    division = denominators.reduce((a, b) => (a * b) / gcd(a, b));
  }
  return [division, measures.length === 0 ? ratio(1) : medianRatio(measures)];
}

/** TupletParser.get_tuplet_duration. */
function tupletDuration(chord: SymbolChord): SymbolDuration | undefined {
  for (const symbol of chord.symbols) {
    if (isNoteOrRest(symbol.rhythm)) {
      const duration = durationOfRhythm(symbol.rhythm);
      if (duration.normalNotes !== duration.actualNotes) {
        return duration;
      }
    }
  }
  return undefined;
}

/** TupletParser.add_tuplets: false when a tuplet in the measure cannot be completed. */
function addTuplets(groups: readonly SymbolChord[]): boolean {
  let cursor = 0;
  while (cursor < groups.length) {
    const first = groups[cursor];
    const duration = first === undefined ? undefined : tupletDuration(first);
    if (duration === undefined) {
      cursor += 1;
      continue;
    }
    const start = cursor;
    while (cursor - start < duration.actualNotes) {
      const group = groups[cursor];
      if (group === undefined) {
        return false;
      }
      const current = tupletDuration(group);
      if (
        current === undefined ||
        current.actualNotes !== duration.actualNotes ||
        current.normalNotes !== duration.normalNotes
      ) {
        return false;
      }
      cursor += 1;
    }
    const opening = groups[start];
    const closing = groups[cursor - 1];
    if (opening !== undefined && closing !== undefined) {
      opening.tupletMark = "start";
      closing.tupletMark = "stop";
    }
  }
  return true;
}

/** TupletParser.parse: per measure, and a measure whose tuplets fail keeps its old marks. */
function addTupletStartStop(groups: SymbolChord[]): SymbolChord[] {
  let measure: SymbolChord[] = [];
  const measures: SymbolChord[][] = [];
  for (const group of groups) {
    measure.push(group);
    if (isBarline(group)) {
      measures.push(measure);
      measure = [];
    }
  }
  if (measure.length > 0) {
    measures.push(measure);
  }
  for (const groupsOfMeasure of measures) {
    const saved = groupsOfMeasure.map((group) => group.tupletMark);
    if (!addTuplets(groupsOfMeasure)) {
      for (const [index, group] of groupsOfMeasure.entries()) {
        group.tupletMark = saved[index] ?? "";
      }
    }
  }
  return groups;
}

class ConversionState {
  /** 4 * constants.duration_of_quarter until a time signature replaces it with the beat count. */
  beats = 64;
  readonly division: number;
  lastVoltaMeasure = -10;
  readonly nominator: Ratio;
  tremoloState: "start" | "stop" = "stop";
  voltaNumber = 1;

  constructor(division: number, nominator: Ratio) {
    this.division = division;
    this.nominator = nominator;
  }

  startVolta(measure: number): number {
    this.voltaNumber =
      measure === this.lastVoltaMeasure + 1 ? this.voltaNumber + 1 : 1;
    return this.voltaNumber;
  }

  stopVolta(measure: number): number {
    this.lastVoltaMeasure = measure;
    return this.voltaNumber;
  }

  toggleTremolo(): "start" | "stop" {
    this.tremoloState = this.tremoloState === "start" ? "stop" : "start";
    return this.tremoloState;
  }
}

const LIFT_TO_ALTER: Readonly<Record<string, number>> = {
  "#": 1,
  "##": 2,
  b: -1,
  bb: -2,
  N: 0,
};

const DURATION_NAMES: Readonly<Record<number, string>> = {
  0: "breve",
  1: "whole",
  2: "half",
  4: "quarter",
  8: "eighth",
  16: "16th",
  32: "32nd",
  64: "64th",
  128: "128th",
};

function durationName(kern: number): string {
  const name = DURATION_NAMES[kern];
  if (name === undefined) {
    throw new MusicXmlError(`KeyError: ${kern}`);
  }
  return name;
}

/** XMLDuration validates its value as a positive number, so a zero is an error, not a file. */
function durationElement(value: number): XmlElement {
  if (value <= 0) {
    throw new MusicXmlError(
      `XMLDuration.value '${value}' must be greater than '0'`
    );
  }
  return leaf("duration", value);
}

const staffOf = (symbol: EncodedSymbol): 1 | 2 =>
  symbol.position === "lower" ? 2 : 1;

/** Four voices per staff: staff 1 is voices 1 to 4, staff 2 voices 5 to 8. */
const xmlVoice = (staff: number, layer: number): number =>
  (staff - 1) * 4 + layer + 1;

function buildArticulations(
  note: XmlElement,
  articulations: string,
  tupletMark: TupletMark,
  state: ConversionState,
  log: LogLine
): void {
  const notations = note.add(new XmlElement("notations"));
  const marks: XmlElement[] = [];
  const ornaments: XmlElement[] = [];
  for (const articulation of articulations.split("_")) {
    switch (articulation) {
      case "":
        break;
      case NONOTE:
        log(`WARNING note without valid articulation ${articulations}`);
        break;
      case "fermata":
      case "arpeggiate":
        notations.add(new XmlElement(articulation));
        break;
      case "accent":
      case "staccato":
      case "staccatissimo":
      case "tenuto":
      case "caesura":
      case "doit":
        marks.push(new XmlElement(articulation));
        break;
      case "breathMark":
        marks.push(new XmlElement("breath-mark"));
        break;
      case "tremolo":
        ornaments.push(
          new XmlElement("tremolo", { type: state.toggleTremolo() }, 3)
        );
        break;
      case "trill":
        ornaments.push(new XmlElement("trill-mark"));
        break;
      case "turn":
        ornaments.push(new XmlElement("inverted-turn"));
        break;
      case "slurStart":
      case "slurStop":
        notations.add(
          new XmlElement("slur", {
            type: articulation === "slurStart" ? "start" : "stop",
          })
        );
        break;
      case "tieStart":
      case "tieStop":
        notations.add(
          new XmlElement("tied", {
            type: articulation === "tieStart" ? "start" : "stop",
          })
        );
        break;
      default:
        throw new MusicXmlError(`Unsupported articulation ${articulation}`);
    }
  }
  if (tupletMark !== "") {
    notations.add(new XmlElement("tuplet", { type: tupletMark }));
  }
  if (marks.length > 0) {
    const parent = notations.add(new XmlElement("articulations"));
    for (const mark of marks) {
      parent.add(mark);
    }
  }
  if (ornaments.length > 0) {
    const parent = notations.add(new XmlElement("ornaments"));
    for (const ornament of ornaments) {
      parent.add(ornament);
    }
  }
}

function buildSlurs(
  note: XmlElement,
  slurs: string,
  number: number,
  log: LogLine
): void {
  const notations =
    note.childrenNamed("notations")[0] ?? note.add(new XmlElement("notations"));
  const slur = (type: string) =>
    notations.add(new XmlElement("slur", { number, type }));
  if (slurs === EMPTY || slurs === "") {
    return;
  }
  if (slurs === NONOTE) {
    log(`WARNING note without valid articulation ${slurs}`);
  } else if (slurs === "slurStart") {
    slur("start");
  } else if (slurs === "slurStop") {
    slur("stop");
  } else if (slurs === "slurStart_slurStop") {
    // Stop before start, or the slur would start and stop on this note.
    slur("stop");
    slur("start");
  } else {
    throw new MusicXmlError(`Unsupported slur ${slurs}`);
  }
}

/** A note's pitch, or its rest when the pitch is empty or missing. */
function pitchOrRest(
  symbol: EncodedSymbol,
  duration: SymbolDuration,
  log: LogLine
): XmlElement {
  if (symbol.pitch === EMPTY) {
    return new XmlElement(
      "rest",
      duration.fraction.num === 0 ? { measure: "yes" } : {}
    );
  }
  if (symbol.pitch === NONOTE) {
    log(`WARNING note without pitch ${symbolText(symbol)}`);
    return new XmlElement("rest");
  }
  const pitch = new XmlElement("pitch");
  pitch.add(leaf("step", symbol.pitch[0] ?? ""));
  pitch.add(leaf("octave", Number.parseInt(symbol.pitch[1] ?? "", 10)));
  if (symbol.lift === NONOTE) {
    log(`WARNING note with invalid lift ${symbolText(symbol)}`);
  } else if (symbol.lift !== EMPTY) {
    const alter = LIFT_TO_ALTER[symbol.lift];
    if (alter === undefined) {
      throw new MusicXmlError(`KeyError: '${symbol.lift}'`);
    }
    pitch.add(leaf("alter", alter));
  }
  return pitch;
}

/** `<grace>` and `<type>`, or `<type>` and `<duration>`; a non-note lasts the time signature's beat count. */
function addTypeAndDuration(
  note: XmlElement,
  symbol: EncodedSymbol,
  duration: SymbolDuration,
  state: ConversionState
): void {
  const { fraction, kern } = duration;
  if (symbol.rhythm.includes("G")) {
    note.add(new XmlElement("grace"));
    note.add(leaf("type", durationName(kern)));
  } else if (fraction.num > 0) {
    note.add(leaf("type", durationName(kern === 0 ? 1 : kern)));
    note.add(
      durationElement(truncRatio(mulRatio(fraction, ratio(state.division))))
    );
  } else {
    note.add(leaf("type", durationName(0)));
    note.add(durationElement(state.beats));
  }
}

function buildNoteOrRest(
  symbol: EncodedSymbol,
  layer: number,
  isChord: boolean,
  state: ConversionState,
  tupletMark: TupletMark,
  log: LogLine
): XmlElement {
  const note = new XmlElement("note");
  if (isChord) {
    note.add(new XmlElement("chord"));
  }
  const duration = durationOf(symbol, log);
  note.add(pitchOrRest(symbol, duration, log));
  addTypeAndDuration(note, symbol, duration, state);

  const staff = staffOf(symbol);
  note.add(leaf("staff", staff));
  note.add(leaf("voice", String(xmlVoice(staff, layer))));
  for (let i = 0; i < duration.dots; i += 1) {
    note.add(new XmlElement("dot"));
  }
  const isTuplet = duration.actualNotes !== duration.normalNotes;
  if (isTuplet) {
    const modification = note.add(new XmlElement("time-modification"));
    modification.add(leaf("actual-notes", duration.actualNotes));
    modification.add(leaf("normal-notes", duration.normalNotes));
  }
  buildArticulations(
    note,
    symbol.articulation,
    isTuplet ? tupletMark : "",
    state,
    log
  );
  buildSlurs(note, symbol.slur, staff, log);
  return note;
}

interface DurationGroup {
  readonly fraction: Ratio;
  readonly notes: EncodedSymbol[];
}

/** _group_notes: by duration, grace notes at 0, a whole-measure rest at the chord's longest. */
function groupNotes(
  notes: readonly EncodedSymbol[],
  log: LogLine
): DurationGroup[] {
  let longest: Ratio | undefined;
  for (const note of notes) {
    const { fraction } = durationOf(note, log);
    if (longest === undefined || compareRatio(fraction, longest) > 0) {
      longest = fraction;
    }
  }
  const groups: DurationGroup[] = [];
  for (const note of notes) {
    let { fraction } = durationOf(note, log);
    if (note.rhythm.includes("G")) {
      fraction = ZERO;
    } else if (fraction.num === 0) {
      fraction = longest ?? ZERO;
    }
    const group = groups.find((g) => sameRatio(g.fraction, fraction));
    if (group === undefined) {
      groups.push({ fraction, notes: [note] });
    } else {
      group.notes.push(note);
    }
  }
  return [...groups].sort((a, b) => compareRatio(a.fraction, b.fraction));
}

function backup(duration: Ratio, state: ConversionState): XmlElement {
  const element = new XmlElement("backup");
  element.add(
    durationElement(truncRatio(mulRatio(duration, ratio(state.division))))
  );
  return element;
}

/** build_note_chord: one layer per duration, a backup between layers and back to the chord's own length. */
function buildNoteChord(
  chord: SymbolChord,
  state: ConversionState,
  duration: Ratio,
  log: LogLine
): XmlElement[] {
  const groups = groupNotes(chord.symbols, log);
  const result: XmlElement[] = [];
  let finalDuration = ZERO;
  for (const [layer, group] of groups.entries()) {
    for (const [index, note] of group.notes.entries()) {
      result.push(
        buildNoteOrRest(note, layer, index > 0, state, chord.tupletMark, log)
      );
    }
    if (layer !== groups.length - 1 && compareRatio(group.fraction, ZERO) > 0) {
      result.push(backup(group.fraction, state));
    }
    finalDuration = group.fraction;
  }
  if (compareRatio(duration, finalDuration) < 0) {
    result.push(backup(subRatio(finalDuration, duration), state));
  }
  return result;
}

const intText = (element: XmlElement | undefined, fallback: number): number =>
  element === undefined ? fallback : Number.parseInt(element.text ?? "", 10);

interface TimedNoteEvent {
  readonly end: number;
  readonly notes: XmlElement[];
  readonly staff: number;
  readonly start: number;
}

/** The first half of rebalance_measure_voices: each note's start and end, a chord's tones joined into one event. */
function timedEvents(measure: XmlElement): TimedNoteEvent[] {
  const events: TimedNoteEvent[] = [];
  let currentTime = 0;
  let lastNoteStart = 0;
  for (const child of measure.children) {
    if (child.name === "backup") {
      currentTime -= intText(child.childrenNamed("duration")[0], 0);
    } else if (child.name === "note") {
      const duration = intText(child.childrenNamed("duration")[0], 0);
      const staff = intText(child.childrenNamed("staff")[0], 1);
      const isChordTone = child.childrenNamed("chord").length > 0;
      const start = isChordTone ? lastNoteStart : currentTime;
      const end = start + duration;
      const last = events.at(-1);
      if (
        isChordTone &&
        last?.staff === staff &&
        last.start === start &&
        last.end === end
      ) {
        last.notes.push(child);
      } else {
        if (!isChordTone) {
          lastNoteStart = start;
          currentTime += duration;
        }
        events.push({ end, notes: [child], staff, start });
      }
    }
  }
  return events;
}

/** One staff's events in (start, end) order, each given the lowest voice not still sounding. */
function assignVoices(staff: number, events: readonly TimedNoteEvent[]): void {
  const sorted = [...events].sort((a, b) => a.start - b.start || a.end - b.end);
  let active: { end: number; voice: number }[] = [];
  for (const event of sorted) {
    active = active.filter((a) => a.end > event.start);
    const used = new Set(active.map((a) => a.voice));
    let voice = 1;
    while (used.has(voice)) {
      voice += 1;
    }
    active.push({ end: event.end, voice });
    for (const note of event.notes) {
      const [element] = note.childrenNamed("voice");
      if (element !== undefined) {
        element.text = String(xmlVoice(staff, voice - 1));
      }
    }
  }
}

/** rebalance_measure_voices: non-overlapping voices per staff for the whole measure. */
function rebalanceMeasureVoices(measure: XmlElement): void {
  const byStaff = new Map<number, TimedNoteEvent[]>();
  for (const event of timedEvents(measure)) {
    byStaff.set(event.staff, [...(byStaff.get(event.staff) ?? []), event]);
  }
  for (const [staff, events] of byStaff) {
    assignVoices(staff, events);
  }
}

function attributesOf(
  measure: XmlElement,
  last: XmlElement | undefined,
  forceNew = false
): XmlElement {
  if (last !== undefined && !forceNew) {
    return last;
  }
  return measure.add(new XmlElement("attributes"));
}

function barlineOf(
  measure: XmlElement,
  location: "left" | "right"
): XmlElement {
  return (
    measure
      .childrenNamed("barline")
      .find((barline) => barline.attribute("location") === location) ??
    measure.add(new XmlElement("barline", { location }))
  );
}

function buildRepeat(rhythm: string, barline: XmlElement, log: LogLine): void {
  if (barline.childrenNamed("repeat").length > 0) {
    log("barline already has a repeat");
    return;
  }
  barline.add(
    new XmlElement("repeat", {
      direction: rhythm === "repeatStart" ? "forward" : "backward",
    })
  );
}

function buildEnding(
  rhythm: string,
  barline: XmlElement,
  number: number
): void {
  let type: string;
  if (rhythm.startsWith("voltaStart")) {
    type = "start";
  } else if (rhythm.startsWith("voltaStop")) {
    type = "stop";
  } else {
    type = "discontinue";
  }
  barline.add(new XmlElement("ending", { number: String(number), type }));
}

function buildMultiMeasureRest(
  symbol: EncodedSymbol,
  attributes: XmlElement,
  log: LogLine
): void {
  if (attributes.childrenNamed("measure-style").length > 0) {
    log("Measure already has a multi rest");
    return;
  }
  const count = Number.parseInt(
    (symbol.rhythm.split("_")[1] ?? "").replace("m", ""),
    10
  );
  attributes
    .add(new XmlElement("measure-style"))
    .add(leaf("multiple-rest", count));
}

function buildClef(symbol: EncodedSymbol, attributes: XmlElement): void {
  const signAndLine = symbol.rhythm.split("_")[1] ?? "";
  const clef = attributes.add(
    new XmlElement("clef", { number: staffOf(symbol) })
  );
  clef.add(leaf("sign", signAndLine[0] ?? ""));
  clef.add(leaf("line", Number.parseInt(signAndLine[1] ?? "", 10)));
}

function buildTime(
  symbol: EncodedSymbol,
  attributes: XmlElement,
  state: ConversionState
): void {
  const denominator = symbol.rhythm.split("/")[1] ?? "";
  const beats = Math.max(
    truncRatio(
      mulRatio(state.nominator, ratio(Number.parseInt(denominator, 10)))
    ),
    1
  );
  const time = attributes.add(new XmlElement("time"));
  time.add(leaf("beats", String(beats)));
  time.add(leaf("beat-type", denominator));
  state.beats = beats;
}

/** A part's measures as they are written, the open one last. */
class MeasureList {
  current = new XmlElement("measure", { number: "1" });
  readonly measures: XmlElement[] = [];
  number = 1;

  barline(location: "left" | "right"): XmlElement {
    return barlineOf(this.current, location);
  }

  /** close_current_measure, then a new measure. */
  next(): void {
    rebalanceMeasureVoices(this.current);
    this.measures.push(this.current);
    this.number += 1;
    this.current = new XmlElement("measure", { number: String(this.number) });
  }

  /** The open measure is kept only if anything was written to it. */
  finish(): XmlElement[] {
    if (this.current.children.length > 0) {
      rebalanceMeasureVoices(this.current);
      this.measures.push(this.current);
    }
    return this.measures;
  }
}

const isMultiRest = (group: SymbolChord, rhythm: string): boolean =>
  group.symbols.length === 1 && rhythm.endsWith("m");

function addNotes(
  list: MeasureList,
  group: SymbolChord,
  state: ConversionState,
  log: LogLine
): void {
  const positions = intoPositions(group);
  for (const [index, position] of positions.entries()) {
    const duration =
      index === positions.length - 1 ? chordDuration(group) : ZERO;
    for (const element of buildNoteChord(position, state, duration, log)) {
      list.current.add(element);
    }
  }
}

/**
 * A multirest, clef, key or time signature: written into the attributes the
 * previous group wrote to, if it wrote to one, and a clef always opens new
 * attributes. Answers the attributes for the next group.
 */
function addAttributes(
  list: MeasureList,
  group: SymbolChord,
  last: XmlElement | undefined,
  state: ConversionState,
  log: LogLine
): XmlElement {
  const [symbol] = group.symbols;
  const rhythm = symbol?.rhythm ?? "";
  const attributes = attributesOf(
    list.current,
    last,
    rhythm.startsWith("clef")
  );
  if (symbol === undefined) {
    return attributes;
  }
  if (rhythm.startsWith("clef")) {
    for (const clef of group.symbols) {
      if (clef.rhythm.startsWith("clef")) {
        buildClef(clef, attributes);
      }
    }
  } else if (rhythm.startsWith("keySignature")) {
    attributes
      .add(new XmlElement("key"))
      .add(leaf("fifths", Number.parseInt(rhythm.split("_")[1] ?? "", 10)));
  } else if (rhythm.startsWith("timeSignature")) {
    buildTime(symbol, attributes, state);
  } else {
    buildMultiMeasureRest(symbol, attributes, log);
  }
  return attributes;
}

/** Barlines, repeats, voltas, a system break, and the log line for anything else. */
function addStructure(
  list: MeasureList,
  symbol: EncodedSymbol,
  isLastGroup: boolean,
  state: ConversionState,
  log: LogLine
): void {
  const { rhythm } = symbol;
  if (rhythm === "newline") {
    if (!isLastGroup) {
      list.current.add(new XmlElement("print", { "new-system": "yes" }));
    }
  } else if (rhythm.includes("barline")) {
    if (rhythm !== "barline") {
      list
        .barline("right")
        .add(
          leaf(
            "bar-style",
            rhythm === "bolddoublebarline" ? "heavy-heavy" : "light-light"
          )
        );
    }
    list.next();
  } else if (rhythm === "repeatStart") {
    list.next();
    // homr puts the forward repeat on the new measure's right barline.
    buildRepeat(rhythm, list.barline("right"), log);
  } else if (rhythm === "repeatEnd") {
    buildRepeat(rhythm, list.barline("right"), log);
    list.next();
  } else if (rhythm === "repeatEndStart") {
    buildRepeat("repeatEnd", list.barline("right"), log);
    list.next();
    buildRepeat("repeatStart", list.barline("right"), log);
  } else if (rhythm.startsWith("voltaStart")) {
    buildEnding(rhythm, list.barline("left"), state.startVolta(list.number));
  } else if (
    rhythm.startsWith("voltaStop") ||
    rhythm.startsWith("voltaDiscontinue")
  ) {
    buildEnding(rhythm, list.barline("right"), state.stopVolta(list.number));
  } else {
    log(`Symbol isn't supported yet  ${symbolText(symbol)}`);
  }
}

const ATTRIBUTE_RHYTHMS = ["clef", "keySignature", "timeSignature"] as const;

function buildMeasures(
  voice: readonly EncodedSymbol[],
  hasTwoStaves: boolean,
  log: LogLine
): XmlElement[] {
  const groups = addTupletStartStop(
    sortTokenChords(voice).map((symbols) => ({ symbols, tupletMark: "" }))
  );
  const [division, nominator] = divisionAndNominator(groups);
  const state = new ConversionState(division, nominator);
  const list = new MeasureList();
  const first = attributesOf(list.current, undefined);
  first.add(leaf("divisions", Math.floor(division / 4)));
  if (hasTwoStaves) {
    first.add(leaf("staves", 2));
    first.add(leaf("part-symbol", "brace"));
  }
  let attributes: XmlElement | undefined = first;
  for (const [groupNo, group] of groups.entries()) {
    const [symbol] = group.symbols;
    const last = attributes;
    attributes = undefined;
    if (symbol === undefined) {
      continue;
    }
    const { rhythm } = symbol;
    if (isNoteOrRest(rhythm) && !isMultiRest(group, rhythm)) {
      addNotes(list, group, state, log);
    } else if (
      isNoteOrRest(rhythm) ||
      ATTRIBUTE_RHYTHMS.some((prefix) => rhythm.startsWith(prefix))
    ) {
      attributes = addAttributes(list, group, last, state, log);
    } else {
      addStructure(list, symbol, groupNo === groups.length - 1, state, log);
    }
  }
  return list.finish();
}

const partId = (index: number): string => `P${index + 1}`;

function buildScorePart(index: number, hasTwoStaves: boolean): XmlElement {
  const [name, sound, program] = hasTwoStaves
    ? ["Piano", "keyboard.piano", 1]
    : ["Voice", "voice", 54];
  const id = partId(index);
  if (index + 1 > 16) {
    throw new MusicXmlError(`midi-channel ${index + 1} is above 16`);
  }
  const part = new XmlElement("score-part", { id });
  part.add(leaf("part-name", name));
  const instrument = part.add(
    new XmlElement("score-instrument", { id: `${id}-I1` })
  );
  instrument.add(leaf("instrument-name", name));
  instrument.add(leaf("instrument-sound", sound));
  const midi = part.add(new XmlElement("midi-instrument", { id: `${id}-I1` }));
  midi.add(leaf("midi-channel", index + 1));
  midi.add(leaf("midi-program", program));
  midi.add(leaf("volume", 100));
  midi.add(leaf("pan", 0));
  return part;
}

/**
 * generate_xml(XmlGeneratorArguments(), voices, title) followed by
 * `write`: one part per voice, a part with any lower-staff symbol being a
 * two-staff piano part. Throws MusicXmlError where homr raises.
 */
export function generateMusicXml(
  voices: readonly (readonly EncodedSymbol[])[],
  title: string,
  log: LogLine = () => undefined
): string {
  const root = new XmlElement("score-partwise", { version: "4.0" });
  root.add(new XmlElement("work")).add(new XmlElement("work-title", {}, title));
  root
    .add(new XmlElement("identification"))
    .add(new XmlElement("encoding"))
    .add(leaf("software", "homr"));
  root.add(new XmlElement("defaults"));
  const twoStaves = voices.map((voice) =>
    voice.some((symbol) => symbol.position === "lower")
  );
  const partList = root.add(new XmlElement("part-list"));
  for (const [index, hasTwo] of twoStaves.entries()) {
    partList.add(buildScorePart(index, hasTwo));
  }
  for (const [index, voice] of voices.entries()) {
    const part = root.add(new XmlElement("part", { id: partId(index) }));
    for (const measure of buildMeasures(
      voice,
      twoStaves[index] ?? false,
      log
    )) {
      part.add(measure);
    }
  }
  return writeXmlDocument(root);
}

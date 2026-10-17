/**
 * The fret numbers of line tablature, read off the tab systems the guard
 * found. homr never sees them: this is the guard's own connected-component
 * pass carried to the end. Every digit-sized mark sitting on a line is cut out
 * with its own ink only, read by RapidOCR's recogniser, and kept as a fret on
 * that line, a technique letter beside it, or an unread mark.
 *
 * Built for born-digital tabs. The prototype (.scratch/tab-sketch, 2026-10-08)
 * read every event of four vector pages right against a hand transcription or
 * the staff above; on a scan whose lines strike through the digits it read 3
 * of 21, so a scan's reading is not to be trusted (docs/design/tab-2-findings.md).
 */

import type { OpenCv } from "../cv/opencv.js";
import { withMatScope } from "../cv/opencv.js";
import type { ColorImage } from "../image/plane.js";
import {
  type Box,
  bandWithoutLines,
  cropOf,
  type DetectedTab,
  inkOf,
  type Mark,
  marksOf,
  type ReadCrops,
  readFret,
  type TabSystem,
} from "./detect.js";
import { glyphsOf, onLines, type PageTextItem, textIn } from "./layer.js";
import {
  type FretColumn,
  rhythmGeometry,
  type TabBarline,
  type TabStem,
} from "./rhythm.js";
import type { TabCapo, TabInstrumentName, TabTuning } from "./tuning.js";

/** One fret on one line: `string` 1 is the top line. */
export interface TabNote {
  readonly fret: number;
  readonly string: number;
}

/** Frets struck together, at page-normalised `x`, one per string, ordered by string. */
export interface TabPlacedEvent {
  readonly notes: readonly [TabNote, ...TabNote[]];
  readonly x: number;
}

/**
 * A fret column with what the page hangs under it for its time value.
 * `grace` is a grace note: frets printed smaller than the system's, slurred
 * into the next event; it takes no time, whatever its stem says.
 */
export interface TabEvent extends TabPlacedEvent {
  readonly grace: boolean;
  readonly stem: TabStem;
}

export const TAB_SLUR_KINDS = [
  "tie",
  "hammer-on",
  "pull-off",
  "slide",
  "slur",
] as const;
/**
 * What an arc between two frets on one string means: `tie` joins two equal
 * frets (one pluck, held), the techniques come from the letter printed by
 * the arc (H, P, S and their TablEdit and LilyPond forms), and `slur` is an
 * arc between two frets with no letter the reader knows, played legato.
 */
export type TabSlurKind = (typeof TAB_SLUR_KINDS)[number];

/** An arc from event `from` to event `to` (indices into `events`) on `string`, with the letter printed by it, if any. */
export interface TabSlur {
  readonly from: number;
  readonly kind: TabSlurKind;
  /** The letter read by the arc, as printed ("P", "Po", "h"); absent when none was found. */
  readonly letter?: string;
  readonly string: number;
  readonly to: number;
}

/** A time signature printed at the start of a system, as two stacked numbers. */
export interface TabTimeSignature {
  readonly den: number;
  readonly num: number;
  readonly text: string;
}

export const TAB_TECHNIQUES = [
  "Sl",
  "Po",
  "H",
  "R",
  "p",
  "h",
  "x",
  "Harm.",
] as const;
/** The playing-technique letters tab software prints; none changes a pitch. */
export type TabTechnique = (typeof TAB_TECHNIQUES)[number];

/** A technique letter on a line, standing alone or stuck to a fret, at page-normalised `x`. */
export interface TabAnnotation {
  readonly string: number;
  readonly technique: TabTechnique;
  readonly x: number;
}

/**
 * A tab system with what is printed on it. `unread` counts the line-sitting
 * marks that read as neither a fret nor a technique: the "TAB" clef letters,
 * a time signature, a symbol the recogniser has no character for.
 * `tuning` and `capo` are what the page's text says about this system
 * (src/tab/text.ts), absent when it says nothing: never a default.
 */
export interface TabReading extends TabSystem {
  readonly annotations: readonly TabAnnotation[];
  /** Barlines left to right; the system's opening line is one. */
  readonly barlines: readonly TabBarline[];
  readonly capo?: TabCapo;
  readonly events: readonly TabEvent[];
  readonly instrumentName?: TabInstrumentName;
  /** Where the frets and letters were read: the PDF's text layer, or the page image by the recogniser. */
  readonly readFrom: "image" | "text_layer";
  readonly slurs: readonly TabSlur[];
  readonly timeSignature?: TabTimeSignature;
  readonly tuning?: TabTuning;
  readonly unread: number;
}

/**
 * Recogniser text to technique, tried before a fret so "Po" is not a P stuck
 * to a 0. TablEdit's "Sl" and "Po" can reach the recogniser as their
 * capital alone: the second letter is a short mark, and joinWords adds it
 * back only when it stands close enough.
 */
const TECHNIQUE_READS: readonly (readonly [RegExp, TabTechnique])[] = [
  // "arm." is x-height: the recogniser has read its crop as "ar"
  [/^har/i, "Harm."],
  [/^[Ss][lI1|]?$/, "Sl"],
  [/^P[oO0]?$/, "Po"],
  [/^H$/, "H"],
  [/^R$/, "R"],
  [/^p$/, "p"],
  [/^h$/, "h"],
  [/^[xX×]$/, "x"],
];

const techniqueOf = (text: string): TabTechnique | undefined =>
  TECHNIQUE_READS.find(([pattern]) => pattern.test(text))?.[1];

type MarkReading =
  | {
      readonly kind: "fret";
      readonly fret: number;
      readonly technique: TabTechnique | undefined;
    }
  | { readonly kind: "technique"; readonly technique: TabTechnique }
  | { readonly kind: "unread" };

export function readMark(text: string): MarkReading {
  const trimmed = text.trim();
  const technique = techniqueOf(trimmed);
  if (technique !== undefined) {
    return { kind: "technique", technique };
  }
  const fret = readFret(trimmed);
  return fret === undefined
    ? { kind: "unread" }
    : { fret: fret.fret, kind: "fret", technique: techniqueOf(fret.letters) };
}

/** Frets closer than this, in line spacings, to an event's first fret belong to it, one per string. */
const CHORD_REACH = 0.45;

interface PlacedFret {
  readonly box?: Box;
  readonly centre: number;
  readonly note: TabNote;
}

interface Column {
  readonly centre: number;
  readonly frets: PlacedFret[];
}

/** Frets left to right into columns: a fret joins the open column when it is within reach and its string is free. */
function columnsOf(frets: readonly PlacedFret[], spacing: number): Column[] {
  const open: Column[] = [];
  for (const fret of [...frets].sort((a, b) => a.centre - b.centre)) {
    const last = open.at(-1);
    if (
      last !== undefined &&
      fret.centre - last.centre < CHORD_REACH * spacing &&
      !last.frets.some((f) => f.note.string === fret.note.string)
    ) {
      last.frets.push(fret);
    } else {
      open.push({ centre: fret.centre, frets: [fret] });
    }
  }
  for (const column of open) {
    column.frets.sort((a, b) => a.note.string - b.note.string);
  }
  return open;
}

const placedOf = (column: Column, pageWidth: number): TabPlacedEvent => ({
  notes: column.frets.map((f) => f.note) as [TabNote, ...TabNote[]],
  x: column.centre / pageWidth,
});

/** Frets left to right into events: a fret joins the open event when it is within reach and its string is free. */
export function eventsOf(
  frets: readonly PlacedFret[],
  spacing: number,
  pageWidth: number
): TabPlacedEvent[] {
  return columnsOf(frets, spacing).map((column) => placedOf(column, pageWidth));
}

interface ReadMark {
  readonly mark: Mark;
  readonly text: string;
}

/** A short mark this close to a letter, in line spacings, continues its word ("H" + "arm" is "Harm"). */
const WORD_GAP = 0.35;
const LETTERS = /^[A-Za-z.]+$/;

/**
 * Tall letter marks with the short letter marks that follow them on their
 * line joined in: the lowercase tail of "Harm." or "Po" is x-height, so the
 * capital comes out as a tall mark and the rest as short ones.
 */
function joinWords(
  tall: readonly ReadMark[],
  short: readonly ReadMark[],
  spacing: number
): { readonly short: ReadMark[]; readonly tall: ReadMark[] } {
  const rest = [...short].sort((a, b) => a.mark.x - b.mark.x);
  const joined = tall.map((word) => {
    if (!LETTERS.test(word.text)) {
      return word;
    }
    let { mark, text } = word;
    for (let k = 0; k < rest.length; ) {
      const next = rest[k] as ReadMark;
      const gap = next.mark.x - (mark.x + mark.w);
      if (
        next.mark.line === mark.line &&
        gap >= -mark.w / 2 &&
        gap < WORD_GAP * spacing &&
        LETTERS.test(next.text)
      ) {
        const right = Math.max(mark.x + mark.w, next.mark.x + next.mark.w);
        const top = Math.min(mark.y, next.mark.y);
        const bottom = Math.max(mark.y + mark.h, next.mark.y + next.mark.h);
        mark = { ...mark, h: bottom - top, w: right - mark.x, y: top };
        text = `${text}${next.text}`;
        rest.splice(k, 1);
      } else {
        k += 1;
      }
    }
    return { mark, text };
  });
  return { short: rest, tall: joined };
}

interface Placement {
  readonly annotations: TabAnnotation[];
  readonly frets: PlacedFret[];
  readonly unread: number;
}

/**
 * A system's frets, annotations and unread count from its read marks. A
 * short mark (under 0.75 of the median height) counts only as a technique
 * letter: it is an x-height "x" or a piece of a pull-off arc, never a fret,
 * and an arc is not unread.
 */
function placementOf(
  tab: DetectedTab,
  tallMarks: readonly ReadMark[],
  shortMarks: readonly ReadMark[],
  pageWidth: number
): Placement {
  const { short, tall } = joinWords(tallMarks, shortMarks, tab.group.spacing);
  const frets: PlacedFret[] = [];
  const annotations: TabAnnotation[] = [];
  let unread = 0;
  const annotate = (mark: Mark, technique: TabTechnique) =>
    annotations.push({
      string: mark.line + 1,
      technique,
      x: (mark.x + mark.w / 2) / pageWidth,
    });
  for (const { mark, text } of tall) {
    const reading = readMark(text);
    if (reading.kind === "unread") {
      unread += 1;
      continue;
    }
    if (reading.technique !== undefined) {
      annotate(mark, reading.technique);
    }
    if (reading.kind === "fret") {
      frets.push({
        box: mark,
        centre: mark.x + mark.w / 2,
        note: { fret: reading.fret, string: mark.line + 1 },
      });
    }
  }
  for (const { mark, text } of short) {
    const reading = readMark(text);
    if (reading.kind === "technique") {
      annotate(mark, reading.technique);
    }
  }
  annotations.sort((a, b) => a.x - b.x || a.string - b.string);
  return { annotations, frets, unread };
}

/** The slur letters the reader knows, as printed beside an arc: TablEdit's Po and Sl, Peghead Nation's capitals, LilyPond's lowercase. */
const SLUR_LETTERS: readonly (readonly [RegExp, TabSlurKind])[] = [
  [/^[Hh]$/, "hammer-on"],
  [/^(?:[Pp]|Po|PO)$/, "pull-off"],
  [/^(?:[Ss]|Sl|SL|S1|SI)$/, "slide"],
];

const slurLetterKind = (text: string): TabSlurKind | undefined =>
  SLUR_LETTERS.find(([pattern]) => pattern.test(text))?.[1];

/** What an arc means, from its letter or, without one the reader knows, from its two frets. */
export function slurKindOf(
  letter: string | undefined,
  fromFret: number,
  toFret: number
): TabSlurKind {
  const known = letter === undefined ? undefined : slurLetterKind(letter);
  if (known !== undefined) {
    return known;
  }
  return fromFret === toFret ? "tie" : "slur";
}

/** A time signature's number, with a repeat dot or a space the text may carry after it. */
const TIME_NUMBER = /^(\d{1,2})[.\s]*$/;
const DENOMINATORS: ReadonlySet<number> = new Set([1, 2, 4, 8, 16, 32]);

/** Two recognised numbers as a time signature, or undefined when they are not one. */
export function timeSignatureOf(
  upper: string,
  lower: string
): TabTimeSignature | undefined {
  const num = upper.trim().match(TIME_NUMBER)?.[1];
  const den = lower.trim().match(TIME_NUMBER)?.[1];
  if (num === undefined || den === undefined) {
    return;
  }
  const signature = { den: Number(den), num: Number(num) };
  return signature.num >= 1 &&
    signature.num <= 16 &&
    DENOMINATORS.has(signature.den)
    ? { ...signature, text: `${num}/${den}` }
    : undefined;
}

/**
 * Whether the mark's line runs into it from both sides: a number printed
 * without a knock-out. A knocked-out number keeps its own ink untouched,
 * since erasing a line through it also takes a digit's middle bar held on
 * one side only (the bar of a 3 or a 5).
 */
function struck(
  ink: Uint8Array,
  width: number,
  tab: DetectedTab,
  mark: Mark
): boolean {
  const line = tab.group.lines[mark.line];
  if (line === undefined) {
    return false;
  }
  const touches = (x: number) => {
    for (let y = line.y0; y <= line.y1; y += 1) {
      if (ink[y * width + x] === 1) {
        return true;
      }
    }
    return false;
  };
  return touches(mark.x - 1) && touches(mark.x + mark.w);
}

/**
 * The page's ink with the tab's lines erased over its band, so a crop of a
 * number the line runs through shows the number alone: the recogniser reads
 * "12" with a line across it as "2".
 */
function withoutLines(
  ink: Uint8Array,
  width: number,
  height: number,
  tab: DetectedTab
): Uint8Array {
  const band = bandWithoutLines(ink, width, height, tab.group);
  const cols = tab.group.x1 - tab.group.x0 + 1;
  const out = ink.slice();
  for (let y = 0; y < band.rows; y += 1) {
    out.set(
      band.data.subarray(y * cols, (y + 1) * cols),
      (y + band.top) * width + tab.group.x0
    );
  }
  return out;
}

/**
 * Each tab's frets, techniques, unread marks and rhythm marks, in the tabs'
 * order. A system's digit-height marks go to `read` in one call, the batch
 * the prototype measured, its short marks in a second, and the letters by
 * its arcs with its time signature's digits in a third. `onSystem` is told
 * after each system.
 */
export async function readTablature(
  cv: OpenCv,
  page: ColorImage,
  tabs: readonly DetectedTab[],
  read: ReadCrops,
  onSystem?: (done: number, total: number) => void,
  layer: readonly PageTextItem[] = []
): Promise<TabReading[]> {
  if (tabs.length === 0) {
    return [];
  }
  const ink = withMatScope(
    (scope) => new Uint8Array(inkOf(cv, scope, page).data)
  );
  const glyphs = glyphsOf(layer, page.width, page.height);
  const readings: TabReading[] = [];
  for (const tab of tabs) {
    const printed = onLines(glyphs, tab.group);
    const fromLayer =
      printed.filter((g) => readFret(g.text) !== undefined).length >=
      LAYER_MIN_FRETS;
    const placement = fromLayer
      ? placementOf(
          tab,
          printed.map((g) => ({
            mark: { ...g.box, line: g.line },
            text: g.text,
          })),
          [],
          page.width
        )
      : // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, one system at a time
        await placementByRecogniser(cv, page, ink, tab, read);
    const readBoxes = async (wanted: readonly Box[]): Promise<string[]> => {
      const inLayer = wanted.map((box) =>
        fromLayer ? textIn(glyphs, box, LAYER_PAD * tab.group.spacing) : ""
      );
      const unprinted = wanted.filter((_, k) => inLayer[k] === "");
      const recognised =
        unprinted.length === 0
          ? []
          : await read(
              unprinted.map((box) =>
                cropOf(page, ink, { ...box, line: 0 }, tab.group.spacing)
              )
            );
      return inLayer.map((text, k) =>
        text === ""
          ? (recognised[inLayer.slice(0, k).filter((t) => t === "").length] ??
            "")
          : text
      );
    };
    const columns = columnsOf(placement.frets, tab.group.spacing);
    const geometry = rhythmGeometry(
      cv,
      ink,
      page.width,
      page.height,
      tab.group,
      columns.map(fretColumnOf)
    );
    const boxes: Box[] = [
      ...geometry.timeDigits.flat(),
      ...geometry.slurs.flatMap((slur) => slur.letters),
    ];
    const texts = (boxes.length === 0 ? [] : await readBoxes(boxes)).map(
      (text) => text.trim()
    );
    const timeOf = (said: readonly string[]) =>
      geometry.timeDigits
        .map((_, k) =>
          timeSignatureOf(said[2 * k] ?? "", said[2 * k + 1] ?? "")
        )
        .find((t) => t !== undefined);
    const time =
      timeOf(texts) ??
      (fromLayer && geometry.timeDigits.length > 0
        ? timeOf(
            await read(
              geometry.timeDigits
                .flat()
                .map((box) =>
                  cropOf(page, ink, { ...box, line: 0 }, tab.group.spacing)
                )
            )
          )
        : undefined);
    let next = 2 * geometry.timeDigits.length;
    const slurs = geometry.slurs.map((slur): TabSlur => {
      const letters = texts.slice(next, next + slur.letters.length);
      next += slur.letters.length;
      const letter = letters.find((text) => slurLetterKind(text) !== undefined);
      const fretOn = (k: number) =>
        columns[k]?.frets.find((f) => f.note.string === slur.string)?.note
          .fret ?? 0;
      return {
        from: slur.from,
        kind: slurKindOf(letter, fretOn(slur.from), fretOn(slur.to)),
        ...(letter === undefined ? {} : { letter }),
        string: slur.string,
        to: slur.to,
      };
    });
    readings.push({
      ...tab.system,
      annotations: placement.annotations,
      barlines: geometry.barlines.map(({ px: _px, ...bar }) => bar),
      events: columns.map((column, k) => ({
        ...placedOf(column, page.width),
        grace: isGrace(column, k, columns, slurs),
        stem: geometry.stems[k] ?? { kind: "none" },
      })),
      readFrom: fromLayer ? "text_layer" : "image",
      slurs,
      ...(time === undefined ? {} : { timeSignature: time }),
      unread: placement.unread,
    });
    onSystem?.(readings.length, tabs.length);
  }
  return readings;
}

/** Grace frets stand under this share of the system's median fret height. */
const GRACE_HEIGHT = 0.88;

/** Smaller than the system's frets, every one of them, and slurred into the next column. */
function isGrace(
  column: Column,
  index: number,
  columns: readonly Column[],
  slurs: readonly TabSlur[]
): boolean {
  const heights = columns
    .flatMap((c) => c.frets.map((f) => f.box?.h ?? 0))
    .sort((a, b) => a - b);
  const median = heights[Math.floor(heights.length / 2)] ?? 0;
  return (
    column.frets.every(
      (f) => f.box !== undefined && f.box.h < GRACE_HEIGHT * median
    ) && slurs.some((slur) => slur.from === index && slur.to === index + 1)
  );
}

/** A text layer speaks for a system when it holds at least this many frets on its lines. */
const LAYER_MIN_FRETS = 4;
/** A letter or digit box takes the layer's glyphs whose centre lies within this many line spacings of it. */
const LAYER_PAD = 0.2;

/** The frets and letters of a system read by the recogniser off the page's ink: the digit-height marks in one call, the short ones in a second. */
async function placementByRecogniser(
  cv: OpenCv,
  page: ColorImage,
  ink: Uint8Array,
  tab: DetectedTab,
  read: ReadCrops
): Promise<Placement> {
  const { marks, short } = marksOf(cv, ink, page.width, page.height, tab.group);
  const lineless = withoutLines(ink, page.width, page.height, tab);
  const readAll = async (list: readonly Mark[]): Promise<ReadMark[]> => {
    const crops = list.map((mark) =>
      cropOf(
        page,
        struck(ink, page.width, tab, mark) ? lineless : ink,
        mark,
        tab.group.spacing
      )
    );
    const texts = crops.length === 0 ? [] : await read(crops);
    return list.map((mark, k) => ({ mark, text: texts[k] ?? "" }));
  };
  const tall = await readAll(marks);
  return placementOf(tab, tall, await readAll(short), page.width);
}

const fretColumnOf = (column: Column): FretColumn => ({
  bottom: Math.max(
    ...column.frets.map((f) => (f.box === undefined ? 0 : f.box.y + f.box.h))
  ),
  centre: column.centre,
  notes: column.frets.flatMap((f) =>
    f.box === undefined ? [] : [{ box: f.box, string: f.note.string }]
  ),
});

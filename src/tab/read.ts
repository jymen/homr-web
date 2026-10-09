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
import type { TabCapo, TabTuning } from "./tuning.js";

/** One fret on one line: `string` 1 is the top line. */
export interface TabNote {
  readonly fret: number;
  readonly string: number;
}

/** Frets struck together, at page-normalised `x`, one per string, ordered by string. */
export interface TabEvent {
  readonly notes: readonly [TabNote, ...TabNote[]];
  readonly x: number;
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
  readonly capo?: TabCapo;
  readonly events: readonly TabEvent[];
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
  readonly centre: number;
  readonly note: TabNote;
}

/** Frets left to right into events: a fret joins the open event when it is within reach and its string is free. */
export function eventsOf(
  frets: readonly PlacedFret[],
  spacing: number,
  pageWidth: number
): TabEvent[] {
  const open: { centre: number; notes: TabNote[] }[] = [];
  for (const fret of [...frets].sort((a, b) => a.centre - b.centre)) {
    const last = open.at(-1);
    if (
      last !== undefined &&
      fret.centre - last.centre < CHORD_REACH * spacing &&
      !last.notes.some((note) => note.string === fret.note.string)
    ) {
      last.notes.push(fret.note);
    } else {
      open.push({ centre: fret.centre, notes: [fret.note] });
    }
  }
  return open.map(({ centre, notes }) => ({
    notes: [...notes].sort((a, b) => a.string - b.string) as [
      TabNote,
      ...TabNote[],
    ],
    x: centre / pageWidth,
  }));
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

/**
 * A system's reading from its read marks. A short mark (under 0.75 of the
 * median height) counts only as a technique letter: it is an x-height "x"
 * or a piece of a pull-off arc, never a fret, and an arc is not unread.
 */
function readingOf(
  tab: DetectedTab,
  tallMarks: readonly ReadMark[],
  shortMarks: readonly ReadMark[],
  pageWidth: number
): TabReading {
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
  return {
    ...tab.system,
    annotations,
    events: eventsOf(frets, tab.group.spacing, pageWidth),
    unread,
  };
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
 * Each tab's frets, techniques and unread marks, in the tabs' order. A
 * system's digit-height marks go to `read` in one call, the batch the
 * prototype measured, and its short marks in a second. `onSystem` is told
 * after each system.
 */
export async function readTablature(
  cv: OpenCv,
  page: ColorImage,
  tabs: readonly DetectedTab[],
  read: ReadCrops,
  onSystem?: (done: number, total: number) => void
): Promise<TabReading[]> {
  if (tabs.length === 0) {
    return [];
  }
  const ink = withMatScope(
    (scope) => new Uint8Array(inkOf(cv, scope, page).data)
  );
  const readings: TabReading[] = [];
  for (const tab of tabs) {
    const { marks, short } = marksOf(
      cv,
      ink,
      page.width,
      page.height,
      tab.group
    );
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
    // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, one system at a time
    const tall = await readAll(marks);
    readings.push(readingOf(tab, tall, await readAll(short), page.width));
    onSystem?.(readings.length, tabs.length);
  }
  return readings;
}

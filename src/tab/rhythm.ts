/**
 * The rhythm a tab system draws around its frets, read with the same
 * classical image processing as the frets: the stem hanging from each fret
 * column and the beams across it, the barlines and their repeat dots, the
 * arcs that tie or slur one fret to the next on a string, and the boxes of
 * the time signature's two digits for the recogniser. Pure geometry over the
 * page's ink; reading letters and digits is left to the caller.
 *
 * Built on the pages of docs/design/tab-4-findings.md: TablEdit, Guitar Pro,
 * LilyPond and Peghead Nation tabs, all of which hang their rhythm below the
 * tab. Stems drawn above it are not looked for.
 */

import type { OpenCv } from "../cv/opencv.js";
import {
  type Band,
  type Box,
  bandWithoutLines,
  componentBoxes,
  type LineGroup,
} from "./detect.js";

/** Rhythm geometry thresholds, in line spacings unless said otherwise. */
const RHYTHM = {
  /** Rows of the band read around a system: above its top line and below its bottom one. */
  above: 4,
  /** An arc: at least this wide, at most this tall, and this many times wider than tall. */
  arcAspect: 2,
  arcMaxHeight: 0.9,
  arcMinWidth: 0.55,
  /** An arc's ends reach this far from the frets they join. */
  arcReach: 0.7,
  /** Strokes of one barline (a repeat's thick and thin bars) stand this close. */
  barGroup: 1.2,
  /** A column is part of a barline when this share of the rows from the top line to the bottom one is ink. */
  barInk: 0.92,
  /** A barline's ink carries on at most this far past the outer lines. */
  barOverhang: 0.35,
  /** A beam may lie this far under the stem's end: a pen's stem stops a row or three short of its beam (Weeping Willow). */
  beamGap: 0.4,
  /** A beam is at least this thick and runs at least this far from its stem. */
  beamMinThickness: 0.1,
  beamReach: 0.4,
  /** Beams are counted this far up from the stem's end. */
  beamWindow: 1.3,
  below: 7,
  /** Repeat dots: this size range, this close to the barline. */
  dotMax: 0.55,
  dotMin: 0.12,
  dotReach: 1.1,
  /** A flag reaches this far right of its stem, its end at most `flagRise` above where it leaves it. */
  flagReach: 0.25,
  flagRise: 0.8,
  /** Letters by an arc: size bounds, clear of the lines by `letterClear`, at most `letterAbove` over the top line or `letterBelow` under the bottom one, centred within `letterReach` of the arc's first fret. */
  letterAbove: 3.5,
  /** Each spacing a letter stands from the tab counts as this much distance across: lyrics hang further down than a slur's letter. */
  letterAway: 0.5,
  letterBelow: 5,
  letterClear: 0.4,
  letterMaxHeight: 1.4,
  letterMaxWidth: 1.3,
  letterMinHeight: 0.4,
  letterMinWidth: 0.15,
  letterReach: 0.9,
  letterTries: 3,
  /** A stem leans at most this many columns per row (about 8 degrees). */
  stemLean: 0.15,
  /** And at most this wide. */
  stemMaxWidth: 0.2,
  /** A stem is at least this long. */
  stemMinLength: 0.8,
  /** A stem is looked for this far either side of the fret column's centre. */
  stemReach: 0.3,
  /** A stem starts at most this far below the bottom line, or below its lowest fret when that is lower. */
  stemStart: 1.5,
  /** And is at least this long: the upright of a letter under the tab is shorter. */
  strayLength: 1.5,
  /** Stray stems are looked for across this row, this far under the bottom line: below a fret on it. */
  strayRow: 1,
  /** A stray stem starts no lower than this under the bottom line, where a fret on it ends: text under a tab starts lower. */
  strayStart: 0.75,
  /** A barline at least this wide is thick. */
  thickBar: 0.25,
  timeMaxHeight: 2.8,
  timeMaxWidth: 2,
  timeMinHeight: 0.85,
  /** Time-signature digits stand at least this many times a fret's height. */
  timeOverFret: 1.1,
  /** Time-signature digit pairs tried, and the digits' size. */
  timePairs: 3,
  /** A component this tall, in line spacings, is two digits touching (the line between them kept). */
  timeStacked: 2.1,
} as const;

/** What one fret column draws below the tab: `none` is no stem, a half note or longer. */
export type TabStem =
  | { readonly kind: "none" }
  | {
      readonly kind: "stem";
      /** Beams or flags across the stem: 0 a quarter, 1 an eighth, 2 a sixteenth. */
      readonly beams: number;
      /** Augmentation dots beside the stem's end. */
      readonly dots: number;
    };

/** A barline, at page-normalised `x` (its centre). Repeat dots stand after it (`repeatStart`) or before it (`repeatEnd`). */
export interface TabBarline {
  readonly repeatEnd: boolean;
  readonly repeatStart: boolean;
  readonly thick: boolean;
  readonly x: number;
}

/** A fret column as the rhythm reader needs it: its centre and its marks' extent, in page pixels. */
export interface FretColumn {
  readonly bottom: number;
  readonly centre: number;
  /** 1-based strings, top line first, and each note's mark box. */
  readonly notes: readonly { readonly box: Box; readonly string: number }[];
}

/** An arc from fret column `from` to column `to` on `string`, with the boxes that may hold the letter printed by it, likeliest first. */
export interface ArcBetween {
  readonly from: number;
  readonly letters: readonly Box[];
  readonly string: number;
  readonly to: number;
}

/**
 * A beamed stem that hangs from no fret column: a note the page draws whose
 * fret was not read. `x` is the stem's column at its top, where its fret
 * stands, and `top` its first row, in page pixels.
 */
export interface StrayStem {
  readonly stem: TabStem;
  readonly top: number;
  readonly x: number;
}

/** A system's rhythm geometry, in page pixels. */
export interface SystemRhythmGeometry {
  readonly barlines: readonly (TabBarline & { readonly px: number })[];
  readonly slurs: readonly ArcBetween[];
  readonly stems: readonly TabStem[];
  /** Beamed stems under no fret column, left to right. */
  readonly strays: readonly StrayStem[];
  /** Candidates for the time signature's two digits, top first, the likeliest first. */
  readonly timeDigits: readonly (readonly [Box, Box])[];
}

interface Window extends Band {
  readonly cols: number;
  readonly left: number;
}

const inkAt = (w: Window, x: number, y: number): boolean =>
  x >= 0 && x < w.cols && y >= 0 && y < w.rows && w.data[y * w.cols + x] === 1;

/** The system's band, lines erased, from `above` spacings over its top line to `below` under its bottom one. */
function windowOf(
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup
): Window {
  const band = bandWithoutLines(ink, width, height, group, {
    above: RHYTHM.above,
    below: RHYTHM.below,
  });
  return { ...band, cols: group.x1 - group.x0 + 1, left: group.x0 };
}

const topOf = (group: LineGroup) => group.lines[0]?.y0 ?? 0;
const bottomOf = (group: LineGroup) => group.lines.at(-1)?.y1 ?? 0;

/** Consecutive ink rows from `y` going `direction`, at any of columns x0..x1. */
function runFrom(
  w: Window,
  x0: number,
  x1: number,
  y: number,
  direction: 1 | -1
): number {
  let length = 0;
  for (let row = y; row >= 0 && row < w.rows; row += direction) {
    let inked = false;
    for (let x = x0; x <= x1 && !inked; x += 1) {
      inked = inkAt(w, x, row);
    }
    if (!inked) {
      break;
    }
    length += 1;
  }
  return length;
}

/**
 * Vertical strokes from the top line to the bottom one that stop there:
 * barlines. A stem crossing the tab carries on below it.
 */
function barlinesOf(
  boxes: readonly Box[],
  w: Window,
  group: LineGroup
): (TabBarline & { readonly px: number })[] {
  const s = group.spacing;
  const y0 = topOf(group) - w.top;
  const y1 = bottomOf(group) - w.top;
  const full = (x: number) => {
    let inked = 0;
    for (let y = y0; y <= y1; y += 1) {
      inked += inkAt(w, x, y) ? 1 : 0;
    }
    return inked >= RHYTHM.barInk * (y1 - y0 + 1);
  };
  const strokes: { x0: number; x1: number }[] = [];
  for (let x = 0; x < w.cols; x += 1) {
    if (!full(x)) {
      continue;
    }
    const last = strokes.at(-1);
    if (last !== undefined && last.x1 === x - 1) {
      last.x1 = x;
    } else {
      strokes.push({ x0: x, x1: x });
    }
  }
  const bars = strokes.filter(
    ({ x0, x1 }) =>
      runFrom(w, x0, x1, y1 + 1, 1) <= RHYTHM.barOverhang * s &&
      runFrom(w, x0, x1, y0 - 1, -1) <= RHYTHM.barOverhang * s
  );
  const grouped: { thick: boolean; x0: number; x1: number }[] = [];
  for (const bar of bars) {
    const thick = bar.x1 - bar.x0 + 1 >= RHYTHM.thickBar * s;
    const last = grouped.at(-1);
    if (last !== undefined && bar.x0 - last.x1 <= RHYTHM.barGroup * s) {
      last.x1 = bar.x1;
      last.thick ||= thick;
    } else {
      grouped.push({ thick, x0: bar.x0, x1: bar.x1 });
    }
  }
  const dots = boxes.filter(
    (box) =>
      box.w >= RHYTHM.dotMin * s &&
      box.w <= RHYTHM.dotMax * s &&
      box.h >= RHYTHM.dotMin * s &&
      box.h <= RHYTHM.dotMax * s &&
      box.y > y0 &&
      box.y + box.h < y1
  );
  const dotsBeside = (from: number, to: number) =>
    dots.filter((d) => d.x + d.w > from && d.x < to).length >= 2;
  return grouped.map((bar) => ({
    px: w.left + (bar.x0 + bar.x1) / 2,
    repeatEnd: dotsBeside(bar.x0 - RHYTHM.dotReach * s, bar.x0),
    repeatStart: dotsBeside(bar.x1, bar.x1 + RHYTHM.dotReach * s),
    thick: bar.thick,
    x: 0,
  }));
}

/** A stem, in window pixels: `x` is its column at its lower end, where its beams are. */
interface Stem {
  readonly bottom: number;
  readonly top: number;
  readonly x: number;
}

/**
 * The ink run down from (x, y), stepping a column aside where the ink does,
 * as long as it leans at most `stemLean`: a pen's stem slants, so a run down
 * one column ends partway (Weeping Willow's stems leave their column after
 * two or three line spacings, above their beams), and skipping a row where
 * the pen's stroke breaks for a pixel (`STEM_GAP`). The column at each row.
 */
function stemPath(w: Window, x: number, y: number): number[] {
  const path: number[] = [];
  let at = x;
  let row = y;
  while (inkAt(w, at, row)) {
    path.push(at);
    let next: number | undefined;
    let step = 1;
    for (; step <= 1 + STEM_GAP && next === undefined; step += 1) {
      const below = row + step;
      next = [at, at - 1, at + 1].find(
        (c) =>
          inkAt(w, c, below) &&
          Math.abs(c - x) <= Math.max(1, RHYTHM.stemLean * (path.length + step))
      );
    }
    if (next === undefined) {
      break;
    }
    // A row the pen skipped counts in the stem, at the column above it.
    for (let gap = 2; gap < step; gap += 1) {
      path.push(at);
    }
    at = next;
    row += step - 1;
  }
  return path;
}

/** Empty rows a pen's stem may skip, a pixel's break in its stroke (Weeping Willow). */
const STEM_GAP = 1;

/**
 * The longest thin run down under a fret column, leaning at most
 * `stemLean`: starting below its lowest fret, no lower than `stemStart`
 * under the bottom line, at least `stemMinLength` long.
 */
function stemUnder(
  w: Window,
  column: FretColumn,
  group: LineGroup
): Stem | undefined {
  const s = group.spacing;
  const from = column.bottom - w.top + 1;
  const latest =
    Math.max(column.bottom, bottomOf(group)) - w.top + RHYTHM.stemStart * s;
  let best: Stem | undefined;
  const reach = Math.round(RHYTHM.stemReach * s);
  const centre = Math.round(column.centre - w.left);
  for (let x = centre - reach; x <= centre + reach; x += 1) {
    let y = from;
    while (y <= latest && !inkAt(w, x, y)) {
      y += 1;
    }
    if (y > latest) {
      continue;
    }
    const path = stemPath(w, x, y);
    const { length } = path;
    const middle = Math.floor(length / 2);
    if (
      length >= RHYTHM.stemMinLength * s &&
      (best === undefined || length > best.bottom - best.top) &&
      widthAt(w, path[middle] ?? x, y + middle) <= RHYTHM.stemMaxWidth * s
    ) {
      best = { bottom: y + length - 1, top: y, x: path.at(-1) ?? x };
    }
  }
  return best;
}

/** The ink run along row `y` through column `x`. */
function widthAt(w: Window, x: number, y: number): number {
  let left = x;
  while (inkAt(w, left - 1, y)) {
    left -= 1;
  }
  let right = x;
  while (inkAt(w, right + 1, y)) {
    right += 1;
  }
  return right - left + 1;
}

/**
 * Beams or flags across a stem's lower end, counted on the side with more.
 * A beam is a horizontal run crossing two columns beside the stem, `near`
 * and `beamReach` away, at least `beamMinThickness` thick, ending at most
 * `beamGap` under the stem's end; a pen's beam frays, so `beamMinThickness`
 * of its rows at the far column is enough.
 * A flag (a lone eighth's, TablEdit's) leaves the stem's end rising to the
 * right: a run crossing the near column with ink `flagReach` from the stem
 * within `flagRise` above it.
 */
function beamsOn(w: Window, stem: Stem, s: number): number {
  const near = Math.max(2, Math.round(0.15 * s));
  const far = Math.round(RHYTHM.beamReach * s);
  const thick = Math.max(2, RHYTHM.beamMinThickness * s);
  const top = Math.max(
    stem.top + Math.round(0.3 * s),
    stem.bottom - Math.round(RHYTHM.beamWindow * s)
  );
  const end = stem.bottom + Math.max(2, Math.round(RHYTHM.beamGap * s));
  const runsAt = (x: number, crossing: (y0: number, y1: number) => boolean) => {
    let runs = 0;
    let start = -1;
    for (let y = top; y <= end + 1; y += 1) {
      if (inkAt(w, x, y) && y <= end) {
        start = start < 0 ? y : start;
        continue;
      }
      if (start >= 0 && y - start >= thick && crossing(start, y - 1)) {
        runs += 1;
      }
      start = -1;
    }
    return runs;
  };
  // A pen's beam thins and frays near its end: `thick` of its rows at the far
  // column is enough.
  const beams = (side: 1 | -1) =>
    runsAt(stem.x + side * near, (y0, y1) => {
      let inked = 0;
      for (let y = y0; y <= y1; y += 1) {
        inked += inkAt(w, stem.x + side * far, y) ? 1 : 0;
      }
      return inked >= Math.min(thick, y1 - y0 + 1);
    });
  const rise = Math.round(RHYTHM.flagRise * s);
  const tip = Math.round(RHYTHM.flagReach * s);
  const flags = runsAt(stem.x + near, (y0, y1) => {
    for (let y = y0 - rise; y <= y1; y += 1) {
      if (inkAt(w, stem.x + tip, y)) {
        return true;
      }
    }
    return false;
  });
  return Math.max(beams(-1), beams(1), flags);
}

/** The top of the stroke through (x, y), followed up a column aside where the ink does, leaning at most `stemLean`. */
function strokeTop(w: Window, x: number, y: number): { x: number; y: number } {
  let at = x;
  let row = y;
  while (row > 0) {
    const climbed = y - row + 1;
    const next = [at, at - 1, at + 1].find(
      (c) =>
        inkAt(w, c, row - 1) &&
        Math.abs(c - x) <= Math.max(1, RHYTHM.stemLean * climbed)
    );
    if (next === undefined) {
      break;
    }
    at = next;
    row -= 1;
  }
  return { x: at, y: row };
}

/**
 * Stems crossing the row `strayRow` under the bottom line that no fret
 * column's stem took, at least `strayLength` long, starting no higher than
 * the top line and no lower than `strayStart` under the bottom one, no wider
 * than a stem, and carrying a beam or a flag at their lower end: the ink of a
 * note whose fret was not read. A stroke with no beam is not taken (a
 * letter, a lyric, a stem of a quarter whose fret is missing alike).
 */
function strayStems(
  w: Window,
  group: LineGroup,
  taken: readonly (Stem | undefined)[]
): { readonly stem: Stem; readonly topX: number }[] {
  const s = group.spacing;
  const row = bottomOf(group) - w.top + Math.round(RHYTHM.strayRow * s);
  const highest = topOf(group) - w.top;
  const lowest = bottomOf(group) - w.top + RHYTHM.strayStart * s;
  const reach = Math.max(3, RHYTHM.stemReach * s);
  const strays: { readonly stem: Stem; readonly topX: number }[] = [];
  let x = 0;
  while (x < w.cols) {
    if (!inkAt(w, x, row)) {
      x += 1;
      continue;
    }
    const start = x;
    while (inkAt(w, x, row)) {
      x += 1;
    }
    const top = strokeTop(w, Math.floor((start + x - 1) / 2), row);
    const path = stemPath(w, top.x, top.y);
    const middle = Math.floor(path.length / 2);
    const stem: Stem = {
      bottom: top.y + path.length - 1,
      top: top.y,
      x: path.at(-1) ?? top.x,
    };
    if (
      x - start <= RHYTHM.stemMaxWidth * s * 2 &&
      top.y >= highest &&
      top.y <= lowest &&
      path.length >= RHYTHM.strayLength * s &&
      widthAt(w, path[middle] ?? top.x, top.y + middle) <=
        RHYTHM.stemMaxWidth * s &&
      !taken.some((t) => t !== undefined && Math.abs(t.x - stem.x) <= reach) &&
      !strays.some((t) => Math.abs(t.stem.x - stem.x) <= reach) &&
      beamsOn(w, stem, s) > 0
    ) {
      strays.push({ stem, topX: top.x });
    }
  }
  return strays;
}

/**
 * The stemless fret column a stray stem hangs from: the one whose marks'
 * extent, widened by `stemReach`, holds the stem's top column. A pen hangs a
 * stem from a wide zero's side, further from its centre than `stemReach`
 * (Weeping Willow). `taken` when that column has a stem already, or when two
 * columns are that near (two voices a few pixels apart on Moon over
 * Shanghai: whose stem it is cannot be told), `none` when no column is near:
 * a note whose fret was not read.
 */
function ownerOf(
  columns: readonly FretColumn[],
  stems: readonly (Stem | undefined)[],
  x: number,
  s: number
): number | "none" | "taken" {
  const reach = RHYTHM.stemReach * s;
  const near = [...columns.entries()].filter(([, column]) => {
    const left = Math.min(...column.notes.map((n) => n.box.x)) - reach;
    const right =
      Math.max(...column.notes.map((n) => n.box.x + n.box.w)) + reach;
    return x >= left && x <= right;
  });
  const [only, ...others] = near;
  if (only === undefined) {
    return "none";
  }
  return others.length > 0 || stems[only[0]] !== undefined ? "taken" : only[0];
}

/** Small round marks just right of a stem's end, clear of its beams. */
function dotsBy(cv: OpenCv, w: Window, stem: Stem, s: number): number {
  const x0 = stem.x + Math.round(0.25 * s);
  const x1 = stem.x + Math.round(1.1 * s);
  const y0 = stem.bottom - Math.round(1 * s);
  const y1 = stem.bottom + Math.round(0.6 * s);
  const cols = x1 - x0 + 1;
  const rows = y1 - y0 + 1;
  const patch = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      patch[y * cols + x] = inkAt(w, x0 + x, y0 + y) ? 1 : 0;
    }
  }
  return componentBoxes(cv, patch, cols, rows).filter(
    (box) =>
      box.x > 0 &&
      box.y > 0 &&
      box.x + box.w < cols &&
      box.y + box.h < rows &&
      box.w >= RHYTHM.dotMin * s &&
      box.w <= RHYTHM.dotMax * 0.8 * s &&
      Math.abs(box.w - box.h) <= Math.max(2, 0.12 * s)
  ).length;
}

/** Thin wide components near the lines, in page pixels: tie and slur arcs. */
function arcsOf(boxes: readonly Box[], w: Window, group: LineGroup): Box[] {
  const s = group.spacing;
  const y0 = topOf(group) - w.top - 1.6 * s;
  const y1 = bottomOf(group) - w.top + 1.1 * s;
  return boxes
    .filter(
      (box) =>
        box.w >= RHYTHM.arcMinWidth * s &&
        box.h <= RHYTHM.arcMaxHeight * s &&
        box.w >= RHYTHM.arcAspect * box.h &&
        box.y >= y0 &&
        box.y + box.h <= y1
    )
    .map((box) => ({ ...box, x: box.x + w.left, y: box.y + w.top }));
}

/** The row with the least ink across `box` within its middle fifth: where two stacked digits of one size touch. */
function thinnestRow(w: Window, box: Box): number {
  let best = box.y + Math.floor(box.h / 2);
  let least = Number.POSITIVE_INFINITY;
  for (
    let y = box.y + Math.floor(0.4 * box.h);
    y <= box.y + 0.6 * box.h;
    y += 1
  ) {
    let inked = 0;
    for (let c = 0; c < box.w; c += 1) {
      inked += inkAt(w, box.x + c, y) ? 1 : 0;
    }
    if (inked < least) {
      least = inked;
      best = y;
    }
  }
  return best;
}

/**
 * Pairs of stacked marks taller than a fret before the first fret column,
 * the nearest to the music first: the time signature is among them, beside
 * a TAB clef's letters; the recogniser decides which pair is digits. A
 * single component twice that tall is two digits touching, cut at its
 * thinnest row.
 */
function timeDigitsOf(
  boxes: readonly Box[],
  w: Window,
  group: LineGroup,
  columns: readonly FretColumn[]
): (readonly [Box, Box])[] {
  const s = group.spacing;
  const firstColumn = columns[0]?.centre ?? group.x1;
  const heights = columns
    .flatMap((c) => c.notes.map((n) => n.box.h))
    .sort((a, b) => a - b);
  const fret = heights[Math.floor(heights.length / 2)] ?? 0;
  const minHeight = Math.max(
    RHYTHM.timeMinHeight * s,
    RHYTHM.timeOverFret * fret
  );
  const top = topOf(group) - w.top - 0.6 * s;
  const bottom = bottomOf(group) - w.top + 0.6 * s;
  const before = boxes.filter(
    (box) =>
      box.w <= RHYTHM.timeMaxWidth * s &&
      box.w >= 0.4 * s &&
      box.y >= top &&
      box.y + box.h <= bottom &&
      box.x + w.left < firstColumn
  );
  const page = (box: Box): Box => ({
    ...box,
    x: box.x + w.left,
    y: box.y + w.top,
  });
  const tall = before
    .filter((box) => box.h >= minHeight && box.h < RHYTHM.timeStacked * s)
    .sort((a, b) => a.y - b.y);
  const pairs: (readonly [Box, Box])[] = [];
  for (const [k, upper] of tall.entries()) {
    for (const lower of tall.slice(k + 1)) {
      const overlap =
        Math.min(lower.x + lower.w, upper.x + upper.w) -
        Math.max(lower.x, upper.x);
      if (
        lower.y >= upper.y + upper.h * 0.6 &&
        overlap > 0.5 * Math.min(lower.w, upper.w)
      ) {
        pairs.push([page(upper), page(lower)]);
      }
    }
  }
  for (const box of before) {
    if (
      box.h >= RHYTHM.timeStacked * s &&
      box.h <= 2 * RHYTHM.timeMaxHeight * s
    ) {
      const split = thinnestRow(w, box);
      pairs.push([
        page({ ...box, h: split - box.y }),
        page({ ...box, h: box.y + box.h - split, y: split }),
      ]);
    }
  }
  return pairs.sort((a, b) => b[0].x - a[0].x).slice(0, RHYTHM.timePairs);
}

/**
 * Each arc tied to the fret columns at its ends and to the string they share
 * nearest the arc, which sits over or under its frets; an arc whose ends find
 * no two columns sharing a string is dropped.
 */
function slursOf(
  arcs: readonly Box[],
  columns: readonly FretColumn[],
  group: LineGroup
): Omit<ArcBetween, "letters">[] {
  const s = group.spacing;
  const reach = RHYTHM.arcReach * s;
  const nearest = (x: number) => {
    let best = -1;
    for (const [k, column] of columns.entries()) {
      const d = Math.abs(column.centre - x);
      if (
        d <= reach &&
        (best < 0 || d < Math.abs((columns[best]?.centre ?? 0) - x))
      ) {
        best = k;
      }
    }
    return best;
  };
  const found = new Map<string, Omit<ArcBetween, "letters">>();
  for (const arc of arcs) {
    const from = nearest(arc.x + 0.15 * arc.w);
    const to = nearest(arc.x + 0.85 * arc.w);
    if (from < 0 || to <= from) {
      continue;
    }
    const shared = (columns[from]?.notes ?? []).filter((note) =>
      columns[to]?.notes.some((other) => other.string === note.string)
    );
    const middle = arc.y + arc.h / 2;
    const [best] = shared.sort(
      (a, b) =>
        Math.abs(a.box.y + a.box.h / 2 - middle) -
        Math.abs(b.box.y + b.box.h / 2 - middle)
    );
    if (best !== undefined) {
      found.set(`${from}:${to}:${best.string}`, {
        from,
        string: best.string,
        to,
      });
    }
  }
  return [...found.values()];
}

/**
 * What may be the letter printed by an arc, above the tab or under its
 * beams: letter-sized components outside the lines whose centre stands
 * within `letterReach` of the arc's first fret, ranked by the distance across
 * to the first fret or the arc's middle (Peghead Nation prints under the
 * first, LilyPond over the middle) plus half the distance from the tab;
 * `letterTries` of them, since lyrics may still rank before the letter.
 */
function lettersBy(
  boxes: readonly Box[],
  w: Window,
  group: LineGroup,
  from: FretColumn,
  to: FretColumn
): Box[] {
  const s = group.spacing;
  const top = topOf(group);
  const bottom = bottomOf(group);
  const outside = (box: Box) => {
    const y0 = box.y + w.top;
    const y1 = y0 + box.h;
    return (
      (y1 < top - RHYTHM.letterClear * s &&
        y0 > top - RHYTHM.letterAbove * s) ||
      (y0 > bottom + RHYTHM.letterClear * s &&
        y1 < bottom + RHYTHM.letterBelow * s)
    );
  };
  const away = (box: Box) => {
    const y = box.y + w.top + box.h / 2;
    return y < top ? top - y : y - bottom;
  };
  const centre = (box: Box) => box.x + w.left + box.w / 2;
  const middle = (from.centre + to.centre) / 2;
  const score = (box: Box) =>
    Math.min(
      Math.abs(centre(box) - from.centre),
      Math.abs(centre(box) - middle)
    ) +
    RHYTHM.letterAway * away(box);
  return boxes
    .filter(
      (box) =>
        box.h >= RHYTHM.letterMinHeight * s &&
        box.h <= RHYTHM.letterMaxHeight * s &&
        box.w >= RHYTHM.letterMinWidth * s &&
        box.w <= RHYTHM.letterMaxWidth * s &&
        centre(box) >= from.centre - RHYTHM.letterReach * s &&
        centre(box) <= to.centre + 0.3 * s &&
        outside(box)
    )
    .sort((a, b) => score(a) - score(b))
    .slice(0, RHYTHM.letterTries)
    .map((box) => ({ ...box, x: box.x + w.left, y: box.y + w.top }));
}

/**
 * The window with every fret's box blanked, so an arc resting on its two
 * frets (Peghead Nation draws them touching) stands alone, and a stem starts
 * where its fret's box ends.
 */
function withoutFrets(w: Window, columns: readonly FretColumn[]): Window {
  const data = w.data.slice();
  for (const { notes } of columns) {
    for (const { box } of notes) {
      for (let y = box.y - w.top; y < box.y - w.top + box.h; y += 1) {
        if (y < 0 || y >= w.rows) {
          continue;
        }
        const x0 = Math.max(0, box.x - w.left);
        const x1 = Math.min(w.cols, box.x - w.left + box.w);
        data.fill(0, y * w.cols + x0, y * w.cols + x1);
      }
    }
  }
  return { ...w, data };
}

/**
 * Everything a system draws for its rhythm, in page pixels: one stem per
 * fret column (in the columns' order), the barlines, the arcs and the time
 * signature's digit boxes.
 */
export function rhythmGeometry(
  cv: OpenCv,
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup,
  columns: readonly FretColumn[]
): SystemRhythmGeometry {
  const w = withoutFrets(windowOf(ink, width, height, group), columns);
  const s = group.spacing;
  const boxes = componentBoxes(cv, w.data, w.cols, w.rows);
  const stems = columns.map((column) => stemUnder(w, column, group));
  const strays = strayStems(w, group, stems).filter(({ stem, topX }) => {
    const owner = ownerOf(columns, stems, w.left + topX, group.spacing);
    if (owner === "none") {
      return true;
    }
    if (owner !== "taken") {
      stems[owner] = stem;
    }
    return false;
  });
  const timeDigits = timeDigitsOf(boxes, w, group, columns);
  const slurs = slursOf(arcsOf(boxes, w, group), columns, group).map(
    (slur): ArcBetween => {
      const from = columns[slur.from];
      const to = columns[slur.to];
      return {
        ...slur,
        letters:
          from === undefined || to === undefined
            ? []
            : lettersBy(boxes, w, group, from, to),
      };
    }
  );
  return {
    barlines: barlinesOf(boxes, w, group).map((bar) => ({
      ...bar,
      x: bar.px / width,
    })),
    slurs,
    stems: stems.map(
      (stem): TabStem =>
        stem === undefined
          ? { kind: "none" }
          : {
              beams: beamsOn(w, stem, s),
              dots: dotsBy(cv, w, stem, s),
              kind: "stem",
            }
    ),
    strays: strays.map(({ stem, topX }) => ({
      stem: {
        beams: beamsOn(w, stem, s),
        dots: dotsBy(cv, w, stem, s),
        kind: "stem",
      },
      top: stem.top + w.top,
      x: w.left + topX,
    })),
    timeDigits,
  };
}

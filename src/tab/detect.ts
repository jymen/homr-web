/**
 * Line tablature on a page, found with classical image processing before homr
 * reads it. homr has no notion of tablature: it takes a tab's lines for a
 * staff and invents notes on them, so the guard has to know where the tabs
 * are first.
 *
 * A system is a run of four to six evenly spaced horizontal lines. Four or six
 * lines is a tab, since a standard staff has five. Five lines is a tab only
 * when the marks sitting on its lines read as numbers: noteheads are wider
 * than a line space and never reach the recogniser, and the few marks of a
 * staff that do (dots, small rests) do not read as numbers.
 *
 * The thresholds are the prototype's (.scratch/tab-sketch, 2026-10-08), which
 * told staves from tabs on every system it found in ten sample pages (digit
 * share 0.00 on staves, 0.87 to 1.00 on born-digital tabs), plus two grouping
 * rules for chord diagrams and beams (docs/decisions.tsv, tab-1).
 */

import type { Mat } from "@techstark/opencv-js";
import { planeToMat } from "../cv/mat-plane.js";
import { type MatScope, type OpenCv, withMatScope } from "../cv/opencv.js";
import { type ColorImage, createColor } from "../image/plane.js";

export const TAB_LINE_COUNTS = [4, 5, 6] as const;
export type TabLineCount = (typeof TAB_LINE_COUNTS)[number];

/** A tab system, page-normalised 0..1 like `StaffBox` over the extent of its lines; `index` in top-to-bottom order among the page's tabs. */
export interface TabSystem {
  readonly cx: number;
  readonly cy: number;
  readonly h: number;
  readonly index: number;
  readonly lines: TabLineCount;
  readonly w: number;
}

/** One horizontal line, in page pixels: `y` is its ink-weighted centre, `y0..y1` its rows, `breakLength` its longest break. */
interface PageLine {
  readonly breakLength: number;
  readonly x0: number;
  readonly x1: number;
  readonly y: number;
  readonly y0: number;
  readonly y1: number;
}

/** Four to six evenly spaced lines, in page pixels. */
export interface LineGroup {
  readonly lines: readonly PageLine[];
  readonly spacing: number;
  readonly x0: number;
  readonly x1: number;
}

/** A tab with the pixel geometry the guard paints over. */
export interface DetectedTab {
  readonly group: LineGroup;
  /** Found by the hand-written second chance (handWrittenMarks): read at HAND_MARK, stems erased. */
  readonly hand?: boolean;
  /** Its frets are knocked out of its lines (knockedOutShare), as tab software prints them: no stroke of a fret ever runs along a line. */
  readonly knocksOut?: boolean;
  readonly system: TabSystem;
}

export interface Box {
  readonly h: number;
  readonly w: number;
  readonly x: number;
  readonly y: number;
}

/** Page rows `top` onwards over a line group's columns, `rows` of them, 1 for ink. */
export interface Band {
  readonly data: Uint8Array;
  readonly rows: number;
  readonly top: number;
}

/** A line-sitting mark, in page pixels, `line` 0-based from the top. */
export interface Mark extends Box {
  readonly line: number;
}

/**
 * What decides a group: its line count alone, or the crops of its
 * line-sitting marks, which only a recogniser can settle. Four and six lines
 * stay a tab on their count: reading them as well loses Killdare Fancy's
 * first system, whose top line runs through its digits and halves every one
 * (2026-10-31), and Moon over Shanghai's last, a one-bar system holding one
 * "0". A staff that loses a line to a beam is handled where lines are
 * grouped (withoutBeamRows, withFaintLine) and, if still taken, by the
 * pipeline painting out only systems read to events.
 */
type Evidence = "tab" | "staff" | readonly ColorImage[];

/** A line group, what decides whether it is a tab, and the share of its marks knocked out (undefined under four marks). */
interface Candidate {
  readonly evidence: Evidence;
  readonly group: LineGroup;
  readonly hand?: boolean;
  readonly knockedOut: number | undefined;
}

/** Reads crops as text; the pipeline's reader is RapidOCR's recogniser, opened on the first call. */
export type ReadCrops = (crops: readonly ColorImage[]) => Promise<string[]>;

const TAB = {
  /** A refused five-line group is drawn like the page's five-line tabs within this share of each one's line spacing (alikeFiveLineTabs). */
  alikeSpacing: 0.1,
  /** Rows of a line touched by a beam: those covered at least this share of the most covered row, the others at most `besideLineShare`. */
  besideLineShare: 0.25,
  /** Crop padding, in line spacings. */
  cropPad: 0.25,
  /** The outer cuts lie this far outside the outer lines, in line spacings (see cutBetweenLines). */
  cutOutside: 0.5,
  /** What a cut leaves at one end narrower than this, in line spacings, is a piece of stem. */
  cutPieceWidth: 0.2,
  /**
   * A line the ink threshold lost, completing a four-line group into a
   * staff: faint ink over this share of the group's width, one gap past its
   * outer line (see faintLine).
   */
  faintLineCover: 0.8,
  gapTolerance: 0.2,
  /**
   * A hand-written tab's frets are as large as a staff's noteheads, but sit
   * on its lines; a staff's noteheads sit in its spaces as often. At most
   * this share of hand-sized marks in the spaces, against those on lines: a
   * finger letter ("M", "T") beside the frets lands in a space.
   */
  handInSpaceShare: 1 / 3,
  /** hangsAStem: at least this tall, the head at least this tall, and this share of the rows below it a stem's width. */
  /** The stem hangs within this share of the head's width from its middle. */
  hangingCentred: 0.2,
  hangingHead: 0.5,
  /** The head's widest row at least this many times the stem's median width. */
  hangingHeadWidth: 2,
  hangingHeight: 1.5,
  /** The head is looked for in this many spacings from the top. */
  hangingReach: 1.3,
  hangingThinShare: 0.8,
  /** A stroke's jog along a line spans at most this many of its widths (bridgeJogs). */
  jogWidths: 2,
  /**
   * A mark is knocked out when its line is blank within this many line
   * spacings on both sides of it ("—10—"), as tab software prints a fret.
   */
  knockOutReach: 0.15,
  lineCoverShare: 0.5,
  /** The band read for marks, in line spacings above the top line and below the bottom one. */
  markBand: 0.75,
  /**
   * The longest break along a line, in line spacings: a fret number's
   * knock-out is about one, while a row of chord diagrams lines up fret rows
   * a dozen spacings apart.
   */
  maxBreak: 5,
  maxFret: 24,
  maxGapShare: 0.03,
  /** At most this many lines of a group may be hidden by beams (finishGroup). */
  maxHidden: 2,
  maxMarkHeight: 1.15,
  maxMarkWidth: 1,
  /**
   * A turned scan spreads a thin line over more rows than a line is thick:
   * up to this many more per page column (a third of a degree) while its ink
   * is still a line's thickness over its length. Weeping Willow's
   * handwritten banjo page spreads its lines over 5 to 7 rows.
   */
  maxSlope: 0.006,
  /** A line is at most this many rows thick: max(4, width / 400). */
  maxThickness: 4,
  /** Two marks on one line merge into one number ("10") across a gap below this. */
  mergeGap: 0.18,
  mergeWidth: 1.2,
  /** Five-line tabs read as numbers on a page before a refused group drawn like them is taken too. */
  minAlikeTabs: 2,
  minDigitShare: 0.5,
  /** A five-line group is a tab with this many numbers, making at least this share of the marks read. */
  minDigits: 4,
  /** Consecutive line gaps: at least 5 px, at most 3% of the page height, within 20% of the first. */
  minGap: 5,
  /** Frets hanging their stems on a system for it to be cut between the lines. */
  minHanging: 2,
  /**
   * The share of a group's marks knocked out for it to count as printed that
   * way: tabs measure 0.81 to 1.00 where their program knocks out, staves
   * with fingering digits 0.00.
   */
  minKnockOutShare: 0.5,
  /** A mark is digit sized: height and width bounds in line spacings. */
  minMarkHeight: 0.35,
  minMarkWidth: 0.08,
  /** Marks shorter than this share of the median (arcs cut by a stem) are dropped. */
  minMedianHeight: 0.75,
  /** Each line and the group's first overlap over this share of the wider: a beam above a staff is shorter. */
  minOverlap: 0.6,
  /** A row is a line row when its long runs cover this share of the width. */
  minRowCover: 0.1,
  minRun: 12,
  /** Line rows: a dark run counts when it is at least this share of the page width (and 12 px). */
  minRunShare: 1 / 45,
  /** A mark sits on a line when its centre is this close to it. */
  onLine: 0.3,
  /** Marks read per recogniser run (RapidOCR's own batch), and at most this many per group. */
  readBatch: 6,
  /**
   * A group whose numbers are not knocked out is dropped only when its line
   * spacing is further than this share from every knocked-out tab's: a staff
   * is drawn smaller than its tab (27 against 41, 22 against 33 px).
   */
  sameSizeTolerance: 0.2,
  sampleSize: 12,
  /** A cut piece with this share of its box inked is a piece of a thick bar. */
  solidShare: 0.8,
  /** A thin stroke this many spacings long is a stem or a barline, never part of a fret (eraseStems). */
  stemLength: 2,
  /** A piece narrower than this, in line spacings, whose ink carries on past both its cuts for `strokeReach` is a stroke of a larger glyph. */
  strokeReach: 0.3,
  strokeWidth: 0.4,
  /** How far, in spacings, a stem's stub is followed past the stem into the fret it hangs from (eraseStems). */
  stubLength: 0.3,
  thicknessShare: 1 / 400,
} as const;

/** A fret number, possibly with technique letters stuck to it ("S0", "7h"); "O" reads as 0, and so does "θ", a handwritten zero with its stroke slanting through it. */
const FRET = /^([A-Za-z~/\\(]*?)(\d{1,2})([A-Za-z~/\\)]*)$/;

/** The fret a recogniser read names, and the letters stuck to it, or undefined when the text is no fret. */
export function readFret(
  text: string
): { readonly fret: number; readonly letters: string } | undefined {
  const match = text
    .trim()
    .replace(/[OoθΘ](?=\d|$)/g, "0")
    .match(FRET);
  if (!match) {
    return;
  }
  const [, before, digits, after] = match;
  const fret = Number(digits);
  return fret > TAB.maxFret
    ? undefined
    : { fret, letters: `${before ?? ""}${after ?? ""}` };
}

export const isFretText = (text: string): boolean =>
  readFret(text) !== undefined;

/** 1 where the page is ink, Otsu-thresholded on cv2's gray, so a gray scan and a clean PDF share one rule. */
export function inkOf(cv: OpenCv, scope: MatScope, page: ColorImage): Mat {
  const gray = scope.keep(new cv.Mat());
  cv.cvtColor(planeToMat(cv, scope, page), gray, cv.COLOR_BGR2GRAY);
  const ink = scope.keep(new cv.Mat());
  cv.threshold(gray, ink, 0, 1, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
  return ink;
}

/**
 * 1 where the page is darker than half way between the ink threshold and the
 * paper: what Otsu leaves out of `inkOf` on an enlarged web image, a staff
 * line spread over five pale rows and the pale stem between two lines
 * (Marche des élèves, 820 by 1061 enlarged to 3508 rows: lines 162 to 176,
 * stems 165 to 172, paper 241). Read only to decide whether a group is a
 * tab, never to read one.
 */
export function faintInkOf(
  cv: OpenCv,
  scope: MatScope,
  page: ColorImage
): Uint8Array {
  const gray = scope.keep(new cv.Mat());
  cv.cvtColor(planeToMat(cv, scope, page), gray, cv.COLOR_BGR2GRAY);
  const ink = scope.keep(new cv.Mat());
  const threshold = cv.threshold(
    gray,
    ink,
    0,
    1,
    cv.THRESH_BINARY_INV + cv.THRESH_OTSU
  );
  const data = gray.data as Uint8Array;
  const histogram = new Uint32Array(256);
  for (const value of data) {
    histogram[value] = (histogram[value] ?? 0) + 1;
  }
  let paper = 255;
  let seen = 0;
  for (let level = 0; level < 256; level += 1) {
    seen += histogram[level] ?? 0;
    if (seen >= data.length / 2) {
      paper = level;
      break;
    }
  }
  const limit = (threshold + Math.max(paper, threshold)) / 2;
  return data.map((value) => (value < limit ? 1 : 0));
}

/**
 * The line at `y` that only `faint` holds: rows within the gap tolerance
 * of it whose faint ink covers `faintLineCover` of the group's width. A
 * staff whose line was too pale for the ink threshold is otherwise a
 * four-line group, which passes for a mandolin tab on its line count alone
 * (Le p'tit Sarny's first system, 2026-10-28).
 */
function faintLine(
  faint: Uint8Array,
  width: number,
  height: number,
  group: LineGroup,
  y: number
): PageLine | undefined {
  const reach = TAB.gapTolerance * group.spacing;
  const span = group.x1 - group.x0 + 1;
  let y0 = -1;
  let y1 = -1;
  for (let row = Math.ceil(y - reach); row <= Math.floor(y + reach); row += 1) {
    if (row < 0 || row >= height) {
      continue;
    }
    let covered = 0;
    for (let x = group.x0; x <= group.x1; x += 1) {
      covered += faint[row * width + x] ?? 0;
    }
    if (covered >= TAB.faintLineCover * span) {
      y0 = y0 < 0 ? row : y0;
      y1 = row;
    }
  }
  if (y0 < 0) {
    return;
  }
  return {
    breakLength: 0,
    x0: group.x0,
    x1: group.x1,
    y: (y0 + y1) / 2,
    y0,
    y1,
  };
}

/** A four-line group made five with a faint line a gap below or above it, or left as it was. */
function withFaintLine(
  faint: Uint8Array,
  width: number,
  height: number,
  group: LineGroup
): LineGroup {
  if (group.lines.length !== 4) {
    return group;
  }
  const first = group.lines[0] as PageLine;
  const last = group.lines.at(-1) as PageLine;
  const below = faintLine(faint, width, height, group, last.y + group.spacing);
  const above =
    below === undefined
      ? faintLine(faint, width, height, group, first.y - group.spacing)
      : undefined;
  if (below === undefined && above === undefined) {
    return group;
  }
  const lines =
    below === undefined
      ? [above as PageLine, ...group.lines]
      : [...group.lines, below];
  return {
    ...group,
    lines,
    spacing:
      ((lines.at(-1) as PageLine).y - (lines[0] as PageLine).y) /
      (lines.length - 1),
  };
}

/**
 * Whether a mark is where a stroke crosses a line: narrower than a stroke,
 * with faint ink in its columns for `strokeReach` on both sides. On an
 * enlarged web image a stem between two lines is too pale for the ink
 * threshold and only its crossings stay, each a "1" on a line to the
 * recogniser: a staff of eighth notes then reads as a tab (Le p'tit Sarny's
 * second system, 12 of 12 marks read "1", 2026-10-28). A fret stands on
 * paper above and below, and a fret stacked over another meets the gap
 * between them.
 */
function crossesAsStroke(
  faint: Uint8Array,
  width: number,
  height: number,
  mark: Mark,
  spacing: number
): boolean {
  if (mark.w >= TAB.strokeWidth * spacing) {
    return false;
  }
  const { w, x: left } = mark;
  const inked = (y: number) => {
    if (y < 0 || y >= height) {
      return false;
    }
    for (let x = left; x < left + w; x += 1) {
      if (faint[y * width + x] === 1) {
        return true;
      }
    }
    return false;
  };
  const reach = Math.max(2, Math.round(TAB.strokeReach * spacing));
  for (let k = 1; k <= reach; k += 1) {
    if (!(inked(mark.y - k) && inked(mark.y + mark.h - 1 + k))) {
      return false;
    }
  }
  return true;
}

/** Per row, the pixels in long horizontal runs and the x extent of those runs. */
function rowCover(ink: Uint8Array, width: number, height: number) {
  const minRun = Math.max(TAB.minRun, Math.round(width * TAB.minRunShare));
  const cover = new Uint32Array(height);
  const rowX0 = new Int32Array(height).fill(width);
  const rowX1 = new Int32Array(height).fill(-1);
  for (let y = 0; y < height; y += 1) {
    let run = 0;
    for (let x = 0; x <= width; x += 1) {
      if (x < width && ink[y * width + x] === 1) {
        run += 1;
        continue;
      }
      if (run >= minRun) {
        cover[y] = (cover[y] ?? 0) + run;
        rowX0[y] = Math.min(rowX0[y] ?? width, x - run);
        rowX1[y] = Math.max(rowX1[y] ?? -1, x - 1);
      }
      run = 0;
    }
  }
  return { cover, rowX0, rowX1 };
}

/** The longest run of columns in x0..x1 with no ink on any of the rows y0..y1. */
function longestBreak(
  ink: Uint8Array,
  width: number,
  y0: number,
  y1: number,
  x0: number,
  x1: number
): number {
  let longest = 0;
  let gap = 0;
  for (let x = x0; x <= x1; x += 1) {
    let dark = false;
    for (let y = y0; y <= y1 && !dark; y += 1) {
      dark = ink[y * width + x] === 1;
    }
    gap = dark ? 0 : gap + 1;
    longest = Math.max(longest, gap);
  }
  return longest;
}

/**
 * The rows of a run of covered rows that make the line: all of them, or,
 * when a beam lies against the line (Peghead Nation beams a fifth-string
 * note just under the bottom line), the rows covered at least half as much
 * as the most covered one, when they are contiguous and every other row is
 * covered a quarter as much at most. A beam is a few spacings long, the
 * line the system's width; a scanned line fading over its edge rows is not
 * split.
 */
function lineRows(
  cover: Uint32Array,
  start: number,
  end: number
): { readonly end: number; readonly start: number } | undefined {
  let most = 0;
  for (let row = start; row < end; row += 1) {
    most = Math.max(most, cover[row] ?? 0);
  }
  const strong = (row: number) =>
    (cover[row] ?? 0) >= TAB.lineCoverShare * most;
  let first = start;
  while (first < end && !strong(first)) {
    first += 1;
  }
  let last = first;
  while (last < end && strong(last)) {
    last += 1;
  }
  for (let row = start; row < end; row += 1) {
    const inside = row >= first && row < last;
    if (
      !inside &&
      (strong(row) || (cover[row] ?? 0) > TAB.besideLineShare * most)
    ) {
      return;
    }
  }
  return { end: last, start: first };
}

/**
 * The thickness of the line a turned scan spread over rows start..end-1, too
 * many for a line: undefined unless they are a few rows too many, at most
 * `maxSlope` per page column, and the ink in their long runs is at most
 * `maxThick` over their length; then the ink down the median column that
 * has any, which `findLines` compares with the page's level lines. Beams
 * slanting across a staff fill such rows too, each as thick as a beam
 * (Moon over Shanghai's sixteenths, 5 rows against lines of 3 or 4).
 */
function spreadThickness(
  ink: Uint8Array,
  cover: Uint32Array,
  rows: { readonly end: number; readonly start: number },
  line: PageLine,
  width: number,
  maxThick: number
): number | undefined {
  const { end, start } = rows;
  if (end - start > maxThick + Math.round(width * TAB.maxSlope)) {
    return;
  }
  let covered = 0;
  for (let row = start; row < end; row += 1) {
    covered += cover[row] ?? 0;
  }
  if (covered / (line.x1 - line.x0 + 1) > maxThick) {
    return;
  }
  const thickness: number[] = [];
  for (let x = line.x0; x <= line.x1; x += 1) {
    let inked = 0;
    for (let row = start; row < end; row += 1) {
      inked += ink[row * width + x] ?? 0;
    }
    if (inked > 0) {
      thickness.push(inked);
    }
  }
  return medianOf(thickness);
}

const medianOf = (values: readonly number[]): number | undefined =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

/**
 * Rows covered by long horizontal runs, clustered into thin lines. A cluster
 * too thick for a line whose core rows are one (a beam lying against it) is
 * kept apart as `hidden`: groupLines takes it only to finish a group. A
 * cluster a few rows too tall that is a line a turned scan spread over them
 * (spreadThickness) is a line when it is no thicker than the page's level
 * lines, or `maxThick` on a page with none.
 */
function findLines(
  ink: Uint8Array,
  width: number,
  height: number
): { readonly hidden: PageLine[]; readonly lines: PageLine[] } {
  const { cover, rowX0, rowX1 } = rowCover(ink, width, height);
  const minCover = width * TAB.minRowCover;
  const maxThick = Math.max(
    TAB.maxThickness,
    Math.round(width * TAB.thicknessShare)
  );
  const lineOf = (start: number, end: number): PageLine => {
    let weight = 0;
    let sumY = 0;
    let x0 = width;
    let x1 = -1;
    for (let row = start; row < end; row += 1) {
      const covered = cover[row] ?? 0;
      weight += covered;
      sumY += row * covered;
      x0 = Math.min(x0, rowX0[row] ?? width);
      x1 = Math.max(x1, rowX1[row] ?? -1);
    }
    return {
      breakLength: longestBreak(ink, width, start, end - 1, x0, x1),
      x0,
      x1,
      y: sumY / weight,
      y0: start,
      y1: end - 1,
    };
  };
  const lines: PageLine[] = [];
  const hidden: PageLine[] = [];
  const spread: { line: PageLine; thickness: number }[] = [];
  let y = 0;
  while (y < height) {
    if ((cover[y] ?? 0) < minCover) {
      y += 1;
      continue;
    }
    let end = y;
    while (end < height && (cover[end] ?? 0) >= minCover) {
      end += 1;
    }
    const core = end - y <= maxThick ? undefined : lineRows(cover, y, end);
    if (end - y <= maxThick) {
      lines.push(lineOf(y, end));
    } else if (core !== undefined && core.end - core.start <= maxThick) {
      hidden.push(lineOf(core.start, core.end));
    } else {
      const line = lineOf(y, end);
      const thickness = spreadThickness(
        ink,
        cover,
        { end, start: y },
        line,
        width,
        maxThick
      );
      if (thickness !== undefined) {
        spread.push({ line, thickness });
      }
    }
    y = end;
  }
  const level = medianOf(lines.map((l) => l.y1 - l.y0 + 1)) ?? maxThick;
  const turned = spread.filter((s) => s.thickness <= level).map((s) => s.line);
  return {
    hidden,
    lines: [...lines, ...turned].sort((a, b) => a.y - b.y),
  };
}

/**
 * A group of three or four lines made five with the hidden lines a gap
 * below its last line or above its first, two at most, or left as it was: Peghead Nation beams notes
 * against a banjo tab's lines (a fifth-string note just under the bottom
 * line, low notes between the top ones), which thickens those lines out of
 * the line rows. Only five lines are taken: a staff missing lines to beams
 * is completed the same way and then fails the digit test, while four would
 * pass for a tab by their count alone.
 */
function finishGroup(
  group: PageLine[],
  hidden: readonly PageLine[],
  gap: number
): void {
  const first = group[0] as PageLine;
  const fits = (line: PageLine, y: number) =>
    Math.abs(line.y - y) <= gap * TAB.gapTolerance &&
    Math.min(line.x1, first.x1) - Math.max(line.x0, first.x0) >=
      TAB.minOverlap * Math.max(line.x1 - line.x0, first.x1 - first.x0) &&
    line.breakLength <= TAB.maxBreak * gap;
  const finished = [...group];
  for (
    let added = 0;
    added < TAB.maxHidden && finished.length < 5;
    added += 1
  ) {
    const below = hidden.find((line) =>
      fits(line, (finished.at(-1) as PageLine).y + gap)
    );
    const above = hidden.find((line) =>
      fits(line, (finished[0] as PageLine).y - gap)
    );
    if (below !== undefined) {
      finished.push(below);
    } else if (above === undefined) {
      break;
    } else {
      finished.unshift(above);
    }
  }
  if (finished.length === 5) {
    group.splice(0, group.length, ...finished);
  }
}

/**
 * The lines with every pair closer than `minGap` cut to the one with the
 * shorter longest break: two lines of a system are never that close, so the
 * other is a beam lying against the line (Reel de Mattawa at 300 dpi: a beam
 * row two pixels under a staff's top line, 11% of the page wide). Kept, the
 * beam row stopped the top line starting a group, could not start one itself,
 * and the staff's other four lines passed for a mandolin tab.
 */
function withoutBeamRows(lines: readonly PageLine[]): PageLine[] {
  const kept: PageLine[] = [];
  for (const line of lines) {
    const last = kept.at(-1);
    if (last === undefined || line.y - last.y >= TAB.minGap) {
      kept.push(line);
    } else if (line.breakLength < last.breakLength) {
      kept[kept.length - 1] = line;
    }
  }
  return kept;
}

/** Runs of 4 to 6 evenly spaced lines of about one extent; longer runs are not systems. */
function groupLines(
  found: ReturnType<typeof findLines>,
  pageHeight: number
): LineGroup[] {
  const { hidden } = found;
  const lines = withoutBeamRows(found.lines);
  const groups: LineGroup[] = [];
  let i = 0;
  while (i < lines.length - 1) {
    const first = lines[i] as PageLine;
    const gap = (lines[i + 1] as PageLine).y - first.y;
    if (
      gap > pageHeight * TAB.maxGapShare ||
      gap < TAB.minGap ||
      first.breakLength > TAB.maxBreak * gap
    ) {
      i += 1;
      continue;
    }
    const group = [first];
    let j = i + 1;
    for (; j < lines.length; j += 1) {
      const line = lines[j] as PageLine;
      const last = group.at(-1) as PageLine;
      const overlap = Math.min(line.x1, first.x1) - Math.max(line.x0, first.x0);
      if (
        Math.abs(line.y - last.y - gap) > gap * TAB.gapTolerance ||
        overlap <
          TAB.minOverlap * Math.max(line.x1 - line.x0, first.x1 - first.x0) ||
        line.breakLength > TAB.maxBreak * gap
      ) {
        break;
      }
      group.push(line);
    }
    if (group.length === 3 || group.length === 4) {
      finishGroup(group, hidden, gap);
    }
    if (group.length >= 4 && group.length <= 6) {
      groups.push({
        lines: group,
        spacing:
          ((group.at(-1) as PageLine).y - (group[0] as PageLine).y) /
          (group.length - 1),
        x0: Math.min(...group.map((l) => l.x0)),
        x1: Math.max(...group.map((l) => l.x1)),
      });
      i = j;
    } else {
      i += 1;
    }
  }
  return groups;
}

/** The band of a group that carries its marks, in page rows. */
function bandOf(
  group: LineGroup,
  height: number,
  margin: number,
  below = margin
) {
  const top = (group.lines[0] as PageLine).y;
  const bottom = (group.lines.at(-1) as PageLine).y;
  return {
    bottom: Math.min(height - 1, Math.ceil(bottom + below * group.spacing)),
    top: Math.max(0, Math.floor(top - margin * group.spacing)),
  };
}

/**
 * The band with its stems erased, in place: ink no wider than a stroke
 * (strokeWidth spacings) that runs on, a column either side allowed per row
 * for a pen's slant, over stemLength spacings. A hand's fret is joined to its
 * stem (Sourwood Mountain's circles hang theirs to a beam under the tab), and
 * fret and stem make one shape too tall to be a fret; with the stem gone the
 * fret stands alone. A circle's side is a stroke too, but shorter, and its
 * top and bottom arcs are wider than one, so a stem stops at the circle.
 */
function eraseStems(band: Band, cols: number, spacing: number): void {
  const { data, rows } = band;
  const thin = thinInk(
    band,
    cols,
    Math.max(3, Math.round(TAB.strokeWidth * spacing))
  );
  const { down, up } = thinRuns(thin, cols, rows);
  const length = TAB.stemLength * spacing;
  const erased = new Uint8Array(cols * rows);
  for (let i = 0; i < data.length; i += 1) {
    if (thin[i] === 1 && (down[i] ?? 0) + (up[i] ?? 0) - 1 >= length) {
      data[i] = 0;
      erased[i] = 1;
    }
  }
  eraseStubs(band, cols, thin, erased, Math.ceil(TAB.stubLength * spacing));
}

/** 1 on the band's ink in horizontal runs no wider than `maxWidth`. */
function thinInk(band: Band, cols: number, maxWidth: number): Uint8Array {
  const { data, rows } = band;
  const thin = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y += 1) {
    let x = 0;
    while (x < cols) {
      if (data[y * cols + x] !== 1) {
        x += 1;
        continue;
      }
      const start = x;
      while (x < cols && data[y * cols + x] === 1) {
        x += 1;
      }
      if (x - start <= maxWidth) {
        thin.fill(1, y * cols + start, y * cols + x);
      }
    }
  }
  return thin;
}

/** The longest slanting run of thin ink through each pixel, a column aside allowed per row: down from above, up from below. */
function thinRuns(
  thin: Uint8Array,
  cols: number,
  rows: number
): { readonly down: Uint16Array; readonly up: Uint16Array } {
  const down = new Uint16Array(cols * rows);
  const up = new Uint16Array(cols * rows);
  const best = (run: Uint16Array, y: number, x: number) =>
    Math.max(
      x > 0 ? (run[y * cols + x - 1] ?? 0) : 0,
      run[y * cols + x] ?? 0,
      x + 1 < cols ? (run[y * cols + x + 1] ?? 0) : 0
    );
  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      if (thin[y * cols + x] === 1) {
        down[y * cols + x] = 1 + (y > 0 ? best(down, y - 1, x) : 0);
      }
    }
  }
  for (let y = rows - 1; y >= 0; y -= 1) {
    for (let x = 0; x < cols; x += 1) {
      if (thin[y * cols + x] === 1) {
        up[y * cols + x] = 1 + (y + 1 < rows ? best(up, y + 1, x) : 0);
      }
    }
  }
  return { down, up };
}

/**
 * A stem's last rows, where it meets the fret, run into the fret's arc and
 * stop counting as thin a few pixels early: the stub left under a circle
 * makes it an omega, too tall and off its line. Thin ink touching an erased
 * stem goes too, row by row, for at most `passes` rows.
 */
function eraseStubs(
  band: Band,
  cols: number,
  thin: Uint8Array,
  erased: Uint8Array,
  passes: number
): void {
  const { data, rows } = band;
  const touchesErased = (y: number, x: number) =>
    [-1, 0, 1].some((dx) =>
      [-1, 1].some((dy) => {
        const yy = y + dy;
        const xx = x + dx;
        return (
          yy >= 0 &&
          yy < rows &&
          xx >= 0 &&
          xx < cols &&
          erased[yy * cols + xx] === 1
        );
      })
    );
  for (let pass = 0; pass < passes; pass += 1) {
    const next: number[] = [];
    for (let y = 0; y < rows; y += 1) {
      for (let x = 0; x < cols; x += 1) {
        const i = y * cols + x;
        if (data[i] === 1 && thin[i] === 1 && touchesErased(y, x)) {
          next.push(i);
        }
      }
    }
    if (next.length === 0) {
      return;
    }
    for (const i of next) {
      data[i] = 0;
      erased[i] = 1;
    }
  }
}

/**
 * The group's band with its lines erased: a line pixel goes when nothing
 * touches the line there from above or below, so a digit the line crosses
 * keeps its middle, and when it belongs to the line itself: a horizontal run
 * at least a line spacing long, or a shorter one with a free end (a stub of
 * line left beside a knock-out). The bar of an H standing on the line inside
 * a knock-out is short and held by a stem at both ends, so it stays. With
 * `jogs`, a stroke turning along the line keeps the line under its turn
 * (bridgeJogs): reading a system whose frets stand over its lines.
 */
export function bandWithoutLines(
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup,
  reach: { readonly above: number; readonly below: number } = {
    above: TAB.markBand,
    below: TAB.markBand,
  },
  jogs = false
): Band {
  const { bottom, top } = bandOf(group, height, reach.above, reach.below);
  const cols = group.x1 - group.x0 + 1;
  const rows = bottom - top + 1;
  const data = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y += 1) {
    const from = (y + top) * width + group.x0;
    data.set(ink.subarray(from, from + cols), y * cols);
  }
  for (const line of group.lines) {
    const a = line.y0 - 1 - top;
    const b = line.y1 + 1 - top;
    const free = freeColumns(
      data,
      cols,
      rows,
      a,
      b,
      jogs ? TAB.strokeWidth * group.spacing : undefined
    );
    const ruled = ruledColumns(data, cols, line.y0 - top, line.y1 - top, {
      free,
      minRun: group.spacing,
    });
    for (let x = 0; x < cols; x += 1) {
      if (ruled[x] === 1 && free[x] === 1) {
        for (let y = Math.max(0, a); y <= Math.min(rows - 1, b); y += 1) {
          data[y * cols + x] = 0;
        }
      }
    }
  }
  return { data, rows, top };
}

/**
 * 1 for the columns where nothing touches the line from the row above `a`
 * or below `b`, but for a stroke's jog along the line shorter than `jogGap`
 * when one is given (bridgeJogs).
 */
function freeColumns(
  data: Uint8Array,
  cols: number,
  rows: number,
  a: number,
  b: number,
  jogGap: number | undefined
): Uint8Array {
  const free = new Uint8Array(cols);
  const above = new Uint8Array(cols);
  const below = new Uint8Array(cols);
  for (let x = 0; x < cols; x += 1) {
    above[x] = a - 1 >= 0 && data[(a - 1) * cols + x] === 1 ? 1 : 0;
    below[x] = b + 1 < rows && data[(b + 1) * cols + x] === 1 ? 1 : 0;
    free[x] = above[x] === 1 || below[x] === 1 ? 0 : 1;
  }
  if (jogGap !== undefined) {
    bridgeJogs(free, above, below, jogGap);
  }
  return free;
}

/**
 * Holds, in place, the free columns of a line between a stroke leaving it
 * upwards and one leaving it downwards, when they are less than `maxGap`
 * apart and at most `jogWidths` times the narrower stroke's width: one
 * stroke of a glyph turning along the line for a pen's width or two. A
 * hand's 5 comes down to the line and starts its bowl four pixels to the
 * right (strokes four wide); erasing the line there cut it in two (Weeping
 * Willow), the top read as a short mark and the bottom as nothing. Strokes
 * touching from one side only (two frets sitting on the line), a stem
 * crossing it, and strokes further apart than a turn (a printed letter and
 * a digit on a line, 9 to 16 px apart for strokes 2 to 3 wide on Cripple
 * Creek) join nothing.
 */
function bridgeJogs(
  free: Uint8Array,
  above: Uint8Array,
  below: Uint8Array,
  maxGap: number
): void {
  const side = (at: number) => {
    if (above[at] === 1 && below[at] !== 1) {
      return above;
    }
    return below[at] === 1 && above[at] !== 1 ? below : undefined;
  };
  const widthFrom = (row: Uint8Array, at: number, step: 1 | -1) => {
    let k = at;
    while (k >= 0 && k < row.length && row[k] === 1) {
      k += step;
    }
    return Math.abs(k - at);
  };
  let x = 0;
  while (x < free.length) {
    if (free[x] !== 1) {
      x += 1;
      continue;
    }
    const start = x;
    while (x < free.length && free[x] === 1) {
      x += 1;
    }
    const left = side(start - 1);
    const right = side(x);
    if (
      left === undefined ||
      right === undefined ||
      left === right ||
      x - start >= maxGap
    ) {
      continue;
    }
    const narrower = Math.min(
      widthFrom(left, start - 1, -1),
      widthFrom(right, x, 1)
    );
    if (x - start <= TAB.jogWidths * narrower) {
      free.fill(0, start, x);
    }
  }
}

/** 1 for the columns of the line's rows that lie in a run belonging to the line (see bandWithoutLines). */
function ruledColumns(
  data: Uint8Array,
  cols: number,
  y0: number,
  y1: number,
  { free, minRun }: { readonly free: Uint8Array; readonly minRun: number }
): Uint8Array {
  const ruled = new Uint8Array(cols);
  for (let y = y0; y <= y1; y += 1) {
    let run = 0;
    for (let x = 0; x <= cols; x += 1) {
      if (x < cols && data[y * cols + x] === 1) {
        run += 1;
        continue;
      }
      if (
        run >= minRun ||
        (run > 0 && (free[x - run] === 1 || free[x - 1] === 1))
      ) {
        ruled.fill(1, x - run, x);
      }
      run = 0;
    }
  }
  return ruled;
}

/** Connected components of a 0/1 band, 4-connected as the prototype's flood fill: boxes, and each pixel's label (box k is label k + 1). */
function componentsOf(
  cv: OpenCv,
  band: Uint8Array,
  cols: number,
  rows: number
): { readonly boxes: Box[]; readonly labels: Int32Array } {
  return withMatScope((scope) => {
    const mat = scope.keep(new cv.Mat(rows, cols, cv.CV_8UC1));
    mat.data.set(band);
    const labels = scope.keep(new cv.Mat());
    const stats = scope.keep(new cv.Mat());
    const centroids = scope.keep(new cv.Mat());
    const count = cv.connectedComponentsWithStats(
      mat,
      labels,
      stats,
      centroids,
      4,
      cv.CV_32S
    );
    const s = stats.data32S;
    const boxes: Box[] = [];
    for (let k = 1; k < count; k += 1) {
      boxes.push({
        h: s[k * 5 + 3] ?? 0,
        w: s[k * 5 + 2] ?? 0,
        x: s[k * 5] ?? 0,
        y: s[k * 5 + 1] ?? 0,
      });
    }
    return { boxes, labels: new Int32Array(labels.data32S) };
  });
}

/** Connected components' boxes of a 0/1 band, 4-connected. */
export const componentBoxes = (
  cv: OpenCv,
  band: Uint8Array,
  cols: number,
  rows: number
): Box[] => componentsOf(cv, band, cols, rows).boxes;

/**
 * Rows blanked half way between the lines and half a spacing outside the
 * outer ones, in place; 1 for a cut row. A fret is centred on its line and no
 * taller than a spacing, so the cuts pass between frets: they part numbers
 * stacked in a chord that touch once the line is erased, a number from the
 * stem hung on its own ink (Peghead Nation), and a number from a slur arc
 * resting on it. What a cut leaves of a stem, a barline or a large glyph
 * reaches a cut at both ends, or is a stem's width at one.
 */
function cutBetweenLines(
  band: Band,
  cols: number,
  group: LineGroup
): Uint8Array {
  const cut = new Uint8Array(band.rows);
  const first = (group.lines[0] as PageLine).y;
  const last = (group.lines.at(-1) as PageLine).y;
  const at = [
    first - TAB.cutOutside * group.spacing,
    ...group.lines
      .slice(1)
      .map((line, k) => (line.y + (group.lines[k] as PageLine).y) / 2),
    last + TAB.cutOutside * group.spacing,
  ];
  for (const y of at) {
    for (const row of new Set([
      Math.floor(y - band.top),
      Math.ceil(y - band.top),
    ])) {
      if (row >= 0 && row < band.rows) {
        cut[row] = 1;
        band.data.fill(0, row * cols, (row + 1) * cols);
      }
    }
  }
  return cut;
}

/**
 * Whether the ink of `box`'s columns carries on across the cut at its top
 * (`direction` -1) or bottom (1) for 0.3 spacing in every row: the stroke
 * of something taller than a fret, such as the stem of the "T" in a TAB clef.
 * A fret stacked on another meets that one's curve and then its hole.
 */
function continuesPast(
  whole: Uint8Array,
  cols: number,
  cut: Uint8Array,
  box: Box,
  direction: -1 | 1,
  spacing: number
): boolean {
  let row = direction < 0 ? box.y - 1 : box.y + box.h;
  while (row >= 0 && row < cut.length && cut[row] === 1) {
    row += direction;
  }
  for (let k = 0; k < Math.round(TAB.strokeReach * spacing); k += 1) {
    const y = row + direction * k;
    if (y < 0 || y >= cut.length) {
      return false;
    }
    let inked = false;
    for (let c = 0; c < box.w && !inked; c += 1) {
      inked = whole[y * cols + box.x + c] === 1;
    }
    if (!inked) {
      return false;
    }
  }
  return true;
}

/**
 * A piece grown back by the one cut row next to it on each side it touches:
 * that row may have shaved the fret's top or bottom, while the row beyond
 * belongs to what the cut parted it from (the fret stacked over it, whose
 * sliver in a crop reads a 0 as "ō").
 */
function grownOverCuts(box: Box, cut: Uint8Array): Box {
  const top = box.y > 0 && cut[box.y - 1] === 1 ? box.y - 1 : box.y;
  const end = box.y + box.h;
  const bottom = end < cut.length && cut[end] === 1 ? end + 1 : end;
  return { ...box, h: bottom - top, y: top };
}

/**
 * Whether a component is a fret with its stem hung from its own ink
 * (Peghead Nation): at least 1.5 spacings tall and no wider than a fret,
 * with a head of a fret's height (0.5 to 1.15 spacings) above rows that are
 * a stem's width all the way down (80% of them), hung from the head's
 * middle, the head at least `hangingHeadWidth` times as wide as the stem. A
 * staff's note hangs its stem from the head's left edge; a "T" of a TAB clef
 * has a head as thin as its bar; and a pen's stem, a little wider than a
 * printed one, has no head at all: on Weeping Willow two of them passed for
 * frets hanging their stems (rows 4 to 7 px wide over 0.2 spacings, 3.7 px),
 * which cut the hand's stems between the lines into pieces read as 1s.
 */
function hangsAStem(band: Band, cols: number, box: Box, s: number): boolean {
  if (box.h < TAB.hangingHeight * s || box.w > TAB.maxMarkWidth * s) {
    return false;
  }
  const inkedOn = (y: number) => {
    let inked = 0;
    for (let c = 0; c < box.w; c += 1) {
      inked += band.data[y * cols + box.x + c] ?? 0;
    }
    return inked;
  };
  const thin = (y: number) => inkedOn(y) <= TAB.cutPieceWidth * s;
  let head = 0;
  for (let y = 0; y < Math.min(box.h, TAB.hangingReach * s); y += 1) {
    head = thin(box.y + y) ? head : y + 1;
  }
  if (head < TAB.hangingHead * s || head > TAB.maxMarkHeight * s) {
    return false;
  }
  let stem = 0;
  let sumX = 0;
  let inked = 0;
  const stemWidths: number[] = [];
  for (let y = box.y + head; y < box.y + box.h; y += 1) {
    stem += thin(y) ? 1 : 0;
    stemWidths.push(inkedOn(y));
    for (let c = 0; c < box.w; c += 1) {
      if (band.data[y * cols + box.x + c] === 1) {
        sumX += c;
        inked += 1;
      }
    }
  }
  const headWidth = Math.max(
    ...Array.from({ length: head }, (_, y) => inkedOn(box.y + y))
  );
  return (
    stem >= TAB.hangingThinShare * (box.h - head) &&
    headWidth >=
      TAB.hangingHeadWidth * (medianOf(stemWidths.filter((w) => w > 0)) ?? 0) &&
    Math.abs(sumX / Math.max(1, inked) - box.w / 2) <=
      TAB.hangingCentred * box.w
  );
}

/** The largest mark taken for a fret, in line spacings. */
export interface MarkSize {
  readonly maxHeight: number;
  readonly maxWidth: number;
}

/** Deciding whether a group is a tab: printed frets, smaller than a staff's noteheads. */
const PRINTED_MARK: MarkSize = {
  maxHeight: TAB.maxMarkHeight,
  maxWidth: TAB.maxMarkWidth,
};

/**
 * Reading a system already known to be a tab: a hand writes its digits
 * larger than a line space (Weeping Willow's zeros are 1.05 to 1.3 wide and
 * up to 1.25 tall), which deciding whether a group is a tab must not take,
 * as a staff's noteheads are that size too. A mark taller than a printed
 * fret must still be wider than a stroke: a piece of stem crossing a line is
 * that tall too (Arkansas Traveler's TablEdit stems read as 1s).
 */
export const READ_MARK: MarkSize = { maxHeight: 1.3, maxWidth: 1.35 };

/**
 * A system found by the hand-written second chance: a hand's frets reach two
 * spacings (Sourwood Mountain's circles are 1.9 to 2.1 tall and wide), hung on
 * stems that eraseStems removes first. Printed systems never take this size:
 * their stem-hanging frets are cut by fretSized's own rules (Billy in the
 * Lowground, Old Joe Clark lost bars when they did).
 */
export const HAND_MARK: MarkSize = { maxHeight: 2.2, maxWidth: 2.4 };

/**
 * The band's components of a fret's size. On a system whose frets hang
 * their stems (at least two do), the components too big to be a fret are
 * cut between the lines (cutBetweenLines) and their pieces grown back over
 * the cuts; the rest stand as they are, and so does every component of a
 * system that hangs no stems (docs/design/tab-4-findings.md). `size` bounds
 * the components of a system that hangs no stems; one that hangs them is
 * printed, and is cut and bounded at the printed size.
 */
function fretSized(
  cv: OpenCv,
  band: Band,
  cols: number,
  group: LineGroup,
  size: MarkSize
): Box[] {
  const s = group.spacing;
  const sizedBy =
    ({ maxHeight, maxWidth }: MarkSize) =>
    (box: Box) =>
      box.h >= TAB.minMarkHeight * s &&
      box.h <= maxHeight * s &&
      (box.h <= TAB.maxMarkHeight * s || box.w >= TAB.strokeWidth * s) &&
      box.w >= Math.max(2, TAB.minMarkWidth * s) &&
      box.w <= maxWidth * s;
  const { boxes, labels } = componentsOf(cv, band.data, cols, band.rows);
  if (
    boxes.filter((box) => hangsAStem(band, cols, box, s)).length <
    TAB.minHanging
  ) {
    return boxes.filter(sizedBy(size));
  }
  const sized = sizedBy(PRINTED_MARK);
  const big = new Set(
    boxes.flatMap((box, k) =>
      box.h > TAB.maxMarkHeight * s || box.w > TAB.maxMarkWidth * s
        ? [k + 1]
        : []
    )
  );
  const whole = band.data.map((_, i) => (big.has(labels[i] ?? 0) ? 1 : 0));
  const cutBand = { ...band, data: whole.slice() };
  const cut = cutBetweenLines(cutBand, cols, group);
  const pieces = componentBoxes(cv, cutBand.data, cols, cutBand.rows).flatMap(
    (piece) => {
      const above = piece.y === 0 || cut[piece.y - 1] === 1;
      const below =
        piece.y + piece.h >= cutBand.rows || cut[piece.y + piece.h] === 1;
      const stroke =
        above &&
        below &&
        piece.w < TAB.strokeWidth * s &&
        continuesPast(whole, cols, cut, piece, -1, s) &&
        continuesPast(whole, cols, cut, piece, 1, s);
      const box = grownOverCuts(piece, cut);
      return ((above || below) && piece.w < TAB.cutPieceWidth * s) ||
        stroke ||
        solid(whole, cols, box) ||
        !sized(box)
        ? []
        : [box];
    }
  );
  return [...boxes.filter(sized), ...pieces];
}

/** Whether `box` is nearly all ink: a piece of a thick barline, never a digit. */
function solid(ink: Uint8Array, cols: number, box: Box): boolean {
  let inked = 0;
  for (let y = 0; y < box.h; y += 1) {
    for (let c = 0; c < box.w; c += 1) {
      inked += ink[(box.y + y) * cols + box.x + c] ?? 0;
    }
  }
  return inked >= TAB.solidShare * box.w * box.h;
}

/**
 * Digit-sized components on a line, touching ones merged ("12"), in page
 * pixels, ordered by line then x. `short` holds the marks under 0.75 of the
 * median height: arcs cut by a stem, and x-height letters ("x", "o").
 * `size` bounds the marks taken.
 */
export function marksOf(
  cv: OpenCv,
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup,
  size: MarkSize = PRINTED_MARK,
  jogs = false
): { readonly marks: Mark[]; readonly short: Mark[] } {
  const s = group.spacing;
  const band = bandWithoutLines(ink, width, height, group, undefined, jogs);
  const cols = group.x1 - group.x0 + 1;
  if (size === HAND_MARK) {
    eraseStems(band, cols, s);
  }
  const onLines: Mark[] = [];
  for (const box of fretSized(cv, band, cols, group, size)) {
    const y = box.y + band.top;
    const centre = y + box.h / 2;
    let line = 0;
    let distance = Number.POSITIVE_INFINITY;
    for (const [k, l] of group.lines.entries()) {
      if (Math.abs(l.y - centre) < distance) {
        distance = Math.abs(l.y - centre);
        line = k;
      }
    }
    if (distance <= TAB.onLine * s) {
      onLines.push({ ...box, line, x: box.x + group.x0, y });
    }
  }
  onLines.sort((a, b) => a.line - b.line || a.x - b.x);
  const merged: Mark[] = [];
  for (const mark of onLines) {
    const last = merged.at(-1);
    if (
      last !== undefined &&
      last.line === mark.line &&
      mark.x - (last.x + last.w) < TAB.mergeGap * s &&
      mark.x + mark.w - last.x < TAB.mergeWidth * s
    ) {
      const x = Math.min(last.x, mark.x);
      const y = Math.min(last.y, mark.y);
      merged[merged.length - 1] = {
        h: Math.max(last.y + last.h, mark.y + mark.h) - y,
        line: last.line,
        w: Math.max(last.x + last.w, mark.x + mark.w) - x,
        x,
        y,
      };
    } else {
      merged.push(mark);
    }
  }
  const heights = merged.map((m) => m.h).sort((a, b) => a - b);
  const median = heights[Math.floor(heights.length / 2)] ?? 0;
  const tall = (m: Mark) => m.h >= TAB.minMedianHeight * median;
  return {
    marks: merged.filter(tall),
    short: merged.filter((m) => !tall(m)),
  };
}

/**
 * Where a five-line group's marks sit at a hand's size (HAND_MARK), its stems
 * erased first: on its lines, as a tab's frets do, or in its spaces, where a
 * staff's noteheads sit as often as on its lines.
 */
function handPlacement(
  cv: OpenCv,
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup
): { readonly inSpaces: number; readonly onLines: number } {
  const s = group.spacing;
  const band = bandWithoutLines(ink, width, height, group);
  const cols = group.x1 - group.x0 + 1;
  eraseStems(band, cols, s);
  const ys = group.lines.map((line) => line.y);
  let onLines = 0;
  let inSpaces = 0;
  for (const box of fretSized(cv, band, cols, group, HAND_MARK)) {
    const centre = box.y + band.top + box.h / 2;
    const toLine = Math.min(...ys.map((y) => Math.abs(y - centre)));
    const toSpace = Math.min(
      ...ys.slice(1).map((y, k) => Math.abs((y + (ys[k] ?? y)) / 2 - centre))
    );
    if (toLine <= TAB.onLine * s) {
      onLines += 1;
    } else if (toSpace <= TAB.onLine * s) {
      inSpaces += 1;
    }
  }
  return { inSpaces, onLines };
}

/** At most handInSpaceShare of the marks on the lines stand in the spaces: no staff's noteheads. */
const sitsOnLines = ({
  inSpaces,
  onLines,
}: ReturnType<typeof handPlacement>): boolean =>
  inSpaces <= TAB.handInSpaceShare * onLines;

/**
 * A five-line group's marks at a hand's size (HAND_MARK), when they sit on its
 * lines as a tab's frets do: at least minDigits on lines, and in its spaces at
 * most handInSpaceShare of that, which a staff's noteheads never manage.
 * Undefined otherwise. Sourwood Mountain's zeros are circles 1.4 spacings
 * wide, over a printed fret's size.
 */
function handWrittenMarks(
  cv: OpenCv,
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup
): Mark[] | undefined {
  const placement = handPlacement(cv, ink, width, height, group);
  if (placement.onLines < TAB.minDigits || !sitsOnLines(placement)) {
    return;
  }
  return marksOf(cv, ink, width, height, group, HAND_MARK).marks;
}

/**
 * The page's ink with the tab's lines erased over its band, so a crop of a
 * number the line runs through shows the number alone: the recogniser reads
 * "12" with a line across it as "2".
 */
export function inkWithoutLines(
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup,
  jogs = false
): Uint8Array {
  const band = bandWithoutLines(ink, width, height, group, undefined, jogs);
  const cols = group.x1 - group.x0 + 1;
  const out = ink.slice();
  for (let y = 0; y < band.rows; y += 1) {
    out.set(
      band.data.subarray(y * cols, (y + 1) * cols),
      (y + band.top) * width + group.x0
    );
  }
  return out;
}

/** The mark's own ink on white, padded, so the recogniser sees a bare number. */
export function cropOf(
  page: ColorImage,
  ink: Uint8Array,
  mark: Mark,
  spacing: number
): ColorImage {
  const { height, width } = page;
  const { h, w, x: left, y: top } = mark;
  const pad = Math.round(spacing * TAB.cropPad);
  const x0 = Math.max(0, left - pad);
  const x1 = Math.min(width - 1, left + w + pad);
  const y0 = Math.max(0, top - pad);
  const y1 = Math.min(height - 1, top + h + pad);
  const crop = createColor(x1 - x0 + 1, y1 - y0 + 1);
  for (let y = top; y < Math.min(top + h, height); y += 1) {
    for (let x = left; x < Math.min(left + w, width); x += 1) {
      if (ink[y * width + x] === 1) {
        const at = ((y - y0) * crop.width + (x - x0)) * 3;
        crop.data.fill(0, at, at + 3);
      }
    }
  }
  return crop;
}

/**
 * Up to `sampleSize` marks spread evenly over the group (marks are ordered by
 * line, then left to right), so the sample sees every line.
 */
function sampleOf<T>(marks: readonly T[]): T[] {
  if (marks.length <= TAB.sampleSize) {
    return [...marks];
  }
  return Array.from(
    { length: TAB.sampleSize },
    (_, k) => marks[Math.floor((k * marks.length) / TAB.sampleSize)] as T
  );
}

/**
 * Whether a five-line group's marks read as numbers: a sample read a batch
 * at a time, stopping as soon as the share read so far decides. A tab's
 * first batch decides alone (the prototype measured shares of 0.87 to 1.00
 * on tabs, 0.00 on staves).
 */
async function marksAreNumbers(
  crops: readonly ColorImage[],
  read: ReadCrops
): Promise<boolean> {
  const sample = sampleOf(crops);
  let digits = 0;
  let seen = 0;
  for (let start = 0; start < sample.length; start += TAB.readBatch) {
    const batch = sample.slice(start, start + TAB.readBatch);
    // biome-ignore lint/performance/noAwaitInLoops: each batch may decide, and the next is then never read
    digits += (await read(batch)).filter(isFretText).length;
    seen += batch.length;
    if (digits < TAB.minDigitShare * seen) {
      return false;
    }
    if (digits >= TAB.minDigits) {
      return true;
    }
  }
  return false;
}

/** The share of the marks whose line is blank just left and just right of them; undefined under `minDigits` marks. */
function knockedOutShare(
  ink: Uint8Array,
  width: number,
  group: LineGroup,
  marks: readonly Mark[]
): number | undefined {
  if (marks.length < TAB.minDigits) {
    return;
  }
  const reach = Math.max(2, Math.round(TAB.knockOutReach * group.spacing));
  const blank = (line: PageLine, x: number) => {
    if (x < 0 || x >= width) {
      return false;
    }
    for (let y = line.y0; y <= line.y1; y += 1) {
      if (ink[y * width + x] === 1) {
        return false;
      }
    }
    return true;
  };
  const blankWithin = (line: PageLine, from: number, step: 1 | -1) =>
    Array.from({ length: reach }, (_, k) => from + step * (k + 1)).some((x) =>
      blank(line, x)
    );
  const knocked = marks.filter((mark) => {
    const line = group.lines[mark.line] as PageLine;
    return (
      blankWithin(line, mark.x, -1) && blankWithin(line, mark.x + mark.w - 1, 1)
    );
  });
  return knocked.length / marks.length;
}

/**
 * One program prints a page's tabs one way and at one size: when a tab on the
 * page knocks its numbers out, a group whose numbers sit on unbroken lines and
 * whose line spacing is another size is a staff (the fingering digits of
 * Guitar Pro's staves, or a staff missing the line a beam thickened). Pages
 * whose tabs print over their lines keep every tab, and so does a tab without
 * knock-outs drawn at the knocked-out tabs' size.
 */
function keepPrintedAlike(tabs: readonly Candidate[]): Candidate[] {
  const knocks = (c: Candidate) =>
    c.knockedOut !== undefined && c.knockedOut >= TAB.minKnockOutShare;
  const sizes = tabs.filter(knocks).map((c) => c.group.spacing);
  const sameSize = (c: Candidate) =>
    sizes.some(
      (size) => Math.abs(c.group.spacing - size) <= TAB.sameSizeTolerance * size
    );
  return tabs.filter(
    (c) =>
      sizes.length === 0 ||
      c.knockedOut === undefined ||
      knocks(c) ||
      sameSize(c)
  );
}

/**
 * The five-line groups the digit test refused that are drawn like the page's
 * five-line tabs, once it has `minAlikeTabs` of them: a line spacing within
 * `alikeSpacing` of theirs, and marks standing on the lines rather than in the
 * spaces (`onLines`), as no staff's noteheads do. A hand writes some lines
 * with too few printed-size digits to pass: Weeping Willow's endings line holds
 * four, one a zero read "日". A program draws a staff over its tab smaller
 * than the tab (17.7 against 26.6 px to 27 against 41 on the local
 * score-and-tab pages), and a staff at the tab's size still has noteheads in
 * its spaces.
 */
function alikeFiveLineTabs(
  tabs: readonly Candidate[],
  refused: readonly Candidate[],
  onLines: (group: LineGroup) => boolean
): Candidate[] {
  const spacings = tabs
    .filter((c) => c.group.lines.length === 5 && c.evidence !== "tab")
    .map((c) => c.group.spacing);
  if (spacings.length < TAB.minAlikeTabs) {
    return [];
  }
  const alike = (spacing: number) =>
    spacings.every((s) => Math.abs(spacing - s) <= TAB.alikeSpacing * s);
  return refused.filter(
    (c) =>
      c.group.lines.length === 5 && alike(c.group.spacing) && onLines(c.group)
  );
}

const toSystem = (
  group: LineGroup,
  index: number,
  width: number,
  height: number
): TabSystem => {
  const top = (group.lines[0] as PageLine).y;
  const bottom = (group.lines.at(-1) as PageLine).y;
  return {
    cx: (group.x0 + group.x1) / 2 / width,
    cy: (top + bottom) / 2 / height,
    h: (bottom - top) / height,
    index,
    lines: group.lines.length as TabLineCount,
    w: (group.x1 - group.x0) / width,
  };
};

/**
 * Every tab system on the page, top to bottom. `read` is called only for a
 * five-line group with at least four line-sitting marks, so a page of
 * ordinary staves never reaches the recogniser.
 */
export async function detectTablature(
  cv: OpenCv,
  page: ColorImage,
  read: ReadCrops
): Promise<DetectedTab[]> {
  const ink = withMatScope(
    (scope) => new Uint8Array(inkOf(cv, scope, page).data)
  );
  const groups = withMatScope((scope) => {
    const faint = faintInkOf(cv, scope, page);
    return groupLines(findLines(ink, page.width, page.height), page.height).map(
      (found): Candidate => {
        const group = withFaintLine(faint, page.width, page.height, found);
        const marks = marksOf(
          cv,
          ink,
          page.width,
          page.height,
          group
        ).marks.filter(
          (mark) =>
            !crossesAsStroke(
              faint,
              page.width,
              page.height,
              mark,
              group.spacing
            )
        );
        const knockedOut = knockedOutShare(ink, page.width, group, marks);
        if (group.lines.length !== 5) {
          return { evidence: "tab", group, knockedOut };
        }
        if (marks.length >= TAB.minDigits) {
          return {
            evidence: marks.map((mark) =>
              cropOf(page, ink, mark, group.spacing)
            ),
            group,
            knockedOut,
          };
        }
        const hand = handWrittenMarks(cv, ink, page.width, page.height, group);
        const lineless =
          hand === undefined
            ? ink
            : inkWithoutLines(ink, page.width, page.height, group);
        return {
          ...(hand === undefined ? {} : { hand: true }),
          evidence:
            hand === undefined
              ? "staff"
              : hand.map((mark) => cropOf(page, lineless, mark, group.spacing)),
          group,
          knockedOut,
        };
      }
    );
  });
  const candidates: Candidate[] = [];
  const unread: Candidate[] = [];
  for (const candidate of groups) {
    const { evidence } = candidate;
    const isTab =
      evidence === "tab" ||
      // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, one batch at a time
      (evidence !== "staff" && (await marksAreNumbers(evidence, read)));
    if (isTab) {
      candidates.push(candidate);
    } else {
      unread.push(candidate);
    }
  }
  const alike = alikeFiveLineTabs(candidates, unread, (group) =>
    sitsOnLines(handPlacement(cv, ink, page.width, page.height, group))
  );
  return keepPrintedAlike(
    [...candidates, ...alike].sort(
      (a, b) =>
        (a.group.lines[0] as PageLine).y - (b.group.lines[0] as PageLine).y
    )
  ).map(({ group, hand, knockedOut }, index) => ({
    group,
    ...(hand === true ? { hand } : {}),
    ...(knockedOut !== undefined && knockedOut >= TAB.minKnockOutShare
      ? { knocksOut: true }
      : {}),
    system: toSystem(group, index, page.width, page.height),
  }));
}

/** Rows painted white over each tab, `margin` line spacings beyond its outer lines; a new page, the input untouched. */
export function whitenTablature(
  page: ColorImage,
  tabs: readonly DetectedTab[],
  margin: number
): ColorImage {
  const out = createColor(page.width, page.height);
  out.data.set(page.data);
  for (const { group } of tabs) {
    const { bottom, top } = bandOf(group, page.height, margin);
    const pad = Math.round(group.spacing);
    const x0 = Math.max(0, group.x0 - pad);
    const x1 = Math.min(page.width - 1, group.x1 + pad);
    for (let y = top; y <= bottom; y += 1) {
      out.data.fill(
        255,
        (y * page.width + x0) * 3,
        (y * page.width + x1 + 1) * 3
      );
    }
  }
  return out;
}

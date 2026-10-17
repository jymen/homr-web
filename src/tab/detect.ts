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
 * line-sitting marks, which only a recogniser can settle.
 */
type Evidence = "tab" | "staff" | readonly ColorImage[];

/** A line group, what decides whether it is a tab, and the share of its marks knocked out (undefined under four marks). */
interface Candidate {
  readonly evidence: Evidence;
  readonly group: LineGroup;
  readonly knockedOut: number | undefined;
}

/** Reads crops as text; the pipeline's reader is RapidOCR's recogniser, opened on the first call. */
export type ReadCrops = (crops: readonly ColorImage[]) => Promise<string[]>;

const TAB = {
  /** Rows of a line touched by a beam: those covered at least this share of the most covered row, the others at most `besideLineShare`. */
  besideLineShare: 0.25,
  /** Crop padding, in line spacings. */
  cropPad: 0.25,
  /** The outer cuts lie this far outside the outer lines, in line spacings (see cutBetweenLines). */
  cutOutside: 0.5,
  /** What a cut leaves at one end narrower than this, in line spacings, is a piece of stem. */
  cutPieceWidth: 0.2,
  gapTolerance: 0.2,
  /** hangsAStem: at least this tall, the head at least this tall, and this share of the rows below it a stem's width. */
  /** The stem hangs within this share of the head's width from its middle. */
  hangingCentred: 0.2,
  hangingHead: 0.5,
  hangingHeight: 1.5,
  /** The head is looked for in this many spacings from the top. */
  hangingReach: 1.3,
  hangingThinShare: 0.8,
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
  /** A line is at most this many rows thick: max(4, width / 400). */
  maxThickness: 4,
  /** Two marks on one line merge into one number ("10") across a gap below this. */
  mergeGap: 0.18,
  mergeWidth: 1.2,
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
  /** A piece narrower than this, in line spacings, whose ink carries on past both its cuts for `strokeReach` is a stroke of a larger glyph. */
  strokeReach: 0.3,
  strokeWidth: 0.4,
  thicknessShare: 1 / 400,
} as const;

/** A fret number, possibly with technique letters stuck to it ("S0", "7h"); "O" reads as 0. */
const FRET = /^([A-Za-z~/\\(]*?)(\d{1,2})([A-Za-z~/\\)]*)$/;

/** The fret a recogniser read names, and the letters stuck to it, or undefined when the text is no fret. */
export function readFret(
  text: string
): { readonly fret: number; readonly letters: string } | undefined {
  const match = text
    .trim()
    .replace(/[Oo](?=\d|$)/g, "0")
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
 * Rows covered by long horizontal runs, clustered into thin lines. A cluster
 * too thick for a line whose core rows are one (a beam lying against it) is
 * kept apart as `hidden`: groupLines takes it only to finish a group.
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
    if (end - y <= maxThick) {
      lines.push(lineOf(y, end));
    } else {
      const core = lineRows(cover, y, end);
      if (core !== undefined && core.end - core.start <= maxThick) {
        hidden.push(lineOf(core.start, core.end));
      }
    }
    y = end;
  }
  return { hidden, lines };
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

/** Runs of 4 to 6 evenly spaced lines of about one extent; longer runs are not systems. */
function groupLines(
  { hidden, lines }: ReturnType<typeof findLines>,
  pageHeight: number
): LineGroup[] {
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
 * The group's band with its lines erased: a line pixel goes when nothing
 * touches the line there from above or below, so a digit the line crosses
 * keeps its middle, and when it belongs to the line itself: a horizontal run
 * at least a line spacing long, or a shorter one with a free end (a stub of
 * line left beside a knock-out). The bar of an H standing on the line inside
 * a knock-out is short and held by a stem at both ends, so it stays.
 */
export function bandWithoutLines(
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup,
  reach: { readonly above: number; readonly below: number } = {
    above: TAB.markBand,
    below: TAB.markBand,
  }
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
    const free = new Uint8Array(cols);
    for (let x = 0; x < cols; x += 1) {
      const above = a - 1 >= 0 && data[(a - 1) * cols + x] === 1;
      const below = b + 1 < rows && data[(b + 1) * cols + x] === 1;
      free[x] = above || below ? 0 : 1;
    }
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
 * middle. A staff's note hangs its stem from the head's left edge; a "T" of
 * a TAB clef has a head as thin as its bar.
 */
function hangsAStem(band: Band, cols: number, box: Box, s: number): boolean {
  if (box.h < TAB.hangingHeight * s || box.w > TAB.maxMarkWidth * s) {
    return false;
  }
  const thin = (y: number) => {
    let inked = 0;
    for (let c = 0; c < box.w; c += 1) {
      inked += band.data[y * cols + box.x + c] ?? 0;
    }
    return inked <= TAB.cutPieceWidth * s;
  };
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
  for (let y = box.y + head; y < box.y + box.h; y += 1) {
    stem += thin(y) ? 1 : 0;
    for (let c = 0; c < box.w; c += 1) {
      if (band.data[y * cols + box.x + c] === 1) {
        sumX += c;
        inked += 1;
      }
    }
  }
  return (
    stem >= TAB.hangingThinShare * (box.h - head) &&
    Math.abs(sumX / Math.max(1, inked) - box.w / 2) <=
      TAB.hangingCentred * box.w
  );
}

/**
 * The band's components of a fret's size. On a system whose frets hang
 * their stems (at least two do), the components too big to be a fret are
 * cut between the lines (cutBetweenLines) and their pieces grown back over
 * the cuts; the rest stand as they are, and so does every component of a
 * system that hangs no stems (docs/design/tab-4-findings.md).
 */
function fretSized(
  cv: OpenCv,
  band: Band,
  cols: number,
  group: LineGroup
): Box[] {
  const s = group.spacing;
  const sized = (box: Box) =>
    box.h >= TAB.minMarkHeight * s &&
    box.h <= TAB.maxMarkHeight * s &&
    box.w >= Math.max(2, TAB.minMarkWidth * s) &&
    box.w <= TAB.maxMarkWidth * s;
  const { boxes, labels } = componentsOf(cv, band.data, cols, band.rows);
  if (
    boxes.filter((box) => hangsAStem(band, cols, box, s)).length <
    TAB.minHanging
  ) {
    return boxes.filter(sized);
  }
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
 */
export function marksOf(
  cv: OpenCv,
  ink: Uint8Array,
  width: number,
  height: number,
  group: LineGroup
): { readonly marks: Mark[]; readonly short: Mark[] } {
  const s = group.spacing;
  const band = bandWithoutLines(ink, width, height, group);
  const cols = group.x1 - group.x0 + 1;
  const onLines: Mark[] = [];
  for (const box of fretSized(cv, band, cols, group)) {
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
  const groups = withMatScope((scope) => {
    const ink = new Uint8Array(inkOf(cv, scope, page).data);
    return groupLines(findLines(ink, page.width, page.height), page.height).map(
      (group): Candidate => {
        const { marks } = marksOf(cv, ink, page.width, page.height, group);
        const knockedOut = knockedOutShare(ink, page.width, group, marks);
        if (group.lines.length !== 5) {
          return { evidence: "tab", group, knockedOut };
        }
        return {
          evidence:
            marks.length < TAB.minDigits
              ? "staff"
              : marks.map((mark) => cropOf(page, ink, mark, group.spacing)),
          group,
          knockedOut,
        };
      }
    );
  });
  const candidates: Candidate[] = [];
  for (const candidate of groups) {
    const { evidence } = candidate;
    const isTab =
      evidence === "tab" ||
      // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, one batch at a time
      (evidence !== "staff" && (await marksAreNumbers(evidence, read)));
    if (isTab) {
      candidates.push(candidate);
    }
  }
  return keepPrintedAlike(candidates).map(({ group }, index) => ({
    group,
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

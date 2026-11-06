/**
 * Melodeon button tab (CADB) under a staff, found with classical image
 * processing before homr reads the page, as the line-tab guard in detect.ts
 * finds string tab. Under a staff, two bands split by one horizontal rule:
 * P (push) above, T (pull) below, the letters P and T at the rule's left end,
 * the button numbers in the bands and a line of left-hand letters under the
 * T band (docs/design/button-tab-findings.md, H1).
 *
 * homr reads none of it as music, but left on the page the numbers sit in
 * the staff's crop and the letter line in the chord strip above the next
 * staff, where the OCR reads it as that staff's chords. So the guard paints
 * the bands and the letter line white before homr sees the page.
 *
 * detect.ts finds lines by whole-row cover, which a phone photo taken at an
 * angle defeats (the book page of the findings turns about 3 degrees and
 * bends): here lines are traced strip by strip and chained across the page,
 * so a line may slope and curve. A rule is a long line with a staff above
 * it, no line near it, and a P above and a T below its left end that the
 * recogniser reads as such.
 */

import { planeToMat } from "../cv/mat-plane.js";
import { type OpenCv, withMatScope } from "../cv/opencv.js";
import { type ColorImage, createColor } from "../image/plane.js";
import { type Box, componentBoxes, cropOf, type ReadCrops } from "./detect.js";

/** A traced line, in page pixels: one point per strip it was seen in, left to right. */
export interface TracedLine {
  readonly points: readonly { readonly x: number; readonly y: number }[];
}

/**
 * One button tab, page-normalised like `StaffBox`: `x0..x1` the rule's ends
 * (the staff's own barlines), `bandTop`, `rule` and `bandBottom` taken at
 * the middle of the rule. `staff` is the index of the staff above it among
 * homr's staves, set by the pipeline once homr has read the page.
 */
export interface ButtonTabBand {
  readonly bandBottom: number;
  readonly bandTop: number;
  readonly rule: number;
  readonly staff?: number;
  readonly x0: number;
  readonly x1: number;
}

/** Rows below the rule the letter line reaches over columns `x0..x1`, or the T band's height where there is none. */
export interface LetterReach {
  readonly reach: number;
  readonly x0: number;
  readonly x1: number;
}

/** A button tab with the pixel geometry the guard paints over. */
export interface DetectedButtonTab {
  readonly band: ButtonTabBand;
  /** Rows of the T band below the rule, in page pixels. */
  readonly down: number;
  /** Rows below the rule the left-hand letter line reaches, chunk by chunk along the rule. */
  readonly letterLine: readonly LetterReach[];
  /** The margin letters, in page pixels: they may stand inside the rule's extent, among the numbers. */
  readonly letters: { readonly p: Box; readonly t: Box };
  /** The leftmost column painted: the P and T letters may stand left of the rule. */
  readonly marginX: number;
  readonly rule: TracedLine;
  /** The staff's line spacing just above the band, in page pixels. */
  readonly spacing: number;
  /** Rows of the P band above the rule, in page pixels. */
  readonly up: number;
  readonly x0: number;
  readonly x1: number;
}

const BUTTON = {
  /** The bands' height in line spacings: each between these. */
  bandMax: 8,
  bandMin: 1,
  /** A segment joins a chain within this many rows of where its slope puts it, plus `travelTolerance` per column travelled. */
  chainTolerance: 3,
  /**
   * Ink is darker than its neighbourhood's mean by `inkContrast` grey
   * levels, over a window of `inkWindowShare` of the page width: a phone
   * photo's paper darkens toward one corner, past a single threshold
   * (detect.ts's Otsu takes the lower quarter of the findings' book page
   * for ink).
   */
  inkContrast: 15,
  inkWindowShare: 1 / 48,
  /**
   * The left-hand letter line: rows of ink under the T band, looked for
   * from `letterFrom` to `letterTo` times the T band's height below the
   * rule in chunks of `letterChunk` spacings; a row is inked with
   * `letterRowInk` pixels in the chunk, and the line is the first inked run
   * (gaps of `letterGap` rows bridged) at least `letterMinRun` spacings
   * tall. A run taller than `letterMaxRun` is not a line of letters, and
   * stops the search.
   */
  letterChunk: 10,
  letterFrom: 0.85,
  letterGap: 2,
  /**
   * The P and T letters: their height and width in line spacings, looked
   * for this many spacings either side of the rule's left end and above or
   * below the rule, the first `lettersRead` from the left on each side read.
   */
  letterMaxHeight: 2.4,
  letterMaxRun: 3,
  letterMaxWidth: 2,
  letterMinHeight: 0.6,
  letterMinRun: 0.4,
  letterMinWidth: 0.25,
  letterReachX: 6,
  letterReachY: 5,
  letterRowInk: 2,
  lettersRead: 2,
  letterTo: 2.5,
  /** A strip row is lit when this share of its columns is ink. */
  litShare: 0.5,
  /** A chain carries on over this many strips with no segment (noteheads, beams and text on a line). */
  maxGapStrips: 6,
  /** A line drops at most this much per column (about 5.7 degrees): the rows a segment may span grow with it. */
  maxSlope: 0.1,
  /** A line is at most this many rows thick: max(4, width / 400), as detect.ts. */
  maxThickness: 4,
  /** A traced line spans at least this share of the page width. */
  minLineShare: 0.2,
  /** Strips are `stripShare` of the page width, at least `minStrip` columns. */
  minStrip: 24,
  /** Paint this share of a band's height beyond it, and this many spacings beyond the letter line and the ends. */
  paintMargin: 0.15,
  paintPad: 0.5,
  /** A segment covers this share of its strip's columns. */
  segmentCover: 0.8,
  /** A chain's slope is measured over this many strips back. */
  slopeReach: 8,
  /** The staff above: five lines evenly spaced within this share of their spacing, the lowest `staffMinGap`..`staffMaxGap` spacings above the rule, and no other line within `staffMinGap` spacings below it. */
  staffMaxGap: 10,
  staffMinGap: 1.5,
  staffTolerance: 0.25,
  stripShare: 1 / 96,
  thicknessShare: 1 / 400,
  travelTolerance: 0.02,
} as const;

/** 1 for ink, 0 for paper, thresholded against the local mean. */
export function localInkOf(cv: OpenCv, page: ColorImage): Uint8Array {
  return withMatScope((scope) => {
    const gray = scope.keep(new cv.Mat());
    cv.cvtColor(planeToMat(cv, scope, page), gray, cv.COLOR_BGR2GRAY);
    const ink = scope.keep(new cv.Mat());
    const window = Math.max(
      15,
      2 * Math.floor((page.width * BUTTON.inkWindowShare) / 2) + 1
    );
    cv.adaptiveThreshold(
      gray,
      ink,
      1,
      cv.ADAPTIVE_THRESH_MEAN_C,
      cv.THRESH_BINARY_INV,
      window,
      BUTTON.inkContrast
    );
    return new Uint8Array(ink.data);
  });
}

interface Segment {
  readonly x: number;
  readonly y: number;
}

/** The ink in each page row over columns x0..x1-1. */
function stripRowCounts(
  ink: Uint8Array,
  width: number,
  height: number,
  x0: number,
  x1: number
): Uint32Array {
  const counts = new Uint32Array(height);
  for (let y = 0; y < height; y += 1) {
    let count = 0;
    for (let x = x0; x < x1; x += 1) {
      count += ink[y * width + x] ?? 0;
    }
    counts[y] = count;
  }
  return counts;
}

/** Rows y0..y1-1 of the strip as a segment at their ink-weighted row, when they ink most of its columns. */
function segmentOf(
  ink: Uint8Array,
  width: number,
  counts: Uint32Array,
  strip: { readonly x0: number; readonly x1: number },
  y0: number,
  y1: number
): Segment | undefined {
  const { x0, x1 } = strip;
  let covered = 0;
  for (let x = x0; x < x1; x += 1) {
    let dark = false;
    for (let row = y0; row < y1 && !dark; row += 1) {
      dark = ink[row * width + x] === 1;
    }
    covered += dark ? 1 : 0;
  }
  if (covered < BUTTON.segmentCover * (x1 - x0)) {
    return;
  }
  let weight = 0;
  let sum = 0;
  for (let row = y0; row < y1; row += 1) {
    weight += counts[row] ?? 0;
    sum += row * (counts[row] ?? 0);
  }
  return { x: (x0 + x1 - 1) / 2, y: sum / weight };
}

/** The thin horizontal segments of one strip: lit rows, a line's thickness tall, ink over most columns. */
function stripSegments(
  ink: Uint8Array,
  width: number,
  height: number,
  strip: { readonly x0: number; readonly x1: number },
  maxRows: number
): Segment[] {
  const counts = stripRowCounts(ink, width, height, strip.x0, strip.x1);
  const lit = (row: number) =>
    (counts[row] ?? 0) >= BUTTON.litShare * (strip.x1 - strip.x0);
  const segments: Segment[] = [];
  let y = 0;
  while (y < height) {
    if (!lit(y)) {
      y += 1;
      continue;
    }
    let end = y;
    while (end < height && lit(end)) {
      end += 1;
    }
    const segment =
      end - y <= maxRows
        ? segmentOf(ink, width, counts, strip, y, end)
        : undefined;
    if (segment !== undefined) {
      segments.push(segment);
    }
    y = end;
  }
  return segments;
}

interface Chain {
  lastStrip: number;
  readonly points: Segment[];
}

const slopeOf = (points: readonly Segment[]): number => {
  const last = points.at(-1) as Segment;
  const from = points[
    Math.max(0, points.length - 1 - BUTTON.slopeReach)
  ] as Segment;
  return last.x === from.x ? 0 : (last.y - from.y) / (last.x - from.x);
};

/**
 * Strip `k`'s segments added to the chains, nearest pairs first: each
 * chain takes at most one, a segment no chain takes starts its own.
 */
function extendChains(
  chains: Chain[],
  segments: readonly Segment[],
  k: number
): void {
  const pairs: { chain: Chain; distance: number; segment: Segment }[] = [];
  for (const chain of chains) {
    if (k - chain.lastStrip > BUTTON.maxGapStrips + 1) {
      continue;
    }
    const last = chain.points.at(-1) as Segment;
    const slope = slopeOf(chain.points);
    for (const segment of segments) {
      const travel = segment.x - last.x;
      const distance = Math.abs(segment.y - (last.y + slope * travel));
      if (distance <= BUTTON.chainTolerance + BUTTON.travelTolerance * travel) {
        pairs.push({ chain, distance, segment });
      }
    }
  }
  pairs.sort((a, b) => a.distance - b.distance);
  const usedChains = new Set<Chain>();
  const usedSegments = new Set<Segment>();
  for (const { chain, segment } of pairs) {
    if (!(usedChains.has(chain) || usedSegments.has(segment))) {
      usedChains.add(chain);
      usedSegments.add(segment);
      chain.points.push(segment);
      chain.lastStrip = k;
    }
  }
  for (const segment of segments) {
    if (!usedSegments.has(segment)) {
      chains.push({ lastStrip: k, points: [segment] });
    }
  }
}

/**
 * Every long, thin, near-horizontal line on the page, traced strip by strip
 * and chained left to right: a segment joins the chain whose slope puts it
 * nearest, within a tolerance that grows with the gap it bridges. Pairs are
 * taken nearest first, so two lines a spacing apart never swap.
 */
export function traceLines(
  ink: Uint8Array,
  width: number,
  height: number
): TracedLine[] {
  const strip = Math.max(
    BUTTON.minStrip,
    Math.round(width * BUTTON.stripShare)
  );
  const maxThick = Math.max(
    BUTTON.maxThickness,
    Math.round(width * BUTTON.thicknessShare)
  );
  const maxRows = maxThick + Math.ceil(BUTTON.maxSlope * strip) + 1;
  const chains: Chain[] = [];
  for (let k = 0; k * strip < width; k += 1) {
    const x0 = k * strip;
    const x1 = Math.min(width, x0 + strip);
    if (x1 - x0 < strip / 2) {
      break;
    }
    extendChains(
      chains,
      stripSegments(ink, width, height, { x0, x1 }, maxRows),
      k
    );
  }
  return chains
    .filter(
      ({ points }) =>
        (points.at(-1) as Segment).x - (points[0] as Segment).x >=
        BUTTON.minLineShare * width
    )
    .map(({ points }) => ({ points }));
}

/** The line's y at column `x`: between its points, or carried on along its end's slope. */
export function lineYAt(line: TracedLine, x: number): number {
  const { points } = line;
  const first = points[0] as Segment;
  const last = points.at(-1) as Segment;
  if (x <= first.x) {
    const next = points[
      Math.min(points.length - 1, BUTTON.slopeReach)
    ] as Segment;
    const slope =
      next.x === first.x ? 0 : (next.y - first.y) / (next.x - first.x);
    return first.y + slope * (x - first.x);
  }
  if (x >= last.x) {
    return last.y + slopeOf(points) * (x - last.x);
  }
  let k = 1;
  while ((points[k] as Segment).x < x) {
    k += 1;
  }
  const a = points[k - 1] as Segment;
  const b = points[k] as Segment;
  return a.y + ((b.y - a.y) * (x - a.x)) / (b.x - a.x);
}

const covers = (line: TracedLine, x: number, slack: number): boolean =>
  (line.points[0] as Segment).x - slack <= x &&
  x <= (line.points.at(-1) as Segment).x + slack;

/**
 * The line spacing of a staff whose bottom line lies `staffMinGap` to
 * `staffMaxGap` spacings above `rule` at column `x`, with no line closer
 * than that below it; undefined when there is none.
 */
function staffSpacingAbove(
  rule: TracedLine,
  lines: readonly TracedLine[],
  x: number,
  slack: number
): number | undefined {
  const y = lineYAt(rule, x);
  const offsets = lines
    .filter((line) => line !== rule && covers(line, x, slack))
    .map((line) => lineYAt(line, x) - y);
  const above = offsets
    .filter((d) => d < 0)
    .map((d) => -d)
    .sort((a, b) => a - b);
  const below = offsets.filter((d) => d > 0).sort((a, b) => a - b);
  if (above.length < 5) {
    return;
  }
  const bottom = above[0] as number;
  const spacing = (above[1] as number) - bottom;
  if (spacing <= 0) {
    return;
  }
  for (let k = 1; k < 5; k += 1) {
    const step = (above[k] as number) - (above[k - 1] as number);
    if (Math.abs(step - spacing) > BUTTON.staffTolerance * spacing) {
      return;
    }
  }
  const [nearBelow] = below;
  if (
    bottom < BUTTON.staffMinGap * spacing ||
    bottom > BUTTON.staffMaxGap * spacing ||
    (nearBelow !== undefined && nearBelow < BUTTON.staffMinGap * spacing)
  ) {
    return;
  }
  return spacing;
}

/**
 * The rule carried to its ends to the pixel: its ink followed outward from
 * its first and last points, re-centred on the ink at every column, over
 * gaps of two columns, the ends added as points. A photographed page bends
 * most at the system's ends, where the traced slope no longer holds.
 */
function withEnds(
  ink: Uint8Array,
  width: number,
  height: number,
  rule: TracedLine,
  reach: number
): TracedLine {
  const follow = (from: Segment, step: number): Segment => {
    let end = from;
    let { y } = from;
    let gap = 0;
    for (
      let x = Math.round(from.x) + step;
      x >= 0 && x < width && gap <= 2;
      x += step
    ) {
      const centre = Math.round(y);
      let sum = 0;
      let count = 0;
      for (
        let row = Math.max(0, centre - reach);
        row <= Math.min(height - 1, centre + reach);
        row += 1
      ) {
        if (ink[row * width + x] === 1) {
          sum += row;
          count += 1;
        }
      }
      if (count === 0) {
        gap += 1;
      } else {
        // A barline crossing the rule fills the whole window: keep the line's row there.
        y = count <= reach ? sum / count : y;
        end = { x, y };
        gap = 0;
      }
    }
    return end;
  };
  const first = rule.points[0] as Segment;
  const last = rule.points.at(-1) as Segment;
  const left = follow(first, -1);
  const right = follow(last, 1);
  return {
    points: [
      ...(left.x < first.x ? [left] : []),
      ...rule.points,
      ...(right.x > last.x ? [right] : []),
    ],
  };
}

const isLetter = (text: string, letter: string): boolean =>
  text.trim().toUpperCase() === letter;

/** The P above the rule's left end and the T below it, each the first of the leftmost few the recogniser reads as that letter. */
async function marginLetters(
  cv: OpenCv,
  page: ColorImage,
  ink: Uint8Array,
  rule: TracedLine,
  x0: number,
  spacing: number,
  read: ReadCrops
): Promise<{ readonly p: Box; readonly t: Box } | undefined> {
  const { height, width } = page;
  const left = Math.max(0, Math.round(x0 - BUTTON.letterReachX * spacing));
  const right = Math.min(
    width - 1,
    Math.round(x0 + BUTTON.letterReachX * spacing)
  );
  const ys = [lineYAt(rule, left), lineYAt(rule, right)];
  const top = Math.max(
    0,
    Math.floor(Math.min(...ys) - BUTTON.letterReachY * spacing)
  );
  const bottom = Math.min(
    height - 1,
    Math.ceil(Math.max(...ys) + BUTTON.letterReachY * spacing)
  );
  const cols = right - left + 1;
  const rows = bottom - top + 1;
  const window = new Uint8Array(cols * rows);
  for (let y = 0; y < rows; y += 1) {
    window.set(
      ink.subarray((y + top) * width + left, (y + top) * width + right + 1),
      y * cols
    );
  }
  const letters = componentBoxes(cv, window, cols, rows)
    .map((box) => ({ ...box, x: box.x + left, y: box.y + top }))
    .filter(
      (box) =>
        box.h >= BUTTON.letterMinHeight * spacing &&
        box.h <= BUTTON.letterMaxHeight * spacing &&
        box.w >= BUTTON.letterMinWidth * spacing &&
        box.w <= BUTTON.letterMaxWidth * spacing
    )
    .sort((a, b) => a.x - b.x);
  const ruleAt = (box: Box) => lineYAt(rule, box.x + box.w / 2);
  const above = letters
    .filter((box) => box.y + box.h < ruleAt(box))
    .slice(0, BUTTON.lettersRead);
  const below = letters
    .filter((box) => box.y > ruleAt(box))
    .slice(0, BUTTON.lettersRead);
  if (above.length === 0 || below.length === 0) {
    return;
  }
  const texts = await read(
    [...above, ...below].map((box) =>
      cropOf(page, ink, { ...box, line: 0 }, spacing)
    )
  );
  const p = above.find((_, k) => isLetter(texts[k] ?? "", "P"));
  const t = below.find((_, k) => isLetter(texts[above.length + k] ?? "", "T"));
  return p === undefined || t === undefined ? undefined : { p, t };
}

/**
 * How far below the rule the left-hand letter line reaches, chunk by chunk
 * along the rule: in each chunk, the first run of inked rows under the T
 * band that is a letter's height, followed along the rule; `down` where
 * there is none. Chunked because the letter line need not run parallel to
 * the rule on a bent page (the findings' book page, last system).
 */
function letterLineReach(
  ink: Uint8Array,
  width: number,
  height: number,
  rule: TracedLine,
  down: number,
  spacing: number
): LetterReach[] {
  const x0 = Math.round((rule.points[0] as Segment).x);
  const x1 = Math.round((rule.points.at(-1) as Segment).x);
  const from = Math.round(BUTTON.letterFrom * down);
  const to = Math.round(BUTTON.letterTo * down);
  const chunk = Math.max(1, Math.round(BUTTON.letterChunk * spacing));
  const reaches: LetterReach[] = [];
  for (let c0 = x0; c0 <= x1; c0 += chunk) {
    const c1 = Math.min(x1, c0 + chunk - 1);
    const counts = new Array<number>(to - from + 1).fill(0);
    for (let x = c0; x <= c1; x += 1) {
      addColumnUnderBand(counts, ink, width, height, {
        base: Math.round(lineYAt(rule, x)),
        from,
        x,
      });
    }
    const end = letterRunEnd(counts, spacing);
    reaches.push({
      reach: end === undefined ? down : Math.max(down, from + end),
      x0: c0,
      x1: c1,
    });
  }
  // A letter across two chunks may be seen in one only.
  return reaches.map((span, k) => ({
    ...span,
    reach: Math.max(
      span.reach,
      reaches[k - 1]?.reach ?? down,
      reaches[k + 1]?.reach ?? down
    ),
  }));
}

/**
 * Column `x`'s ink at offsets `from` onwards below the rule's row `base`,
 * added to `counts`, less the stroke that comes down out of the band: a bar
 * runs on a little past the band's bottom.
 */
function addColumnUnderBand(
  counts: number[],
  ink: Uint8Array,
  width: number,
  height: number,
  column: { readonly base: number; readonly from: number; readonly x: number }
): void {
  const { base, from, x } = column;
  const inked = (offset: number) => {
    const y = base + from + offset;
    return y >= 0 && y < height && ink[y * width + x] === 1;
  };
  let k = 0;
  while (k < counts.length && inked(k)) {
    k += 1;
  }
  for (; k < counts.length; k += 1) {
    counts[k] = (counts[k] ?? 0) + (inked(k) ? 1 : 0);
  }
}

/** The last row of the first run of inked rows a letter's height, gaps of `letterGap` bridged; undefined before a run too tall for letters, or with none. */
function letterRunEnd(
  counts: readonly number[],
  spacing: number
): number | undefined {
  const inked = (k: number) => (counts[k] ?? 0) >= BUTTON.letterRowInk;
  let start = 0;
  while (start < counts.length) {
    if (!inked(start)) {
      start += 1;
      continue;
    }
    let end = start;
    for (
      let k = start;
      k < counts.length && k - end <= BUTTON.letterGap + 1;
      k += 1
    ) {
      end = inked(k) ? k : end;
    }
    const tall = end - start + 1;
    if (tall > BUTTON.letterMaxRun * spacing) {
      return;
    }
    if (tall >= BUTTON.letterMinRun * spacing) {
      return end;
    }
    start = end + 1;
  }
}

/**
 * Every button tab on the page, top to bottom. `read` is called only for a
 * rule with a staff above it, so a page of ordinary staves, whose long
 * lines all have a neighbour a spacing away, never reaches the recogniser.
 */
export async function detectButtonTabs(
  cv: OpenCv,
  page: ColorImage,
  read: ReadCrops
): Promise<DetectedButtonTab[]> {
  const { height, width } = page;
  const ink = localInkOf(cv, page);
  const lines = traceLines(ink, width, height);
  const slack = Math.max(
    BUTTON.minStrip,
    Math.round(width * BUTTON.stripShare)
  );
  const maxThick = Math.max(
    BUTTON.maxThickness,
    Math.round(width * BUTTON.thicknessShare)
  );
  const found: DetectedButtonTab[] = [];
  for (const rule of lines) {
    const first = (rule.points[0] as Segment).x;
    const last = (rule.points.at(-1) as Segment).x;
    const spacing = [0.25, 0.5, 0.75]
      .map((share) =>
        staffSpacingAbove(rule, lines, first + share * (last - first), slack)
      )
      .find((s) => s !== undefined);
    if (spacing === undefined) {
      continue;
    }
    const ended = withEnds(ink, width, height, rule, maxThick);
    const ends = {
      x0: Math.round((ended.points[0] as Segment).x),
      x1: Math.round((ended.points.at(-1) as Segment).x),
    };
    // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, a rule at a time
    const letters = await marginLetters(
      cv,
      page,
      ink,
      ended,
      ends.x0,
      spacing,
      read
    );
    if (letters === undefined) {
      continue;
    }
    const { p, t } = letters;
    const up = 2 * (lineYAt(ended, p.x + p.w / 2) - (p.y + p.h / 2));
    const down = 2 * (t.y + t.h / 2 - lineYAt(ended, t.x + t.w / 2));
    if (
      Math.min(up, down) < BUTTON.bandMin * spacing ||
      Math.max(up, down) > BUTTON.bandMax * spacing
    ) {
      continue;
    }
    const middle = lineYAt(ended, (ends.x0 + ends.x1) / 2);
    found.push({
      band: {
        bandBottom: (middle + down) / height,
        bandTop: (middle - up) / height,
        rule: middle / height,
        x0: ends.x0 / width,
        x1: ends.x1 / width,
      },
      down,
      letterLine: letterLineReach(ink, width, height, ended, down, spacing),
      letters,
      marginX: Math.min(ends.x0, p.x, t.x),
      rule: ended,
      spacing,
      up,
      x0: ends.x0,
      x1: ends.x1,
    });
  }
  return found.sort((a, b) => a.band.rule - b.band.rule);
}

/**
 * The bands and the letter line under each tab painted white, following the
 * rule's slope, `paintMargin` of each band's height beyond it and
 * `paintPad` spacings beyond the letter line and the ends; a new page, the
 * input untouched.
 */
export function whitenButtonTabs(
  page: ColorImage,
  tabs: readonly DetectedButtonTab[]
): ColorImage {
  const { height, width } = page;
  const out = createColor(width, height);
  out.data.set(page.data);
  for (const tab of tabs) {
    const pad = Math.round(BUTTON.paintPad * tab.spacing);
    const above = tab.up * (1 + BUTTON.paintMargin);
    const band = tab.down * (1 + BUTTON.paintMargin);
    const x0 = Math.max(0, tab.marginX - pad);
    const x1 = Math.min(width - 1, tab.x1 + pad);
    for (let x = x0; x <= x1; x += 1) {
      const letters = tab.letterLine
        .filter((span) => span.x0 - pad <= x && x <= span.x1 + pad)
        .map((span) => (span.reach > tab.down ? span.reach + pad : 0));
      const below = Math.max(band, ...letters);
      const y = lineYAt(tab.rule, x);
      const top = Math.max(0, Math.floor(y - above));
      const bottom = Math.min(height - 1, Math.ceil(y + below));
      for (let row = top; row <= bottom; row += 1) {
        out.data.fill(255, (row * width + x) * 3, (row * width + x + 1) * 3);
      }
    }
  }
  return out;
}

/** A staff box, page-normalised: what `withStaves` needs of the pipeline's StaffBox. */
interface StaffExtent {
  readonly cx: number;
  readonly cy: number;
  readonly h: number;
  readonly index: number;
  readonly w: number;
}

/**
 * Each tab with the index of the staff it hangs under: the lowest staff
 * whose bottom lies above the rule and whose extent overlaps the rule's;
 * none when homr found no such staff.
 */
export function withStaves(
  tabs: readonly DetectedButtonTab[],
  staves: readonly StaffExtent[]
): DetectedButtonTab[] {
  return tabs.map((tab) => {
    const { band } = tab;
    let best: StaffExtent | undefined;
    for (const staff of staves) {
      const bottom = staff.cy + staff.h / 2;
      const overlaps =
        staff.cx - staff.w / 2 < band.x1 && staff.cx + staff.w / 2 > band.x0;
      if (
        overlaps &&
        bottom < band.rule &&
        (best === undefined || staff.cy > best.cy)
      ) {
        best = staff;
      }
    }
    return best === undefined
      ? tab
      : { ...tab, band: { ...band, staff: best.index } };
  });
}

/**
 * The numbers of a melodeon button tab (CADB), read band by band once the
 * guard (src/tab/button.ts) has found the bands: for each number its button,
 * its row mark (`'` or `''`), a trailing `-` (held) and its column, the
 * bellows from the band it stands in (docs/design/button-tab-findings.md, H2).
 *
 * The band is straightened first, one page column at a time along the
 * traced rule, so a photographed page that turns and bends reads like a
 * typeset one: in the straightened band the rule is one row, the numbers
 * stand on two rows, and the string-tab reader's machinery applies as it is.
 * The rule is erased with `bandWithoutLines`, the marks are `marksInBand`'s
 * (touching digits merge into "10" as frets do), and each is read by the
 * recogniser with the tab digit classifier's second reading (`fuseDigit`).
 * The suffix is kept apart: it is too small for a mark, and is found
 * beside each number by its shape.
 */

import type { OpenCv } from "../cv/opencv.js";
import { type ColorImage, createColor } from "../image/plane.js";
import { type DetectedButtonTab, lineYAt, localInkOf } from "./button.js";
import { type ButtonTabLetter, readLeftHand } from "./button-letters.js";
import {
  type Band,
  type Box,
  bandWithoutLines,
  componentBoxes,
  cropOf,
  type LineGroup,
  type Mark,
  marksInBand,
  type ReadCrops,
} from "./detect.js";
import {
  type ClassifyDigits,
  type DigitGuess,
  fuseDigit,
  NOT_A_DIGIT,
} from "./digits.js";

/** One printed button number, page-normalised like a string tab's events. */
export interface ButtonTabMark {
  readonly bellows: "push" | "pull";
  /** As printed, from 1. */
  readonly button: number;
  /** A trailing `-`: the button stays down into what follows. */
  readonly held: boolean;
  /** 0 with no mark, 1 for `'`, 2 for `''`. */
  readonly row: 0 | 1 | 2;
  /** The centre of the number's digits, the suffix left out. */
  readonly x: number;
}

/** One button tab read: the marks of both bands left to right. */
export interface ButtonTabReading {
  /** The left-hand line under the T band, left to right, as printed. */
  readonly leftHand: readonly ButtonTabLetter[];
  readonly marks: readonly ButtonTabMark[];
  /** The staff the tab hangs under, among homr's staves; the tab's order on the page when homr placed no staff above it. */
  readonly staff: number;
  /** Number-sized marks in a band that read as no button. */
  readonly unread: number;
}

/** A straightened band: page columns kept, rows counted from the band's top along the rule. */
export interface StraightBand {
  /** The page row of straightened row 0 at column x is `base[x - x0]`. */
  readonly base: Int32Array;
  /** 1 for ink, `width` (the page's) by `rows`; only columns `x0..x1` are filled. */
  readonly ink: Uint8Array;
  readonly rows: number;
  /** The rule's row, and its first and last inked rows. */
  readonly rule: number;
  readonly ruleBottom: number;
  readonly ruleTop: number;
  readonly width: number;
  readonly x0: number;
  readonly x1: number;
}

const READ = {
  /**
   * The suffix pieces, in digit heights: an apostrophe is 0.2 to 0.65 tall,
   * at most 0.35 wide and taller than wide, its top within `apostropheTop`
   * of the number's; a dash at most 0.25 tall, 0.1 to 0.9 wide and wider
   * than tall, its centre between `dashFrom` and `dashTo` of the number's
   * height. Either starts within `suffixReach` of the number's right edge,
   * and belongs to the nearest number on its left.
   */
  apostropheMaxHeight: 0.65,
  apostropheMaxWidth: 0.35,
  apostropheMinHeight: 0.2,
  apostropheTop: 0.3,
  /** attachedSuffix: a one-piece mark wider than `attachedFrom` and narrower than `attachedTo` median marks; split past `attachedSplit` of one; its right part an apostrophe when it ends above `attachedApostrophe` of the mark's height. */
  attachedApostrophe: 0.6,
  attachedFrom: 1.2,
  attachedSplit: 0.7,
  attachedTo: 1.7,
  /** trustsClassifier: the classifier's least probability, and the widest mark, in median marks, it reads alone. */
  classifierSure: 0.8,
  classifierWidth: 1.2,
  dashFrom: 0.25,
  dashMaxHeight: 0.25,
  dashMaxWidth: 0.9,
  dashMinWidth: 0.1,
  dashTo: 0.85,
  /** A component is a digit when its height is this share of the margin letters' or more, and at most `digitMax`. */
  digitMax: 1.5,
  digitMin: 0.75,
  /** The band is taken this share of each half's height beyond it. */
  margin: 0.15,
  /**
   * marksInBand's spacing, in digit heights. Its merge gap (0.18 spacing)
   * then joins digits up to 0.4 digit heights apart: "10" is 0.36 apart in
   * the typeset fixture and 0.16 in the book photo, while two numbers
   * stand 0.55 apart at their closest there.
   */
  markSpacing: 2.2,
  /**
   * The largest button read. The plan's books number to 12 per row; a
   * larger number is a misread, counted unread.
   */
  maxButton: 12,
  /** A row of the rule is inked over this share of its columns. */
  ruleCover: 0.3,
  /** The rule is looked for this many rows either side of its traced row. */
  ruleReach: 6,
  suffixReach: 0.8,
} as const;

/** The tab's band straightened along its rule, from `ink` (1 for ink, the page's size). */
export function straightBand(
  ink: Uint8Array,
  width: number,
  height: number,
  tab: DetectedButtonTab
): StraightBand {
  const above = Math.ceil(tab.up * (1 + READ.margin));
  const below = Math.ceil(tab.down * (1 + READ.margin));
  const rows = above + below + 1;
  const { x0, x1 } = tab;
  const out = new Uint8Array(width * rows);
  const base = new Int32Array(x1 - x0 + 1);
  for (let x = x0; x <= x1; x += 1) {
    const top = Math.round(lineYAt(tab.rule, x)) - above;
    base[x - x0] = top;
    for (let r = 0; r < rows; r += 1) {
      const y = top + r;
      if (y >= 0 && y < height) {
        out[r * width + x] = ink[y * width + x] ?? 0;
      }
    }
  }
  const cover = (r: number) => {
    let count = 0;
    for (let x = x0; x <= x1; x += 1) {
      count += out[r * width + x] ?? 0;
    }
    return count;
  };
  const wanted = READ.ruleCover * (x1 - x0 + 1);
  let ruleTop = above;
  let ruleBottom = above;
  while (ruleTop > above - READ.ruleReach && cover(ruleTop - 1) >= wanted) {
    ruleTop -= 1;
  }
  while (
    ruleBottom < above + READ.ruleReach &&
    cover(ruleBottom + 1) >= wanted
  ) {
    ruleBottom += 1;
  }
  return {
    base,
    ink: out,
    rows,
    rule: above,
    ruleBottom,
    ruleTop,
    width,
    x0,
    x1,
  };
}

/**
 * The straightened band with its rule erased where nothing crosses it, as a
 * string tab's lines are. The band spans every straightened row (its `top`
 * is 0), so its rows are the straightened band's.
 */
export function bandWithoutRule(straight: StraightBand, spacing: number): Band {
  const { rows, rule, ruleBottom, ruleTop, width, x0, x1 } = straight;
  const group: LineGroup = {
    lines: [{ breakLength: 0, x0, x1, y: rule, y0: ruleTop, y1: ruleBottom }],
    spacing,
    x0,
    x1,
  };
  return bandWithoutLines(straight.ink, width, rows, group, {
    above: rule / spacing,
    below: (rows - 1 - rule) / spacing,
  });
}

/** A component of the band that is part of a number's suffix. */
interface SuffixPiece extends Box {
  readonly kind: "apostrophe" | "dash";
  /** The vertical strokes it holds: `''` may print as one `"`. */
  readonly strokes: number;
}

/** The runs of inked columns in a box of the band: one per apostrophe stroke. */
function strokesOf(band: Band, cols: number, box: Box): number {
  let runs = 0;
  let inRun = false;
  for (let c = box.x; c < box.x + box.w; c += 1) {
    let inked = false;
    for (let r = box.y; r < box.y + box.h && !inked; r += 1) {
      inked = band.data[r * cols + c] === 1;
    }
    runs += inked && !inRun ? 1 : 0;
    inRun = inked;
  }
  return Math.max(1, runs);
}

function suffixKind(box: Box, digit: number): SuffixPiece["kind"] | undefined {
  const { h, w } = box;
  if (
    h >= READ.apostropheMinHeight * digit &&
    h <= READ.apostropheMaxHeight * digit &&
    w <= READ.apostropheMaxWidth * digit &&
    w < h
  ) {
    return "apostrophe";
  }
  if (
    h <= READ.dashMaxHeight * digit &&
    w >= READ.dashMinWidth * digit &&
    w <= READ.dashMaxWidth * digit &&
    w > h
  ) {
    return "dash";
  }
}

/** A number as read: its digits' box in straightened rows, its band, its text. */
interface ReadNumber {
  readonly bellows: ButtonTabMark["bellows"];
  readonly mark: Mark;
  readonly readBy: ButtonTabNumber["readBy"];
  readonly text: string;
}

const NUMBER = /^(\d{1,2})\s*(['’‘`´"”]*)\s*([-–—_]?)$/;

/** The button a recogniser read names, with any suffix it read too; undefined when the text is no button. */
export function readButton(
  text: string
):
  | { readonly button: number; readonly held: boolean; readonly row: number }
  | undefined {
  const match = text
    .trim()
    .replace(/[Oo]/g, "0")
    .replace(/[Il|]/g, "1")
    .match(NUMBER);
  if (!match) {
    return;
  }
  const [, digits, marks, dash] = match;
  const button = Number(digits);
  if (button < 1 || button > READ.maxButton) {
    return;
  }
  const row = [...(marks ?? "")].reduce(
    (sum, mark) => sum + (mark === '"' || mark === "”" ? 2 : 1),
    0
  );
  return { button, held: (dash ?? "") !== "", row };
}

/**
 * The suffix pieces standing after a number: its row mark's strokes and
 * whether a dash follows. A piece belongs to the nearest number on its left
 * in its band only, so a number never takes the next one's apostrophe.
 */
function suffixOf(
  number: Mark,
  numbers: readonly Mark[],
  pieces: readonly SuffixPiece[],
  digit: number
): { readonly held: boolean; readonly strokes: number } {
  const right = number.x + number.w;
  const ownerOf = (piece: SuffixPiece) =>
    numbers
      .filter(
        (other) =>
          other.line === number.line &&
          other.x + other.w <= piece.x + 1 &&
          piece.y < other.y + other.h &&
          piece.y + piece.h > other.y - READ.apostropheTop * digit
      )
      .reduce<Mark | undefined>(
        (best, other) =>
          best === undefined || other.x + other.w > best.x + best.w
            ? other
            : best,
        undefined
      );
  const after = pieces.filter(
    (piece) =>
      piece.x >= right - 1 &&
      piece.x - right <= READ.suffixReach * digit &&
      (piece.kind === "apostrophe"
        ? Math.abs(piece.y - number.y) <= READ.apostropheTop * digit
        : piece.y + piece.h / 2 >= number.y + READ.dashFrom * number.h &&
          piece.y + piece.h / 2 <= number.y + READ.dashTo * number.h) &&
      ownerOf(piece) === number
  );
  return {
    held: after.some((piece) => piece.kind === "dash"),
    strokes: after
      .filter((piece) => piece.kind === "apostrophe")
      .reduce((sum, piece) => sum + piece.strokes, 0),
  };
}

/**
 * A suffix touching its number, as a phone photo's ink can join an
 * apostrophe to a 7's bar: a mark of one piece (no blank column across it)
 * wider than `attachedFrom` times the band's median mark, but narrower than
 * two digits, whose part right of its thinnest column (past `attachedSplit`
 * of a median mark) stands in the number's top part (an apostrophe) or is a
 * thin stroke at its middle (a dash). The piece is returned with the column
 * it starts at; undefined for anything else.
 */
function attachedSuffix(
  band: Band,
  cols: number,
  x0: number,
  mark: Mark,
  single: number
): { readonly piece: SuffixPiece } | undefined {
  if (
    mark.w <= READ.attachedFrom * single ||
    mark.w >= READ.attachedTo * single
  ) {
    return;
  }
  const left = mark.x - x0;
  const inkIn = (c: number) => {
    let count = 0;
    for (let r = mark.y; r < mark.y + mark.h; r += 1) {
      count += band.data[r * cols + c] ?? 0;
    }
    return count;
  };
  const counts = Array.from({ length: mark.w }, (_, k) => inkIn(left + k));
  if (counts.some((count) => count === 0)) {
    return;
  }
  let split = Math.ceil(READ.attachedSplit * single);
  for (let k = split; k < mark.w - 2; k += 1) {
    split = (counts[k] ?? 0) < (counts[split] ?? 0) ? k : split;
  }
  let top = mark.y + mark.h;
  let bottom = mark.y - 1;
  for (let r = mark.y; r < mark.y + mark.h; r += 1) {
    for (let k = split + 1; k < mark.w; k += 1) {
      if (band.data[r * cols + left + k] === 1) {
        top = Math.min(top, r);
        bottom = Math.max(bottom, r);
      }
    }
  }
  const box = {
    h: bottom - top + 1,
    w: mark.w - split - 1,
    x: mark.x + split + 1,
    y: top,
  };
  if (bottom <= mark.y + READ.attachedApostrophe * mark.h) {
    const strokes = strokesOf(band, cols, { ...box, x: box.x - x0 });
    return { piece: { ...box, kind: "apostrophe", strokes } };
  }
  const middle = (top + bottom) / 2;
  if (
    box.h <= READ.dashMaxHeight * mark.h &&
    middle >= mark.y + READ.dashFrom * mark.h &&
    middle <= mark.y + READ.dashTo * mark.h
  ) {
    return { piece: { ...box, kind: "dash", strokes: 0 } };
  }
}

/**
 * Whether the digit classifier alone reads this mark: one piece (no blank
 * column across it, so no "10" or "11", whose digits stand apart), no wider
 * than `classifierWidth` median marks (no suffix or second digit joined to
 * it), and a digit from 1 to 9 the classifier gives at least
 * `classifierSure`. Anything else goes to the recogniser.
 */
function trustsClassifier(
  band: Band,
  cols: number,
  x0: number,
  mark: Mark,
  single: number,
  guess: DigitGuess | undefined
): boolean {
  if (
    guess === undefined ||
    guess.digit === NOT_A_DIGIT ||
    guess.digit === 0 ||
    guess.probability < READ.classifierSure ||
    mark.w > READ.classifierWidth * single
  ) {
    return false;
  }
  for (let c = mark.x - x0; c < mark.x - x0 + mark.w; c += 1) {
    let inked = false;
    for (let r = mark.y; r < mark.y + mark.h && !inked; r += 1) {
      inked = band.data[r * cols + c] === 1;
    }
    if (!inked) {
      return false;
    }
  }
  return true;
}

const overlaps = (a: Box, b: Box): boolean =>
  a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

/** A number-sized mark of a band: what the recogniser said, and the mark it makes when that names a button. */
export interface ButtonTabNumber {
  readonly bellows: ButtonTabMark["bellows"];
  readonly mark?: ButtonTabMark;
  /** Who read the digits: the classifier alone (trustsClassifier), or the recogniser. */
  readonly readBy: "classifier" | "recogniser";
  readonly text: string;
  readonly x: number;
}

/** One tab's number-sized marks, left to right: suffixes kept apart, digits read. */
export async function readButtonTabNumbers(
  cv: OpenCv,
  page: ColorImage,
  ink: Uint8Array,
  tab: DetectedButtonTab,
  read: ReadCrops,
  classify: ClassifyDigits | undefined
): Promise<ButtonTabNumber[]> {
  const straight = straightBand(ink, page.width, page.height, tab);
  const { rule, width, x0, x1 } = straight;
  const cols = x1 - x0 + 1;
  const band = bandWithoutRule(straight, tab.spacing);
  // The margin letters, where they stand inside the rule's extent, in straightened rows.
  const letters = [tab.letters.p, tab.letters.t].flatMap((box) => {
    const at = Math.min(x1, Math.max(x0, Math.round(box.x + box.w / 2)));
    return box.x + box.w < x0
      ? []
      : [{ ...box, y: box.y - (straight.base[at - x0] ?? 0) }];
  });
  const boxes = componentBoxes(cv, band.data, cols, band.rows);
  const letterHeight = (tab.letters.p.h + tab.letters.t.h) / 2;
  const digits = boxes
    .map((box) => box.h)
    .filter(
      (h) =>
        h >= READ.digitMin * letterHeight && h <= READ.digitMax * letterHeight
    )
    .sort((a, b) => a - b);
  const digit = digits[Math.floor(digits.length / 2)] ?? letterHeight;
  const pieces: SuffixPiece[] = [];
  const clean: Band = { ...band, data: band.data.slice() };
  for (const box of boxes) {
    const kind = suffixKind(box, digit);
    if (kind !== undefined) {
      pieces.push({
        ...box,
        kind,
        strokes: kind === "apostrophe" ? strokesOf(band, cols, box) : 0,
        x: box.x + x0,
      });
      for (let r = box.y; r < box.y + box.h; r += 1) {
        clean.data.fill(0, r * cols + box.x, r * cols + box.x + box.w);
      }
    }
  }
  // The two bands' middles and the rule, in straightened rows: a mark is
  // pushed or pulled by the one it sits on, and one on the rule is no number.
  const line = (y: number) => ({ breakLength: 0, x0, x1, y, y0: y, y1: y });
  const group: LineGroup = {
    lines: [line(rule - tab.up / 2), line(rule), line(rule + tab.down / 2)],
    spacing: READ.markSpacing * digit,
    x0,
    x1,
  };
  const found = marksInBand(cv, clean, group).marks.filter(
    (mark) =>
      mark.line !== 1 && !letters.some((letter) => overlaps(letter, mark))
  );
  const widths = found.map((mark) => mark.w).sort((a, b) => a - b);
  const single = widths[Math.floor(widths.length / 2)] ?? digit;
  const marks = found.map((mark) => {
    const split = attachedSuffix(clean, cols, x0, mark, single);
    if (split === undefined) {
      return mark;
    }
    pieces.push(split.piece);
    return { ...mark, w: split.piece.x - mark.x };
  });
  const cleanInk = new Uint8Array(width * straight.rows);
  for (let r = 0; r < band.rows; r += 1) {
    cleanInk.set(clean.data.subarray(r * cols, (r + 1) * cols), r * width + x0);
  }
  const sheet = createColor(width, straight.rows);
  const crops = marks.map((mark) => cropOf(sheet, cleanInk, mark, digit));
  const guesses =
    classify === undefined || crops.length === 0 ? [] : await classify(crops);
  const trusted = marks.map((mark, k) =>
    trustsClassifier(clean, cols, x0, mark, single, guesses[k])
  );
  const doubtful = crops.filter((_, k) => !trusted[k]);
  const said = doubtful.length === 0 ? [] : await read(doubtful);
  let next = 0;
  const numbers: ReadNumber[] = marks.map((mark, k) => {
    const guess = guesses[k];
    const bellows = mark.line === 0 ? "push" : "pull";
    if (trusted[k] && guess !== undefined) {
      return { bellows, mark, readBy: "classifier", text: String(guess.digit) };
    }
    const text = said[next] ?? "";
    next += 1;
    return {
      bellows,
      mark,
      readBy: "recogniser",
      text:
        guess === undefined
          ? text
          : fuseDigit(text, readButton(text) !== undefined, guess),
    };
  });
  return numbers
    .map(({ bellows, mark, readBy, text }): ButtonTabNumber => {
      const x = (mark.x + mark.w / 2) / page.width;
      const button = readButton(text);
      if (button === undefined) {
        return { bellows, readBy, text, x };
      }
      const suffix = suffixOf(mark, marks, pieces, digit);
      const strokes = suffix.strokes > 0 ? suffix.strokes : button.row;
      return {
        bellows,
        mark: {
          bellows,
          button: button.button,
          held: suffix.held || button.held,
          row: Math.min(2, strokes) as ButtonTabMark["row"],
          x,
        },
        readBy,
        text,
        x,
      };
    })
    .sort((a, b) => a.x - b.x);
}

/**
 * Every tab's numbers, in the tabs' order. `read` is the recogniser the
 * guard used; `classify`, the tab digit classifier, second-reads each
 * number as it does a fret (fuseDigit); the left-hand line is read
 * under each tab (readLeftHand).
 */
export async function readButtonTabs(
  cv: OpenCv,
  page: ColorImage,
  tabs: readonly DetectedButtonTab[],
  read: ReadCrops,
  classify?: ClassifyDigits
): Promise<ButtonTabReading[]> {
  if (tabs.length === 0) {
    return [];
  }
  const ink = localInkOf(cv, page);
  const readings: ButtonTabReading[] = [];
  for (const [k, tab] of tabs.entries()) {
    // biome-ignore lint/performance/noAwaitInLoops: one recogniser session, a tab at a time
    const numbers = await readButtonTabNumbers(
      cv,
      page,
      ink,
      tab,
      read,
      classify
    );
    const { letters } = await readLeftHand(cv, page, ink, tab, read);
    readings.push({
      leftHand: letters,
      marks: numbers.flatMap(({ mark }) => (mark === undefined ? [] : [mark])),
      staff: tab.band.staff ?? k,
      unread: numbers.filter(({ mark }) => mark === undefined).length,
    });
  }
  return readings;
}

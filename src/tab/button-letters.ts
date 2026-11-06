/**
 * The left-hand line of a melodeon button tab (CADB): the letters printed
 * under the T band, each with its column (docs/design/button-tab-findings.md,
 * H3). They are read as printed, case and all ("C", "Am", "g"): what they
 * mean (bass or chord, and that the staff's chord symbols rule) is the app's.
 *
 * The strip under the T band is straightened along the rule, as the bands
 * are, and its letter-sized pieces grouped into words. The words are packed
 * into one short line, a fixed gap apart, and that line is read by the
 * recogniser in one run. It reads no spaces there, so the line is taken
 * one character per piece; when the counts differ (a broken letter, a
 * glyph read as two), each word is read on its own instead.
 */

import type { OpenCv } from "../cv/opencv.js";
import { type ColorImage, createColor } from "../image/plane.js";
import { type DetectedButtonTab, lineYAt } from "./button.js";
import { type Box, componentBoxes, type ReadCrops } from "./detect.js";

/** A left-hand letter (or word: "Am"), page-normalised like a mark. */
export interface ButtonTabLetter {
  readonly text: string;
  /** The centre of the word's ink. */
  readonly x: number;
}

const LETTERS = {
  /** The strip read: from `from` times the T band's height under the rule to the letter line's reach plus `pad` letter heights. */
  from: 0.85,
  /** In the packed line, words stand this many letter heights apart, with this much white around. */
  gap: 0.8,
  /** Pieces closer than this many letter heights make one word ("Am", a broken letter). */
  joinGap: 0.25,
  /** A piece is a letter's when its height is this share of the margin T's, and at most `maxHeight`; at most `maxWidth` wide. */
  maxHeight: 1.4,
  maxWidth: 2,
  minHeight: 0.35,
  pad: 0.5,
  /** A piece narrower than this many letter heights that starts on the strip's first row is a bar coming out of the band. */
  stubWidth: 0.3,
} as const;

/** The straightened strip under the T band: 1 for ink, `cols` by `rows`, row 0 at `top` rows under the rule. */
interface Strip {
  readonly cols: number;
  readonly data: Uint8Array;
  readonly rows: number;
}

/** The recogniser's text in plain characters: it answers some capitals in their full-width form ("Ｇ"). */
const printed = (text: string): string => text.normalize("NFKC").trim();

function stripUnder(
  ink: Uint8Array,
  width: number,
  height: number,
  tab: DetectedButtonTab,
  top: number,
  bottom: number
): Strip {
  const cols = tab.x1 - tab.x0 + 1;
  const rows = bottom - top + 1;
  const data = new Uint8Array(cols * rows);
  for (let c = 0; c < cols; c += 1) {
    const x = tab.x0 + c;
    const base = Math.round(lineYAt(tab.rule, x)) + top;
    for (let r = 0; r < rows; r += 1) {
      const y = base + r;
      if (y >= 0 && y < height) {
        data[r * cols + c] = ink[y * width + x] ?? 0;
      }
    }
  }
  return { cols, data, rows };
}

/** Pieces left to right into words: a piece joins the open word when the gap to it is under `gap` px. */
function wordsOf(pieces: readonly Box[], gap: number): Box[] {
  const words: Box[] = [];
  for (const piece of [...pieces].sort((a, b) => a.x - b.x)) {
    const last = words.at(-1);
    if (last !== undefined && piece.x - (last.x + last.w) < gap) {
      const x = Math.min(last.x, piece.x);
      const y = Math.min(last.y, piece.y);
      words[words.length - 1] = {
        h: Math.max(last.y + last.h, piece.y + piece.h) - y,
        w: Math.max(last.x + last.w, piece.x + piece.w) - x,
        x,
        y,
      };
    } else {
      words.push(piece);
    }
  }
  return words;
}

/** The ink of `boxes` (strip coordinates) side by side on white, `gap` px apart, top-aligned on one row band. */
function packed(strip: Strip, boxes: readonly Box[], gap: number): ColorImage {
  const top = Math.min(...boxes.map((b) => b.y));
  const bottom = Math.max(...boxes.map((b) => b.y + b.h));
  const width =
    boxes.reduce((sum, b) => sum + b.w, 0) + gap * (boxes.length + 1);
  const height = bottom - top + 2 * gap;
  const image = createColor(width, height);
  image.data.fill(255);
  let at = gap;
  for (const box of boxes) {
    for (let r = 0; r < box.h; r += 1) {
      for (let c = 0; c < box.w; c += 1) {
        if (strip.data[(box.y + r) * strip.cols + box.x + c] === 1) {
          const p = ((box.y - top + gap + r) * width + at + c) * 3;
          image.data.fill(0, p, p + 3);
        }
      }
    }
    at += box.w + gap;
  }
  return image;
}

/** The letters a tab's left-hand line holds, left to right, and how they were read. */
export async function readLeftHand(
  cv: OpenCv,
  page: ColorImage,
  ink: Uint8Array,
  tab: DetectedButtonTab,
  read: ReadCrops
): Promise<{
  readonly letters: ButtonTabLetter[];
  readonly readAs: "line" | "words" | "none";
}> {
  const letterHeight = tab.letters.t.h;
  const reach = Math.max(tab.down, ...tab.letterLine.map((l) => l.reach));
  const top = Math.round(LETTERS.from * tab.down);
  const bottom = Math.ceil(reach + LETTERS.pad * letterHeight);
  if (bottom <= top) {
    return { letters: [], readAs: "none" };
  }
  const strip = stripUnder(ink, page.width, page.height, tab, top, bottom);
  const pieces = componentBoxes(cv, strip.data, strip.cols, strip.rows).filter(
    (box) =>
      box.h >= LETTERS.minHeight * letterHeight &&
      box.h <= LETTERS.maxHeight * letterHeight &&
      box.w <= LETTERS.maxWidth * letterHeight &&
      !(box.y === 0 && box.w < LETTERS.stubWidth * letterHeight)
  );
  const words = wordsOf(pieces, LETTERS.joinGap * letterHeight);
  if (words.length === 0) {
    return { letters: [], readAs: "none" };
  }
  const gap = Math.round(LETTERS.gap * letterHeight);
  // The recogniser drops the spaces between words, so the line is taken
  // glyph by glyph: one character per piece, left to right.
  const [line] = await read([packed(strip, words, gap)]);
  const glyphs = [...printed(line ?? "").replace(/\s+/g, "")];
  const ordered = [...pieces].sort((a, b) => a.x - b.x);
  let readAs: "line" | "words" = "line";
  let texts: string[];
  if (glyphs.length === ordered.length) {
    texts = words.map((word) =>
      ordered
        .flatMap((piece, k) =>
          piece.x >= word.x && piece.x + piece.w <= word.x + word.w
            ? [glyphs[k] ?? ""]
            : []
        )
        .join("")
    );
  } else {
    readAs = "words";
    texts = (await read(words.map((word) => packed(strip, [word], gap)))).map(
      (text) => printed(text).replace(/\s+/g, "")
    );
  }
  const letters = words.flatMap((word, k) => {
    const text = texts[k] ?? "";
    return text === ""
      ? []
      : [{ text, x: (tab.x0 + word.x + word.w / 2) / page.width }];
  });
  return { letters, readAs };
}

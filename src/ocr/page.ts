/**
 * The two readings of text a page gets: the server's chord strips, one per
 * staff, and homr's title, from the band above the first staff.
 */

import { pySliceBounds, toFloat32 } from "../image/numeric.js";
import { type ColorImage, slicePlane } from "../image/plane.js";
import type { Staff } from "../model/staff.js";
import type { PageText, StaffBox } from "../result.js";
import type { RapidOcr } from "./rapid-ocr.js";
import { chordStrip, cropStrip, pageText, sortPageTexts } from "./strips.js";

/** A crop RapidOCR cannot read: an empty slice, on which the Python divides by zero. */
export class OcrError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OcrError";
  }
}

function nonEmpty(image: ColorImage, what: string): ColorImage {
  if (image.width === 0 || image.height === 0) {
    throw new OcrError(`${what} is ${image.width}x${image.height}`);
  }
  return image;
}

/**
 * The texts the server's route answers for this page and these staves. The
 * page is the image as decoded, before homr's crop and resize, as the script
 * reads it from disk.
 */
export async function readStripTexts(
  ocr: RapidOcr,
  page: ColorImage,
  staves: readonly StaffBox[],
  options: {
    readonly minTextScore?: number;
    readonly onStrip?: (done: number) => void;
  } = {}
): Promise<PageText[]> {
  const texts: PageText[] = [];
  for (const [i, staff] of staves.entries()) {
    const strip = chordStrip(staff, page);
    const lines = await ocr.read(
      nonEmpty(cropStrip(page, strip), `the strip above staff ${staff.index}`),
      options.minTextScore
    );
    texts.push(...lines.map((line) => pageText(line, strip, page)));
    options.onStrip?.(i + 1);
  }
  return sortPageTexts(texts);
}

const MIN_TITLE_LETTERS = 4;
const LETTER = /\p{L}/u;
/** Everything a title does not keep: not a letter, mark, digit, apostrophe or hyphen. */
const NOT_TITLE_TEXT = /[^\p{L}\p{M}\p{N}'’-]+/gu;

/**
 * title_detection.is_tempo_marking: under four characters, or under four
 * letters. Deliberately not homr's: homr counts only a to z, so a title in
 * another alphabet read as a tempo marking and was dropped. Any letter counts
 * here, in any alphabet. A tempo marking still has none or one: the
 * RapidOCR models read the note of "♩=85" as 小, one letter.
 */
export function isTempoMarking(text: string): boolean {
  const characters = [...text];
  if (characters.length < MIN_TITLE_LETTERS) {
    return true;
  }
  const letters = characters.filter((c) => LETTER.test(c)).length;
  return letters < MIN_TITLE_LETTERS;
}

/**
 * title_detection.cleanup_text, deliberately not homr's. homr keeps only a to
 * z and digits, which turned "Marche des élèves" into "Marche des l ves" and
 * "Le p'tit Sarny" into "Le p tit Sarny". This keeps letters and digits of
 * any alphabet, apostrophes and hyphens, and reduces everything else to
 * single spaces, as homr does. NFC first, so an accent read as a separate
 * combining mark stays on its letter. Titles in a to z come out exactly as
 * homr's (docs/decisions.tsv, 2026-10-06).
 */
export const cleanupText = (text: string): string =>
  text.normalize("NFC").replace(NOT_TITLE_TEXT, " ").trim();

/** The crop _detect_title_task reads: 15 unit sizes above the first staff, 50 px wider each side. */
export function titleCrop(original: ColorImage, top: Staff): ColorImage {
  let height = Math.trunc(15 * top.averageUnitSize);
  const y = Math.max(Math.trunc(top.minY) - height, 0);
  const x = Math.max(Math.trunc(top.minX) - 50, 0);
  const width = Math.min(
    Math.trunc(top.maxX - top.minX) + 100,
    original.width - x
  );
  height = Math.min(height, Math.trunc(top.minY) - y);
  const rows = pySliceBounds(y, y + height, original.height);
  const columns = pySliceBounds(x, x + width, original.width);
  return slicePlane(
    original,
    columns.start,
    rows.start,
    columns.stop,
    rows.stop
  );
}

/**
 * homr's _detect_title_task: the line with the tallest characters that is
 * not a tempo marking, reduced to letters, digits and single spaces.
 * `original` is the autocropped and resized page homr's Debug holds.
 */
export async function detectTitle(
  ocr: RapidOcr,
  original: ColorImage,
  top: Staff
): Promise<string> {
  const lines = await ocr.read(
    nonEmpty(titleCrop(original, top), "the band above the first staff")
  );
  let best: { readonly size: number; readonly text: string } | undefined;
  for (const line of lines) {
    if (isTempoMarking(line.text)) {
      continue;
    }
    const ys = line.box.map(([, y]) => y);
    const size = toFloat32(
      toFloat32(Math.max(...ys) - Math.min(...ys)) / [...line.text].length
    );
    if (best === undefined || size > best.size) {
      best = { size, text: line.text };
    }
  }
  return best === undefined ? "" : cleanupText(best.text);
}

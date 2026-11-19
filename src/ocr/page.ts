/**
 * The two readings of text a page gets: the server's chord strips, one per
 * staff, and homr's title, from the band above the first staff.
 */

import { pySliceBounds, toFloat32 } from "../image/numeric.js";
import { type ColorImage, slicePlane } from "../image/plane.js";
import type { Staff } from "../model/staff.js";
import type { PageText, StaffBox } from "../result.js";
import type { OcrLine, RapidOcr } from "./rapid-ocr.js";
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

/**
 * The crop _detect_title_task reads: 15 unit sizes above the first staff, 50 px
 * wider each side. `toPageEdge` runs it on to the page's left edge, keeping
 * its right edge, for a title that starts further left than that.
 */
export function titleCrop(
  original: ColorImage,
  top: Staff,
  toPageEdge = false
): ColorImage {
  let height = Math.trunc(15 * top.averageUnitSize);
  const y = Math.max(Math.trunc(top.minY) - height, 0);
  const homrX = titleLeft(top);
  const x = toPageEdge ? 0 : homrX;
  const width = Math.min(
    Math.trunc(top.maxX - top.minX) + 100 + homrX - x,
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

/** Where homr's title crop starts on the page. */
const titleLeft = (top: Staff): number =>
  Math.max(Math.trunc(top.minX) - 50, 0);

/**
 * A line whose box starts this close to the crop's left edge was cut by it:
 * "The Sailor's Bonnet" printed left of an indented first staff came out
 * "he Sailor's Bonnet" (2026-11-19).
 */
const CUT_AT_LEFT_PX = 2;

const xsOf = (line: OcrLine) => line.box.map(([x]) => x);
const ysOf = (line: OcrLine) => line.box.map(([, y]) => y);

/** The line with the tallest characters that is not a tempo marking. */
function tallestLine(lines: readonly OcrLine[]): OcrLine | undefined {
  let best: { readonly line: OcrLine; readonly size: number } | undefined;
  for (const line of lines) {
    if (isTempoMarking(line.text)) {
      continue;
    }
    const ys = ysOf(line);
    const size = toFloat32(
      toFloat32(Math.max(...ys) - Math.min(...ys)) / [...line.text].length
    );
    if (best === undefined || size > best.size) {
      best = { line, size };
    }
  }
  return best?.line;
}

/**
 * The line of `wider` (read from the crop run on to the page's left edge)
 * that is `cut` made whole: the one sharing most of its rows, among those
 * that reach into its columns. `shift` is how much further left the wider
 * crop starts.
 */
function wholeLine(
  cut: OcrLine,
  wider: readonly OcrLine[],
  shift: number
): OcrLine | undefined {
  const right = Math.max(...xsOf(cut)) + shift;
  const top = Math.min(...ysOf(cut));
  const bottom = Math.max(...ysOf(cut));
  const sharedRows = (line: OcrLine): number => {
    const ys = ysOf(line);
    const reaches = Math.min(...xsOf(line)) < right;
    return reaches
      ? Math.min(bottom, Math.max(...ys)) - Math.max(top, Math.min(...ys))
      : 0;
  };
  let whole: OcrLine | undefined;
  for (const line of wider) {
    if (sharedRows(line) > (whole === undefined ? 0 : sharedRows(whole))) {
      whole = line;
    }
  }
  return whole;
}

/**
 * homr's _detect_title_task: the line with the tallest characters that is
 * not a tempo marking, reduced to letters, digits and single spaces.
 * `original` is the autocropped and resized page homr's Debug holds.
 *
 * One departure: when that line starts at the crop's left edge, it is read
 * again from a crop run on to the page's edge. Only a cut line is re-read,
 * because a crop that wide from the start takes a tuning or a part label at
 * the left margin for the title (two clawhammer pages, Amazing Grace).
 */
export async function detectTitle(
  ocr: RapidOcr,
  original: ColorImage,
  top: Staff
): Promise<string> {
  const what = "the band above the first staff";
  const best = tallestLine(
    await ocr.read(nonEmpty(titleCrop(original, top), what))
  );
  if (best === undefined) {
    return "";
  }
  const shift = titleLeft(top);
  if (shift === 0 || Math.min(...xsOf(best)) > CUT_AT_LEFT_PX) {
    return cleanupText(best.text);
  }
  const wider = await ocr.read(nonEmpty(titleCrop(original, top, true), what));
  return cleanupText((wholeLine(best, wider, shift) ?? best).text);
}

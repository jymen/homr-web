/**
 * The AbcMusicStudio server's chord OCR (AbcGoDb abcsql/omr_chord_ocr.py and
 * runOmrChordOcr in omr.go): the strip from 1.9 to 0.05 staff heights above
 * each staff, 3 % of the page wider on the left and 2 % on the right, read by
 * RapidOCR, every box normalised to the page, texts sorted by staff then x0.
 */

import { pyRound, pySliceBounds } from "../image/numeric.js";
import { type ColorImage, slicePlane } from "../image/plane.js";
import type { PageText, StaffBox } from "../result.js";
import type { OcrLine } from "./rapid-ocr.js";

/** A strip in page pixels: rows [top, bottom) and columns [x0, x1) once Python's slice clamps them. */
export interface ChordStrip {
  readonly bottom: number;
  readonly staff: number;
  /** As the script computes it, possibly negative; the crop starts at max(top, 0). */
  readonly top: number;
  readonly x0: number;
  readonly x1: number;
}

export function chordStrip(
  staff: StaffBox,
  page: { readonly height: number; readonly width: number }
): ChordStrip {
  const { cx, cy, h, w } = staff;
  return {
    bottom: Math.trunc((cy - h / 2 - 0.05 * h) * page.height),
    staff: staff.index,
    top: Math.trunc((cy - h / 2 - 1.9 * h) * page.height),
    x0: Math.max(Math.trunc((cx - w / 2 - 0.03) * page.width), 0),
    x1: Math.trunc((cx + w / 2 + 0.02) * page.width),
  };
}

/** `img[max(top, 0):bot, x0:x1]` with numpy's slice clamping. */
export function cropStrip(page: ColorImage, strip: ChordStrip): ColorImage {
  const rows = pySliceBounds(Math.max(strip.top, 0), strip.bottom, page.height);
  const columns = pySliceBounds(strip.x0, strip.x1, page.width);
  return slicePlane(page, columns.start, rows.start, columns.stop, rows.stop);
}

/** One RapidOCR line as the script writes it: rounded to 4 digits in page units, the score to 3. */
export function pageText(
  line: OcrLine,
  strip: ChordStrip,
  page: { readonly height: number; readonly width: number }
): PageText {
  const xs = line.box.map(([x]) => x);
  const ys = line.box.map(([, y]) => y);
  const top = Math.max(strip.top, 0);
  return {
    score: pyRound(line.score, 3),
    staff: strip.staff,
    text: line.text,
    x0: pyRound((Math.min(...xs) + strip.x0) / page.width, 4),
    x1: pyRound((Math.max(...xs) + strip.x0) / page.width, 4),
    y0: pyRound((Math.min(...ys) + top) / page.height, 4),
    y1: pyRound((Math.max(...ys) + top) / page.height, 4),
  };
}

/** runOmrChordOcr's sort.SliceStable: by staff, then by x0. */
export const sortPageTexts = (texts: readonly PageText[]): PageText[] =>
  [...texts].sort((a, b) => a.staff - b.staff || a.x0 - b.x0);

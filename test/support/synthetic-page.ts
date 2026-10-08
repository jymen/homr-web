/**
 * Pages drawn in the test with opencv.js, so the tab guard's fixtures carry
 * no third-party rights: horizontal lines, fret numbers on a white knock-out
 * the way tab software prints them, and filled noteheads with stems.
 * Coordinates are page pixels at 150 dpi (1240 x 1754).
 */

import type { Mat } from "@techstark/opencv-js";
import {
  type MatScope,
  type OpenCv,
  withMatScope,
} from "../../src/cv/opencv.js";
import { type ColorImage, planeFromBytes } from "../../src/image/plane.js";

export const PAGE_WIDTH = 1240;
export const PAGE_HEIGHT = 1754;
const X0 = 90;
const X1 = PAGE_WIDTH - 90;
const BLACK = [0, 0, 0, 255];
const WHITE = [255, 255, 255, 255];

/** A system to draw: `top` is the first line's row. */
export type Drawn =
  | {
      readonly kind: "tab";
      readonly lines: 4 | 5 | 6;
      readonly spacing: number;
      readonly top: number;
    }
  | { readonly kind: "staff"; readonly spacing: number; readonly top: number };

function drawLines(
  cv: OpenCv,
  page: Mat,
  top: number,
  count: number,
  spacing: number
): void {
  for (let k = 0; k < count; k += 1) {
    const y = top + k * spacing;
    cv.line(page, new cv.Point(X0, y), new cv.Point(X1, y), BLACK, 2);
  }
  for (const x of [X0, X1]) {
    cv.line(
      page,
      new cv.Point(x, top),
      new cv.Point(x, top + (count - 1) * spacing),
      BLACK,
      2
    );
  }
}

/** Fret numbers 0 to 12 across the system, one per column, cycling over the lines. */
function drawFrets(
  cv: OpenCv,
  page: Mat,
  top: number,
  lines: number,
  spacing: number
): void {
  const scale = spacing / 30;
  for (let column = 0; column < 32; column += 1) {
    const text = String((column * 7) % 13);
    const line = column % lines;
    const x = X0 + 40 + column * 32;
    const y = top + line * spacing;
    // Hershey simplex digits are about 22 px tall and 20 px wide at scale 1; this build has no getTextSize.
    const height = Math.round(22 * scale);
    const width = Math.round(20 * scale * text.length);
    cv.rectangle(
      page,
      new cv.Point(x - 2, y - height / 2 - 2),
      new cv.Point(x + width + 2, y + height / 2 + 2),
      WHITE,
      -1
    );
    cv.putText(
      page,
      text,
      new cv.Point(x, Math.round(y + height / 2)),
      cv.FONT_HERSHEY_SIMPLEX,
      scale,
      BLACK,
      2
    );
  }
}

/** Filled noteheads on lines and spaces, stems up, a bar line every four. */
function drawNotes(cv: OpenCv, page: Mat, top: number, spacing: number): void {
  for (let k = 0; k < 28; k += 1) {
    const step = (k * 5) % 9;
    const x = X0 + 60 + k * 36;
    const y = Math.round(top + 4 * spacing - (step * spacing) / 2);
    cv.ellipse(
      page,
      new cv.Point(x, y),
      new cv.Size(Math.round(spacing * 0.65), Math.round(spacing * 0.45)),
      -20,
      0,
      360,
      BLACK,
      -1
    );
    const stemX = x + Math.round(spacing * 0.6);
    cv.line(
      page,
      new cv.Point(stemX, y),
      new cv.Point(stemX, y - Math.round(3.5 * spacing)),
      BLACK,
      2
    );
    if (k % 4 === 3) {
      const barX = x + 20;
      cv.line(
        page,
        new cv.Point(barX, top),
        new cv.Point(barX, top + 4 * spacing),
        BLACK,
        2
      );
    }
  }
}

function toColorImage(page: Mat): ColorImage {
  const rgba = page.data;
  const bgr = new Uint8Array(PAGE_WIDTH * PAGE_HEIGHT * 3);
  for (let i = 0; i < PAGE_WIDTH * PAGE_HEIGHT; i += 1) {
    bgr[i * 3] = rgba[i * 4 + 2] ?? 0;
    bgr[i * 3 + 1] = rgba[i * 4 + 1] ?? 0;
    bgr[i * 3 + 2] = rgba[i * 4] ?? 0;
  }
  return planeFromBytes("bgr", PAGE_WIDTH, PAGE_HEIGHT, bgr);
}

export function drawPage(cv: OpenCv, systems: readonly Drawn[]): ColorImage {
  return withMatScope((scope: MatScope) => {
    const page = scope.keep(
      new cv.Mat(PAGE_HEIGHT, PAGE_WIDTH, cv.CV_8UC4, new cv.Scalar(...WHITE))
    );
    for (const system of systems) {
      if (system.kind === "tab") {
        drawLines(cv, page, system.top, system.lines, system.spacing);
        drawFrets(cv, page, system.top, system.lines, system.spacing);
      } else {
        drawLines(cv, page, system.top, 5, system.spacing);
        drawNotes(cv, page, system.top, system.spacing);
      }
    }
    return toColorImage(page);
  });
}

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
      /** Line thickness in pixels; 2 when absent. */
      readonly thickness?: number;
      /** Rows each line drops from its left end to its right end, as a scan turned a little; 0 when absent. */
      readonly tilt?: number;
      /** Zeros as a hand writes them, hollow ovals about 1.2 line spacings wide, in place of the printed frets. */
      readonly wideZeros?: boolean;
      /** The lines and bar lines alone, no fret on them. */
      readonly empty?: boolean;
      readonly top: number;
    }
  | { readonly kind: "staff"; readonly spacing: number; readonly top: number }
  | {
      /** Polylines drawn as a pen draws them, over whatever else is on the page. */
      readonly kind: "strokes";
      readonly strokes: readonly {
        readonly points: readonly (readonly [number, number])[];
        readonly thickness: number;
      }[];
    };

function drawLines(
  cv: OpenCv,
  page: Mat,
  top: number,
  count: number,
  spacing: number,
  { thickness = 2, tilt = 0 }: { thickness?: number; tilt?: number } = {}
): void {
  for (let k = 0; k < count; k += 1) {
    const y = top + k * spacing;
    cv.line(
      page,
      new cv.Point(X0, y),
      new cv.Point(X1, y + tilt),
      BLACK,
      thickness
    );
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

/** What `drawPage` prints on a tab: one fret per column, 0 to 12, cycling over the lines; `string` 1 is the top line. */
export const drawnFrets = (lines: number) =>
  Array.from({ length: 32 }, (_, column) => ({
    fret: (column * 7) % 13,
    string: (column % lines) + 1,
  }));

function drawFrets(
  cv: OpenCv,
  page: Mat,
  top: number,
  lines: number,
  spacing: number
): void {
  const scale = spacing / 30;
  for (const [column, { fret, string }] of drawnFrets(lines).entries()) {
    const text = String(fret);
    const line = string - 1;
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

/** Hollow ovals about 1.2 spacings wide and 0.95 tall, one per column on each line in turn, on white. */
function drawWideZeros(
  cv: OpenCv,
  page: Mat,
  top: number,
  lines: number,
  spacing: number
): void {
  for (let column = 0; column < 24; column += 1) {
    const x = X0 + 60 + column * 40;
    const y = top + (column % lines) * spacing;
    const axes = new cv.Size(
      Math.round(spacing * 0.55),
      Math.round(spacing * 0.45)
    );
    cv.ellipse(page, new cv.Point(x, y), axes, -15, 0, 360, WHITE, -1);
    cv.ellipse(page, new cv.Point(x, y), axes, -15, 0, 360, BLACK, 2);
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

function drawStrokes(
  cv: OpenCv,
  page: Mat,
  strokes: Extract<Drawn, { kind: "strokes" }>["strokes"]
): void {
  for (const { points, thickness } of strokes) {
    for (const [k, [x, y]] of points.slice(1).entries()) {
      const [px, py] = points[k] ?? [x, y];
      cv.line(page, new cv.Point(px, py), new cv.Point(x, y), BLACK, thickness);
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
      if (system.kind === "strokes") {
        drawStrokes(cv, page, system.strokes);
      } else if (system.kind === "tab") {
        drawLines(cv, page, system.top, system.lines, system.spacing, system);
        if (system.empty === true) {
          continue;
        }
        if (system.wideZeros === true) {
          drawWideZeros(cv, page, system.top, system.lines, system.spacing);
        } else {
          drawFrets(cv, page, system.top, system.lines, system.spacing);
        }
      } else {
        drawLines(cv, page, system.top, 5, system.spacing);
        drawNotes(cv, page, system.top, system.spacing);
      }
    }
    return toColorImage(page);
  });
}

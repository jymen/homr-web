/**
 * A born-digital PDF's own text, used in place of the recogniser where it
 * holds the tab's numbers. The page image still gives the geometry (lines,
 * stems, beams, arcs, barlines); the text layer gives what the recogniser
 * would read: the frets, the technique letters, the time signature and the
 * header text. Measured on the eleven tab pages that carry a text layer
 * (docs/design/tab-4-findings.md, decisions.tsv tab-4): never worse than the
 * recogniser against the exported MusicXML, better on three pages, and no
 * recogniser run for the frets at all. A scan, a photo or a PDF whose glyphs
 * are outlines has no such text; the recogniser reads those.
 */

import type { OcrLine } from "../ocr/rapid-ocr.js";
import type { Box, LineGroup } from "./detect.js";

/**
 * One run of text from the PDF, as pdf.js's getTextContent gives it, as a
 * box in page-normalised coordinates (0..1 of the page's width and height, y
 * downward): from the run's start to its end across, and from its baseline
 * up by its font size (`y1` is the baseline). A run turned a quarter, a
 * tuning printed up the margin, is given by the box around it. An app
 * holding pdf.js items converts each with the page's viewport (README,
 * "Tablature").
 */
export interface PageTextItem {
  readonly text: string;
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

/**
 * A glyph's box reaches this share of its font size above the baseline (a
 * digit's cap height, 0.7, and its overshoot) and this share below it, so
 * the box holds all of the digit's ink and a stem under it starts outside.
 */
const ABOVE_BASELINE = 0.78;
const BELOW_BASELINE = 0.08;

/** An item in page pixels, its glyph box from cap height to baseline. */
export interface LayerGlyph {
  readonly box: Box;
  readonly text: string;
}

/**
 * Characters of the Unicode private use areas: a music font's time-signature
 * digits and clefs (TablEdit's), which carry no text. The recogniser reads
 * those from the image.
 */
const PRIVATE_USE = /[\uE000-\uF8FF]|[\u{F0000}-\u{10FFFD}]/gu;

/**
 * The runs as glyph boxes in page pixels. A run holding a space ("0 2", two
 * frets a slur joins, set as one run) is split at its spaces, each word
 * placed across the run in proportion to its characters, so each fret is a
 * glyph of its own.
 */
export function glyphsOf(
  items: readonly PageTextItem[],
  width: number,
  height: number
): LayerGlyph[] {
  return items.flatMap((item) => {
    const baseline = item.y1 * height;
    const size = (item.y1 - item.y0) * height;
    const top = Math.round(baseline - ABOVE_BASELINE * size);
    const h = Math.max(1, Math.round(baseline + BELOW_BASELINE * size) - top);
    const x0 = item.x0 * width;
    const perChar =
      ((item.x1 - item.x0) * width) / Math.max(1, item.text.length);
    return [...item.text.matchAll(/\S+/g)].flatMap((word) => {
      const text = word[0].replace(PRIVATE_USE, "");
      if (text === "") {
        return [];
      }
      const left = x0 + (word.index ?? 0) * perChar;
      return [
        {
          box: {
            h,
            w: Math.max(1, Math.round(word[0].length * perChar)),
            x: Math.round(left),
            y: top,
          },
          text,
        },
      ];
    });
  });
}

const centreX = (box: Box) => box.x + box.w / 2;
const centreY = (box: Box) => box.y + box.h / 2;

/** Glyphs on the system's lines no taller than a fret: what the recogniser would be given, `line` 0-based from the top. */
export function onLines(
  glyphs: readonly LayerGlyph[],
  group: LineGroup
): (LayerGlyph & { readonly line: number })[] {
  const s = group.spacing;
  return glyphs.flatMap((glyph) => {
    const x = centreX(glyph.box);
    if (
      x < group.x0 - 0.5 * s ||
      x > group.x1 + 0.5 * s ||
      glyph.box.h > 1.2 * s
    ) {
      return [];
    }
    const y = centreY(glyph.box);
    let line = -1;
    let distance = Number.POSITIVE_INFINITY;
    for (const [k, l] of group.lines.entries()) {
      if (Math.abs(l.y - y) < distance) {
        distance = Math.abs(l.y - y);
        line = k;
      }
    }
    return distance <= 0.3 * s ? [{ ...glyph, line }] : [];
  });
}

/** The text of the glyphs whose centre lies in `box` grown by `pad` pixels, left to right: what the recogniser would read in that crop. */
export function textIn(
  glyphs: readonly LayerGlyph[],
  box: Box,
  pad: number
): string {
  return glyphs
    .filter(
      (g) =>
        centreX(g.box) >= box.x - pad &&
        centreX(g.box) <= box.x + box.w + pad &&
        centreY(g.box) >= box.y - pad &&
        centreY(g.box) <= box.y + box.h + pad
    )
    .sort((a, b) => a.box.x - b.box.x)
    .map((g) => g.text)
    .join("");
}

/**
 * The page's text in a region of page pixels as recogniser lines in the
 * region's own pixels: items on one baseline joined left to right when the
 * gap between them is under a font size, score 1.
 */
export function linesIn(
  glyphs: readonly LayerGlyph[],
  region: {
    readonly x0: number;
    readonly x1: number;
    readonly y0: number;
    readonly y1: number;
  }
): OcrLine[] {
  const inside = glyphs
    .filter(
      (g) =>
        centreX(g.box) >= region.x0 &&
        centreX(g.box) < region.x1 &&
        centreY(g.box) >= region.y0 &&
        centreY(g.box) < region.y1
    )
    .sort(
      (a, b) => a.box.y + a.box.h - (b.box.y + b.box.h) || a.box.x - b.box.x
    );
  const lines: { box: Box; text: string }[] = [];
  for (const glyph of inside) {
    const baseline = glyph.box.y + glyph.box.h;
    const line = lines.find(
      (l) =>
        Math.abs(l.box.y + l.box.h - baseline) < 0.3 * glyph.box.h &&
        glyph.box.x - (l.box.x + l.box.w) < 1.5 * glyph.box.h &&
        glyph.box.x >= l.box.x
    );
    if (line === undefined) {
      lines.push({ box: glyph.box, text: glyph.text });
      continue;
    }
    const gap = glyph.box.x - (line.box.x + line.box.w);
    line.text = `${line.text}${gap > 0.15 * glyph.box.h ? " " : ""}${glyph.text}`;
    const right = Math.max(line.box.x + line.box.w, glyph.box.x + glyph.box.w);
    const y = Math.min(line.box.y, glyph.box.y);
    line.box = {
      h: Math.max(line.box.y + line.box.h, glyph.box.y + glyph.box.h) - y,
      w: right - line.box.x,
      x: line.box.x,
      y,
    };
  }
  return lines.map(({ box, text }) => {
    const x0 = box.x - region.x0;
    const y0 = box.y - region.y0;
    return {
      box: [
        [x0, y0],
        [x0 + box.w, y0],
        [x0 + box.w, y0 + box.h],
        [x0, y0 + box.h],
      ],
      score: 1,
      text,
    };
  });
}

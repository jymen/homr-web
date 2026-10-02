/**
 * Which of the bar-line candidates are bar lines. Port of
 * bar_line_detection.detect_bar_lines and of the filter main.py applies
 * before calling it.
 */

import { barLineMaxWidth, barLineMinHeight } from "../model/constants.js";
import type { NoteheadWithStem } from "../model/symbols.js";
import { type BoxOps, overlapsAny } from "./box-ops.js";
import type { RotatedBox } from "./boxes.js";

/**
 * main.py 255-262: the candidates overlapping no notehead and no stem. The
 * noteheads are the un-thickened ones; a stem two noteheads share is tested
 * twice, as in Python.
 */
export function barLineCandidates(
  ops: BoxOps,
  barLines: readonly RotatedBox[],
  noteheads: readonly NoteheadWithStem[]
): RotatedBox[] {
  const heads = noteheads.map((one) => one.notehead);
  const stems = noteheads.flatMap((one) =>
    one.stem === null ? [] : [one.stem.box]
  );
  return barLines.filter(
    (line) => !(overlapsAny(ops, line, heads) || overlapsAny(ops, line, stems))
  );
}

/**
 * detect_bar_lines: keeps a box unless it is shorter than
 * barLineMinHeight(unit) or wider than barLineMaxWidth(unit); equality keeps.
 * `unitSize` is the median notehead height, not a staff unit.
 */
export function detectBarLines(
  candidates: readonly RotatedBox[],
  unitSize: number
): RotatedBox[] {
  return candidates.filter(
    (line) =>
      !(
        line.rect.h < barLineMinHeight(unitSize) ||
        line.rect.w > barLineMaxWidth(unitSize)
      )
  );
}

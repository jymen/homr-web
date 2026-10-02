/**
 * Noteheads, their stems, and the notes they become on a staff. Port of
 * note_detection.py and of the median main.py takes of the noteheads.
 *
 * In homr 0.7.0 nothing downstream reads a note's position, stem or direction;
 * they exist for notes.json and for the symbols lists in multistaffs.json. The
 * median notehead height is the load-bearing half: it is the unit bar lines
 * are measured in.
 */

import {
  floorDiv,
  median,
  pySliceBounds,
  roundHalfEven,
  truncToInt,
} from "../image/numeric.js";
import { type Mask, nonzeroRowBounds } from "../image/plane.js";
import { NOTEHEAD_SIZE_RATIO } from "../model/constants.js";
import {
  findPositionInUnitSizes,
  isOnStaffZone,
  type Staff,
  staffPointAt,
  withSymbols,
} from "../model/staff.js";
import {
  createNote,
  type Note,
  type NoteheadWithStem,
  STEM_DIRECTIONS,
} from "../model/symbols.js";
import type { BoxOps } from "./box-ops.js";
import {
  cornersOf,
  type Ellipse,
  legacyConventionRectOf,
  type RotatedBox,
} from "./boxes.js";

/** How far a notehead is grown, on both axes, before looking for its stem. */
export const STEM_SEARCH_THICKNESS = 15;

/**
 * combine_noteheads_with_stems. Output order is the noteheads sorted by centre
 * y, stable. The first overlapping stem in input order wins and a stem can be
 * claimed by several noteheads. UP when the stem's centre is strictly above.
 */
export function combineNoteheadsWithStems(
  ops: BoxOps,
  noteheads: readonly Ellipse[],
  stems: readonly RotatedBox[]
): NoteheadWithStem[] {
  return [...noteheads]
    .sort((a, b) => a.rect.cy - b.rect.cy)
    .map((notehead) => {
      const thickened = ops.thicker(notehead, STEM_SEARCH_THICKNESS);
      const box = stems.find((stem) => ops.overlaps(stem, thickened));
      if (box === undefined) {
        return { notehead, stem: null };
      }
      const direction =
        box.rect.cy < notehead.rect.cy
          ? STEM_DIRECTIONS.up
          : STEM_DIRECTIONS.down;
      return { notehead, stem: { box, direction } };
    });
}

/** `float(np.median([n.notehead.size[1] ...]))`. NaN on an empty list; the caller has already refused one. */
export function averageNoteheadHeight(
  noteheads: readonly NoteheadWithStem[]
): number {
  return median(noteheads.map((one) => one.notehead.rect.h));
}

/** note_detection's `bbox`: integer pixel bounds, x1 and y1 exclusive. */
export interface PixelBox {
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

/**
 * adjust_bbox: the box shrunk to the rows of `noteheads` that are set inside
 * it, then grown by one row each way; unchanged when none is.
 *
 * The region is a Python slice on both axes, so a `y0` of -1, which this
 * function itself produces for a notehead on row 0, starts at the last row of
 * the mask and selects nothing.
 */
export function adjustBbox(bbox: PixelBox, noteheads: Mask): PixelBox {
  const rows = pySliceBounds(bbox.y0, bbox.y1, noteheads.height);
  const columns = pySliceBounds(bbox.x0, bbox.x1, noteheads.width);
  const set = nonzeroRowBounds(
    noteheads,
    columns.start,
    rows.start,
    columns.stop,
    rows.stop
  );
  if (set === null) {
    return bbox;
  }
  // note_detection.py:34 adds bbox[1] itself to an index counted from the slice's start.
  const offset = bbox.y0 - rows.start;
  return {
    x0: bbox.x0,
    x1: bbox.x1,
    y0: set.minY + offset - 1,
    y1: set.maxY + offset + 1,
  };
}

/** get_center's x: `int(round((x0 + x1) / 2))`, half to even. */
const centerOfSpan = (from: number, to: number): number =>
  roundHalfEven((from + to) / 2);

/** check_bbox_size's else branch: `round(h / unit)` boxes of `h // n` rows, the remainder left off the bottom. */
function splitByHeight(bbox: PixelBox, unitSize: number): PixelBox[] {
  const height = bbox.y1 - bbox.y0;
  const count = roundHalfEven(height / unitSize);
  const boxes: PixelBox[] = [];
  const step = count > 0 ? floorDiv(height, count) : 0;
  for (let i = 0; i < count; i += 1) {
    boxes.push({
      x0: bbox.x0,
      x1: bbox.x1,
      y0: bbox.y0 + i * step,
      y1: bbox.y0 + (i + 1) * step,
    });
  }
  return boxes;
}

/**
 * check_bbox_size: a box nearer two notehead widths than one is cut at its
 * centre and each half checked; otherwise it is cut into `round(h / unit)`
 * rows.
 *
 * Every box a width split yields goes through this function a second time
 * (note_detection.py:69-73). For a unit of ten pixels or more the second pass
 * returns what it was given; it drops a box only where `h // n` is half a
 * unit or less, which takes a unit of about two pixels. A width split that
 * yields nothing falls through to the height split of the box itself.
 */
export function checkBboxSize(
  bbox: PixelBox,
  noteheads: Mask,
  unitSize: number
): PixelBox[] {
  const width = bbox.x1 - bbox.x0;
  const noteWidth = NOTEHEAD_SIZE_RATIO * unitSize;
  const halves: PixelBox[] = [];
  if (Math.abs(width - noteWidth) > Math.abs(width - noteWidth * 2)) {
    const middle = centerOfSpan(bbox.x0, bbox.x1);
    for (const half of [
      { ...bbox, x1: middle },
      { ...bbox, x0: middle },
    ]) {
      const adjusted = adjustBbox(half, noteheads);
      halves.push(...checkBboxSize(adjusted, noteheads, unitSize));
    }
  }
  if (halves.length === 0) {
    return splitByHeight(bbox, unitSize);
  }
  return halves.flatMap((box) => checkBboxSize(box, noteheads, unitSize));
}

/**
 * split_clumps_of_noteheads. One box or none returns `[chunk]`, the original
 * object, even when the one box differs from it. Each piece is an ellipse at
 * angle 0 on integer centre and size, carrying the chunk's contour, debugId
 * and stem.
 */
export function splitClumpsOfNoteheads(
  ops: BoxOps,
  chunk: NoteheadWithStem,
  noteheads: Mask,
  staff: Staff
): NoteheadWithStem[] {
  const { bottomRight, topLeft } = cornersOf(chunk.notehead.rect);
  const boxes = checkBboxSize(
    {
      x0: truncToInt(topLeft.x),
      x1: truncToInt(bottomRight.x),
      y0: truncToInt(topLeft.y),
      y1: truncToInt(bottomRight.y),
    },
    noteheads,
    staff.averageUnitSize
  );
  if (boxes.length <= 1) {
    return [chunk];
  }
  return boxes.map((box) => ({
    notehead: ops.ellipseFromRect(
      legacyConventionRectOf({
        angle: 0,
        cx: centerOfSpan(box.x0, box.x1),
        cy: centerOfSpan(box.y0, box.y1),
        h: box.y1 - box.y0,
        w: box.x1 - box.x0,
      }),
      chunk.notehead.contour,
      chunk.notehead.debugId
    ),
    stem: chunk.stem,
  }));
}

/** The notes of one staff, in notehead order. */
function notesOnStaff(
  ops: BoxOps,
  staff: Staff,
  noteheads: readonly NoteheadWithStem[],
  noteheadMask: Mask
): Note[] {
  const notes: Note[] = [];
  for (const chunk of noteheads) {
    // note_detection.py:167 looks the point up again per piece, at the chunk's x and not the piece's, so one lookup serves both.
    const point = isOnStaffZone(staff, chunk.notehead)
      ? staffPointAt(staff, chunk.notehead.rect.cx)
      : null;
    if (point === null) {
      continue;
    }
    const unit = point.averageUnitSize;
    const { h, w } = chunk.notehead.rect;
    if (w < 0.5 * unit || h < 0.5 * unit) {
      continue;
    }
    for (const piece of splitClumpsOfNoteheads(
      ops,
      chunk,
      noteheadMask,
      staff
    )) {
      const size = piece.notehead.rect;
      if (
        size.w < 0.5 * unit ||
        size.w > 3 * unit ||
        size.h < 0.5 * unit ||
        size.h > 2 * unit
      ) {
        continue;
      }
      notes.push(
        createNote(
          piece.notehead,
          findPositionInUnitSizes(point, piece.notehead),
          piece.stem
        )
      );
    }
  }
  return notes;
}

/**
 * add_notes_to_staffs, without the mutation. homr appends each note to its
 * staff's `symbols`; this returns the staffs again, same order, each carrying
 * its notes after whatever symbols it already had.
 *
 * Every staff of the result is a new object, so code that tells staffs apart
 * by identity must use the returned list throughout. `notes` is staff-major,
 * then notehead order, and a notehead inside two staffs' tolerance becomes a
 * note on both. Python's unused `symbols` mask argument is not taken.
 */
export function addNotesToStaffs(
  ops: BoxOps,
  staffs: readonly Staff[],
  noteheads: readonly NoteheadWithStem[],
  noteheadMask: Mask
): { readonly notes: Note[]; readonly staffs: Staff[] } {
  const perStaff = staffs.map((staff) =>
    notesOnStaff(ops, staff, noteheads, noteheadMask)
  );
  return {
    notes: perStaff.flat(),
    staffs: staffs.map((staff, i) =>
      withSymbols(staff, [...staff.symbols, ...(perStaff[i] ?? [])])
    ),
  };
}

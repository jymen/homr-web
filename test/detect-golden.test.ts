/**
 * The whole stage once, from the raw masks Python's segnet stage produced to
 * multistaffs.json, notes.json and staff-positions.txt. This one chains the
 * port's own boxes through every step, so it carries the wider tolerance; the
 * stage rows of the other golden tests are where a difference is located.
 *
 * Two things differ from Python here and nowhere else, and both are pinned by
 * name below rather than tolerated: the two noteheads opencv.js fits another
 * rectangle to, and the order of noteheads whose centres are one float32 ulp
 * apart in y.
 */

import { describe, expect, it, vi } from "vitest";
import { MatScope } from "../src/cv/opencv.js";
import { pointCount } from "../src/geometry/boxes.js";
import { BOX_RECT_TOLERANCE } from "../src/golden/box-tolerance.js";
import type { GoldenPage } from "../src/golden/page.js";
import {
  assertGoldenMatches,
  compareMultiStaffLists,
  compareNoteLists,
  STAFF_TOLERANCES,
} from "../src/golden/staff-tolerance.js";
import { median } from "../src/image/numeric.js";
import { createMask } from "../src/image/plane.js";
import {
  createInputPredictions,
  DetectionError,
  type InputPredictions,
  type PageDetection,
} from "../src/model/pipeline.js";
import {
  type MultiStaff,
  type Staff,
  withSymbols,
} from "../src/model/staff.js";
import type { Note } from "../src/model/symbols.js";
import { detectStaffsInImage } from "../src/pipeline/detect.js";
import { formatStaffPositions } from "../src/pipeline/staff-positions.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";

/** Raw masks, not the filtered ones: the noise filter and makeLinesStronger are inside the stage. */
const segnetOutput = (page: GoldenPage): InputPredictions =>
  createInputPredictions(page.resized(), page.preprocessed(), {
    clefsKeys: page.mask("clefsKeys"),
    notehead: page.mask("notehead"),
    staff: page.mask("staff"),
    stemsRest: page.mask("stemsRest"),
    symbols: page.mask("symbols"),
  });

const TOLERANCE = STAFF_TOLERANCES.fromOwnBoxes;

/**
 * Notes the port emits at another index than Python, as `[index in the port's
 * list, index in notes.json]`.
 *
 * combine_noteheads_with_stems sorts the noteheads by centre y. Noteheads on
 * one staff height have centres one or two float32 ulps apart, and the port's
 * centres differ from Python's by the same amount (BOX_RECT_TOLERANCE's
 * cause), so the sort puts such neighbours in another order. The set of notes
 * is the same. A pair listed here must still be out of order and no other
 * note may be, so a change in either direction fails.
 *
 * `[83, 84]` of the grand-staff page is the one the notehead-18 divergence
 * causes: its centre moves from 1499 to 1499.25, past a neighbour at
 * 1499.00012.
 */
const REORDERED: Record<string, readonly (readonly [number, number])[]> = {
  "chord-study-300dpi": [
    [12, 13],
    [13, 12],
    [32, 33],
    [33, 32],
    [46, 47],
    [47, 46],
  ],
  "grand-staff-300dpi": [
    [5, 6],
    [6, 5],
    [83, 84],
    [84, 83],
  ],
  "the-kesh-300dpi": [
    [4, 6],
    [5, 4],
    [6, 5],
    [8, 9],
    [9, 8],
    [15, 16],
    [16, 15],
    [24, 26],
    [25, 24],
    [26, 25],
    [75, 76],
    [76, 75],
  ],
};

/**
 * notes.json entries whose ellipse the port fits differently: noteheads 18
 * and 48 of boxes-noteheads.json, pinned in test/boxes-golden.test.ts, where
 * they are notes 83 and 53. Each must still differ in its box and in nothing
 * else.
 */
const REFITTED: Record<string, readonly number[]> = {
  "grand-staff-300dpi": [53, 83],
};

const sameContour = (a: Note, b: Note): boolean =>
  pointCount(a.box.contour) === pointCount(b.box.contour) &&
  a.box.contour.every((value, i) => value === b.box.contour[i]);

/**
 * The port's notes in Python's order, matched by contour, which is exact or
 * the note is not the same note. `moved` lists every note found at another
 * index.
 */
function alignNotes(
  got: readonly Note[],
  want: readonly Note[]
): { readonly aligned: Note[]; readonly moved: [number, number][] } {
  const aligned: (Note | undefined)[] = want.map(() => undefined);
  const moved: [number, number][] = [];
  for (const [i, note] of got.entries()) {
    const at = want.findIndex(
      (other, k) => aligned[k] === undefined && sameContour(note, other)
    );
    if (at < 0) {
      throw new Error(`notes[${i}] has a contour notes.json does not hold`);
    }
    aligned[at] = note;
    if (at !== i) {
      moved.push([i, at]);
    }
  }
  return { aligned: aligned.flatMap((note) => note ?? []), moved };
}

/**
 * Grid points the port adds when it detects staffs from its own staff
 * fragments: `[multi staff, staff, index]`, the index of a point that repeats
 * the x of the point before it. From Python's fragments detect_staff is exact
 * on these pages (test/staffs-golden.test.ts), so the cause is the fragment
 * polygons opencv.js fits within 1 px of opencv-python's
 * (docs/design/phase-5-minarearect.md). Each point must still be there.
 */
const REGRIDDED: Record<
  string,
  readonly (readonly [number, number, number])[]
> = {
  "chord-study-300dpi": [[1, 0, 176]],
};

function withoutRegridded(
  multiStaffs: readonly MultiStaff[],
  extra: readonly (readonly [number, number, number])[]
): MultiStaff[] {
  return multiStaffs.map((multiStaff, m) => {
    const [head, ...tail] = multiStaff.staffs.map((staff, s): Staff => {
      const drop = extra
        .filter(([em, es]) => em === m && es === s)
        .map(([, , i]) => i);
      if (drop.length === 0) {
        return staff;
      }
      for (const i of drop) {
        if (staff.grid[i]?.x !== staff.grid[i - 1]?.x) {
          throw new Error(
            `multistaffs[${m}].staffs[${s}].grid[${i}] no longer repeats the point before it: remove it from REGRIDDED`
          );
        }
      }
      const [first, ...rest] = staff.grid.filter((_, i) => !drop.includes(i));
      if (first === undefined) {
        throw new Error("a regridded staff lost every point");
      }
      return {
        ...staff,
        averageUnitSize: median([first, ...rest].map((p) => p.averageUnitSize)),
        grid: [first, ...rest],
      };
    });
    return head === undefined
      ? multiStaff
      : { connections: multiStaff.connections, staffs: [head, ...tail] };
  });
}

/** The multi staffs with `notes` dealt back out to the staffs, in order, by each staff's own count. */
function withNotes(
  multiStaffs: readonly MultiStaff[],
  notes: readonly Note[]
): MultiStaff[] {
  let next = 0;
  return multiStaffs.map((multiStaff) => {
    const [first, ...rest] = multiStaff.staffs.map((staff) => {
      const symbols = notes.slice(next, next + staff.symbols.length);
      next += staff.symbols.length;
      return withSymbols(staff, symbols);
    });
    return first === undefined
      ? multiStaff
      : { connections: multiStaff.connections, staffs: [first, ...rest] };
  });
}

/** detectStaffsInImage with the Mats of its largest scope counted. */
function measured(
  cv: Awaited<ReturnType<typeof testOpenCv>>,
  input: InputPredictions
): {
  readonly detection: PageDetection;
  readonly mats: number;
  readonly ms: number;
} {
  const perScope = new Map<MatScope, number>();
  const { keep } = MatScope.prototype;
  const spy = vi
    .spyOn(MatScope.prototype, "keep")
    .mockImplementation(function counted(this: MatScope, value) {
      perScope.set(this, (perScope.get(this) ?? 0) + 1);
      return keep.call(this, value);
    } as typeof keep);
  try {
    const started = performance.now();
    const detection = detectStaffsInImage(cv, input);
    const ms = performance.now() - started;
    return { detection, mats: Math.max(...perScope.values()), ms };
  } finally {
    spy.mockRestore();
  }
}

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);
  const reordered = REORDERED[fixture.name] ?? [];
  const refitted = REFITTED[fixture.name] ?? [];

  describe(`detect_staffs_in_image ${fixture.name}`, () => {
    it("reproduces multistaffs.json, notes.json and staff-positions.txt from the raw masks", async () => {
      const cv = await testOpenCv();
      const { detection, mats, ms } = measured(cv, segnetOutput(page));
      const want = page.notes();
      expect(detection.noise.kind).toBe(page.noise().outcome);

      const { aligned, moved } = alignNotes(detection.notes, want);
      expect(moved).toEqual(reordered);
      for (const [i, at] of moved) {
        expect(
          Math.abs((want[i]?.center.y ?? 0) - (want[at]?.center.y ?? 0)),
          `the port has notes.json[${at}] at ${i}, where Python's note is at another height`
        ).not.toBeGreaterThan(BOX_RECT_TOLERANCE.center);
      }

      for (const index of refitted) {
        const [mine, python] = [aligned[index], want[index]];
        if (mine === undefined || python === undefined) {
          throw new Error(`no note ${index}`);
        }
        const alone = compareNoteLists(`notes[${index}]`, [mine], [python]);
        expect(
          alone.boxes.failures.length,
          `notes[${index}] now has Python's ellipse: remove it from REFITTED and from KNOWN_DIVERGENCES`
        ).toBeGreaterThan(0);
        expect(alone.positionsExact).toBe(1);
        expect(alone.directionsExact).toBe(1);
        expect(alone.stems.failures).toEqual([]);
      }
      const notes = aligned.map((note, i) =>
        refitted.includes(i) ? (want[i] ?? note) : note
      );
      const noteReport = compareNoteLists("notes", notes, want);
      assertGoldenMatches(noteReport);

      const staffNotes = detection.multiStaffs.flatMap((multiStaff) =>
        multiStaff.staffs.flatMap((staff) => staff.symbols)
      );
      expect(staffNotes).toHaveLength(detection.notes.length);
      for (const [i, symbol] of staffNotes.entries()) {
        expect(symbol).toBe(detection.notes[i]);
      }
      const report = compareMultiStaffLists(
        "multistaffs",
        withoutRegridded(
          withNotes(detection.multiStaffs, notes),
          REGRIDDED[fixture.name] ?? []
        ),
        page.multiStaffs(),
        TOLERANCE
      );
      assertGoldenMatches(report);

      const positions = formatStaffPositions(
        detection.multiStaffs,
        detection.preprocessed
      )
        .split("\n")
        .map((line) => line.split(" "));
      const wanted = page
        .staffPositionsText()
        .split("\n")
        .map((line) => line.split(" "));
      expect(positions).toHaveLength(wanted.length);
      const bound = (2 * TOLERANCE.y) / detection.preprocessed.height;
      for (const [i, line] of wanted.entries()) {
        const mine = positions[i] ?? [];
        // The class, the centre x and the width come from grid x values, which are exact.
        expect([mine[0], mine[1], mine[3]], `line ${i}`).toEqual([
          line[0],
          line[1],
          line[3],
        ]);
        for (const column of [2, 4]) {
          expect(
            Math.abs(Number(mine[column]) - Number(line[column])),
            `line ${i} column ${column}`
          ).not.toBeGreaterThan(bound);
        }
      }

      process.stdout.write(
        `detectStaffsInImage ${fixture.name}: ${ms.toFixed(0)} ms, ${mats} Mats in the one scope, ${report.grouping.got.join(" ")} staffs per multi staff, ${detection.notes.length} notes of which ${moved.length} reordered and ${refitted.length} refitted, y ${Math.max(0, ...report.staffs.map((one) => one.worstY)).toExponential(1)}, angle ${Math.max(0, ...report.staffs.map((one) => one.worstAngle)).toExponential(1)}, note centre ${noteReport.boxes.worstCenter.toExponential(1)}\n`
      );
    });
  });
}

describe("a page detection refuses", () => {
  const [fixture] = listGoldenFixtures();
  if (fixture === undefined) {
    throw new Error("no golden fixture");
  }
  const input = segnetOutput(goldenPageOf(fixture));
  const { height, width } = input.preprocessed;
  const without = (name: "notehead" | "staff"): InputPredictions => ({
    ...input,
    masks: { ...input.masks, [name]: createMask(width, height) },
  });
  const failureOf = async (page: InputPredictions): Promise<unknown> => {
    const cv = await testOpenCv();
    try {
      return detectStaffsInImage(cv, page);
    } catch (error) {
      return error;
    }
  };

  it("has no noteheads, with homr's own message", async () => {
    const error = await failureOf(without("notehead"));
    expect(error).toBeInstanceOf(DetectionError);
    expect((error as DetectionError).code).toBe("no-noteheads");
    expect((error as DetectionError).message).toBe("No noteheads found");
  });

  it("has no staffs, with homr's own message", async () => {
    const error = await failureOf(without("staff"));
    expect(error).toBeInstanceOf(DetectionError);
    expect((error as DetectionError).code).toBe("no-staffs");
    expect((error as DetectionError).message).toBe("No staffs found");
  });
});

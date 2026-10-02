/**
 * Noteheads with their stems, the bar lines that are neither, and the notes on
 * each staff, against Python: each row from Python's value of the row before.
 */

import { describe, expect, it } from "vitest";
import { withBoxOps } from "../src/cv/box-ops.js";
import { barLineCandidates, detectBarLines } from "../src/geometry/barlines.js";
import { rotatedBoxFromRect } from "../src/geometry/box-transforms.js";
import {
  legacyConventionRectOf,
  pointListFromPairs,
} from "../src/geometry/boxes.js";
import {
  addNotesToStaffs,
  adjustBbox,
  averageNoteheadHeight,
  checkBboxSize,
  combineNoteheadsWithStems,
  type PixelBox,
  splitClumpsOfNoteheads,
} from "../src/geometry/notes.js";
import { compareBoxLists } from "../src/golden/box-tolerance.js";
import {
  decodeNoteheadSplits,
  decodeNoteheadsWithStems,
  decodeNotes,
  decodeStaffs,
} from "../src/golden/decode.js";
import {
  assertGoldenMatches,
  compareNoteheadLists,
  compareNoteLists,
} from "../src/golden/staff-tolerance.js";
import { isOnStaffZone, staffPointAt } from "../src/model/staff.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";
import { maskOfRows, vectorSet } from "./support/vectors.js";

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`noteheads and bar lines ${fixture.name}`, () => {
    it("reproduces combine_noteheads_with_stems", async () => {
      const cv = await testOpenCv();
      const want = page.noteheadsWithStems();
      const report = withBoxOps(cv, (ops) =>
        compareNoteheadLists(
          "noteheads",
          combineNoteheadsWithStems(
            ops,
            page.noteheads(),
            page.boxes("stemsRest")
          ),
          want
        )
      );
      process.stdout.write(
        `${fixture.name} noteheads: ${report.count.got}/${report.count.want}, ${report.stems.count.want} stemmed, directions ${report.directionsExact}\n`
      );
      expect(report.failures).toEqual([]);
      expect(report.directionsExact).toBe(want.length);
      expect(report.boxes.worstCenter + report.stems.worstCenter).toBe(0);
    });

    it("reproduces the median notehead height and detect_bar_lines", async () => {
      const cv = await testOpenCv();
      const want = page.barLines();
      const noteheads = page.noteheadsWithStems();
      const unit = averageNoteheadHeight(noteheads);
      expect(unit).toBe(want.averageNoteHeadHeight);
      const candidates = page.boxes("barLines");
      const got = withBoxOps(cv, (ops) =>
        detectBarLines(barLineCandidates(ops, candidates, noteheads), unit)
      );
      process.stdout.write(
        `${fixture.name} bar lines: ${got.length}/${want.barLines.length} of ${candidates.length} candidates\n`
      );
      expect(compareBoxLists("barLines", got, want.barLines).failures).toEqual(
        []
      );
      for (const line of got) {
        expect(candidates).toContain(line);
      }
    });

    it("reproduces add_notes_to_staffs without touching the staffs it was given", async () => {
      const cv = await testOpenCv();
      const given = page.staffs();
      const { notes, staffs } = withBoxOps(cv, (ops) =>
        addNotesToStaffs(
          ops,
          given,
          page.noteheadsWithStems(),
          page.mask("notehead", true)
        )
      );
      const report = compareNoteLists("notes", notes, page.notes());
      process.stdout.write(
        `${fixture.name} notes: ${report.count.got}/${report.count.want}, positions ${report.positionsExact}, per staff ${staffs.map((staff) => staff.symbols.length).join(" ")}\n`
      );
      assertGoldenMatches(report);
      expect(report.boxes.worstCenter + report.boxes.worstSize).toBe(0);
      expect(staffs.map((staff) => staff.symbols.length)).toEqual(
        page.braces().notesPerStaff
      );
      for (const [i, symbol] of staffs
        .flatMap((staff) => staff.symbols)
        .entries()) {
        expect(symbol).toBe(notes[i]);
      }
      expect(given.every((staff) => staff.symbols.length === 0)).toBe(true);
      expect(staffs.map((staff) => staff.grid)).toEqual(
        given.map((staff) => staff.grid)
      );
    });

    it("splits the noteheads notehead-splits.json lists, and no other", async () => {
      const cv = await testOpenCv();
      const want = page.noteheadSplits();
      const noteheads = page.noteheadsWithStems();
      const mask = page.mask("notehead", true);
      const got = withBoxOps(cv, (ops) =>
        page.staffs().flatMap((staff, staffIndex) =>
          noteheads.flatMap((chunk, notehead) => {
            const point = staffPointAt(staff, chunk.notehead.rect.cx);
            if (
              !isOnStaffZone(staff, chunk.notehead) ||
              point === null ||
              chunk.notehead.rect.w < 0.5 * point.averageUnitSize ||
              chunk.notehead.rect.h < 0.5 * point.averageUnitSize
            ) {
              return [];
            }
            const pieces = splitClumpsOfNoteheads(ops, chunk, mask, staff);
            if (pieces.length === 1) {
              expect(pieces[0]).toBe(chunk);
              return [];
            }
            return [{ notehead, pieces, staff: staffIndex }];
          })
        )
      );
      expect(got.map(({ notehead, staff }) => [staff, notehead])).toEqual(
        want.map(({ notehead, staff }) => [staff, notehead])
      );
      for (const [i, split] of got.entries()) {
        expect(
          compareBoxLists(
            `splits[${i}]`,
            split.pieces.map((piece) => piece.notehead),
            want[i]?.pieces ?? []
          ).failures
        ).toEqual([]);
      }
    });
  });
}

describe("bbox-split.json", () => {
  const { cases } = vectorSet("bbox-split");
  const boxOf = (value: unknown): PixelBox => {
    const [x0 = 0, y0 = 0, x1 = 0, y1 = 0] = value as number[];
    return { x0, x1, y0, y1 };
  };
  const tuple = (box: PixelBox): number[] => [box.x0, box.y0, box.x1, box.y1];

  it("holds a width split, a height split, a box the second pass drops and a box on row 0", () => {
    const names = cases.map((one) => String(one.name));
    for (const part of [
      "side by side",
      "stacked",
      "dropped by the second pass",
      "row 0",
    ]) {
      expect(names.some((name) => name.includes(part))).toBe(true);
    }
  });

  it("adjustBbox and checkBboxSize are homr's on every case", () => {
    for (const one of cases) {
      const mask = maskOfRows(one.mask, `${one.name}.mask`);
      const bbox = boxOf(one.bbox);
      expect(tuple(adjustBbox(bbox, mask)), `${one.name}: adjusted`).toEqual(
        one.adjusted
      );
      expect(
        checkBboxSize(bbox, mask, one.unitSize as number).map(tuple),
        `${one.name}: boxes`
      ).toEqual(one.boxes);
    }
  });
});

describe("detectBarLines", () => {
  const line = rotatedBoxFromRect(
    legacyConventionRectOf({ angle: 0, cx: 50, cy: 50, h: 30, w: 20 }),
    pointListFromPairs([]),
    0
  );

  it("keeps a box exactly three units tall and two wide", () => {
    expect(detectBarLines([line], 10)).toEqual([line]);
  });

  it("drops a box under three units tall, or over two wide", () => {
    expect(detectBarLines([line], 10.01)).toEqual([]);
    expect(detectBarLines([line], 9.99)).toEqual([]);
  });
});

describe("notehead-clumps.json", () => {
  const cases = vectorSet("notehead-clumps").cases.map((one) => ({
    mask: maskOfRows(one.mask, `${one.name}.mask`),
    name: String(one.name),
    noteheads: decodeNoteheadsWithStems(one.noteheads, `${one.name}.noteheads`),
    notes: decodeNotes(one.notes, `${one.name}.notes`),
    notesPerStaff: one.notesPerStaff,
    splits: decodeNoteheadSplits(one.splits, `${one.name}.splits`),
    staffs: decodeStaffs(one.staffs, `${one.name}.staffs`),
  }));

  it("holds splits in two and in four, a note on two staffs, and noteheads no staff takes", () => {
    const pieces = cases.map((one) => one.splits[0]?.pieces.length ?? 1);
    expect(pieces).toContain(2);
    expect(pieces).toContain(4);
    expect(cases.some((one) => one.staffs.length === 2)).toBe(true);
    expect(cases.some((one) => one.notes.length === 0)).toBe(true);
    expect(
      cases.some((one) => one.notes.length > 0 && one.splits.length === 0)
    ).toBe(true);
  });

  it("splitClumpsOfNoteheads builds homr's pieces, each with the clump's contour, id and stem", async () => {
    const cv = await testOpenCv();
    for (const one of cases) {
      for (const split of one.splits) {
        const chunk = one.noteheads[split.notehead];
        const staff = one.staffs[split.staff];
        if (chunk === undefined || staff === undefined) {
          throw new Error(`${one.name}: a split outside the case`);
        }
        const pieces = withBoxOps(cv, (ops) =>
          splitClumpsOfNoteheads(ops, chunk, one.mask, staff)
        );
        const report = compareBoxLists(
          one.name,
          pieces.map((piece) => piece.notehead),
          split.pieces
        );
        expect(report.failures, one.name).toEqual([]);
        expect(report.worstCenter + report.worstSize, one.name).toBe(0);
        expect(report.polygonsExact, one.name).toBe(split.pieces.length);
        expect(pieces.every((piece) => piece.stem === chunk.stem)).toBe(true);
      }
    }
  });

  it("addNotesToStaffs makes homr's notes, staff by staff", async () => {
    const cv = await testOpenCv();
    for (const one of cases) {
      const { notes, staffs } = withBoxOps(cv, (ops) =>
        addNotesToStaffs(ops, one.staffs, one.noteheads, one.mask)
      );
      const report = compareNoteLists(one.name, notes, one.notes);
      assertGoldenMatches(report);
      expect(report.boxes.worstCenter + report.boxes.worstSize, one.name).toBe(
        0
      );
      expect(report.boxes.polygonsExact, one.name).toBe(one.notes.length);
      expect(
        staffs.map((staff) => staff.symbols.length),
        one.name
      ).toEqual(one.notesPerStaff);
    }
  });
});

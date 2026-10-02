/**
 * The comparison the stage goldens will be judged by, judged first: Python's
 * own output must compare equal to itself, and a copy changed in one place
 * must fail with a line that names the place.
 */

import { describe, expect, it } from "vitest";
import {
  type Ellipse,
  ellipseFromParts,
  type RotatedRect,
} from "../src/geometry/boxes.js";
import { rawStaffFromParts } from "../src/geometry/raw-staffs.js";
import { createStaffAnchor } from "../src/geometry/staff-anchors.js";
import {
  assertGoldenMatches,
  compareAnchorLists,
  compareMultiStaffLists,
  compareNoteheadLists,
  compareNoteLists,
  compareRawStaffLists,
  compareStaffLists,
  describeStaffComparison,
  STAFF_TOLERANCES,
  StaffGoldenMismatch,
} from "../src/golden/staff-tolerance.js";
import {
  createMultiStaff,
  createStaff,
  createStaffPoint,
  type MultiStaff,
  type Staff,
  type StaffPoint,
  withSymbols,
} from "../src/model/staff.js";
import {
  createNote,
  type Note,
  type NoteheadWithStem,
} from "../src/model/symbols.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";

const { fromOwnBoxes, fromPythonInputs } = STAFF_TOLERANCES;

const replaced = <T>(items: readonly T[], index: number, item: T): T[] =>
  items.map((old, i) => (i === index ? item : old));

const withPoint = (
  staff: Staff,
  index: number,
  change: (point: StaffPoint) => StaffPoint
): Staff => {
  const point = staff.grid[index];
  if (point === undefined) {
    throw new Error(`the staff has no grid[${index}]`);
  }
  return createStaff(replaced(staff.grid, index, change(point)), {
    isGrandstaff: staff.isGrandstaff,
    space: staff.space,
    symbols: staff.symbols,
  });
};

const shiftedLine = (line: number, by: number) => (point: StaffPoint) =>
  createStaffPoint(
    point.x,
    point.y.map((y, i) => (i === line ? y + by : y)),
    point.angle
  );

const movedBox = (box: Ellipse, dx: number): Ellipse =>
  ellipseFromParts(
    { ...box.rect, cx: box.rect.cx + dx } as RotatedRect,
    box.polygon,
    box.contour,
    box.debugId
  );

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`staffs.json ${fixture.name}`, () => {
    const staffs = page.staffs();
    const changed = (change: (staff: Staff) => Staff): Staff[] => {
      const [, second] = staffs;
      if (second === undefined) {
        throw new Error("the fixture has fewer than two staffs");
      }
      return replaced(staffs, 1, change(second));
    };
    const failuresOf = (got: readonly Staff[], tolerance = fromPythonInputs) =>
      compareStaffLists("staffs", got, staffs, tolerance).failures;

    it("compares equal to itself, with nothing to report", () => {
      const report = compareStaffLists(
        "staffs",
        page.staffs(),
        staffs,
        fromPythonInputs
      );
      expect(report.failures).toEqual([]);
      expect(report).toMatchObject({
        count: { got: staffs.length, want: staffs.length },
        worstAngle: 0,
        worstUnit: 0,
        worstY: 0,
        xMismatches: 0,
      });
      expect(describeStaffComparison(report)).toContain(
        `staffs: ${staffs.length}/${staffs.length}`
      );
      expect(() => assertGoldenMatches(report)).not.toThrow();
    });

    it("names the line ordinate that moved", () => {
      const moved = changed((staff) =>
        withPoint(staff, 7, shiftedLine(2, 1e-6))
      );
      const failures = failuresOf(moved);
      expect(failures.join("\n")).toContain("staffs[1].grid[7].y[2]");
      expect(() =>
        assertGoldenMatches(
          compareStaffLists("staffs", moved, staffs, fromPythonInputs)
        )
      ).toThrow(StaffGoldenMismatch);
    });

    it("accepts a move inside the bound and reports its size", () => {
      const moved = changed((staff) =>
        withPoint(staff, 7, shiftedLine(2, 1e-4))
      );
      const report = compareStaffLists("staffs", moved, staffs, fromOwnBoxes);
      expect(report.failures).toEqual([]);
      expect(report.worstY).toBeGreaterThan(9e-5);
      expect(report.worstY).toBeLessThan(1.1e-4);
    });

    it("names the grid point whose x or angle changed, and counts a shifted grid once", () => {
      const angled = changed((staff) =>
        withPoint(staff, 3, (point) =>
          createStaffPoint(point.x, point.y, point.angle + 0.5)
        )
      );
      expect(failuresOf(angled).join("\n")).toContain(
        "staffs[1].grid[3].angle"
      );
      const shifted = changed((staff) =>
        createStaff(
          staff.grid.map((point) =>
            createStaffPoint(point.x + 10, point.y, point.angle)
          )
        )
      );
      const report = compareStaffLists("staffs", shifted, staffs, fromOwnBoxes);
      expect(report.xMismatches).toBe(shifted[1]?.grid.length);
      const aboutGrid = report.failures.filter((line) =>
        line.includes(".grid")
      );
      expect(aboutGrid).toHaveLength(1);
      expect(aboutGrid[0]).toContain("staffs[1].grid[0]: x ");
    });

    it("refuses a missing staff, a missing point and a staff that became a grand staff", () => {
      expect(failuresOf(staffs.slice(1))).toEqual([
        `staffs: ${staffs.length - 1} staffs, Python found ${staffs.length}`,
      ]);
      const shorter = changed((staff) => createStaff(staff.grid.slice(1)));
      expect(failuresOf(shorter).join("\n")).toContain("staffs[1].grid");
      const grand = changed((staff) =>
        createStaff(staff.grid, { isGrandstaff: true })
      );
      expect(failuresOf(grand).join("\n")).toContain("staffs[1].isGrandstaff");
    });
  });

  describe(`notes.json ${fixture.name}`, () => {
    const notes = page.notes();
    const stemmed = notes.findIndex((note) => note.stem !== null);
    const changed = (index: number, change: (note: Note) => Note): Note[] => {
      const note = notes[index];
      if (note === undefined) {
        throw new Error(`the fixture has no notes[${index}]`);
      }
      return replaced(notes, index, change(note));
    };
    const failuresOf = (got: readonly Note[]) =>
      compareNoteLists("notes", got, notes).failures.join("\n");

    it("compares equal to itself", () => {
      const report = compareNoteLists("notes", page.notes(), notes);
      expect(report.failures).toEqual([]);
      expect(report.positionsExact).toBe(notes.length);
      expect(report.directionsExact).toBe(notes.length);
      expect(report.stems.count.want).toBe(
        notes.filter((note) => note.stem !== null).length
      );
    });

    it("names the note whose position, stem, direction or ellipse changed", () => {
      expect(
        failuresOf(
          changed(5, (note) =>
            createNote(note.box, note.position + 1, note.stem)
          )
        )
      ).toContain("notes[5].position");
      expect(
        failuresOf(
          changed(stemmed, (note) => createNote(note.box, note.position, null))
        )
      ).toContain(`notes[${stemmed}].stem`);
      expect(
        failuresOf(
          changed(stemmed, (note) =>
            createNote(
              note.box,
              note.position,
              note.stem && {
                box: note.stem.box,
                direction: note.stem.direction === "UP" ? "DOWN" : "UP",
              }
            )
          )
        )
      ).toContain(`notes[${stemmed}].stem.direction`);
      expect(
        failuresOf(
          changed(5, (note) =>
            createNote(movedBox(note.box, 1), note.position, note.stem)
          )
        )
      ).toContain("notes[5].box");
      expect(failuresOf(notes.slice(0, -1))).toContain(
        `notes: ${notes.length - 1} notes, Python found ${notes.length}`
      );
    });

    it("compares noteheads-with-stems.json equal to itself and names a lost stem", () => {
      const noteheads = page.noteheadsWithStems();
      const report = compareNoteheadLists(
        "noteheads",
        page.noteheadsWithStems(),
        noteheads
      );
      expect(report.failures).toEqual([]);
      const index = noteheads.findIndex((one) => one.stem !== null);
      const lost: NoteheadWithStem[] = replaced(noteheads, index, {
        notehead: (noteheads[index] as NoteheadWithStem).notehead,
        stem: null,
      });
      expect(
        compareNoteheadLists("noteheads", lost, noteheads).failures.join("\n")
      ).toContain(`noteheads[${index}].stem`);
    });
  });

  describe(`multistaffs.json ${fixture.name}`, () => {
    const multiStaffs = page.multiStaffs();
    const failuresOf = (got: readonly MultiStaff[]) =>
      compareMultiStaffLists(
        "multistaffs",
        got,
        multiStaffs,
        fromPythonInputs
      ).failures.join("\n");
    const [first] = multiStaffs;
    if (first === undefined) {
      throw new Error("the fixture has no multi staff");
    }

    it("compares equal to itself", () => {
      const report = compareMultiStaffLists(
        "multistaffs",
        page.multiStaffs(),
        multiStaffs,
        fromPythonInputs
      );
      expect(report.failures).toEqual([]);
      expect(report.grouping.got).toEqual(
        multiStaffs.map((multi) => multi.staffs.length)
      );
      expect(report.notes.flat()).toHaveLength(
        multiStaffs.flatMap((multi) => multi.staffs).length
      );
    });

    it("names the staff, the note and the grouping that changed", () => {
      const moved = replaced(
        multiStaffs,
        0,
        createMultiStaff(
          replaced(
            first.staffs,
            0,
            withPoint(first.staffs[0], 3, shiftedLine(0, 1e-6))
          ),
          first.connections
        )
      );
      expect(failuresOf(moved)).toContain(
        "multistaffs[0].staffs[0].grid[3].y[0]"
      );
      const bare = replaced(
        multiStaffs,
        0,
        createMultiStaff(
          replaced(first.staffs, 0, withSymbols(first.staffs[0], [])),
          first.connections
        )
      );
      expect(failuresOf(bare)).toContain("multistaffs[0].staffs[0].symbols");
      const regrouped = [
        createMultiStaff([...first.staffs, first.staffs[0]], first.connections),
        ...multiStaffs.slice(1),
      ];
      expect(failuresOf(regrouped)).toContain("multistaffs[0]: 2 staffs");
      expect(failuresOf(multiStaffs.slice(1))).toContain(
        `multistaffs: ${multiStaffs.length - 1} multi staffs, Python found ${multiStaffs.length}`
      );
    });
  });
}

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`staff-anchors.json ${fixture.name}`, () => {
    const { kept } = page.staffAnchors();
    const [first, second] = kept;
    if (first === undefined || second === undefined) {
      throw new Error("the fixture has at least two anchors");
    }

    it("compares equal to itself", () => {
      const report = compareAnchorLists(
        "anchors",
        kept,
        kept,
        fromPythonInputs
      );
      expect(report.failures).toEqual([]);
      expect([report.linesExact, report.zonesExact]).toEqual([
        kept.length,
        kept.length,
      ]);
      expect(report.symbols.count.want).toBe(kept.length);
    });

    it("names an anchor found at another symbol, over other lines or one fewer", () => {
      const otherSymbol = createStaffAnchor(first.lines, second.symbol);
      expect(
        compareAnchorLists(
          "anchors",
          replaced(kept, 0, otherSymbol),
          kept,
          fromPythonInputs
        ).failures.some((line) => line.startsWith("anchors[0].symbol"))
      ).toBe(true);
      const [top, ...rest] = first.lines;
      const reordered = createStaffAnchor(
        [rest[0], top, rest[1], rest[2], rest[3]],
        first.symbol
      );
      expect(
        compareAnchorLists(
          "anchors",
          replaced(kept, 0, reordered),
          kept,
          fromPythonInputs
        ).failures
      ).toContain("anchors[0].lines: not the fragments Python connected");
      expect(
        compareAnchorLists("anchors", kept.slice(1), kept, fromPythonInputs)
          .failures
      ).toEqual([
        `anchors: ${kept.length - 1} anchors, Python found ${kept.length}`,
      ]);
    });

    it("names a zone or a unit size that moved", () => {
      const moved = {
        ...first,
        averageUnitSize: first.averageUnitSize + 1e-6,
        zone: { start: first.zone.start, stop: first.zone.stop + 1 },
      };
      const { failures } = compareAnchorLists(
        "anchors",
        replaced(kept, 0, moved),
        kept,
        fromPythonInputs
      );
      expect(failures.some((line) => line.startsWith("anchors[0].zone"))).toBe(
        true
      );
      expect(
        failures.some((line) => line.startsWith("anchors[0].averageUnitSize"))
      ).toBe(true);
    });
  });

  describe(`raw-staffs.json ${fixture.name}`, () => {
    const { connected } = page.rawStaffs();
    const [first, second] = connected;
    if (first === undefined || second === undefined) {
      throw new Error("the fixture has at least two raw staffs");
    }

    it("compares equal to itself", () => {
      const report = compareRawStaffLists("raw", connected, connected);
      expect(report.failures).toEqual([]);
      expect([report.linesExact, report.anchorsExact]).toEqual([
        connected.length,
        connected.length,
      ]);
      expect(report.boxes.worstCenter).toBe(0);
    });

    it("names a staff in another place, with other lines or with its anchors in another order", () => {
      expect(
        compareRawStaffLists(
          "raw",
          [second, first, ...connected.slice(2)],
          connected
        ).failures.length
      ).toBeGreaterThan(0);
      const [head, ...tail] = first.anchors;
      const lastAnchor = tail.pop();
      if (lastAnchor === undefined) {
        throw new Error("the first raw staff has at least two anchors");
      }
      const reordered = rawStaffFromParts(first.box, first.lines, [
        lastAnchor,
        ...tail,
        head,
      ]);
      expect(
        compareRawStaffLists(
          "raw",
          replaced(connected, 0, reordered),
          connected
        ).failures
      ).toEqual([
        `raw[0].anchors: ${first.anchors.length}, not Python's ${first.anchors.length} in Python's order`,
      ]);
      const otherLines = rawStaffFromParts(
        first.box,
        second.lines,
        first.anchors
      );
      expect(
        compareRawStaffLists(
          "raw",
          replaced(connected, 0, otherLines),
          connected
        ).failures
      ).toContain("raw[0].lines: not the fragments Python's staff holds");
    });
  });
}

/** save_staff_positions against the bytes homr wrote, from Python's multi staffs. */

import { describe, expect, it } from "vitest";
import { createStaff, createStaffPoint } from "../src/model/staff.js";
import {
  formatStaffPositions,
  staffBoxes,
  staffPositions,
} from "../src/pipeline/staff-positions.js";
import {
  goldenPageOf,
  listGoldenFixtures,
  readerFor,
} from "./support/golden.js";

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`save_staff_positions ${fixture.name}`, () => {
    it("writes staff-positions.txt byte for byte", () => {
      const want = page.staffPositionsText();
      expect(want.endsWith("\n")).toBe(true);
      expect(
        formatStaffPositions(page.multiStaffs(), page.preprocessed())
      ).toBe(want);
    });

    it("gives the app server's staves.json, parsed by its own Go code", () => {
      const want: unknown = JSON.parse(readerFor(fixture).text("staves.json"));
      expect(
        staffBoxes(staffPositions(page.multiStaffs(), page.preprocessed()))
      ).toEqual(want);
    });
  });
}

describe("staffBoxes", () => {
  it("sorts by cy stably and numbers after the sort", () => {
    const at = (cy: number, cx: number) => ({
      cx,
      cy,
      h: 0.1,
      isGrandstaff: false,
      w: 0.5,
    });
    expect(
      staffBoxes([at(0.5, 1), at(0.2, 2), at(0.5, 3)]).map((b) => [
        b.index,
        b.cx,
      ])
    ).toEqual([
      [0, 2],
      [1, 1],
      [2, 3],
    ]);
  });
});

describe("formatStaffPositions", () => {
  it("writes nothing for no staff", () => {
    expect(formatStaffPositions([], { height: 10, width: 10 })).toBe("");
  });

  it("writes Python's str(float): a whole number keeps its .0 and a small one its two-digit exponent", () => {
    const staff = createStaff(
      [0, 10].map((x) => createStaffPoint(x, [0, 0.01, 0.02, 0.03, 0.04], 0))
    );
    expect(
      formatStaffPositions([{ connections: [], staffs: [staff] }], {
        height: 1000,
        width: 10,
      })
    ).toBe("0 0.5 2e-05 1.0 4e-05\n");
  });

  it("takes the centre as `y1 + height / 2`, where `(y1 + y2) / 2` prints another number", () => {
    const staff = createStaff(
      [0, 10].map((x) => createStaffPoint(x, [0.1, 0.25, 0.4, 0.55, 0.7], 0))
    );
    expect((0.1 + 0.7) / 2 / 1000).not.toBe((0.1 + (0.7 - 0.1) / 2) / 1000);
    expect(
      formatStaffPositions([{ connections: [], staffs: [staff] }], {
        height: 1000,
        width: 10,
      })
    ).toBe("0 0.5 0.0004 1.0 0.0006\n");
  });
});

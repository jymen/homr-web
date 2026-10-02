/**
 * The Staff, StaffPoint and MultiStaff methods of model.py as free functions:
 * against Python's own merges, against every note Python placed on the public
 * fixtures, and on hand-built grids for the ordering rules no page isolates.
 */

import { describe, expect, it } from "vitest";
import {
  createAxisBox,
  ellipseFromParts,
  legacyConventionRectOf,
  normalizeRotatedRect,
  pointListFromPairs,
  type RotatedBox,
  rotatedBoxFromParts,
} from "../src/geometry/boxes.js";
import { decodeStaff } from "../src/golden/decode.js";
import { DetectionError } from "../src/model/pipeline.js";
import {
  createMultiStaff,
  createStaff,
  createStaffPoint,
  findPositionInUnitSizes,
  isOnStaffZone,
  mergeMultiStaffs,
  mergeStaffPoints,
  mergeStaffs,
  StaffError,
  staffPointAt,
  staffPointToAxisBox,
  withSymbols,
  yDistanceTo,
} from "../src/model/staff.js";
import { createNote, symbolsOfKind } from "../src/model/symbols.js";
import {
  goldenPageOf,
  listGoldenFixtures,
  readerFor,
} from "./support/golden.js";
import { type VectorCase, vectorSet } from "./support/vectors.js";

interface StoredPoint {
  readonly angle: number;
  readonly average_unit_size: number;
  readonly x: number;
  readonly y: readonly number[];
}

interface StoredStaff {
  readonly average_unit_size: number;
  readonly grid: readonly StoredPoint[];
  readonly max_x: number;
  readonly max_y: number;
  readonly min_x: number;
  readonly min_y: number;
}

const FIVE = [100, 110, 120, 130, 140];
const flatStaff = (xs: readonly number[], top = 100) =>
  createStaff(
    xs.map((x) =>
      createStaffPoint(
        x,
        FIVE.map((y) => y - 100 + top),
        0
      )
    )
  );

const noContour = pointListFromPairs([]);
const ellipseAt = (cx: number, cy: number) =>
  ellipseFromParts(
    normalizeRotatedRect(
      legacyConventionRectOf({ angle: 0, cx, cy, h: 8, w: 10 })
    ),
    noContour,
    noContour,
    0
  );
const connectionAt = (cx: number, debugId: number): RotatedBox =>
  rotatedBoxFromParts(
    normalizeRotatedRect(
      legacyConventionRectOf({ angle: 0, cx, cy: 200, h: 120, w: 6 })
    ),
    noContour,
    noContour,
    debugId
  );

describe("staff-merge.json", () => {
  const { cases } = vectorSet("staff-merge");
  const staffOf = (one: VectorCase, key: "a" | "b") =>
    decodeStaff(one[key], {}, `staff-merge.${String(one.name)}.${key}`);

  for (const one of cases.filter((entry) => "merged" in entry)) {
    it(`mergeStaffs is Staff.merge, bit for bit: ${String(one.name)}`, () => {
      const stored = one.merged as StoredStaff;
      const merged = mergeStaffs(staffOf(one, "a"), staffOf(one, "b"));
      expect(merged.isGrandstaff).toBe(true);
      expect(
        merged.grid.map((point) => ({
          angle: point.angle,
          average_unit_size: point.averageUnitSize,
          x: point.x,
          y: [...point.y],
        }))
      ).toEqual(
        stored.grid.map(({ angle, average_unit_size, x, y }) => ({
          angle,
          average_unit_size,
          x,
          y,
        }))
      );
      expect({
        average_unit_size: merged.averageUnitSize,
        max_x: merged.maxX,
        max_y: merged.maxY,
        min_x: merged.minX,
        min_y: merged.minY,
      }).toEqual({
        average_unit_size: stored.average_unit_size,
        max_x: stored.max_x,
        max_y: stored.max_y,
        min_x: stored.min_x,
        min_y: stored.min_y,
      });
    });
  }

  it("throws staff-without-points where Python's Staff([]) raises IndexError", () => {
    const failing = cases.filter((entry) => entry.error === "IndexError");
    expect(failing).toHaveLength(1);
    for (const one of failing) {
      const merge = () => mergeStaffs(staffOf(one, "a"), staffOf(one, "b"));
      expect(merge).toThrow(DetectionError);
      expect(merge).toThrow("A staff has no position with all five lines");
    }
  });

  it("throws a StaffError where Python's StaffPoint.merge raises ValueError", () => {
    const failing = cases.filter((entry) => entry.error === "ValueError");
    expect(failing).toHaveLength(1);
    for (const one of failing) {
      expect(() => mergeStaffs(staffOf(one, "a"), staffOf(one, "b"))).toThrow(
        StaffError
      );
    }
  });

  it("keys a grid point by Python's round: 100.5 and 101.4 share no key, 102.5 and 101.6 share 102", () => {
    expect(() =>
      mergeStaffs(flatStaff([100.5]), flatStaff([101.4], 300))
    ).toThrow(DetectionError);
    expect(() =>
      mergeStaffs(flatStaff([102.5]), flatStaff([101.6], 300))
    ).toThrow(StaffError);
  });

  it("carries the symbols of both staffs, self first", () => {
    const [first] = cases;
    if (first === undefined) {
      throw new Error("staff-merge.json is empty");
    }
    const upper = createNote(ellipseAt(100, 210), 3, null);
    const lower = createNote(ellipseAt(110, 340), 5, null);
    const merged = mergeStaffs(
      withSymbols(staffOf(first, "a"), [upper]),
      withSymbols(staffOf(first, "b"), [lower])
    );
    expect(merged.symbols).toEqual([upper, lower]);
    expect(merged.symbols[0]).toBe(upper);
  });
});

describe("mergeStaffPoints", () => {
  it("sorts the ten ordinates and averages the two angles", () => {
    const merged = mergeStaffPoints(
      createStaffPoint(50, [300, 310, 320, 330, 340], 1),
      createStaffPoint(50.0005, FIVE, -0.5)
    );
    expect(merged.x).toBe(50);
    expect([...merged.y]).toEqual([...FIVE, 300, 310, 320, 330, 340]);
    expect(merged.angle).toBe(0.25);
  });
});

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`staff model ${fixture.name}`, () => {
    it("createStaffPoint reproduces Python's stored average_unit_size on every point, bit for bit", () => {
      const stored: StoredPoint[] = [];
      const collect = (value: unknown): void => {
        if (Array.isArray(value)) {
          value.forEach(collect);
        } else if (typeof value === "object" && value !== null) {
          const object = value as Record<string, unknown>;
          if (object.__class__ === "StaffPoint") {
            stored.push(object as unknown as StoredPoint);
          }
          Object.values(object).forEach(collect);
        }
      };
      const reader = readerFor(fixture);
      collect(JSON.parse(reader.text("staffs.json")));
      collect(JSON.parse(reader.text("multistaffs.json")));
      const wrong = stored.filter(
        (point) =>
          createStaffPoint(point.x, point.y, point.angle).averageUnitSize !==
          point.average_unit_size
      );
      expect(stored.length).toBeGreaterThan(0);
      expect(wrong).toEqual([]);
    });

    it("staffPointAt returns the first grid point at each x, in grid order", () => {
      for (const staff of page.staffs()) {
        for (const point of staff.grid) {
          expect(staffPointAt(staff, point.x)).toBe(
            staff.grid.find((other) => other.x === point.x)
          );
        }
        expect(staffPointAt(staff, staff.grid[0].x - 50)).not.toBeNull();
        expect(
          staffPointAt(staff, Math.min(...staff.grid.map((p) => p.x)) - 50.5)
        ).toBeNull();
      }
    });

    it("places every note where Python did: on its staff's zone, at its position", () => {
      const wrong: string[] = [];
      let notes = 0;
      for (const [s, staff] of page.staffsWithNotes().entries()) {
        for (const [n, note] of symbolsOfKind(
          staff.symbols,
          "note"
        ).entries()) {
          notes += 1;
          const point = staffPointAt(staff, note.box.rect.cx);
          if (point === null || !isOnStaffZone(staff, note.box)) {
            wrong.push(`staffs[${s}].symbols[${n}]: not on the staff zone`);
            continue;
          }
          const position = findPositionInUnitSizes(point, note.box);
          if (position !== note.position) {
            wrong.push(
              `staffs[${s}].symbols[${n}]: position ${position}, Python ${note.position}`
            );
          }
          if (yDistanceTo(staff, note.center) > 4 * staff.averageUnitSize) {
            wrong.push(`staffs[${s}].symbols[${n}]: further than the zone`);
          }
        }
      }
      expect(notes).toBe(page.notes().length);
      expect(wrong).toEqual([]);
    });
  });
}

describe("staffPointAt", () => {
  const staff = flatStaff([100, 110, 110, 105, 120]);

  it("takes the first minimum over an unsorted grid with duplicate x", () => {
    expect(staffPointAt(staff, 110)).toBe(staff.grid[1]);
    expect(staffPointAt(staff, 107.5)).toBe(staff.grid[1]);
    expect(staffPointAt(staff, 102.5)).toBe(staff.grid[0]);
  });

  it("answers null only beyond 50 px of the nearest point", () => {
    expect(staffPointAt(staff, 170)).toBe(staff.grid[4]);
    expect(staffPointAt(staff, 170.001)).toBeNull();
    expect(staffPointAt(staff, 50)).toBe(staff.grid[0]);
    expect(staffPointAt(staff, 49.999)).toBeNull();
  });
});

describe("yDistanceTo and isOnStaffZone", () => {
  const staff = flatStaff([100, 110]);

  it("measures to the nearest line, and 1e10 off the end of the staff", () => {
    expect(yDistanceTo(staff, { x: 104, y: 123 })).toBe(3);
    expect(yDistanceTo(staff, { x: 104, y: 90 })).toBe(10);
    expect(yDistanceTo(staff, { x: 400, y: 120 })).toBe(1e10);
  });

  it("accepts a centre exactly four units outside the outer lines and nothing further", () => {
    expect(isOnStaffZone(staff, ellipseAt(105, 60))).toBe(true);
    expect(isOnStaffZone(staff, ellipseAt(105, 59.999))).toBe(false);
    expect(isOnStaffZone(staff, ellipseAt(105, 180))).toBe(true);
    expect(isOnStaffZone(staff, ellipseAt(105, 180.001))).toBe(false);
    expect(isOnStaffZone(staff, ellipseAt(400, 120))).toBe(false);
  });
});

describe("findPositionInUnitSizes", () => {
  const [point] = flatStaff([100]).grid;

  it.each([
    [140, 1],
    [135, 2],
    [130, 3],
    [100, 9],
    [150, -1],
    [90, 11],
    [137.5, 1],
    [132.5, 3],
    [142.5, 1],
    [147.5, -1],
  ])(
    "a centre at y %s is position %s, halves rounding to even",
    (cy, position) => {
      expect(findPositionInUnitSizes(point, ellipseAt(100, cy))).toBe(position);
    }
  );
});

describe("staffPointToAxisBox", () => {
  it("is a zero-width box from the first line to the last, truncated toward zero", () => {
    const point = createStaffPoint(-12.7, [-3.9, 6.2, 16.2, 26.2, 36.8], 0);
    expect(staffPointToAxisBox(point)).toEqual(
      createAxisBox(-12, -3, -12, 36, pointListFromPairs([]), -2)
    );
  });
});

describe("mergeMultiStaffs", () => {
  const upper = flatStaff([100, 110], 100);
  const middle = flatStaff([100, 110], 300);
  const lower = flatStaff([100, 110], 500);

  it("keeps each staff once by identity and sorts by minY", () => {
    const twin = flatStaff([100, 110], 300);
    const merged = mergeMultiStaffs(
      createMultiStaff([middle, lower]),
      createMultiStaff([upper, middle, twin])
    );
    expect(merged.staffs).toHaveLength(4);
    expect(merged.staffs.map((staff) => staff.minY)).toEqual([
      100, 300, 300, 500,
    ]);
    expect(merged.staffs[1]).toBe(middle);
    expect(merged.staffs[2]).toBe(twin);
  });

  it("keeps each connection once by the value of its rect, first seen first", () => {
    const first = connectionAt(90, 1);
    const sameRect = connectionAt(90, 2);
    const other = connectionAt(95, 3);
    const merged = mergeMultiStaffs(
      createMultiStaff([upper, middle], [first, other]),
      createMultiStaff([middle, lower], [sameRect, other])
    );
    expect(merged.connections).toEqual([first, other]);
    expect(merged.connections[0]).toBe(first);
  });
});

describe("withSymbols", () => {
  it("returns a second staff and leaves the first without symbols", () => {
    const staff = flatStaff([100, 110]);
    const note = createNote(ellipseAt(100, 120), 5, null);
    const filled = withSymbols(staff, [note]);
    expect(filled).not.toBe(staff);
    expect(staff.symbols).toHaveLength(0);
    expect(filled.symbols).toEqual([note]);
    expect(filled.grid).toBe(staff.grid);
    expect(filled.averageUnitSize).toBe(staff.averageUnitSize);
  });
});

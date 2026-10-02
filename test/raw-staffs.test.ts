/**
 * The ordering rules of the raw-staff stage that neither public page reaches:
 * remove_duplicate_staffs' three branches (both pages keep every staff), the
 * line-index rule of get_staff_for_anchor, and where a merged staff lands. The
 * rules are about order and counts, not geometry, so the overlap answers are
 * given rather than computed and no opencv.js is loaded.
 */

import { describe, expect, it } from "vitest";
import type { BoxOps } from "../src/geometry/box-ops.js";
import { rotatedBoxFromRect } from "../src/geometry/box-transforms.js";
import {
  legacyConventionRectOf,
  pointListFromPairs,
  type RotatedBox,
} from "../src/geometry/boxes.js";
import {
  findRawStaffsByConnectingLineFragments,
  mergeRawStaffs,
  type RawStaff,
  rawStaffFromParts,
  removeDuplicateStaffs,
  staffForAnchor,
} from "../src/geometry/raw-staffs.js";
import {
  createStaffAnchor,
  type StaffAnchor,
} from "../src/geometry/staff-anchors.js";
import {
  asFiveLines,
  createStaffLineSegment,
  type FiveLines,
} from "../src/geometry/staff-lines.js";
import { vectorSet } from "./support/vectors.js";

const UNIT = 18;

const boxAt = (cx: number, cy: number, w: number, h: number): RotatedBox =>
  rotatedBoxFromRect(
    legacyConventionRectOf({ angle: 0, cx, cy, h, w }),
    pointListFromPairs([[cx, cy]]),
    0
  );

/** One fragment per line, 100 px wide, the top line at `top`. */
const fragmentsAt = (cx: number, top: number): RotatedBox[] =>
  [0, 1, 2, 3, 4].map((line) => boxAt(cx, top + line * UNIT, 100, 3));

function fiveLines(rows: readonly (readonly RotatedBox[])[]): FiveLines {
  const lines = asFiveLines(rows.map(createStaffLineSegment));
  if (lines === null) {
    throw new Error("a test staff has five lines");
  }
  return lines;
}

const anchorOver = (fragments: readonly RotatedBox[], x: number): StaffAnchor =>
  createStaffAnchor(
    fiveLines(fragments.map((fragment) => [fragment])),
    boxAt(x, (fragments[2]?.rect.cy ?? 0) + 0, 4, 4 * UNIT)
  );

/** A staff whose box is told apart by `id` alone. */
function staffOf(id: number, anchorCount: number): RawStaff {
  const fragments = fragmentsAt(100, 100 + id * 200);
  const [first, ...rest] = Array.from({ length: anchorCount }, (_, k) =>
    anchorOver(fragments, 60 + k * 10)
  );
  if (first === undefined) {
    throw new Error("a test staff has an anchor");
  }
  return rawStaffFromParts(
    boxAt(100, 136 + id * 200, 100, 80),
    fiveLines(fragments.map((fragment) => [fragment])),
    [first, ...rest]
  );
}

const refuse = (): never => {
  throw new Error("this rule must not ask for that");
};

/** A BoxOps that answers `overlaps` from a list of staff pairs and refuses everything else. */
const opsWhereOverlap = (
  pairs: ReadonlyArray<readonly [RawStaff, RawStaff]>
): BoxOps => ({
  ellipseFromRect: refuse,
  fitRotatedBox: refuse,
  intersects: refuse,
  overlaps: (a, b) =>
    pairs.some(
      ([x, y]) => (x.box === a && y.box === b) || (x.box === b && y.box === a)
    ),
  thicker: refuse,
});

/** A BoxOps whose only answer is a fitted box that carries the staff id as its centre, so two staffs never compare equal. */
const opsThatFit: BoxOps = {
  ellipseFromRect: refuse,
  fitRotatedBox: (contour, debugId) => ({
    ...boxAt(debugId, 0, 10, 10),
    contour,
    debugId,
  }),
  intersects: refuse,
  overlaps: refuse,
  thicker: refuse,
};

describe("removeDuplicateStaffs", () => {
  const few = staffOf(0, 1);
  const some = staffOf(1, 2);
  const many = staffOf(2, 3);

  it("keeps every staff when none overlaps", () => {
    expect(
      removeDuplicateStaffs(opsWhereOverlap([]), [few, some, many])
    ).toEqual([few, some, many]);
  });

  it("replaces a kept staff by one with more anchors, at the end", () => {
    const ops = opsWhereOverlap([[few, many]]);
    expect(removeDuplicateStaffs(ops, [few, some, many])).toEqual([some, many]);
  });

  it("keeps the first of two that overlap when the later has no more anchors", () => {
    const other = staffOf(3, 2);
    expect(
      removeDuplicateStaffs(opsWhereOverlap([[some, other]]), [some, other])
    ).toEqual([some]);
    expect(
      removeDuplicateStaffs(opsWhereOverlap([[many, few]]), [many, few])
    ).toEqual([many]);
  });

  it("drops a staff overlapping two kept ones, whatever its anchors", () => {
    const ops = opsWhereOverlap([
      [few, many],
      [some, many],
    ]);
    expect(removeDuplicateStaffs(ops, [few, some, many])).toEqual([few, some]);
  });
});

describe("staffForAnchor", () => {
  const fragments = fragmentsAt(100, 100);
  const staff = rawStaffFromParts(
    boxAt(100, 136, 100, 80),
    fiveLines(fragments.map((fragment) => [fragment])),
    [anchorOver(fragments, 100)]
  );

  it("matches on one line at its own index", () => {
    const elsewhere = fragmentsAt(100, 500);
    const sharingLineTwo = elsewhere.map((fragment, i) =>
      i === 2 ? (fragments[2] ?? fragment) : fragment
    );
    expect(staffForAnchor(anchorOver(sharingLineTwo, 100), [staff])).toBe(
      staff
    );
  });

  it("does not match a shared fragment at another line index", () => {
    const elsewhere = fragmentsAt(100, 500);
    const lineTwoAsLineThree = elsewhere.map((fragment, i) =>
      i === 3 ? (fragments[2] ?? fragment) : fragment
    );
    expect(
      staffForAnchor(anchorOver(lineTwoAsLineThree, 100), [staff])
    ).toBeNull();
  });

  it("returns the first of two staffs that match", () => {
    const twin = rawStaffFromParts(boxAt(100, 136, 100, 81), staff.lines, [
      anchorOver(fragments, 120),
    ]);
    expect(staffForAnchor(anchorOver(fragments, 100), [twin, staff])).toBe(
      twin
    );
  });
});

describe("mergeRawStaffs", () => {
  const left = fragmentsAt(100, 100);
  const right = fragmentsAt(210, 100);
  const self = rawStaffFromParts(
    { ...boxAt(0, 0, 10, 10), debugId: 7 },
    fiveLines(left.map((fragment) => [fragment])),
    [anchorOver(left, 100)]
  );

  it("unites the fragments line by line, its own anchors first, under its own id", () => {
    const other = rawStaffFromParts(
      { ...boxAt(1, 0, 10, 10), debugId: 9 },
      fiveLines(left.map((fragment, i) => [fragment, right[i] ?? fragment])),
      [anchorOver(right, 210)]
    );
    const merged = mergeRawStaffs(opsThatFit, self, other);
    expect(merged.box.debugId).toBe(7);
    expect(merged.anchors).toEqual([...self.anchors, ...other.anchors]);
    expect(merged.lines.map((line) => line.fragments)).toEqual(
      left.map((fragment, i) => [fragment, right[i]])
    );
    expect(Array.from(merged.box.contour)).toEqual(
      left.flatMap((fragment, i) => [
        fragment.rect.cx,
        fragment.rect.cy,
        right[i]?.rect.cx,
        right[i]?.rect.cy,
      ])
    );
  });
});

describe("raw-staff-merge.json", () => {
  type RectJson = [[number, number], [number, number], number];
  interface StaffJson {
    readonly lines: number[][];
    readonly staffId: number;
  }
  const boxOfRect = ([[cx, cy], [w, h]]: RectJson): RotatedBox =>
    boxAt(cx, cy, w, h);

  it("mergeRawStaffs is homr's RawStaff.merge where two fragments share a centre x: the other staff's comes first", () => {
    for (const one of vectorSet("raw-staff-merge").cases) {
      const fragments = (one.fragments as RectJson[]).map(boxOfRect);
      const linesOf = (lines: number[][]): FiveLines =>
        fiveLines(lines.map((line) => line.flatMap((k) => fragments[k] ?? [])));
      const anchorJson = one.anchor as { lines: number[][]; symbol: RectJson };
      const anchor = createStaffAnchor(
        linesOf(anchorJson.lines),
        boxOfRect(anchorJson.symbol)
      );
      const staffOfJson = ({ lines, staffId }: StaffJson, at: number) =>
        rawStaffFromParts(
          { ...boxAt(at, 0, 10, 10), debugId: staffId },
          linesOf(lines),
          [anchor]
        );
      const want = one.merged as StaffJson & { anchors: number };
      const merged = mergeRawStaffs(
        opsThatFit,
        staffOfJson(one.self as StaffJson, 0),
        staffOfJson(one.other as StaffJson, 1)
      );
      expect(
        merged.lines.map((line) =>
          line.fragments.map((fragment) => fragments.indexOf(fragment))
        ),
        String(one.name)
      ).toEqual(want.lines);
      expect(want.lines[0]).toEqual([5, 0]);
      expect(merged.box.debugId).toBe(want.staffId);
      expect(merged.anchors).toHaveLength(want.anchors);
    }
  });
});

describe("findRawStaffsByConnectingLineFragments", () => {
  const upper = fragmentsAt(100, 100);
  const lower = fragmentsAt(100, 600);
  const anchors = [
    anchorOver(upper, 80),
    anchorOver(lower, 80),
    anchorOver(upper, 120),
  ];

  it("moves a staff a later anchor merges into to the end of the list", () => {
    const staffs = findRawStaffsByConnectingLineFragments(opsThatFit, anchors, [
      ...upper,
      ...lower,
    ]);
    expect(staffs.map((staff) => staff.anchors)).toEqual([
      [anchors[1]],
      [anchors[0], anchors[2]],
    ]);
    expect(staffs.map((staff) => staff.box.debugId)).toEqual([1, 0]);
  });
});

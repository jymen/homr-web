/**
 * The rules of the staff chain that neither public page reaches, on inputs
 * small enough to read. Inputs and expected values both come from
 * test/golden/vectors/, where tools/dump-vectors.py wrote what homr 0.7.0
 * returned; nothing here is derived from the port.
 */

import { describe, expect, it } from "vitest";
import { withBoxOps } from "../src/cv/box-ops.js";
import { rotatedBoxFromRect } from "../src/geometry/box-transforms.js";
import {
  legacyConventionRectOf,
  pointListFromPairs,
  type RotatedBox,
} from "../src/geometry/boxes.js";
import { groupLinePeaks } from "../src/geometry/other-clefs.js";
import { rawStaffFromParts } from "../src/geometry/raw-staffs.js";
import { filterEdgeOfVision, resampleStaff } from "../src/geometry/resample.js";
import {
  type AnchorSymbolKind,
  createStaffAnchor,
  findStaffAnchors,
  type StaffAnchor,
} from "../src/geometry/staff-anchors.js";
import {
  asFiveLines,
  connectStaffLines,
  createStaffLineSegment,
  type FiveLines,
} from "../src/geometry/staff-lines.js";
import { decodeStaff, decodeStaffs } from "../src/golden/decode.js";
import {
  compareStaffLists,
  STAFF_TOLERANCES,
} from "../src/golden/staff-tolerance.js";
import { mean } from "../src/image/numeric.js";
import { testOpenCv } from "./support/opencv.js";
import { type VectorCase, vectorSet } from "./support/vectors.js";

type RectJson = [[number, number], [number, number], number];

/** A vector file's `[[cx, cy], [w, h], angle]` as a box with no contour: none of these rules reads one. */
function boxOfRect(value: unknown): RotatedBox {
  const [[cx, cy], [w, h], angle] = value as RectJson;
  return rotatedBoxFromRect(
    legacyConventionRectOf({ angle, cx, cy, h, w }),
    pointListFromPairs([]),
    0
  );
}

const boxesOf = (one: VectorCase): RotatedBox[] =>
  (one.fragments as unknown[]).map(boxOfRect);

function fiveLines(
  fragments: readonly RotatedBox[],
  lines: unknown
): FiveLines {
  const five = asFiveLines(
    (lines as number[][]).map((line) =>
      createStaffLineSegment(line.flatMap((k) => fragments[k] ?? []))
    )
  );
  if (five === null) {
    throw new Error("a vector staff has five lines");
  }
  return five;
}

interface AnchorJson {
  readonly averageUnitSize: number;
  readonly lines: number[][];
  readonly symbol: RectJson;
}

const anchorOf = (
  fragments: readonly RotatedBox[],
  anchor: AnchorJson
): StaffAnchor =>
  createStaffAnchor(
    fiveLines(fragments, anchor.lines),
    boxOfRect(anchor.symbol)
  );

const rectOf = ({ rect }: RotatedBox): RectJson => [
  [rect.cx, rect.cy],
  [rect.w, rect.h],
  rect.angle,
];

describe("connect-lines-cleanup.json", () => {
  it("connectStaffLines joins two fragments 100 px apart only when a fragment too short to keep has run the clean-up in between", () => {
    const { cases } = vectorSet("connect-lines-cleanup");
    expect(cases.map((one) => (one.fragments as unknown[]).length)).toEqual([
      2, 3,
    ]);
    for (const one of cases) {
      const fragments = boxesOf(one);
      expect(
        connectStaffLines(fragments, one.unitSize as number).map((line) =>
          line.fragments.map((fragment) => fragments.indexOf(fragment))
        ),
        String(one.name)
      ).toEqual(one.lines);
    }
  });
});

describe("find-anchors.json", () => {
  const { cases } = vectorSet("find-anchors");

  it("holds a sixth line either side of two units, and a bar line on and off a whole unit from its lines", () => {
    const counts = cases.map((one) => (one.anchors as unknown[]).length);
    expect(new Set(counts.slice(0, 4)).size).toBe(2);
    expect(counts.slice(4)).toEqual([0, 4, 5]);
  });

  it("findStaffAnchors is homr's find_staff_anchors on every case", async () => {
    const cv = await testOpenCv();
    for (const one of cases) {
      const fragments = boxesOf(one);
      const got = withBoxOps(cv, (ops) =>
        findStaffAnchors(
          ops,
          fragments,
          [boxOfRect(one.symbol)],
          one.kind as AnchorSymbolKind
        )
      );
      expect(
        got.map((anchor) => ({
          averageUnitSize: anchor.averageUnitSize,
          lines: anchor.lines.map((line) =>
            line.fragments.map((fragment) => fragments.indexOf(fragment))
          ),
          symbol: rectOf(anchor.symbol),
        })),
        String(one.name)
      ).toEqual(one.anchors);
    }
  });
});

describe("resample.json", () => {
  const { cases } = vectorSet("resample");

  it("holds a staff whose top line is missing beside a too-close pair, and a sloped one", () => {
    const [shifted, sloped] = cases.map((one) =>
      decodeStaff(one.staff, {}, String(one.name))
    );
    expect(new Set(shifted?.grid.map((point) => point.y.join(" "))).size).toBe(
      2
    );
    expect(sloped?.grid[0].angle).toBe(6);
    expect(sloped?.grid[0].x).toBeLessThan(0);
  });

  it("resampleStaff is homr's resample_staff on every case", () => {
    for (const one of cases) {
      const fragments = boxesOf(one);
      const anchor = anchorOf(fragments, one.anchor as AnchorJson);
      expect(anchor.averageUnitSize, `${one.name}: anchor unit`).toBe(
        (one.anchor as AnchorJson).averageUnitSize
      );
      const staff = rawStaffFromParts(
        boxOfRect(one.box),
        fiveLines(fragments, one.lines),
        [anchor]
      );
      const report = compareStaffLists(
        String(one.name),
        [resampleStaff(staff)],
        [decodeStaff(one.staff, {}, String(one.name))],
        STAFF_TOLERANCES.fromPythonInputs
      );
      expect(report.failures).toEqual([]);
    }
  });
});

describe("edge-of-vision.json", () => {
  it("filterEdgeOfVision drops what homr drops and nothing else", () => {
    for (const one of vectorSet("edge-of-vision").cases) {
      const staffs = decodeStaffs(one.staffs, String(one.name));
      expect(mean(staffs.map((staff) => staff.maxX - staff.minX))).toBe(
        one.usualWidth
      );
      const kept = filterEdgeOfVision(staffs, {
        height: one.height as number,
        width: one.width as number,
      });
      const labels = one.labels as string[];
      expect(kept.map((staff) => labels[staffs.indexOf(staff)])).toEqual(
        (one.kept as number[]).map((k) => labels[k])
      );
    }
  });
});

describe("line-peak-groups.json", () => {
  const { cases } = vectorSet("line-peak-groups");

  it("holds three groups, one group and a lone peak in group -1", () => {
    expect(cases.map((one) => new Set(one.groups as number[]).size)).toContain(
      3
    );
    expect(cases.map((one) => one.groups)).toContainEqual([-1]);
  });

  it("groupLinePeaks is filter_line_peaks' groups on every case", () => {
    for (const one of cases) {
      expect(groupLinePeaks(one.peaks as number[]), String(one.name)).toEqual(
        one.groups
      );
    }
  });
});

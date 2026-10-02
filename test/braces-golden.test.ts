/**
 * Braces and grand staffs against Python, on both pages and on the hand-built
 * staffs of vectors/braces.json: each row from Python's value of the row
 * before, then the whole function.
 */

import { describe, expect, it } from "vitest";
import { withBoxOps } from "../src/cv/box-ops.js";
import type { BoxOps } from "../src/geometry/box-ops.js";
import { rotatedBoxFromRect } from "../src/geometry/box-transforms.js";
import {
  legacyConventionRectOf,
  pointListFromPairs,
  type RotatedBox,
} from "../src/geometry/boxes.js";
import {
  connectionsBetweenStaffs,
  createGrandstaffs,
  filterForTallElements,
  findBracesBracketsAndGrandStaffLines,
  mergeMultiStaffsSharingAStaff,
  scoreBraceWithStaffPair,
} from "../src/geometry/braces.js";
import {
  compareBoxLists,
  describeBoxComparison,
} from "../src/golden/box-tolerance.js";
import {
  decodeBraces,
  decodeMultiStaffs,
  decodeRotatedBoxes,
  decodeStaffs,
  type GoldenBraces,
} from "../src/golden/decode.js";
import {
  assertGoldenMatches,
  compareMultiStaffLists,
  STAFF_TOLERANCES,
} from "../src/golden/staff-tolerance.js";
import { createMask, fillRectInPlace } from "../src/image/plane.js";
import {
  createMultiStaff,
  createStaff,
  createStaffPoint,
  type MultiStaff,
  type Staff,
} from "../src/model/staff.js";
import { predictBraceDot } from "../src/pipeline/detect.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";
import { vectorSet } from "./support/vectors.js";

type Staffs = readonly [Staff, ...Staff[]];

function nonEmpty(staffs: readonly Staff[]): Staffs {
  const [first, ...rest] = staffs;
  if (first === undefined) {
    throw new Error("a brace case has a staff");
  }
  return [first, ...rest];
}

/** find_braces_brackets_and_grand_staff_lines' loop, recording what braces.json records. */
function connectionsOf(
  ops: BoxOps,
  staffs: Staffs,
  braceDot: readonly RotatedBox[],
  tall: readonly RotatedBox[]
): GoldenBraces["connections"] {
  return staffs.flatMap((staff, i) =>
    [i - 1, i + 1].flatMap((neighbour) => {
      const other = staffs[neighbour];
      if (other === undefined) {
        return [];
      }
      const symbols = connectionsBetweenStaffs(ops, staff, other, tall).map(
        (symbol) => braceDot.indexOf(symbol)
      );
      return symbols.length > 0 ? [{ neighbour, staff: i, symbols }] : [];
    })
  );
}

/** The list Python hands _merge_multi_staff_if_they_share_a_staff, rebuilt from braces.json. */
function unmergedOf(
  staffs: Staffs,
  braceDot: readonly RotatedBox[],
  want: GoldenBraces
): MultiStaff[] {
  return staffs.flatMap((staff, i) => {
    const mine = want.connections.filter((one) => one.staff === i);
    if (mine.length === 0) {
      return [createMultiStaff([staff])];
    }
    return mine.map((one) =>
      createMultiStaff(
        [staff, staffs[one.neighbour] ?? staff],
        one.symbols.flatMap((index) => braceDot[index] ?? [])
      )
    );
  });
}

/** What the port answers at each row of braces.json, each from Python's value of the row before, and for the whole function. */
function bracesOf(
  ops: BoxOps,
  given: readonly Staff[],
  braceDot: readonly RotatedBox[],
  want: GoldenBraces
) {
  const staffs = nonEmpty(given);
  const pythonTall = want.tall.flatMap((index) => braceDot[index] ?? []);
  const merged = mergeMultiStaffsSharingAStaff(
    unmergedOf(staffs, braceDot, want)
  );
  return {
    connections: connectionsOf(ops, staffs, braceDot, pythonTall),
    grandStaffs: merged.map((multi) => createGrandstaffs(multi, pythonTall)),
    merged: merged.map((multi) =>
      multi.staffs.map((staff) => staffs.indexOf(staff))
    ),
    tall: filterForTallElements(braceDot, staffs).map((symbol) =>
      braceDot.indexOf(symbol)
    ),
    whole: findBracesBracketsAndGrandStaffLines(ops, staffs, braceDot),
  };
}

const EXACT = STAFF_TOLERANCES.fromPythonInputs;

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`braces and grand staffs ${fixture.name}`, () => {
    it("fits the brace_dot boxes of boxes-brace_dot.json", async () => {
      const cv = await testOpenCv();
      const report = compareBoxLists(
        "brace_dot",
        predictBraceDot(cv, {
          staff: page.mask("staff", true),
          symbols: page.mask("symbols", true),
        }),
        page.boxes("braceDot")
      );
      process.stdout.write(`${describeBoxComparison(report)}\n`);
      expect(report.failures).toEqual([]);
      expect(report.contoursExact).toBe(report.count.want);
    });

    it("reproduces braces.json row by row, then multistaffs.json", async () => {
      const cv = await testOpenCv();
      const want = page.braces();
      const multiStaffs = page.multiStaffs();
      const got = withBoxOps(cv, (ops) =>
        bracesOf(ops, page.staffsWithNotes(), page.boxes("braceDot"), want)
      );
      expect(got.tall).toEqual(want.tall);
      expect(got.connections).toEqual(want.connections);
      expect(got.merged).toEqual(want.merged);
      assertGoldenMatches(
        compareMultiStaffLists(
          "grand staffs",
          got.grandStaffs,
          multiStaffs,
          EXACT
        )
      );
      assertGoldenMatches(
        compareMultiStaffLists("multistaffs", got.whole, multiStaffs, EXACT)
      );
      process.stdout.write(
        `${fixture.name} braces: ${want.tall.length} tall of ${page.boxes("braceDot").length}, ${want.connections.length} connected pairs, ${multiStaffs.length} multi staffs, ${multiStaffs.flatMap((multi) => multi.staffs).filter((staff) => staff.isGrandstaff).length} grand staffs\n`
      );
    });
  });
}

describe("braces.json and braces-units.json", () => {
  const cases = [
    ...vectorSet("braces").cases,
    ...vectorSet("braces-units").cases,
  ];
  const decoded = cases.map((one) => ({
    braceDot: decodeRotatedBoxes(one.braceDot, `${one.name}.braceDot`),
    multiStaffs: decodeMultiStaffs(one.multiStaffs, `${one.name}.multiStaffs`),
    name: String(one.name),
    staffs: decodeStaffs(one.staffs, `${one.name}.staffs`),
    want: decodeBraces(one, String(one.name)),
  }));
  const named = (part: string) => {
    const found = decoded.find((one) => one.name.includes(part));
    if (found === undefined) {
      throw new Error(`no brace case named ${part}`);
    }
    return found;
  };

  it("hold a chain of three, a brace that connects and scores 0, a pair connected one way only and staffs of unlike units", () => {
    expect(
      new Set(
        named("rough limits").staffs.map((staff) => staff.averageUnitSize)
      ).size
    ).toBe(2);
    expect(named("three staffs in a chain").want.merged).toEqual([[0, 1, 2]]);
    const midStaff = named("scores 0");
    expect(midStaff.want.merged).toEqual([[0, 1]]);
    expect(midStaff.multiStaffs[0]?.staffs).toHaveLength(2);
    expect(
      named("unit sizes differ").want.connections.map((one) => [
        one.staff,
        one.neighbour,
      ])
    ).toEqual([[0, 1]]);
  });

  it("every intermediate and every result is homr's", async () => {
    const cv = await testOpenCv();
    for (const one of decoded) {
      const got = withBoxOps(cv, (ops) =>
        bracesOf(ops, one.staffs, one.braceDot, one.want)
      );
      expect(got.tall, `${one.name}: tall`).toEqual(one.want.tall);
      expect(got.connections, `${one.name}: connections`).toEqual(
        one.want.connections
      );
      expect(got.merged, `${one.name}: merged`).toEqual(one.want.merged);
      for (const [row, multiStaffs] of [
        ["grand staffs", got.grandStaffs],
        ["result", got.whole],
      ] as const) {
        assertGoldenMatches(
          compareMultiStaffLists(
            `${one.name}: ${row}`,
            multiStaffs,
            one.multiStaffs,
            EXACT
          )
        );
      }
    }
  });

  it("scores a brace at the left edge above 0 and the same brace in mid staff 0", () => {
    const [upper, lower] = named("left edge").staffs;
    const [brace] = named("left edge").braceDot;
    const [farBrace] = named("scores 0").braceDot;
    if (
      upper === undefined ||
      lower === undefined ||
      brace === undefined ||
      farBrace === undefined
    ) {
      throw new Error("the two cases have two staffs and a brace");
    }
    expect(scoreBraceWithStaffPair(brace, upper, lower)).toBe(194);
    expect(scoreBraceWithStaffPair(farBrace, upper, lower)).toBe(0);
  });
});

describe("multi-staff-merge.json", () => {
  interface Entry {
    readonly connections: readonly number[];
    readonly staffs: readonly number[];
  }
  /** The staffs and connections the file's meta describes; the last connection has the first one's rect in another object. */
  const staffs = [0, 1, 2, 3, 4].map((i) =>
    createStaff(
      [100, 110, 120].map((x) =>
        createStaffPoint(
          x,
          [0, 1, 2, 3, 4].map((line) => 200 + 130 * i + 18 * line),
          0
        )
      )
    )
  );
  const connections = [0, 1, 2, 0].map((k, debugId) =>
    rotatedBoxFromRect(
      legacyConventionRectOf({
        angle: 0,
        cx: 92,
        cy: 301 + 130 * k,
        h: 205,
        w: 12,
      }),
      pointListFromPairs([]),
      debugId
    )
  );
  const indexOfRect = (box: RotatedBox): number =>
    connections.findIndex(
      (one) => one.rect.cy === box.rect.cy && one.rect.cx === box.rect.cx
    );

  it("mergeMultiStaffsSharingAStaff groups, orders and deduplicates as homr does", () => {
    const { cases } = vectorSet("multi-staff-merge");
    expect(cases.map((one) => String(one.name)).join()).toContain(
      "moves to the end"
    );
    for (const one of cases) {
      const given = (one.given as Entry[]).map((entry) =>
        createMultiStaff(
          entry.staffs.flatMap((i) => staffs[i] ?? []),
          entry.connections.flatMap((k) => connections[k] ?? [])
        )
      );
      expect(
        mergeMultiStaffsSharingAStaff(given).map((multi) => ({
          connections: multi.connections.map(indexOfRect),
          staffs: multi.staffs.map((staff) => staffs.indexOf(staff)),
        })),
        String(one.name)
      ).toEqual(one.merged);
    }
  });
});

describe("grand-staffs.json", () => {
  const cases = vectorSet("grand-staffs").cases.map((one) => ({
    braces: decodeRotatedBoxes(one.braces, `${one.name}.braces`),
    name: String(one.name),
    scores: one.scores,
    staffs: decodeStaffs(one.staffs, `${one.name}.staffs`),
    want: decodeMultiStaffs([one.multiStaff], `${one.name}.multiStaff`),
  }));

  it("scoreBraceWithStaffPair is homr's for every pair and brace", () => {
    for (const one of cases) {
      expect(
        one.staffs
          .slice(1)
          .map((lower, i) =>
            one.braces.map((brace) =>
              scoreBraceWithStaffPair(brace, one.staffs[i] ?? lower, lower)
            )
          ),
        one.name
      ).toEqual(one.scores);
    }
  });

  it("createGrandstaffs fuses the pairs homr fuses", () => {
    for (const one of cases) {
      const given = createMultiStaff(one.staffs);
      const got = createGrandstaffs(given, one.braces);
      assertGoldenMatches(
        compareMultiStaffLists(one.name, [got], one.want, EXACT)
      );
      if (one.want[0]?.staffs.every((staff) => !staff.isGrandstaff)) {
        expect(got, one.name).toBe(given);
      }
    }
  });
});

describe("predictBraceDot", () => {
  it("drops a blob wider than 100 px and keeps one of any height", async () => {
    const cv = await testOpenCv();
    const symbols = createMask(400, 400);
    fillRectInPlace(symbols, 20, 20, 170, 60, 1);
    fillRectInPlace(symbols, 300, 100, 320, 380, 1);
    const boxes = predictBraceDot(cv, {
      staff: createMask(400, 400),
      symbols,
    });
    expect(boxes).toHaveLength(1);
    expect(boxes[0]?.rect.cx).toBeGreaterThan(300);
    expect(boxes[0]?.rect.h).toBeGreaterThan(280);
  });
});

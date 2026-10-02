/**
 * detect_staff against Python, stage by stage. Every row computes one stage
 * from **Python's** value of the stage before it, so a stage that drifts
 * fails on its own row and nothing downstream moves with it.
 */

import { describe, expect, it } from "vitest";
import { withBoxOps } from "../src/cv/box-ops.js";
import type { BoxOps } from "../src/geometry/box-ops.js";
import { breakWideFragments } from "../src/geometry/break-fragments.js";
import {
  findHorizontalLines,
  initZones,
  predictOtherAnchorsFromClefs,
  zoneColumns,
} from "../src/geometry/other-clefs.js";
import {
  findRawStaffsByConnectingLineFragments,
  removeDuplicateStaffs,
} from "../src/geometry/raw-staffs.js";
import {
  filterEdgeOfVision,
  resampleStaffs,
  sortStaffsTopToBottom,
} from "../src/geometry/resample.js";
import {
  filterUnusualAnchors,
  findStaffAnchors,
} from "../src/geometry/staff-anchors.js";
import {
  compareBoxLists,
  describeBoxComparison,
} from "../src/golden/box-tolerance.js";
import type { GoldenPage } from "../src/golden/page.js";
import {
  type AnchorListComparison,
  compareAnchorLists,
  compareRawStaffLists,
  compareStaffLists,
  describeStaffComparison,
  type RawStaffListComparison,
  STAFF_TOLERANCES,
} from "../src/golden/staff-tolerance.js";
import { mean } from "../src/image/numeric.js";
import { detectStaff } from "../src/pipeline/detect-staff.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";

/** What a stage's check hands the test: the lines that fail it, the values that must be equal, and one line to print. */
interface Row {
  readonly failures: readonly string[];
  /** `[what, got, want]`. */
  readonly same?: ReadonlyArray<readonly [string, unknown, unknown]>;
  readonly summary: string;
}

const EXACT = STAFF_TOLERANCES.fromPythonInputs;

const anchorRow = (report: AnchorListComparison): Row => ({
  failures: report.failures,
  same: [
    ["lines exact", report.linesExact, report.count.want],
    ["zones exact", report.zonesExact, report.count.want],
  ],
  summary: `${report.label}: ${report.count.got}/${report.count.want}, lines ${report.linesExact}, zones ${report.zonesExact}, unit ${report.worstUnit.toExponential(1)}, y ${report.worstY.toExponential(1)}, symbol centre ${report.symbols.worstCenter.toExponential(1)}, symbol polygons ${report.symbols.polygonsExact} exact`,
});

const rawStaffRow = (report: RawStaffListComparison): Row => ({
  failures: report.failures,
  same: [
    ["lines exact", report.linesExact, report.count.want],
    ["anchors exact", report.anchorsExact, report.count.want],
    ["contours exact", report.boxes.contoursExact, report.count.want],
    ["staff ids exact", report.boxes.debugIdsExact, report.count.want],
  ],
  summary: `${report.label}: ${report.count.got}/${report.count.want}, anchors ${report.anchorCounts.map((one) => `${one.got}/${one.want}`).join(" ")}, box centre ${report.boxes.worstCenter.toExponential(1)}, size ${report.boxes.worstSize.toExponential(1)}, angle ${report.boxes.worstAngle.toExponential(1)}, polygons ${report.boxes.polygonsExact} exact + ${report.boxes.slackUsed} within 1 px`,
});

/** The stage's name in homr, and its report against Python. */
const STAGES = [
  [
    "break_wide_fragments",
    (ops: BoxOps, page: GoldenPage): Row => {
      const report = compareBoxLists(
        "staff_fragments-broken",
        breakWideFragments(ops, page.boxes("staffFragments")),
        page.boxes("staffFragmentsBroken")
      );
      return {
        failures: report.failures,
        same: [
          ["contours exact", report.contoursExact, report.count.want],
          ["ids exact", report.debugIdsExact, report.count.want],
        ],
        summary: describeBoxComparison(report),
      };
    },
  ],
  [
    "find_staff_anchors (clefs)",
    (ops: BoxOps, page: GoldenPage): Row =>
      anchorRow(
        compareAnchorLists(
          "clef anchors",
          findStaffAnchors(
            ops,
            page.boxes("staffFragmentsBroken"),
            page.boxes("clefsKeys"),
            "clef"
          ),
          page.staffAnchors().clefs,
          EXACT
        )
      ),
  ],
  [
    "init_zone and find_horizontal_lines",
    (_ops: BoxOps, page: GoldenPage): Row => {
      const { clefs, zones } = page.staffAnchors();
      const staff = page.mask("staff", true);
      const unit = mean(clefs.map((anchor) => anchor.averageUnitSize));
      const got = initZones(clefs, staff.width).map((zone) => ({
        ...zone,
        lines: findHorizontalLines(zoneColumns(staff, zone), unit),
      }));
      return {
        failures: [],
        same: [["zones", got, zones]],
        summary: `zones: ${got.length}, groups of five ${got.map((zone) => zone.lines.length).join(" ")}`,
      };
    },
  ],
  [
    "predict_other_anchors_from_clefs",
    (ops: BoxOps, page: GoldenPage): Row => {
      const report = compareBoxLists(
        "other clefs",
        predictOtherAnchorsFromClefs(
          ops,
          page.staffAnchors().clefs,
          page.mask("staff", true)
        ),
        page.staffAnchors().otherClefSymbols
      );
      return { ...report, summary: describeBoxComparison(report) };
    },
  ],
  [
    "predict_other_anchors_from_clefs, before its overlap filter",
    (ops: BoxOps, page: GoldenPage): Row => {
      const report = compareBoxLists(
        "other-clef candidates",
        predictOtherAnchorsFromClefs(
          { ...ops, overlaps: () => false },
          page.staffAnchors().clefs,
          page.mask("staff", true)
        ),
        page.otherClefCandidates()
      );
      return {
        ...report,
        same: [
          ["polygons exact", report.polygonsExact, report.count.want],
          ["centre delta", report.worstCenter, 0],
          ["size delta", report.worstSize, 0],
        ],
        summary: describeBoxComparison(report),
      };
    },
  ],
  [
    "find_staff_anchors (other clefs)",
    (ops: BoxOps, page: GoldenPage): Row =>
      anchorRow(
        compareAnchorLists(
          "other-clef anchors",
          findStaffAnchors(
            ops,
            page.boxes("staffFragmentsBroken"),
            page.staffAnchors().otherClefSymbols,
            "clef"
          ),
          page.staffAnchors().otherClefs,
          EXACT
        )
      ),
  ],
  [
    "find_staff_anchors (bar lines)",
    (ops: BoxOps, page: GoldenPage): Row =>
      anchorRow(
        compareAnchorLists(
          "bar-line anchors",
          findStaffAnchors(
            ops,
            page.boxes("staffFragmentsBroken"),
            page.barLines().barLines,
            "barLine"
          ),
          page.staffAnchors().barLines,
          EXACT
        )
      ),
  ],
  [
    "filter_unusual_anchors",
    (_ops: BoxOps, page: GoldenPage): Row => {
      const { barLines, clefs, kept, otherClefs } = page.staffAnchors();
      const all = [...clefs, ...otherClefs, ...barLines];
      const got = filterUnusualAnchors(all);
      return {
        failures: [],
        same: [
          [
            "kept, as positions in clefs + otherClefs + barLines",
            got.map((anchor) => all.indexOf(anchor)),
            kept.map((anchor) => all.indexOf(anchor)),
          ],
        ],
        summary: `kept anchors: ${got.length} of ${all.length}`,
      };
    },
  ],
  [
    "find_raw_staffs_by_connecting_line_fragments",
    (ops: BoxOps, page: GoldenPage): Row =>
      rawStaffRow(
        compareRawStaffLists(
          "raw staffs",
          findRawStaffsByConnectingLineFragments(
            ops,
            page.staffAnchors().kept,
            page.boxes("staffFragmentsBroken")
          ),
          page.rawStaffs().connected
        )
      ),
  ],
  [
    "remove_duplicate_staffs",
    (ops: BoxOps, page: GoldenPage): Row => {
      const { connected, deduplicated } = page.rawStaffs();
      const got = removeDuplicateStaffs(ops, connected);
      return {
        failures: [],
        same: [
          [
            "deduplicated, as positions in connected",
            got.map((staff) => connected.indexOf(staff)),
            deduplicated.map((staff) => connected.indexOf(staff)),
          ],
        ],
        summary: `deduplicated: ${got.length} of ${connected.length}`,
      };
    },
  ],
  [
    "resample_staffs, filter_edge_of_vision, sort_staffs_top_to_bottom",
    (_ops: BoxOps, page: GoldenPage): Row => {
      const { deduplicated, droppedAtEdge, resampledFrom } = page.rawStaffs();
      const resampled = resampleStaffs(deduplicated);
      const kept = filterEdgeOfVision(resampled, page.mask("staff", true));
      const sorted = sortStaffsTopToBottom(kept);
      const report = compareStaffLists("staffs", sorted, page.staffs(), EXACT);
      return {
        failures: report.failures,
        same: [
          [
            "dropped at the edge",
            resampled.flatMap((staff, i) => (kept.includes(staff) ? [] : [i])),
            droppedAtEdge,
          ],
          [
            "resampled from",
            sorted.map((staff) => resampled.indexOf(staff)),
            resampledFrom,
          ],
        ],
        summary: describeStaffComparison(report),
      };
    },
  ],
  [
    "detect_staff",
    (ops: BoxOps, page: GoldenPage): Row => {
      const report = compareStaffLists(
        "detect_staff",
        detectStaff(
          ops,
          page.mask("staff", true),
          page.boxes("staffFragmentsBroken"),
          page.boxes("clefsKeys"),
          page.barLines().barLines
        ),
        page.staffs(),
        EXACT
      );
      return { ...report, summary: describeStaffComparison(report) };
    },
  ],
] as const satisfies ReadonlyArray<
  readonly [string, (ops: BoxOps, page: GoldenPage) => Row]
>;

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);
  describe(`detect_staff ${fixture.name}`, () => {
    for (const [stage, check] of STAGES) {
      it(`reproduces ${stage}`, async () => {
        const cv = await testOpenCv();
        const row = withBoxOps(cv, (ops) => check(ops, page));
        process.stdout.write(`${fixture.name} ${row.summary}\n`);
        expect(row.failures).toEqual([]);
        for (const [what, got, want] of row.same ?? []) {
          expect(got, what).toEqual(want);
        }
      });
    }
  });
}

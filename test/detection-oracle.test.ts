/**
 * The phase 5 oracle: every staff-detection intermediate the dumper writes
 * decodes, agrees with the files it indexes into, and every vector file has
 * the shape its consumer will read. No detection code runs here; each later
 * stage starts from these files.
 */

import { describe, expect, it } from "vitest";
import { decodeStaffPositions } from "../src/golden/decode.js";
import { planeAgreement } from "../src/image/plane.js";
import { symbolsOfKind } from "../src/model/symbols.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { checkVectorCase, VECTOR_FILES, vectorSet } from "./support/vectors.js";

/** Counts measured when each public fixture was dumped (homr 0.7.0, arm64, 2026-10-02). */
const KNOWN_COUNTS: Record<
  string,
  {
    barLineAnchors: number;
    braceDot: number;
    clefAnchors: number;
    connections: number;
    grandStaffs: number;
    keptAnchors: number;
    rawStaffs: number;
    tall: number;
    zoneGroups: number;
  }
> = {
  "grand-staff-300dpi": {
    barLineAnchors: 147,
    braceDot: 25,
    clefAnchors: 22,
    connections: 8,
    grandStaffs: 4,
    keptAnchors: 168,
    rawStaffs: 8,
    tall: 21,
    zoneGroups: 3,
  },
  "the-kesh-300dpi": {
    barLineAnchors: 100,
    braceDot: 0,
    clefAnchors: 24,
    connections: 0,
    grandStaffs: 0,
    keptAnchors: 120,
    rawStaffs: 4,
    tall: 0,
    zoneGroups: 1,
  },
};

const NEW_STAGES = [
  "braces.json",
  "mask-denoised-staff.png",
  "noise.json",
  "notehead-splits.json",
  "other-clefs.json",
  "raw-staffs.json",
  "staff-anchors.json",
];

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`detection oracle ${fixture.name}`, () => {
    it("records the oracle and lists the new stage files", () => {
      const meta = page.meta();
      expect(meta.oracle).toBeDefined();
      for (const value of Object.values(meta.oracle ?? {})) {
        expect(value).not.toBe("");
      }
      for (const stage of NEW_STAGES) {
        expect(meta.stages).toContain(stage);
      }
      expect(meta.stages.includes("mask-noise.png")).toBe(
        page.noise().outcome === "masked"
      );
    });

    it("decodes noise.json, and the denoised staff mask is the raw one under the noise mask", () => {
      const noise = page.noise();
      const raw = page.mask("staff");
      const denoised = page.denoisedStaffMask();
      const kept = page.noiseMask();
      expect(kept === null).toBe(noise.outcome !== "masked");
      const expected =
        kept === null
          ? raw
          : { ...raw, data: raw.data.map((v, i) => v * (kept.data[i] ?? 0)) };
      expect(planeAgreement(denoised, expected)).toBe(1);
    });

    it("decodes staff-anchors.json into anchors of five lines over the broken fragments", () => {
      const anchors = page.staffAnchors();
      const fragments = new Set(page.boxes("staffFragmentsBroken"));
      const all = [
        ...anchors.clefs,
        ...anchors.otherClefs,
        ...anchors.barLines,
      ];
      expect(all.length).toBeGreaterThan(0);
      for (const anchor of all) {
        expect(anchor.lines).toHaveLength(5);
        for (const fragment of anchor.lines.flatMap((line) => line.fragments)) {
          expect(fragments.has(fragment)).toBe(true);
        }
        expect(anchor.averageUnitSize).toBeGreaterThan(0);
        expect(anchor.zone.start).toBeLessThan(anchor.minY);
        expect(anchor.zone.stop).toBeGreaterThan(anchor.maxY);
      }
      const members = new Set(all);
      expect(new Set(anchors.kept).size).toBe(anchors.kept.length);
      for (const anchor of anchors.kept) {
        expect(members.has(anchor)).toBe(true);
      }
      expect(anchors.otherClefs.length).toBeLessThanOrEqual(
        anchors.otherClefSymbols.length * 6
      );
      const { width } = page.denoisedStaffMask();
      for (const zone of anchors.zones) {
        expect(zone.stop).toBeLessThanOrEqual(width);
      }
      const candidates = page.otherClefCandidates();
      expect(candidates).toHaveLength(
        anchors.zones.reduce((n, zone) => n + zone.lines.length, 0)
      );
      for (const symbol of anchors.otherClefSymbols) {
        expect(candidates).toContainEqual(symbol);
      }
    });

    it("decodes raw-staffs.json, and every kept anchor belongs to exactly one connected staff", () => {
      const raw = page.rawStaffs();
      const { kept } = page.staffAnchors();
      const owned = raw.connected.flatMap((staff) => staff.anchors);
      expect(owned).toHaveLength(kept.length);
      expect(new Set(owned)).toEqual(new Set(kept));
      for (const staff of raw.connected) {
        expect(staff.lines).toHaveLength(5);
      }
      expect(raw.resampledFrom).toHaveLength(page.staffs().length);
      expect(raw.resampledFrom.length + raw.droppedAtEdge.length).toBe(
        raw.deduplicated.length
      );
    });

    it("decodes notehead-splits.json", () => {
      for (const split of page.noteheadSplits()) {
        expect(split.pieces.length).toBeGreaterThan(1);
      }
    });

    it("decodes braces.json, which accounts for every staff of multistaffs.json", () => {
      const braces = page.braces();
      const multiStaffs = page.multiStaffs();
      expect(multiStaffs).toHaveLength(braces.merged.length);
      for (const [i, multi] of multiStaffs.entries()) {
        const fiveLineStaffs = multi.staffs.reduce(
          (n, staff) => n + (staff.isGrandstaff ? 2 : 1),
          0
        );
        expect(fiveLineStaffs).toBe(braces.merged[i]?.length);
      }
      const brace = page.braceDotMask();
      const staffMask = page.denoisedStaffMask();
      expect([brace.width, brace.height]).toEqual([
        staffMask.width,
        staffMask.height,
      ]);
    });

    it("deals the notes back onto the staffs, as new staffs on every call", () => {
      const withNotes = page.staffsWithNotes();
      const counts = withNotes.map(
        (staff) => symbolsOfKind(staff.symbols, "note").length
      );
      expect(counts).toEqual(page.braces().notesPerStaff);
      expect(withNotes.flatMap((staff) => staff.symbols)).toEqual(page.notes());
      const again = page.staffsWithNotes();
      for (const [i, staff] of withNotes.entries()) {
        expect(again[i]).not.toBe(staff);
        expect(again[i]?.grid[0]).toBe(staff.grid[0]);
      }
      for (const staff of page.staffs()) {
        expect(staff.symbols).toHaveLength(0);
      }
    });

    it("keeps staff-positions.txt as text beside its parsed form", () => {
      const text = page.staffPositionsText();
      expect(text.endsWith("\n")).toBe(true);
      expect(decodeStaffPositions(text)).toEqual(page.staffPositions());
    });

    const known = KNOWN_COUNTS[fixture.name];
    it("has the counts it was dumped with, when they were recorded", () => {
      if (known === undefined) {
        return;
      }
      const anchors = page.staffAnchors();
      const braces = page.braces();
      expect({
        barLineAnchors: anchors.barLines.length,
        braceDot: page.boxes("braceDot").length,
        clefAnchors: anchors.clefs.length,
        connections: braces.connections.length,
        grandStaffs: page
          .multiStaffs()
          .flatMap((multi) => multi.staffs)
          .filter((staff) => staff.isGrandstaff).length,
        keptAnchors: anchors.kept.length,
        rawStaffs: page.rawStaffs().connected.length,
        tall: braces.tall.length,
        zoneGroups: anchors.zones.reduce((n, zone) => n + zone.lines.length, 0),
      }).toEqual(known);
    });
  });
}

describe("vectors", () => {
  for (const name of VECTOR_FILES) {
    it(`${name}.json names its oracle and every case has the shape its test will read`, () => {
      const { cases, meta } = vectorSet(name);
      for (const key of ["numpy", "opencv", "python", "machine"]) {
        expect(meta[key], `meta.${key}`).toBeTruthy();
      }
      expect(cases.length).toBeGreaterThan(0);
      for (const [i, one] of cases.entries()) {
        checkVectorCase(name, one, `${name}.cases[${i}]`);
      }
    });
  }

  it("argsort.json holds an input that reaches numpy's heapsort fallback", () => {
    const reached = vectorSet("argsort").cases.filter(
      (one) => typeof one.heapsorted === "number" && one.heapsorted > 0
    );
    expect(reached.length).toBeGreaterThan(0);
  });
});

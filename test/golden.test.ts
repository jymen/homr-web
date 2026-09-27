/**
 * Phase 1's test: every golden stage file of every fixture parses, at the
 * boundary, into the domain types, the derived fields Python stored agree
 * with the port's derivations (the decoders assert that on every object),
 * and the vocabulary tables equal Python's index for index.
 */

import { describe, expect, it } from "vitest";
import { cornersOf, pointCount } from "../src/geometry/boxes.js";
import { decodeVocabulary } from "../src/golden/decode.js";
import { GOLDEN_BOX_FILES } from "../src/golden/page.js";
import { planeAgreement } from "../src/image/plane.js";
import { ENCODER_CANVAS, MASK_CLASS_NAMES } from "../src/model/pipeline.js";
import { yTolerance } from "../src/model/staff.js";
import { symbolsOfKind } from "../src/model/symbols.js";
import { isDecodedSymbol, NEWLINE } from "../src/transformer/symbol.js";
import { VOCABULARIES } from "../src/transformer/vocabulary.js";
import {
  goldenPageOf,
  listGoldenFixtures,
  readGoldenJson,
} from "./support/golden.js";

/** Counts measured on the public fixture when it was dumped; a re-dump that changes them is a model change, not a port change. */
const KNOWN_COUNTS: Record<
  string,
  {
    staffs: number;
    notes: number;
    noteheads: number;
    stemsRest: number;
    barLineBoxes: number;
  }
> = {
  "the-kesh-300dpi": {
    barLineBoxes: 103,
    noteheads: 81,
    notes: 80,
    staffs: 4,
    stemsRest: 377,
  },
};

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`golden ${fixture.name}`, () => {
    it("has a meta file naming the fixture and the three models", () => {
      const meta = page.meta();
      expect(meta.fixture).toBe(`${fixture.name}.png`);
      expect(Object.keys(meta.models).sort()).toEqual([
        "decoder",
        "encoder",
        "segnet",
      ]);
    });

    for (const kind of Object.keys(
      GOLDEN_BOX_FILES
    ) as (keyof typeof GOLDEN_BOX_FILES)[]) {
      it(`decodes ${GOLDEN_BOX_FILES[kind]} into normalised rotated boxes`, () => {
        for (const box of page.boxes(kind)) {
          expect(box.rect.angle).toBeGreaterThanOrEqual(-45);
          expect(box.rect.angle).toBeLessThanOrEqual(45);
          expect(pointCount(box.polygon)).toBe(4);
          const corners = cornersOf(box.rect);
          expect(corners.bottomRight.x - corners.topLeft.x).toBeCloseTo(
            box.rect.w,
            9
          );
        }
      });
    }

    it("decodes noteheads as ellipses with many-point outlines", () => {
      for (const ellipse of page.noteheads()) {
        expect(ellipse.kind).toBe("ellipse");
        expect(pointCount(ellipse.polygon)).toBeGreaterThan(4);
      }
    });

    it("pairs a stem with its direction, or has neither", () => {
      for (const n of page.noteheadsWithStems()) {
        if (n.stem !== null) {
          expect(["UP", "DOWN"]).toContain(n.stem.direction);
        }
      }
      expect(page.barLines().averageNoteHeadHeight).toBeGreaterThan(0);
    });

    it("decodes staffs with the bounds and tolerance Python stored", () => {
      const staffs = page.staffs();
      expect(staffs.length).toBeGreaterThan(0);
      for (const staff of staffs) {
        expect(staff.space).toBe("page");
        expect(staff.minX).toBe(staff.grid[0].x);
        expect(yTolerance(staff)).toBeGreaterThan(0);
        expect(staff.symbols).toHaveLength(0);
      }
    });

    it("decodes notes and multi staffs, and the notes are on the staffs", () => {
      const notes = page.notes();
      const onStaffs = page
        .multiStaffs()
        .flatMap((ms) =>
          ms.staffs.flatMap((s) => symbolsOfKind(s.symbols, "note"))
        );
      expect(onStaffs).toHaveLength(notes.length);
      for (const note of notes) {
        expect(note.box.kind).toBe("ellipse");
      }
    });

    it("decodes every per-staff canvas, its canvas-space staff and its tokens", () => {
      const count = page.staffCount();
      expect(count).toBeGreaterThan(0);
      for (let index = 0; index < count; index += 1) {
        const canvas = page.canvas(index);
        expect([canvas.width, canvas.height]).toEqual([
          ENCODER_CANVAS.width,
          ENCODER_CANVAS.height,
        ]);
        const staff = page.canvasStaff(index);
        expect(staff.space).toBe("canvas");
        expect(staff.maxX).toBeLessThanOrEqual(ENCODER_CANVAS.width);
        const tokens = page.tokens(index);
        expect(tokens.length).toBeGreaterThan(0);
        for (const token of tokens) {
          expect(token.coordinates).not.toBeNull();
          expect(isDecodedSymbol(token)).toBe(true);
        }
      }
    });

    it("decodes voices, which carry the newline homr inserts between staffs", () => {
      const voices = page.voices();
      expect(voices.length).toBeGreaterThan(0);
      const rhythms = voices.flat().map((s) => s.rhythm);
      expect(rhythms).toContain(NEWLINE);
    });

    it("reads one staff position per parsed staff", () => {
      const positions = page.staffPositions();
      expect(positions).toHaveLength(
        page.multiStaffs().reduce((n, ms) => n + ms.staffs.length, 0)
      );
      for (const p of positions) {
        expect(p.cx).toBeGreaterThan(0);
        expect(p.cx).toBeLessThan(1);
      }
    });

    it("loads the five raw and five filtered masks at the preprocessed page's size", () => {
      const preprocessed = page.preprocessed();
      for (const name of MASK_CLASS_NAMES) {
        for (const filtered of [false, true]) {
          const mask = page.mask(name, filtered);
          expect([mask.width, mask.height]).toEqual([
            preprocessed.width,
            preprocessed.height,
          ]);
          expect(planeAgreement(mask, mask)).toBe(1);
        }
      }
      expect(page.musicXml()).toContain("<score-partwise");
    });

    const known = KNOWN_COUNTS[fixture.name];
    if (known !== undefined) {
      it("still holds the counts measured when it was dumped", () => {
        expect(page.staffs()).toHaveLength(known.staffs);
        expect(page.notes()).toHaveLength(known.notes);
        expect(page.noteheads()).toHaveLength(known.noteheads);
        expect(page.boxes("stemsRest")).toHaveLength(known.stemsRest);
        expect(page.boxes("barLines")).toHaveLength(known.barLineBoxes);
      });
    }
  });
}

describe("vocabulary", () => {
  it("has the sizes homr 0.7.0 builds", () => {
    expect(VOCABULARIES.rhythm).toHaveLength(259);
    expect(VOCABULARIES.pitch).toHaveLength(72);
    expect(VOCABULARIES.lift).toHaveLength(7);
    expect(VOCABULARIES.articulation).toHaveLength(54);
    expect(VOCABULARIES.slur).toHaveLength(5);
    expect(VOCABULARIES.position).toHaveLength(3);
  });

  it("matches Python index for index", () => {
    const python = decodeVocabulary(readGoldenJson("vocabulary.json"));
    for (const head of Object.keys(
      VOCABULARIES
    ) as (keyof typeof VOCABULARIES)[]) {
      expect([...VOCABULARIES[head]]).toEqual([...python[head]]);
    }
  });
});

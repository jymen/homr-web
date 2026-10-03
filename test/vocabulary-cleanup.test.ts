/**
 * The vocabulary half of phase 7, no model: remove_duplicated_symbols, the
 * durations and tuplets behind it and ConvertToArray against Python's vectors;
 * parse_staffs' voice join from Python's tokens-<n>.json against voices.json;
 * and predict_best's position filter.
 */

import { describe, expect, it } from "vitest";
import { createGray } from "../src/image/plane.js";
import { float16FromFloat32 } from "../src/models/dtype.js";
import { filterPositions, joinVoice } from "../src/pipeline/parse-staffs.js";
import {
  durationOfRhythm,
  priorPowerOfTwo,
} from "../src/transformer/duration.js";
import {
  encoderInput,
  NORMALIZED_FLOAT32,
} from "../src/transformer/normalize.js";
import {
  durationOfMeasure,
  removeDuplicatedSymbols,
  typicalDurationOfMeasures,
} from "../src/transformer/remove-duplicated-symbols.js";
import {
  createEncodedSymbol,
  type DecodedSymbol,
  type EncodedSymbol,
  removeTuplet,
} from "../src/transformer/symbol.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { symbolRows, vectorSet } from "./support/vectors.js";

const symbolOf = ([
  rhythm = ".",
  pitch = ".",
  lift = ".",
  articulation = ".",
  slur = ".",
  position = ".",
]: readonly string[]): EncodedSymbol =>
  createEncodedSymbol(rhythm, {
    articulation,
    lift: lift as EncodedSymbol["lift"],
    pitch: pitch as EncodedSymbol["pitch"],
    position: position as EncodedSymbol["position"],
    slur,
  });

const fieldsOf = (s: EncodedSymbol): string[] => [
  s.rhythm,
  s.pitch,
  s.lift,
  s.articulation,
  s.slur,
  s.position,
];

const cases = (kind: string) =>
  vectorSet("vocabulary-cleanup").cases.filter((one) => one.kind === kind);

describe("remove_duplicated_symbols against Python", () => {
  for (const one of cases("remove")) {
    it(`${String(one.name)}, cleanup_tuplets=${String(one.cleanupTuplets)}`, () => {
      const symbols = symbolRows(one.symbols, "symbols").map(symbolOf);
      const result = removeDuplicatedSymbols(
        symbols,
        one.cleanupTuplets === true
      );
      expect(result.map(fieldsOf)).toEqual(symbolRows(one.result, "result"));
    });
  }

  it("remove_tuplet on every vector rhythm", () => {
    for (const one of cases("tuplet")) {
      const rhythm = String(one.rhythm);
      expect(removeTuplet(createEncodedSymbol(rhythm)).rhythm, rhythm).toBe(
        one.result
      );
    }
  });

  it("get_duration on every vector rhythm", () => {
    for (const one of cases("duration")) {
      const rhythm = String(one.rhythm);
      const d = durationOfRhythm(rhythm);
      expect(
        {
          actualNotes: d.actualNotes,
          base: [d.baseDuration.num, d.baseDuration.den],
          dots: d.dots,
          fraction: [d.fraction.num, d.fraction.den],
          kern: d.kern,
          normalNotes: d.normalNotes,
        },
        rhythm
      ).toEqual({
        actualNotes: one.actualNotes,
        base: one.base,
        dots: one.dots,
        fraction: one.fraction,
        kern: one.kern,
        normalNotes: one.normalNotes,
      });
    }
  });

  it("_get_duration_of_measure and _get_typical_duration_of_measures", () => {
    const measureOf = (value: unknown, at: string) =>
      (Array.isArray(value) ? value : []).map((chord, i) =>
        symbolRows(chord, `${at}[${i}]`).map(symbolOf)
      );
    for (const one of cases("measure")) {
      const d = durationOfMeasure(measureOf(one.measure, "measure"));
      expect([d.num, d.den], String(one.name)).toEqual(one.result);
    }
    for (const one of cases("typical")) {
      const measures = (Array.isArray(one.measures) ? one.measures : []).map(
        (m, i) => measureOf(m, `measures[${i}]`)
      );
      const d = typicalDurationOfMeasures(measures);
      expect([d.num, d.den], String(one.name)).toEqual(one.result);
    }
  });

  it("prior_power_of_two", () => {
    for (const one of cases("priorPowerOfTwo")) {
      expect(priorPowerOfTwo(Number(one.n)), String(one.n)).toBe(one.result);
    }
  });
});

describe("ConvertToArray against Python", () => {
  const rows = vectorSet("normalize").cases;

  it("the float32 table is numpy's, value for value", () => {
    expect([...NORMALIZED_FLOAT32]).toEqual(rows.map((one) => one.float32));
  });

  it("the half table is numpy's astype(float16) of it", () => {
    expect([...NORMALIZED_FLOAT32].map(float16FromFloat32)).toEqual(
      rows.map((one) => one.float16Bits)
    );
  });

  it.each([
    ["float32", "float32"],
    ["float16", "float16Bits"],
  ] as const)(
    "encoderInput writes a %s [1, 1, 256, 1280] tensor pixel by pixel",
    async (type, key) => {
      const canvas = createGray(1280, 256);
      for (let i = 0; i < canvas.data.length; i += 1) {
        canvas.data[i] = (i * 7) % 256;
      }
      const tensor = encoderInput(canvas, type);
      expect(tensor.dims).toEqual([1, 1, 256, 1280]);
      expect(tensor.type).toBe(type);
      const want = rows.map((one) => one[key]);
      expect([...(await tensor.getData())]).toEqual(
        [...canvas.data].map((pixel) => want[pixel])
      );
    }
  );
});

describe("parse_staffs' voice join from Python's tokens", () => {
  for (const fixture of listGoldenFixtures()) {
    it(`${fixture.name}: tokens-<n>.json joined per voice give voices.json`, () => {
      const golden = goldenPageOf(fixture);
      const voices = golden.voices();
      const perVoice = golden.staffCount() / voices.length;
      for (const [v, want] of voices.entries()) {
        const staffs = Array.from({ length: perVoice }, (_, i) =>
          golden.tokens(v * perVoice + i)
        );
        expect(joinVoice(staffs)).toEqual(want);
      }
    });
  }
});

describe("predict_best's position filter", () => {
  const symbols: DecodedSymbol[] = (["upper", "lower", "."] as const).map(
    (position) => ({
      articulation: "_",
      coordinates: null,
      lift: "_",
      pitch: "C4",
      position,
      rhythm: "note_4",
      slur: "_",
    })
  );

  it("drops lower symbols on a single staff", () => {
    expect(filterPositions(symbols, false).map((s) => s.position)).toEqual([
      "upper",
      ".",
    ]);
  });

  it("keeps everything on a grand staff", () => {
    expect(filterPositions(symbols, true)).toEqual(symbols);
  });
});

describe("parse_staffs skips an empty staff", () => {
  it("adds no newline for it", () => {
    const note = createEncodedSymbol("note_4", {
      pitch: "C4",
      position: "upper",
    });
    expect(joinVoice([[], [note], []]).map((s) => s.rhythm)).toEqual([
      "note_4",
      "newline",
    ]);
  });
});

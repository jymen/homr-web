/**
 * The transformer on Python's canvases: every staff of every fixture, encoder
 * and decoder fp32 on WebAssembly, against tokens-<n>.json. The six heads must
 * be equal exactly; the attention coordinates are float32 outputs of the same
 * graph computed by another build of onnxruntime, so they get a tolerance.
 */

import { expect, it } from "vitest";
import { createStaffCanvas } from "../src/model/pipeline.js";
import {
  parseStaffCanvas,
  parseStaffs,
  type TransformerSessions,
} from "../src/pipeline/parse-staffs.js";
import type { EncodedSymbol } from "../src/transformer/symbol.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

/**
 * Pixels on the 1280 x 256 canvas. Native onnxruntime moves these by up to
 * 5.13 px against itself when only its optimisation level and thread count
 * change, tokens unchanged (docs/decisions.tsv, 2026-10-03).
 */
const COORDINATE_TOLERANCE = 6;
const STAFF_TIMEOUT_MS = 120_000;

const heads = (s: EncodedSymbol): string[] => [
  s.rhythm,
  s.pitch,
  s.lift,
  s.articulation,
  s.slur,
  s.position,
];

const report = (line: string): void => {
  process.stdout.write(`${line}\n`);
};

describeWithModels("transformer on Python's staff canvases", () => {
  for (const fixture of listGoldenFixtures()) {
    const golden = goldenPageOf(fixture);
    for (let index = 0; index < golden.staffCount(); index += 1) {
      it(
        `${fixture.name} canvas ${index} gives tokens-${index}.json`,
        async () => {
          const store = await storeOn(CPU);
          try {
            const sessions: TransformerSessions = {
              decoder: await store.open("decoder"),
              encoder: await store.open("encoder"),
            };
            let steps = 0;
            let firstStepAt = 0;
            const began = performance.now();
            const tokens = await parseStaffCanvas(
              sessions,
              createStaffCanvas(
                golden.canvas(index),
                golden.canvasStaff(index)
              ),
              {
                onStep: (step) => {
                  steps = step;
                  if (step === 1) {
                    firstStepAt = performance.now();
                  }
                },
              }
            );
            const ended = performance.now();
            const perStep = (ended - firstStepAt) / (steps - 1);
            const expected = golden.tokens(index);
            let worst = 0;
            for (const [i, token] of tokens.entries()) {
              const want = expected[i];
              if (want?.coordinates && token.coordinates !== null) {
                worst = Math.max(
                  worst,
                  Math.abs(token.coordinates.x - want.coordinates.x),
                  Math.abs(token.coordinates.y - want.coordinates.y)
                );
              }
            }
            report(
              `${fixture.name} canvas ${index}: ${tokens.length} tokens, ${steps} steps, ${Math.round(ended - began)} ms; encoder and step 0 ${Math.round(firstStepAt - began)} ms, then ${perStep.toFixed(1)} ms/step; worst coordinate ${worst.toFixed(3)} px`
            );
            expect(tokens.map(heads)).toEqual(expected.map(heads));
            expect(worst).toBeLessThanOrEqual(COORDINATE_TOLERANCE);
          } finally {
            await store.close();
          }
        },
        STAFF_TIMEOUT_MS
      );
    }
  }

  for (const fixture of listGoldenFixtures()) {
    it(
      `${fixture.name}: parseStaffs from multistaffs.json and preprocessed.png gives voices.json`,
      async () => {
        const golden = goldenPageOf(fixture);
        const cv = await testOpenCv();
        const store = await storeOn(CPU);
        try {
          const voices = await parseStaffs(
            cv,
            {
              decoder: await store.open("decoder"),
              encoder: await store.open("encoder"),
            },
            golden.multiStaffs(),
            golden.preprocessed()
          );
          expect(voices.map((v) => v.map(heads))).toEqual(
            golden.voices().map((v) => v.map(heads))
          );
        } finally {
          await store.close();
        }
      },
      STAFF_TIMEOUT_MS
    );
  }
});

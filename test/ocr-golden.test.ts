/**
 * The chord strips and the title against the oracle: every RapidOCR stage of
 * every strip against ocr.json (tools/dump-ocr.py), the page's texts against
 * texts.json, which the app server's own code wrote (tools/dump-texts.sh),
 * and the title against title.json (tools/dump-golden.py).
 */

import { afterAll, beforeAll, expect, it } from "vitest";
import type { ModelStore } from "../src/models/store.js";
import { detectTitle, readStripTexts, titleCrop } from "../src/ocr/page.js";
import { RapidOcr } from "../src/ocr/rapid-ocr.js";
import { chordStrip, cropStrip } from "../src/ocr/strips.js";
import { staffBoxes } from "../src/pipeline/staff-positions.js";
import type { PageText } from "../src/result.js";
import {
  fixtureImageOf,
  goldenPageOf,
  listGoldenFixtures,
  readerFor,
} from "./support/golden.js";
import {
  CPU,
  describeWithModels,
  required,
  storeOn,
} from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

/**
 * Score drift the port is allowed. The oracle itself moves by up to 0.002
 * between two native OpenCV builds on these pages, and the port, which
 * resizes with opencv.js, a third build, moves by up to 0.006, on the
 * least confident read of the chord page (docs/decisions.tsv, phase 11).
 */
const SCORE_TOLERANCE = 0.01;
/** Page-normalised box drift: one pixel of a 2481-px-wide page is 4.0e-4, of a 3509-px-high one 2.8e-4, and the oracle rounds to 1e-4. */
const BOX_TOLERANCE = 5e-4;

type Corner = readonly [number, number];
interface StripStage {
  readonly bottom: number;
  readonly cls?: readonly (readonly [string, number])[];
  readonly det: { readonly boxes: readonly (readonly Corner[])[] };
  readonly final: { readonly txts: readonly string[] };
  readonly line: number;
  readonly paddingTop: number;
  readonly preprocess: {
    readonly height: number;
    readonly ratioH: number;
    readonly ratioW: number;
    readonly width: number;
  };
  readonly rec?: readonly (readonly [string, number])[];
  readonly top: number;
  readonly x0: number;
  readonly x1: number;
}

let store: ModelStore;
let ocr: RapidOcr;

describeWithModels("chord and title OCR against the server's oracle", () => {
  beforeAll(async () => {
    store = await storeOn(CPU);
    ocr = new RapidOcr(await testOpenCv(), {
      classify: await store.open("ocrClassify"),
      detect: await store.open("ocrDetect"),
      recognize: await store.open("ocrRecognize"),
    });
  }, 120_000);
  afterAll(async () => {
    await store.close();
  });

  for (const fixture of listGoldenFixtures()) {
    const reader = readerFor(fixture);
    const page = goldenPageOf(fixture);
    const { strips } = JSON.parse(reader.text("ocr.json")) as {
      strips: readonly StripStage[];
    };
    const positions = page.staffPositions();
    const staves = staffBoxes(positions);
    const staffOfLine = (line: number): number =>
      required(
        staves.find(
          (s) => s.cy === required(positions[line], `line ${line}`).cy
        ),
        `the staff of line ${line}`
      ).index;

    it(`${fixture.name}: every RapidOCR stage of every strip`, async () => {
      const image = fixtureImageOf(fixture);
      let worst = 0;
      for (const want of strips) {
        const staff = required(staves[staffOfLine(want.line)], "staff");
        const strip = chordStrip(staff, image);
        expect([strip.top, strip.bottom, strip.x0, strip.x1]).toEqual([
          want.top,
          want.bottom,
          want.x0,
          want.x1,
        ]);
        const { lines, trace } = await ocr.trace(cropStrip(image, strip));
        const at = `line ${want.line}`;
        expect(
          [trace.preprocessed.image.height, trace.preprocessed.image.width],
          at
        ).toEqual([want.preprocess.height, want.preprocess.width]);
        expect(trace.preprocessed.ratioH, at).toBe(want.preprocess.ratioH);
        expect(trace.preprocessed.ratioW, at).toBe(want.preprocess.ratioW);
        expect(trace.paddingTop, at).toBe(want.paddingTop);
        expect(
          trace.boxes.map((box) => box.map(([x, y]) => [x, y])),
          `${at} detection boxes`
        ).toEqual(want.det.boxes);
        expect(
          trace.labels.map(([label]) => label),
          at
        ).toEqual((want.cls ?? []).map(([label]) => label));
        const rec = want.rec ?? [];
        expect(
          trace.recognized.map((r) => r.text),
          at
        ).toEqual(rec.map(([text]) => text));
        for (const [i, r] of trace.recognized.entries()) {
          worst = Math.max(worst, Math.abs(r.score - (rec[i]?.[1] ?? 0)));
          expect(
            Math.abs(r.score - (rec[i]?.[1] ?? Number.NaN)),
            `${at} score ${i}`
          ).toBeLessThanOrEqual(SCORE_TOLERANCE);
        }
        expect(
          lines.map((line) => line.text),
          at
        ).toEqual(want.final.txts);
      }
      process.stdout.write(
        `ocr stages ${fixture.name}: ${strips.length} strips, worst recognition score ${worst.toFixed(5)} against ocr.json\n`
      );
    }, 300_000);

    it(`${fixture.name}: the texts the server's route answers`, async () => {
      const started = performance.now();
      const got = await readStripTexts(ocr, fixtureImageOf(fixture), staves);
      const ms = performance.now() - started;
      const want = JSON.parse(reader.text("texts.json")) as PageText[];
      expect(got.map((t) => [t.staff, t.text])).toEqual(
        want.map((t) => [t.staff, t.text])
      );
      let box = 0;
      let score = 0;
      const moved: string[] = [];
      for (const [i, text] of got.entries()) {
        const other = required(want[i], `texts.json[${i}]`);
        score = Math.max(score, Math.abs(text.score - other.score));
        if (text.score !== other.score) {
          moved.push(`${text.text} ${text.score} (${other.score})`);
        }
        for (const key of ["x0", "y0", "x1", "y1"] as const) {
          box = Math.max(box, Math.abs(text[key] - other[key]));
        }
      }
      process.stdout.write(
        `ocr ${fixture.name}: ${got.length} texts in ${staves.length} strips, ${ms.toFixed(0)} ms, worst box ${box.toExponential(1)}, worst score ${score.toFixed(3)}; scores off: ${moved.join(", ") || "none"}\n`
      );
      expect(box).toBeLessThanOrEqual(BOX_TOLERANCE);
      expect(score).toBeLessThanOrEqual(SCORE_TOLERANCE);
    }, 300_000);

    it(`${fixture.name}: homr's title`, async () => {
      const want = JSON.parse(reader.text("title.json")) as {
        crop: { height: number; width: number; x: number; y: number };
        title: string;
      };
      const top = required(page.staffs()[0], "a first staff");
      const crop = titleCrop(page.resized(), top);
      expect([crop.width, crop.height]).toEqual([
        want.crop.width,
        want.crop.height,
      ]);
      expect(await detectTitle(ocr, page.resized(), top)).toBe(want.title);
    }, 300_000);
  }
});

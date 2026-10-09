/**
 * The tab reader on pages we typeset ourselves (tools/typeset-tab.mjs, truth
 * in test/fixtures/tab/truth.json): every event, string and fret exactly, the
 * technique letters as annotations, the pull-off arcs and the TAB clef never
 * read as frets.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { expect, it } from "vitest";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { detectTablature, type ReadCrops } from "../src/tab/detect.js";
import { midiOfPitch, pitchTab } from "../src/tab/pitch.js";
import {
  readTablature,
  type TabNote,
  type TabTechnique,
} from "../src/tab/read.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const FIXTURES = join(import.meta.dirname, "fixtures", "tab");
const PAGE_TIMEOUT_MS = 120_000;

interface TruthSystem {
  readonly annotations: readonly {
    readonly string: number;
    readonly technique: TabTechnique;
    readonly x: number;
  }[];
  readonly events: readonly { readonly notes: TabNote[]; readonly x: number }[];
  readonly lines: number;
  readonly spacing: number;
}

interface TruthPage {
  readonly capo: number;
  readonly file: string;
  readonly height: number;
  readonly systems: readonly TruthSystem[];
  readonly tuning: readonly string[];
  readonly width: number;
}

const truth: { readonly pages: readonly TruthPage[] } = JSON.parse(
  readFileSync(join(FIXTURES, "truth.json"), "utf8")
);

function pageOf(file: string): ColorImage {
  const png = PNG.sync.read(readFileSync(join(FIXTURES, file)));
  return colorImageFromRgba(png.width, png.height, png.data);
}

describeWithModels("readTablature on typeset tab pages", () => {
  for (const page of truth.pages) {
    it(
      `${page.file}: ${page.systems.length} systems, ${page.systems.reduce((n, s) => n + s.events.length, 0)} events, every fret`,
      async () => {
        const cv = await testOpenCv();
        const store = await storeOn(CPU);
        try {
          const session = await store.open("ocrRecognize");
          const characters = ctcCharacters(
            session.metadata.get("character") ?? ""
          );
          const read: ReadCrops = async (crops) =>
            (await recognizeCrops(cv, session, characters, crops)).map(
              (r) => r.text
            );
          const image = pageOf(page.file);
          const readings = await readTablature(
            cv,
            image,
            await detectTablature(cv, image, read),
            read
          );

          expect(readings.map((r) => r.lines)).toEqual(
            page.systems.map((s) => s.lines)
          );
          for (const [k, system] of page.systems.entries()) {
            const reading = readings[k];
            const near = system.spacing / page.width;
            expect(
              reading?.events.map((e) => e.notes),
              `system ${k + 1} events`
            ).toEqual(system.events.map((e) => e.notes));
            for (const [i, event] of system.events.entries()) {
              expect(
                Math.abs((reading?.events[i]?.x ?? 0) - event.x / page.width),
                `system ${k + 1} event ${i + 1} x`
              ).toBeLessThan(near);
            }
            expect(
              reading?.annotations.map(({ string, technique }) => ({
                string,
                technique,
              })),
              `system ${k + 1} annotations`
            ).toEqual(
              system.annotations.map(({ string, technique }) => ({
                string,
                technique,
              }))
            );
            const strings = page.tuning.map(midiOfPitch);
            const pitched = pitchTab(reading ?? { events: [], lines: 4 }, {
              capo: page.capo,
              strings,
            });
            expect(pitched.flatMap((e) => e.notes.map((n) => n.midi))).toEqual(
              system.events.flatMap((e) =>
                e.notes.map(
                  (n) => (strings[n.string - 1] as number) + page.capo + n.fret
                )
              )
            );
          }
        } finally {
          await store.close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

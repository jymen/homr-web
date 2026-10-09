/**
 * Tuning and capo read off pages we typeset ourselves (tools/typeset-tab.mjs,
 * `headers` in test/fixtures/tab/truth.json): a subtitle, a line above the
 * system at the left or right, a label turned a quarter in the margin, one
 * string name per line, a line under the system; English and French, banjo,
 * guitar and mandolin, a tuning for the wrong string count, an unknown name.
 * The three fret pages print neither, and must read as neither.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import { beforeAll, expect, it } from "vitest";
import type { OpenCv } from "../src/cv/opencv.js";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { RapidOcr, recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { detectTablature, type ReadCrops } from "../src/tab/detect.js";
import { pitchTab } from "../src/tab/pitch.js";
import { readTabText, type TabText } from "../src/tab/text.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const FIXTURES = join(import.meta.dirname, "fixtures", "tab");
const PAGE_TIMEOUT_MS = 120_000;

interface HeaderTruth {
  readonly expect: {
    readonly capo: number | null;
    readonly status?: "string_count" | "unknown_name";
    readonly strings: readonly string[] | null;
  };
  readonly file: string;
  readonly lines: number;
  readonly texts: readonly { readonly at: string; readonly text: string }[];
}

const truth: {
  readonly headers: readonly HeaderTruth[];
  readonly pages: readonly { readonly file: string }[];
} = JSON.parse(readFileSync(join(FIXTURES, "truth.json"), "utf8"));

function pageOf(file: string): ColorImage {
  const png = PNG.sync.read(readFileSync(join(FIXTURES, file)));
  return colorImageFromRgba(png.width, png.height, png.data);
}

describeWithModels("readTabText on typeset tab pages", () => {
  let cv: OpenCv;
  let ocr: RapidOcr;
  let read: ReadCrops;
  let close: () => Promise<void>;

  beforeAll(async () => {
    cv = await testOpenCv();
    const store = await storeOn(CPU);
    const recognize = await store.open("ocrRecognize");
    const characters = ctcCharacters(recognize.metadata.get("character") ?? "");
    ocr = new RapidOcr(cv, {
      classify: await store.open("ocrClassify"),
      detect: await store.open("ocrDetect"),
      recognize,
    });
    read = async (crops) =>
      (await recognizeCrops(cv, recognize, characters, crops)).map(
        (r) => r.text
      );
    close = () => store.close();
    return () => close();
  }, PAGE_TIMEOUT_MS);

  async function textsOf(file: string) {
    const page = pageOf(file);
    const tabs = await detectTablature(cv, page, read);
    const texts = await readTabText(
      (image, minScore, limit) => ocr.read(image, minScore, limit),
      page,
      tabs
    );
    return { tabs, texts };
  }

  for (const header of truth.headers) {
    const { capo, status, strings } = header.expect;
    it(
      `${header.file}: ${header.texts.map((t) => `${JSON.stringify(t.text)} (${t.at})`).join(", ")}`,
      async () => {
        const { tabs, texts } = await textsOf(header.file);
        expect(tabs.map((t) => t.system.lines)).toEqual([header.lines]);
        const [text] = texts as [TabText];
        expect(text.capo?.fret ?? null, "capo").toBe(capo);
        if (status === "unknown_name") {
          expect(text.tuning).toMatchObject({ status });
          return;
        }
        expect(text.tuning, "tuning").toMatchObject({
          status: status ?? "read",
          strings,
        });
        expect(text.tuning?.confidence).toBeGreaterThan(0.5);
        if (status === undefined) {
          const pitched = pitchTab({
            ...text,
            events: [{ notes: [{ fret: 0, string: 1 }], x: 0 }],
            lines: tabs[0]?.system.lines ?? 4,
          });
          expect(pitched[0]?.notes[0]?.midi).toBeGreaterThan(40);
        }
      },
      PAGE_TIMEOUT_MS
    );
  }

  for (const { file } of truth.pages) {
    it(
      `${file}: no tuning and no capo printed, none read`,
      async () => {
        const { tabs, texts } = await textsOf(file);
        expect(tabs.length).toBeGreaterThan(0);
        expect(texts).toEqual(tabs.map(() => ({})));
      },
      PAGE_TIMEOUT_MS
    );
  }
});

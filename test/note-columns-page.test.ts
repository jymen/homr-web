/**
 * Note columns on a real page: on the Kesh fixture every staff's decoded
 * note count equals its notehead column count and the k-th note's pitch
 * sits on the k-th column (docs/design/button-tab-findings.md, Method), so
 * pairing the k-th note with the k-th column is the truth, and the
 * alignment must find exactly that.
 *
 * NOTE_COLUMN_PAGES, a ":"-separated list of PNG paths, adds pages that may
 * not be committed (the findings' book photo is one): each is checked the
 * same way, and named in the output by its index only.
 */

import { readFileSync } from "node:fs";
import { PNG } from "pngjs";
import { expect, it } from "vitest";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { createInputPredictions } from "../src/model/pipeline.js";
import { symbolsOfKind } from "../src/model/symbols.js";
import { detectStaffsInImage } from "../src/pipeline/detect.js";
import {
  alignNoteEvents,
  groupNoteheadColumns,
  noteEventsOfStaff,
  pitchCost,
  withNoteColumns,
} from "../src/pipeline/note-columns.js";
import { parseStaffCanvas } from "../src/pipeline/parse-staffs.js";
import {
  ensureSameNumberOfStaffs,
  prepareStaffImageAndPageX,
  staffRegions,
} from "../src/pipeline/staff-image.js";
import { preprocessPage } from "../src/segmentation/preprocess.js";
import {
  DEFAULT_SEGNET_BATCH,
  segmentPage,
} from "../src/segmentation/segment.js";
import { fixtureImageOf, listGoldenFixtures } from "./support/golden.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const PAGE_TIMEOUT_MS = 900_000;

interface Page {
  readonly image: () => ColorImage;
  readonly name: string;
}

const pngPage = (path: string): ColorImage => {
  const png = PNG.sync.read(readFileSync(path));
  return colorImageFromRgba(png.width, png.height, png.data);
};

const PAGES: readonly Page[] = [
  ...listGoldenFixtures()
    .filter((fixture) => fixture.name === "the-kesh-300dpi")
    .map((fixture) => ({
      image: () => fixtureImageOf(fixture),
      name: fixture.name,
    })),
  ...(process.env.NOTE_COLUMN_PAGES ?? "")
    .split(":")
    .filter((path) => path !== "")
    .map((path, k) => ({
      image: () => pngPage(path),
      name: `local page ${k + 1}`,
    })),
];

describeWithModels("note columns on a real page", () => {
  for (const page of PAGES) {
    it(
      `${page.name}: the k-th note on the k-th notehead column, on every staff`,
      async () => {
        const cv = await testOpenCv();
        const store = await storeOn(CPU);
        try {
          const { preprocessed, resized } = await preprocessPage(
            page.image(),
            cv
          );
          const segnet = await store.open("segnet", {
            batch: DEFAULT_SEGNET_BATCH,
          });
          const { masks } = await segmentPage(segnet, preprocessed, {
            batch: DEFAULT_SEGNET_BATCH,
          });
          const detection = detectStaffsInImage(
            cv,
            createInputPredictions(resized, preprocessed, masks)
          );
          const sessions = {
            decoder: await store.open("decoder"),
            encoder: await store.open("encoder"),
          };
          const gray = detection.preprocessed;
          const systems = ensureSameNumberOfStaffs(
            detection.multiStaffs,
            gray.height
          );
          const regions = staffRegions(systems);
          const lines: string[] = [];
          for (const system of systems) {
            for (const staff of system.staffs) {
              const { canvas, toPageX } = prepareStaffImageAndPageX(
                cv,
                staff,
                gray,
                regions
              );
              // biome-ignore lint/performance/noAwaitInLoops: one staff at a time, the sessions share one arena
              const symbols = await parseStaffCanvas(sessions, canvas);
              const columns = groupNoteheadColumns(
                symbolsOfKind(staff.symbols, "note").map((note) => ({
                  position: note.position,
                  x: note.center.x,
                })),
                staff.averageUnitSize
              );
              const { events } = noteEventsOfStaff(symbols, toPageX);
              const paired = alignNoteEvents(events, columns);
              const right = paired.filter(
                (j, k) =>
                  j === k &&
                  pitchCost(
                    events[k] ?? { positions: [], x: null },
                    columns[k] ?? { positions: [], x: 0 }
                  ) === 0
              ).length;
              lines.push(
                `${right}/${Math.max(events.length, columns.length)} (${events.length} events, ${columns.length} columns)`
              );
              expect(events.length).toBe(columns.length);
              expect(right).toBe(events.length);
              const placed = withNoteColumns(
                symbols,
                staff,
                toPageX,
                gray.width
              ).flatMap((s) => (s.column === undefined ? [] : [s.column]));
              for (const [k, x] of placed.entries()) {
                expect(x).toBeGreaterThanOrEqual(placed[k - 1] ?? 0);
                expect(x).toBeLessThan(1);
              }
            }
          }
          process.stdout.write(
            `note columns ${page.name}: ${lines.join(", ")}\n`
          );
        } finally {
          await store.close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

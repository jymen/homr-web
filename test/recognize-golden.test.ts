/**
 * Phase 9's end-to-end check on Node: each fixture page, from its own PNG,
 * through every stage of the port on WebAssembly, against homr's
 * page.musicxml and the app server's staves.json. Nothing here reads a
 * Python intermediate; this is the one test where a tolerance accepted in an
 * earlier stage could show up, and the MusicXML is compared exactly
 * (canonically) all the same.
 */

import { describe, expect, it } from "vitest";
import { ModelError } from "../src/models/errors.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import type { Progress } from "../src/result.js";
import { canonicalXml } from "./support/canonical-xml.js";
import {
  fixtureImageOf,
  listGoldenFixtures,
  readerFor,
} from "./support/golden.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

const PAGE_TIMEOUT_MS = 900_000;
/** Phase 5: the port's staffs agree with homr's to 2.5e-4 px from its own boxes; normalised by a 1920-px page that is far below 1e-6. */
const STAFF_DIGITS = 6;

describeWithModels("recognizePage from the fixture PNG", () => {
  for (const fixture of listGoldenFixtures()) {
    it(
      `${fixture.name}: homr's MusicXML and the server's staves`,
      async () => {
        const store = await storeOn(CPU);
        const stages: Progress[] = [];
        try {
          const result = await recognizePage(
            fixtureImageOf(fixture),
            {
              backend: "wasm",
              cv: await testOpenCv(),
              open: (role, batch) =>
                store.open(role, batch === undefined ? {} : { batch }),
            },
            { onProgress: (progress) => stages.push(progress) }
          );
          process.stdout.write(
            `recognizePage ${fixture.name}: ${result.durationMs} ms, ok ${result.ok}, log:\n${result.log}\n`
          );
          expect(result.ok).toBe(true);
          const reader = readerFor(fixture);
          expect(canonicalXml(result.musicXml)).toEqual(
            canonicalXml(reader.text("page.musicxml"))
          );
          const want = JSON.parse(reader.text("staves.json")) as {
            cx: number;
            cy: number;
            h: number;
            index: number;
            w: number;
          }[];
          expect(result.staves.map((s) => s.index)).toEqual(
            want.map((s) => s.index)
          );
          for (const [i, staff] of result.staves.entries()) {
            for (const key of ["cx", "cy", "w", "h"] as const) {
              expect(staff[key]).toBeCloseTo(
                want[i]?.[key] ?? Number.NaN,
                STAFF_DIGITS
              );
            }
          }
          expect(result.texts).toEqual([]);
          expect(stages.at(-1)).toEqual({ done: 1, stage: "xml", total: 1 });
          expect(new Set(stages.map((p) => p.stage))).toEqual(
            new Set(["segment", "detect", "dewarp", "staff", "xml"])
          );
        } finally {
          await store.close();
        }
      },
      PAGE_TIMEOUT_MS
    );
  }
});

describe("recognizePage's failures are results", () => {
  const engineOpening = (opened: string[]) => async () => ({
    backend: "wasm" as const,
    cv: await testOpenCv(),
    open: (role: string) => {
      opened.push(role);
      return Promise.reject(
        new ModelError("fetch-failed", `no ${role} in this test`)
      );
    },
  });
  const page = () => fixtureImageOf(firstFixture());

  it("a model that cannot be fetched is engine_missing, and nothing after it is opened", async () => {
    const opened: string[] = [];
    const result = await recognizePage(page(), await engineOpening(opened)());
    expect(result).toMatchObject({
      error: "engine_missing",
      musicXml: "",
      ok: false,
      staves: [],
    });
    expect(result.log).toContain("no segnet in this test");
    expect(opened).toEqual(["segnet"]);
  });

  it("an aborted signal is cancelled, or timeout for a TimeoutError, before any model", async () => {
    const opened: string[] = [];
    const engine = await engineOpening(opened)();
    const cancelled = new AbortController();
    cancelled.abort();
    expect(
      (await recognizePage(page(), engine, { signal: cancelled.signal })).error
    ).toBe("cancelled");
    const timedOut = AbortSignal.abort(
      new DOMException("late", "TimeoutError")
    );
    expect(
      (await recognizePage(page(), engine, { signal: timedOut })).error
    ).toBe("timeout");
    expect(opened).toEqual([]);
  });

  it("hands the page's signal to every model it opens", async () => {
    const signals: (AbortSignal | undefined)[] = [];
    const controller = new AbortController();
    await recognizePage(
      page(),
      {
        backend: "wasm",
        cv: await testOpenCv(),
        open: (_role, _batch, signal) => {
          signals.push(signal);
          controller.abort();
          return Promise.reject(new ModelError("fetch-failed", "aborted"));
        },
      },
      { signal: controller.signal }
    );
    expect(signals).toEqual([controller.signal]);
  });
});

function firstFixture() {
  const [first] = listGoldenFixtures();
  if (first === undefined) {
    throw new Error("the test needs a golden fixture");
  }
  return first;
}

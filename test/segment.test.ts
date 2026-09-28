/**
 * Phase 3's own criterion: the five masks this port computes against the five
 * `tools/dump-golden.py` dumped out of homr, per class, at testing.md's bar.
 *
 * Segnet is the slowest thing in the library on Node. Measured here over six
 * runs, 54 tiles of the Kesh page at a batch of 8 on the wasm provider with one
 * thread: **55 to 105 seconds**, 1.0 to 1.9 s a tile depending on what else the
 * machine is doing, plus half a second to open the session. The 159 ms a tile
 * recorded in `src/models/manifest.ts`'s comment does not reproduce and should be
 * read as unverified.
 *
 * So the whole page goes through segnet exactly once by default, in the golden
 * test, and everything else runs on a 1920x320 band of the fixture: six tiles
 * instead of 54. `resize_image` is a no-op at exactly the target width, so a
 * 1920-wide band reaches segnet at its own height, and anything *narrower* is
 * upscaled to 1920 and costs more tiles rather than fewer. Six is the floor for
 * any page that goes through preprocess.
 */

import {
  MessageChannel,
  type MessagePort as NodeMessagePort,
} from "node:worker_threads";
import { describe, expect, it } from "vitest";
import {
  cropPlane,
  planeAgreement,
  planeFromBytes,
} from "../src/image/plane.js";
import {
  MASK_CLASS_NAMES,
  type SegmentationMasks,
} from "../src/model/pipeline.js";
import { memoryCache } from "../src/models/cache.js";
import { ModelError } from "../src/models/errors.js";
import { segmentPage } from "../src/segmentation/segment.js";
import { tileGrid } from "../src/segmentation/tiles.js";
import {
  type SegmentationPort,
  SegmentationWorker,
  type SegmentationWorkerOptions,
  serveSegmentation,
  type WorkerEvent,
} from "../src/segmentation/worker.js";
import {
  fixtureImageOf,
  type GoldenFixture,
  goldenPageOf,
  listGoldenFixtures,
} from "./support/golden.js";
import {
  CPU,
  describeWithModels,
  FP16_ON_WASM,
  localModels,
  required,
  storeOn,
} from "./support/models.js";
import { nodeOpenCvSource } from "./support/opencv.js";

/** homr's own batch_size, and what the segnet sessions below are opened with. */
const BATCH = 8;
/** testing.md's mask criterion, per class. */
const MASK_AGREEMENT = 0.999;
const PAGE_TIMEOUT_MS = 600_000;
/** Six tiles at 2 s each, plus the session open, plus preprocess. Comfortably inside this and nowhere near the shared 30 s budget. */
const BAND_TIMEOUT_MS = 180_000;
/** A 1920-wide band of the fixture, in the *source* page's coordinates: two rows of ink from the middle of the Kesh page. */
const BAND = { bottom: 1520, left: 280, right: 2200, top: 1200 } as const;
const BAND_WIDTH = BAND.right - BAND.left;
const BAND_HEIGHT = BAND.bottom - BAND.top;
/** A contested band proves the wiring; a blank one would agree with anything. */
const MIN_CLASSES = 2;
const IS_CLOSED = /is closed/;

/** Straight to stdout rather than through vitest's per-test console buffer, so the figures appear in the order they were measured. */
const report = (line: string): void => {
  process.stdout.write(`${line}\n`);
};

const firstFixture = (): GoldenFixture =>
  required(listGoldenFixtures()[0], "a golden fixture to segment");

const bandOf = (fixture: GoldenFixture) =>
  cropPlane(
    fixtureImageOf(fixture),
    BAND.left,
    BAND.top,
    BAND.right,
    BAND.bottom
  );

const workerOn = (
  extra: Partial<SegmentationWorkerOptions> = {}
): SegmentationWorker =>
  new SegmentationWorker({
    baseUrl: "file:///models/",
    cache: memoryCache(),
    fetchBytes: localModels(),
    maxBackend: "wasm",
    openCv: nodeOpenCvSource(),
    ...extra,
  });

/**
 * Every per-class agreement printed before any of them is asserted, so the five
 * numbers are on the record whether they pass or fail. A boolean tells the next
 * reader that the masks were once good enough and nothing about how close they
 * were.
 */
function reportAgreements(
  label: string,
  fixture: GoldenFixture,
  masks: SegmentationMasks
): number[] {
  const golden = goldenPageOf(fixture);
  const agreements: number[] = [];
  for (const name of MASK_CLASS_NAMES) {
    const mine = masks[name];
    const agreement = planeAgreement(mine, golden.mask(name));
    agreements.push(agreement);
    const wrong = Math.round((1 - agreement) * mine.data.length);
    report(
      `  ${label} ${name}: ${agreement.toFixed(9)} (${wrong} px of ${mine.data.length})`
    );
  }
  return agreements;
}

describeWithModels("segnet over the whole golden page", () => {
  it(
    "reproduces every Python mask from the Python preprocessed page",
    async () => {
      const fixture = firstFixture();
      const golden = goldenPageOf(fixture);
      // The *Python* stage output, never this port's own preprocess: phase 4 reads
      // the golden the same way, so a preprocess regression cannot hide here.
      const page = golden.preprocessed();
      const tiles = tileGrid(page.width, page.height).length;
      const store = await storeOn(CPU);
      try {
        const segnet = await store.open("segnet", { batch: BATCH });
        const began = performance.now();
        const result = await segmentPage(segnet, page, { batch: BATCH });
        const ms = performance.now() - began;
        report(
          `fp32 segnet on ${fixture.name}: ${page.width}x${page.height}, ${tiles} tiles, batch ${BATCH}, ${Math.round(ms)} ms (${Math.round(ms / tiles)} ms/tile)`
        );
        expect(result.width).toBe(page.width);
        expect(result.height).toBe(page.height);

        const agreements = reportAgreements("fp32", fixture, result.masks);
        for (const agreement of agreements) {
          expect(agreement).toBeGreaterThanOrEqual(MASK_AGREEMENT);
        }
      } finally {
        await store.close();
      }
    },
    PAGE_TIMEOUT_MS
  );
});

describeWithModels("the segmentation Worker", () => {
  it(
    "preprocesses and segments a BGR page, reporting both costs",
    async () => {
      const band = bandOf(firstFixture());
      expect(`${band.width}x${band.height}`).toBe(
        `${BAND_WIDTH}x${BAND_HEIGHT}`
      );
      const progress: string[] = [];
      const worker = workerOn({
        onProgress: (done, total) => {
          progress.push(`${done}/${total}`);
        },
      });
      try {
        const started = await worker.start();
        // Identity, not equality: an equal-but-new report is what a second open
        // would produce, which is the thing being ruled out.
        expect(await worker.start()).toBe(started);
        report(
          `worker: ${started.backend}, ${started.numThreads} thread(s), batch ${started.batch}, ${started.modelReason}`
        );
        report(`worker runtime reason: ${started.reason}`);

        const page = await worker.segment(band);
        report(
          `worker band ${page.result.width}x${page.result.height}: preprocess ${Math.round(page.preprocessMs)} ms, segment ${Math.round(page.segmentMs)} ms`
        );

        expect(page.result.width).toBe(BAND_WIDTH);
        expect(page.result.height).toBe(BAND_HEIGHT);
        for (const name of MASK_CLASS_NAMES) {
          expect(`${name} ${page.result.masks[name].width}`).toBe(
            `${name} ${BAND_WIDTH}`
          );
          expect(`${name} ${page.result.masks[name].height}`).toBe(
            `${name} ${BAND_HEIGHT}`
          );
        }
        expect(new Set(page.result.classes.data).size).toBeGreaterThanOrEqual(
          MIN_CLASSES
        );
        expect(page.preprocessMs).toBeGreaterThan(0);
        expect(page.segmentMs).toBeGreaterThan(0);
        expect(page.durationMs).toBeGreaterThanOrEqual(page.segmentMs);
        // The width and height assertions above are what make the band's own
        // dimensions the right input here: they prove autocrop left it alone.
        const bandTiles = tileGrid(BAND_WIDTH, BAND_HEIGHT).length;
        expect(progress.at(-1)).toBe(`${bandTiles}/${bandTiles}`);
      } finally {
        await worker.close();
      }
    },
    BAND_TIMEOUT_MS
  );
});

/**
 * The browser half of the transport, proved at compile time because nothing on
 * Node exercises it: a worker script's `serveSegmentation(self, worker)` and a
 * page's `serveSegmentation(channel.port1, worker)` both have to type-check.
 *
 * It is not a formality. A `DedicatedWorkerGlobalScope` was *not* assignable to
 * `SegmentationPort` while its `postMessage` took an optional transfer list,
 * because the DOM's two-argument overload declares `transfer: Transferable[]`
 * with no `undefined` in it, while `@types/node`'s takes `transferList?`. So the
 * Node tests below passed green over a browser entry that could not compile, and
 * these two lines are what caught it.
 */
type FitsPort<T> = T extends SegmentationPort ? true : false;
const BROWSER_PORTS_FIT: [
  FitsPort<DedicatedWorkerGlobalScope>,
  FitsPort<MessagePort>,
] = [true, true];

/** A `model` or `progress` event: a notification about the worker rather than a reply to one command, so it carries no correlation id. */
const isNotification = (data: unknown): boolean =>
  typeof data === "object" &&
  data !== null &&
  "kind" in data &&
  (data.kind === "model" || data.kind === "progress");

/**
 * The next *reply* on this port, collecting into `seen` every notification it
 * steps over. Node's `once` hands the deserialized value straight over, so there
 * is nothing to narrow that the assertions do not narrow.
 *
 * The skipping is not tidiness. Notifications carry no id, so a reply assertion
 * that did not step over them would read whichever arrived first, and the store's
 * `opened` event arrives in the middle of `start`. Collecting them here is also
 * the only coverage the notification arms of `WorkerEvent` get.
 */
function nextReply(port: NodeMessagePort, seen: unknown[]): Promise<unknown> {
  return new Promise((resolve) => {
    const take = (data: unknown): void => {
      if (isNotification(data)) {
        seen.push(data);
        port.once("message", take);
        return;
      }
      resolve(data);
    };
    port.once("message", take);
  });
}

describeWithModels("the segmentation transport", () => {
  it(
    "answers each command with its own id, and names one it cannot serve",
    async () => {
      const band = bandOf(firstFixture());
      const { port1, port2 } = new MessageChannel();
      const notified: unknown[] = [];
      const worker = workerOn({
        onEvent: (event) => {
          port2.postMessage({ event, kind: "model" } satisfies WorkerEvent, []);
        },
      });
      const detach = serveSegmentation(port2, worker);
      try {
        port1.postMessage({ id: 7, kind: "start" });
        expect(await nextReply(port1, notified)).toMatchObject({
          id: 7,
          kind: "started",
        });
        // The store's own events reached the port as notifications while the start
        // was in flight, which is what the `model` arm exists for.
        expect(notified.length).toBeGreaterThan(0);
        expect(notified).toContainEqual(
          expect.objectContaining({ kind: "model" })
        );

        port1.postMessage({ id: 8, kind: "segment", page: band });
        expect(await nextReply(port1, notified)).toMatchObject({
          id: 8,
          kind: "segmented",
          page: {
            result: {
              height: BAND_HEIGHT,
              masks: { staff: { height: BAND_HEIGHT, width: BAND_WIDTH } },
              width: BAND_WIDTH,
            },
          },
        });

        port1.postMessage({ id: 9, kind: "enhance" });
        expect(await nextReply(port1, notified)).toEqual({
          error: 'a segmentation worker cannot serve the command "enhance"',
          id: 9,
          kind: "failed",
        });

        port1.postMessage({ id: 10, kind: "segment", page: "a jpeg" });
        expect(await nextReply(port1, notified)).toEqual({
          error:
            "a segmentation worker cannot serve a segment command whose page is not a BGR image",
          id: 10,
          kind: "failed",
        });

        port1.postMessage({ kind: "start" });
        expect(await nextReply(port1, notified)).toEqual({
          error:
            "a segmentation worker cannot serve a message with no integer id (members: kind)",
          id: null,
          kind: "failed",
        });

        port1.postMessage({ id: 11, kind: "close" });
        expect(await nextReply(port1, notified)).toEqual({
          id: 11,
          kind: "closed",
        });
      } finally {
        detach();
        await worker.close();
        port1.close();
        port2.close();
      }
    },
    BAND_TIMEOUT_MS
  );
});

describe("the transport's port type", () => {
  it("is satisfied by a browser worker global and a browser MessagePort", () => {
    // The work is the annotation above; this keeps the constant read and says so.
    expect(BROWSER_PORTS_FIT).toEqual([true, true]);
  });
});

describeWithModels("closing the segmentation Worker", () => {
  it("refuses the start it was closed during", async () => {
    const worker = workerOn();
    const starting = worker.start();
    const closing = worker.close();
    // Whichever of the runtime, the store and the download the close landed in,
    // the open is refused rather than left to finish into a worker nobody holds.
    await expect(starting).rejects.toThrow(ModelError);
    await expect(closing).resolves.toBeUndefined();
  });

  it("answers two concurrent closes and stays closed", async () => {
    const worker = workerOn();
    await worker.start();
    // Both must resolve. Before `#closing` held the one shutdown, the second call
    // saw the closed flag and returned while the first was still releasing.
    await expect(
      Promise.all([worker.close(), worker.close()])
    ).resolves.toEqual([undefined, undefined]);
    await expect(worker.start()).rejects.toThrow(IS_CLOSED);
  });
});

describe("a closed segmentation Worker", () => {
  const onePixelPage = () =>
    planeFromBytes("bgr", 1, 1, new Uint8Array([255, 255, 255]));

  it("closes twice without complaint, and then opens and segments nothing", async () => {
    const worker = workerOn();
    await worker.close();
    await worker.close();
    await expect(worker.start()).rejects.toThrow(ModelError);
    await expect(worker.segment(onePixelPage())).rejects.toThrow(IS_CLOSED);
  });
});

/**
 * The phase's real risk, and measurable here with no GPU: `Placement` splits the
 * artifacts from the provider so the fp16 files run on the wasm EP. It is two
 * more page runs, 135 to 180 s measured, so it is opt-in, and the skip says how
 * to run it because a silent skip proves nothing.
 */
function describeFp16Page(title: string, suite: () => void): void {
  if (process.env.HOMR_FP16_PAGE === "1") {
    describeWithModels(title, suite);
    return;
  }
  describe.skip(
    `${title} (set HOMR_FP16_PAGE=1 to run it: two more page runs, about 135 to 180 s)`,
    suite
  );
}

describeFp16Page("the fp16 segnet against the fp32 one, page-wide", () => {
  it(
    "agrees with the fp32 artifact and with the Python masks on every class",
    async () => {
      const fixture = firstFixture();
      const page = goldenPageOf(fixture).preprocessed();
      const fp32Store = await storeOn(CPU);
      const fp16Store = await storeOn(FP16_ON_WASM);
      try {
        const fp32 = await fp32Store.open("segnet", { batch: BATCH });
        const fp16 = await fp16Store.open("segnet", { batch: BATCH });
        expect(fp16.inputSpec("input").type).toBe("float16");

        const plain = await segmentPage(fp32, page, { batch: BATCH });
        const half = await segmentPage(fp16, page, { batch: BATCH });

        const classMaps = planeAgreement(plain.classes, half.classes);
        report(
          `fp16 vs fp32 class maps, whole page: ${classMaps.toFixed(9)} (${Math.round((1 - classMaps) * page.data.length)} px of ${page.data.length})`
        );
        // session.test.ts measured 0.999961 on the inkiest single tile of this
        // page. Page-wide, measured 2026-09-28: 0.999990, 51 px of 5.2 M, so that
        // tile was the worst case rather than a typical one.
        const agreements = [
          ...reportAgreements("fp32", fixture, plain.masks),
          ...reportAgreements("fp16", fixture, half.masks),
          classMaps,
        ];
        for (const agreement of agreements) {
          expect(agreement).toBeGreaterThanOrEqual(MASK_AGREEMENT);
        }
      } finally {
        await Promise.all([fp32Store.close(), fp16Store.close()]);
      }
    },
    PAGE_TIMEOUT_MS
  );
});

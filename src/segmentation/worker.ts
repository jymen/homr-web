/**
 * The segmentation Worker: one class that owns the runtime, the store, the
 * opencv handle and the one segnet session, and a message transport over it.
 *
 * The two are kept apart deliberately. `SegmentationWorker` is an ordinary
 * object a Node test constructs and calls, so the class is covered without a
 * Worker anywhere; `serveSegmentation` is the `postMessage` wiring, covered on
 * Node through `node:worker_threads`' `MessageChannel`.
 *
 * Two differences from the sketch in `docs/design/phase-2-models.md`:
 *
 * `segment` takes the **BGR page**, not an already-preprocessed `GrayImage`.
 * The point of a Worker is that the whole CPU-heavy job leaves the main thread,
 * and autocrop plus resize plus CLAHE is about 0.7 s of it. `segmentPage` in
 * segment.ts is the pure function on a gray page that the sketch was describing,
 * and it is what the golden test drives with the Python `preprocessed.png`.
 *
 * `start()` takes no argument and reads its configuration from the constructor,
 * so a second `start()` cannot ask for different settings than the first. The
 * runtime is one-shot global state and `startRuntime` already refuses a
 * disagreement; there is no reason to offer a caller a way to trip it.
 *
 * There is also no `self.addEventListener` here, and it is not an omission. An
 * auto-installing entry would have to answer a `start` command *carrying* the
 * configuration, which is what the sketch did and what the point above removed:
 * `baseUrl` and `cache` now reach the class through its constructor, so only the
 * worker script can build it. Importing this module therefore does nothing at
 * all, under Node or anywhere else, and a browser worker script's whole body is
 *
 * ```ts
 * const port = self;
 * const worker = new SegmentationWorker({
 *   baseUrl: "/models/",
 *   cache: await browserCache(),
 *   onEvent: (event) => { port.postMessage({ event, kind: "model" } satisfies WorkerEvent); },
 *   onProgress: (done, total) => { port.postMessage({ done, kind: "progress", total } satisfies WorkerEvent); },
 * });
 * serveSegmentation(port, worker);
 * ```
 */

import type { OpenCv, OpenCvSource } from "../cv/opencv.js";
import { loadOpenCv } from "../cv/opencv.js";
import type { ColorImage } from "../image/plane.js";
import { channelsOf, PLANE_KINDS } from "../image/plane.js";
import {
  MASK_CLASS_NAMES,
  type SegmentationResult,
} from "../model/pipeline.js";
import { startRuntime } from "../models/backend.js";
import type { ModelCache } from "../models/cache.js";
import { ModelError } from "../models/errors.js";
import type { ModelSession } from "../models/session.js";
import {
  type FetchBytes,
  type ModelEvent,
  ModelStore,
} from "../models/store.js";
import type { Backend } from "../result.js";
import { preprocessPage } from "./preprocess.js";
import { segmentPage } from "./segment.js";

/**
 * homr's own `extract(batch_size=8)`. The plan measures 16 and 32 on the bench,
 * which is a later unit: a bigger batch trades 1.2 MB of wasm arena per extra
 * tile against one fewer round trip, and only a measurement says where that
 * turns.
 */
export const DEFAULT_SEGNET_BATCH = 8;

export interface SegmentationWorkerOptions {
  readonly baseUrl: string;
  readonly batch?: number;
  readonly cache: ModelCache;
  readonly fetchBytes?: FetchBytes;
  readonly maxBackend?: Backend;
  readonly onEvent?: (event: ModelEvent) => void;
  readonly onProgress?: (done: number, total: number) => void;
  /** How opencv.js is acquired. Injected for the Node tests, where the library's default dynamic import never resolves under vitest. */
  readonly openCv?: OpenCvSource;
  readonly wasmPaths?: string;
}

/** What `start()` answers. The go/no-go gate quotes `backend`, `reason` and `modelReason` verbatim, and `numThreads` is what the runtime *applied*, which is the figure a timing is read against. */
export interface StartedReport {
  readonly backend: Backend;
  readonly batch: number;
  readonly modelReason: string;
  readonly numThreads: number;
  readonly reason: string;
}

export interface SegmentedPage {
  readonly durationMs: number;
  readonly preprocessMs: number;
  readonly result: SegmentationResult;
  readonly segmentMs: number;
}

/** Everything `start()` opened that a segment needs. The store is not in here: `close()` has to reach it while an open is still in flight. */
interface OpenWorker {
  readonly cv: OpenCv;
  readonly report: StartedReport;
  readonly segnet: ModelSession;
}

export class SegmentationWorker {
  readonly #batch: number;
  /** The one shutdown, so a second `close()` waits for the first instead of answering before anything has been released. */
  #closing: Promise<void> | undefined;
  #opening: Promise<OpenWorker> | undefined;
  readonly #options: SegmentationWorkerOptions;
  /**
   * Segments run one after another, chained here. The second reason is the
   * load-bearing one: a page's tiles already saturate one session and one wasm
   * arena, and `close()` must not release the session while a run holds it, which
   * frees the handle ort is reading rather than cancelling a job.
   */
  #queue: Promise<unknown> = Promise.resolve();
  /** Held outside `#opening` so `close()` can reach it mid-open: closing the store is what aborts a 57 MB model download nobody wants any more. */
  #store: ModelStore | undefined;

  constructor(options: SegmentationWorkerOptions) {
    const batch = options.batch ?? DEFAULT_SEGNET_BATCH;
    // Checked here and not on the first page, because `start()` reports the batch
    // to the go/no-go gate and would report a 0 as though it meant something.
    if (!Number.isInteger(batch) || batch < 1) {
      throw new ModelError(
        "bad-tuning",
        `a segnet batch is a positive integer, got ${batch}`,
        { actual: String(batch) }
      );
    }
    this.#batch = batch;
    this.#options = options;
  }

  /** Opens the runtime, the store and the segnet session. Idempotent, and safe to call twice concurrently. */
  async start(): Promise<StartedReport> {
    return (await this.#ready()).report;
  }

  /** Autocrop, resize, CLAHE, then segnet. Queued behind any segment already running. */
  async segment(page: ColorImage): Promise<SegmentedPage> {
    const open = await this.#ready();
    const run = this.#queue.then(() => this.#segmentOne(open, page));
    // The chain must not carry a rejection forward, or one failed page would fail
    // every page queued behind it.
    this.#queue = run.catch(() => undefined);
    return await run;
  }

  /**
   * Idempotent; releases the session and the store.
   *
   * A segment in flight owns the session, so a close waits for the page it is on.
   * A close during the model download does not wait: there is no session yet, so
   * the queue is idle and closing the store aborts the fetch.
   */
  async close(): Promise<void> {
    this.#closing ??= this.#shutdown();
    await this.#closing;
  }

  async #open(): Promise<OpenWorker> {
    const {
      baseUrl,
      cache,
      fetchBytes,
      maxBackend,
      onEvent,
      openCv,
      wasmPaths,
    } = this.#options;
    const runtime = await startRuntime({
      ...(maxBackend === undefined ? {} : { maxBackend }),
      ...(wasmPaths === undefined ? {} : { wasmPaths }),
    });
    const store = new ModelStore({
      baseUrl,
      cache,
      ...(fetchBytes === undefined ? {} : { fetchBytes }),
      ...(onEvent === undefined ? {} : { onEvent }),
      runtime,
    });
    this.#store = store;
    // `close()` can land while the runtime is starting, when there is no store for
    // it to reach; nothing else would then stop this from downloading into one.
    if (this.#closing !== undefined) {
      await store.close();
      throw new ModelError(
        "store-closed",
        "this SegmentationWorker was closed while its runtime was starting"
      );
    }
    // Sequential rather than concurrent: opencv.js is memoised per realm and owns
    // nothing releasable, so a session that fails to open after it leaks nothing,
    // whereas a Promise.all that half-failed would leave a store to close on an
    // error path.
    const cv = await loadOpenCv(openCv);
    const segnet = await store.open("segnet", { batch: this.#batch });
    return {
      cv,
      report: {
        backend: runtime.backend,
        batch: this.#batch,
        modelReason: segnet.plan.reason,
        numThreads: runtime.numThreads,
        reason: runtime.reason,
      },
      segnet,
    };
  }

  #ready(): Promise<OpenWorker> {
    if (this.#closing !== undefined) {
      throw new ModelError(
        "store-closed",
        "this SegmentationWorker is closed, so it can open nothing and segment nothing"
      );
    }
    const existing = this.#opening;
    if (existing !== undefined) {
      return existing;
    }
    const opening = this.#open();
    this.#opening = opening;
    return opening.catch((cause: unknown) => {
      if (this.#opening === opening) {
        this.#opening = undefined;
      }
      throw cause;
    });
  }

  /**
   * `durationMs` is `preprocessMs + segmentMs`, and all three are measured from
   * after `start()` resolved, so a first call that had to open the session does
   * not charge its half second to the page.
   */
  async #segmentOne(
    open: OpenWorker,
    page: ColorImage
  ): Promise<SegmentedPage> {
    const { onProgress } = this.#options;
    const began = performance.now();
    // preprocess also returns the resized colour page, homr's debug-overlay copy,
    // which nothing in this port reads yet; carrying it in SegmentedPage would put
    // 11 MB through a postMessage for no reader.
    const { preprocessed } = await preprocessPage(page, open.cv);
    const preprocessMs = performance.now() - began;
    const result = await segmentPage(open.segnet, preprocessed, {
      batch: this.#batch,
      ...(onProgress === undefined ? {} : { onProgress }),
    });
    const durationMs = performance.now() - began;
    return {
      durationMs,
      preprocessMs,
      result,
      segmentMs: durationMs - preprocessMs,
    };
  }

  async #shutdown(): Promise<void> {
    const opening = this.#opening;
    this.#opening = undefined;
    await this.#queue;
    await this.#store?.close();
    // The open, if there was one, has now either settled or been refused by the
    // closed store; either way its failure belongs to whoever asked for it.
    await opening?.catch(() => undefined);
  }
}

export type WorkerCommand =
  | { readonly id: number; readonly kind: "close" }
  | { readonly id: number; readonly kind: "segment"; readonly page: ColorImage }
  | { readonly id: number; readonly kind: "start" };

/**
 * A reply carries the id of the command it answers. `model` and `progress` are
 * notifications: they belong to the worker rather than to one command, so they
 * have no id. `failed`'s id is null when the message was too malformed to carry
 * one.
 */
export type WorkerEvent =
  | { readonly done: number; readonly kind: "progress"; readonly total: number }
  | {
      readonly error: string;
      readonly id: number | null;
      readonly kind: "failed";
    }
  | { readonly event: ModelEvent; readonly kind: "model" }
  | { readonly id: number; readonly kind: "closed" }
  | {
      readonly id: number;
      readonly kind: "segmented";
      readonly page: SegmentedPage;
    }
  | {
      readonly id: number;
      readonly kind: "started";
      readonly report: StartedReport;
    };

/**
 * The half of a port this transport uses. Declared here rather than taken as a
 * `MessagePort`, because a browser worker global, a browser `MessagePort` and
 * `node:worker_threads`' port are three unrelated types with these members. The
 * listener's argument is `unknown` for the same reason and because it is true: a
 * message event's `data` has to be narrowed whatever delivered it.
 */
export interface SegmentationPort {
  readonly addEventListener: (
    type: "message",
    listener: (event: unknown) => void
  ) => void;
  /**
   * `transfer` is required rather than optional, and mutable rather than
   * readonly, because those are the shapes that make a real port assignable to
   * this type. `DedicatedWorkerGlobalScope.postMessage`'s two-argument overload
   * declares `transfer: Transferable[]`, so an optional parameter here refuses a
   * worker global outright, and a `readonly ArrayBuffer[]` refuses it too.
   * Callers with nothing to transfer pass an empty array.
   */
  readonly postMessage: (message: unknown, transfer: ArrayBuffer[]) => void;
  readonly removeEventListener: (
    type: "message",
    listener: (event: unknown) => void
  ) => void;
  /** A `MessagePort` queues messages until this is called; a worker global has no such method. */
  readonly start?: () => void;
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

function isColorImage(value: unknown): value is ColorImage {
  if (!isRecord(value)) {
    return false;
  }
  const { channels, data, height, kind, width } = value;
  const bands = channelsOf(PLANE_KINDS.bgr);
  return (
    kind === PLANE_KINDS.bgr &&
    channels === bands &&
    typeof width === "number" &&
    typeof height === "number" &&
    Number.isInteger(width) &&
    Number.isInteger(height) &&
    width > 0 &&
    height > 0 &&
    data instanceof Uint8Array &&
    data.length === width * height * bands
  );
}

/**
 * A command, or why this is not one. The refusal carries the id when the message
 * had one, so a caller waiting on a correlation id still hears back, and it
 * carries the reason so the `failed` event says what arrived: an unrecognised
 * `kind`, a `segment` whose page is not an image and a message with no id are
 * three different mistakes, and one shared "malformed" would send whoever sent it
 * looking in the wrong place.
 */
type ParsedCommand =
  | { readonly command: WorkerCommand; readonly ok: true }
  | { readonly id: number | null; readonly ok: false; readonly why: string };

const membersOf = (data: Record<string, unknown>): string =>
  Object.keys(data).join(", ") || "none";

function parseCommand(data: unknown): ParsedCommand {
  if (!isRecord(data)) {
    return {
      id: null,
      ok: false,
      why: `a message that is ${data === null ? "null" : typeof data}`,
    };
  }
  const { id, kind, page } = data;
  if (typeof id !== "number" || !Number.isInteger(id)) {
    return {
      id: null,
      ok: false,
      why: `a message with no integer id (members: ${membersOf(data)})`,
    };
  }
  if (kind === "close" || kind === "start") {
    return { command: { id, kind }, ok: true };
  }
  if (kind === "segment") {
    return isColorImage(page)
      ? { command: { id, kind, page }, ok: true }
      : {
          id,
          ok: false,
          why: "a segment command whose page is not a BGR image",
        };
  }
  if (typeof kind === "string") {
    return { id, ok: false, why: `the command ${JSON.stringify(kind)}` };
  }
  return {
    id,
    ok: false,
    why:
      kind === undefined
        ? `a message with no kind (members: ${membersOf(data)})`
        : // JSON.stringify answers undefined for a function or a symbol, where the
          // type name is the only thing left to say.
          `a message whose kind is ${JSON.stringify(kind) ?? typeof kind}`,
  };
}

const messageOf = (cause: unknown): string =>
  cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);

/**
 * The result's six planes' buffers, for the transfer list. A transferred buffer
 * is **detached on the sending side**, which is what a Worker wants here: it is
 * done with the masks the moment it has posted them, and the alternative is a
 * 31 MB copy the page then waits for (six planes of a 1920x2716 page). A plane
 * over a `SharedArrayBuffer`
 * cannot be transferred and is left to be copied.
 */
function transferableBuffersOf(result: SegmentationResult): ArrayBuffer[] {
  const buffers = new Set<ArrayBuffer>();
  const planes = [
    result.classes,
    ...MASK_CLASS_NAMES.map((name) => result.masks[name]),
  ];
  for (const plane of planes) {
    const { buffer, byteLength, byteOffset } = plane.data;
    // Only a view that spans its whole buffer. Every plane in a SegmentationResult
    // is freshly allocated today, but `planeFromBytes` takes a caller's
    // Uint8Array, so a plane that was a subarray of a pooled buffer would
    // otherwise detach memory it does not own.
    if (
      buffer instanceof ArrayBuffer &&
      byteOffset === 0 &&
      byteLength === buffer.byteLength
    ) {
      buffers.add(buffer);
    }
  }
  return [...buffers];
}

const post = (
  port: SegmentationPort,
  event: WorkerEvent,
  transfer: ArrayBuffer[] = []
): void => {
  try {
    port.postMessage(event, transfer);
  } catch {
    // Swallowed on purpose. A closed or detached port is the only way this throws,
    // and there is then no channel left to report it on; the alternative is
    // throwing out of a message listener, into nothing.
  }
};

async function runCommand(
  port: SegmentationPort,
  worker: SegmentationWorker,
  command: WorkerCommand
): Promise<void> {
  if (command.kind === "start") {
    post(port, {
      id: command.id,
      kind: "started",
      report: await worker.start(),
    });
    return;
  }
  if (command.kind === "segment") {
    const page = await worker.segment(command.page);
    post(
      port,
      { id: command.id, kind: "segmented", page },
      transferableBuffersOf(page.result)
    );
    return;
  }
  await worker.close();
  post(port, { id: command.id, kind: "closed" });
}

/**
 * Serves `worker` over `port` until the returned function detaches the listener.
 *
 * Nothing here throws into the void: an unparseable message and a rejected
 * command both answer with a `failed` event, because a caller waiting on a
 * correlation id has no other way to learn that its command died.
 */
export function serveSegmentation(
  port: SegmentationPort,
  worker: SegmentationWorker
): () => void {
  const listener = (event: unknown): void => {
    const parsed = parseCommand(isRecord(event) ? event.data : undefined);
    if (!parsed.ok) {
      post(port, {
        error: `a segmentation worker cannot serve ${parsed.why}`,
        id: parsed.id,
        kind: "failed",
      });
      return;
    }
    const { command } = parsed;
    runCommand(port, worker, command).catch((cause: unknown) => {
      post(port, { error: messageOf(cause), id: command.id, kind: "failed" });
    });
  };
  port.addEventListener("message", listener);
  port.start?.();
  return () => {
    port.removeEventListener("message", listener);
  };
}

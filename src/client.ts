/**
 * The main-thread half: `createRecognizer` starts the Worker and answers a
 * `Recognizer` once the Worker has chosen its backend; `recognizePage` posts
 * one page and settles with the Worker's result. Nothing here throws after
 * `createRecognizer` resolves.
 */

import {
  type Backend,
  failedResult,
  isTimeoutAbort,
  type Progress,
  type RecognizeError,
  type RecognizeResult,
  type StaffBox,
} from "./result.js";
import {
  type HostCommand,
  type HostEvent,
  isPageInput,
  type PageInput,
  type PageTask,
  parseEvent,
  type RuntimeReport,
} from "./worker-protocol.js";

export type { PageInput } from "./worker-protocol.js";

export interface RecognizerOptions {
  /** Where the models are served: `${baseUrl}/${sha256}/${filename}`. Relative to the page. */
  readonly baseUrl: string | URL;
  /** For a bundler that cannot resolve `new URL("./worker.js", import.meta.url)`, and for tests. */
  readonly createWorker?: () => Worker;
  /** The best backend to try; the Worker steps down from it. Default "webgpu". */
  readonly prefer?: Backend;
  /** Where onnxruntime-web's .wasm and .mjs files are served. Default: onnxruntime-web's own resolution. Relative to the page. */
  readonly wasmPaths?: string | URL;
}

export interface TextOptions {
  /**
   * RapidOCR's text_score for the chord strips: a line scoring below it is
   * dropped. Default 0.5, the server's, so `texts` equals the server route's.
   * The AbcMusicStudio app keeps texts at 0.6 and above.
   */
  readonly minTextScore?: number;
  readonly onProgress?: (progress: Progress) => void;
  /** Aborting settles the page at once as `cancelled`, or `timeout` when the reason is a TimeoutError. */
  readonly signal?: AbortSignal;
}

export interface RecognizeOptions extends TextOptions {
  /** Read the chord strips and the title. Default true; false leaves `texts` empty and `work-title` blank, and never loads the OCR models but the recogniser, which the tab guard and reader need on a page with tablature or a five-line tablature candidate. */
  readonly ocr?: boolean;
}

const DEFAULT_MIN_TEXT_SCORE = 0.5;

export interface Recognizer {
  /** Chosen by the Worker before any model byte moved; fixed for this recognizer's life. */
  readonly backend: Backend;
  /** Why that backend, in one line. */
  readonly backendReason: string;
  /** Idempotent. Terminates the Worker; a page being read settles as `cancelled`. */
  dispose: () => Promise<void>;
  /** WebAssembly threads the runtime applied; 1 when the page is not cross-origin isolated. */
  readonly numThreads: number;
  /**
   * The chord strips above `staves` alone, as the server's route reads them:
   * `musicXml` is empty and `staves` are the ones given. Shares the one-page
   * rule and never rejects.
   */
  readTextStrips: (
    page: PageInput,
    staves: readonly StaffBox[],
    options?: TextOptions
  ) => Promise<RecognizeResult>;
  /**
   * Never rejects. One page at a time: a second call while one runs answers
   * `busy`. A page asked for right after a cancel is accepted and waits for
   * the Worker to finish the cancelled page's current step.
   */
  recognizePage: (
    page: PageInput,
    options?: RecognizeOptions
  ) => Promise<RecognizeResult>;
}

/** How long dispose waits for the Worker to release its sessions before terminating it. */
const CLOSE_GRACE_MS = 2000;
/** How long createRecognizer waits for the Worker to answer init: a WebGPU probe and opencv.js take about a second. */
const INIT_TIMEOUT_MS = 30_000;

const pageBase = (): string | undefined =>
  (globalThis as { document?: { baseURI?: string } }).document?.baseURI ??
  (globalThis as { location?: { href?: string } }).location?.href;

const absolute = (url: string | URL): string => new URL(url, pageBase()).href;

interface LiveJob {
  readonly id: number;
  readonly onProgress: ((progress: Progress) => void) | undefined;
  readonly settle: (result: RecognizeResult) => void;
}

class WorkerRecognizer implements Recognizer {
  readonly backend: Backend;
  readonly backendReason: string;
  readonly numThreads: number;
  #closed: (() => void) | undefined;
  #dead: string | undefined;
  #disposing: Promise<void> | undefined;
  /** The job whose result the Worker still owes, cancelled or not, and what resolves when it arrives. */
  #inWorker: { readonly id: number; readonly done: () => void } | undefined;
  #live: LiveJob | undefined;
  #nextId = 1;
  #workerIdle: Promise<void> = Promise.resolve();
  readonly #worker: Worker;

  constructor(worker: Worker, report: RuntimeReport) {
    this.backend = report.backend;
    this.backendReason = report.reason;
    this.numThreads = report.numThreads;
    this.#worker = worker;
    worker.addEventListener("message", (event) => this.#onMessage(event.data));
    worker.addEventListener("error", (event) =>
      this.#die(`the worker failed: ${event.message || "no message"}`)
    );
    worker.addEventListener("messageerror", () =>
      this.#die("the worker sent a message that could not be read")
    );
  }

  recognizePage(
    page: PageInput,
    options: RecognizeOptions = {}
  ): Promise<RecognizeResult> {
    return this.#read(page, options, {
      kind: "page",
      minTextScore: options.minTextScore ?? DEFAULT_MIN_TEXT_SCORE,
      ocr: options.ocr ?? true,
    });
  }

  readTextStrips(
    page: PageInput,
    staves: readonly StaffBox[],
    options: TextOptions = {}
  ): Promise<RecognizeResult> {
    return this.#read(page, options, {
      kind: "texts",
      minTextScore: options.minTextScore ?? DEFAULT_MIN_TEXT_SCORE,
      staves: staves.map(({ cx, cy, h, index, w }) => ({
        cx,
        cy,
        h,
        index,
        w,
      })),
    });
  }

  #read(
    page: PageInput,
    options: TextOptions,
    task: PageTask
  ): Promise<RecognizeResult> {
    const { onProgress, signal } = options;
    const fail = (
      error: Exclude<RecognizeError, "tablature_only">,
      log: string
    ) => Promise.resolve(failedResult(this.backend, error, log));
    if (this.#disposing !== undefined) {
      return fail("cancelled", "the recognizer was disposed");
    }
    if (this.#dead !== undefined) {
      return fail("worker_lost", this.#dead);
    }
    if (!isPageInput(page)) {
      return fail(
        "bad_input",
        "the page is not a Blob, ImageBitmap or ImageData"
      );
    }
    if (signal?.aborted) {
      return fail(
        isTimeoutAbort(signal) ? "timeout" : "cancelled",
        "aborted before it started"
      );
    }
    if (this.#live !== undefined) {
      return fail("busy", `page ${this.#live.id} is still being read`);
    }
    const id = this.#nextId;
    this.#nextId += 1;
    return new Promise((resolve) => {
      const onAbort = (): void => {
        const timeout = signal !== undefined && isTimeoutAbort(signal);
        this.#post({ id, kind: "cancel", timeout });
        this.#settle(
          id,
          failedResult(
            this.backend,
            timeout ? "timeout" : "cancelled",
            "aborted by the caller"
          )
        );
      };
      this.#live = {
        id,
        onProgress,
        settle: (result) => {
          signal?.removeEventListener("abort", onAbort);
          resolve(result);
        },
      };
      signal?.addEventListener("abort", onAbort, { once: true });
      // A cancelled page may still be running in the Worker; this one waits for it.
      this.#workerIdle
        .then(() => this.#start(id, page, task))
        .catch(() => undefined);
    });
  }

  dispose(): Promise<void> {
    this.#disposing ??= this.#shutdown();
    return this.#disposing;
  }

  #start(id: number, page: PageInput, task: PageTask): void {
    if (this.#live?.id !== id || this.#dead !== undefined) {
      return;
    }
    let done = (): void => undefined;
    this.#workerIdle = new Promise((resolve) => {
      done = resolve;
    });
    this.#inWorker = { done, id };
    this.#post({ id, kind: "recognize", page, task });
  }

  #settle(id: number, result: RecognizeResult): void {
    const live = this.#live;
    if (live?.id === id) {
      this.#live = undefined;
      live.settle(result);
    }
  }

  #onMessage(data: unknown): void {
    const parsed = parseEvent(data);
    if (!parsed.ok) {
      this.#die(`the worker sent ${parsed.why}`);
      return;
    }
    const event: HostEvent = parsed.value;
    if (event.kind === "progress") {
      if (this.#live?.id === event.id) {
        this.#live.onProgress?.(event.progress);
      }
    } else if (event.kind === "result") {
      if (this.#inWorker?.id === event.id) {
        this.#inWorker.done();
        this.#inWorker = undefined;
      }
      this.#settle(event.id, event.result);
    } else if (event.kind === "error") {
      this.#die(`protocol error from the worker: ${event.message}`);
    } else if (event.kind === "closed") {
      this.#closed?.();
    }
  }

  #post(command: HostCommand): void {
    this.#worker.postMessage(command);
  }

  /** The Worker cannot be trusted any more: the page in flight fails and every later call answers at once. */
  #die(why: string): void {
    if (this.#dead !== undefined) {
      return;
    }
    this.#dead = why;
    const live = this.#live;
    if (live !== undefined) {
      this.#settle(live.id, failedResult(this.backend, "worker_lost", why));
    }
    this.#inWorker?.done();
    this.#closed?.();
    this.#worker.terminate();
  }

  async #shutdown(): Promise<void> {
    const live = this.#live;
    if (live !== undefined) {
      this.#settle(
        live.id,
        failedResult(this.backend, "cancelled", "the recognizer was disposed")
      );
    }
    if (this.#dead === undefined) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const closed = new Promise<void>((resolve) => {
        this.#closed = resolve;
        timer = setTimeout(resolve, CLOSE_GRACE_MS);
      });
      this.#post({ kind: "close" });
      await closed;
      clearTimeout(timer);
    }
    this.#dead ??= "the recognizer was disposed";
    this.#worker.terminate();
  }
}

/** Waits for the Worker's answer to `init`; any other first answer is a failed start. */
function started(worker: Worker): Promise<RuntimeReport> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      cleanup();
      reject(
        new Error(
          `homr-web's worker did not answer init within ${INIT_TIMEOUT_MS / 1000} s`
        )
      );
    }, INIT_TIMEOUT_MS);
    const onMessage = (event: MessageEvent): void => {
      const parsed = parseEvent(event.data);
      cleanup();
      if (parsed.ok && parsed.value.kind === "ready") {
        resolve(parsed.value);
      } else if (parsed.ok && parsed.value.kind === "init-failed") {
        reject(new Error(`homr-web could not start: ${parsed.value.message}`));
      } else {
        reject(
          new Error(
            `homr-web's worker answered init with ${parsed.ok ? parsed.value.kind : parsed.why}`
          )
        );
      }
    };
    const onError = (event: ErrorEvent): void => {
      cleanup();
      reject(
        new Error(
          `homr-web's worker did not load: ${event.message || "no message"}`
        )
      );
    };
    const cleanup = (): void => {
      clearTimeout(timer);
      worker.removeEventListener("message", onMessage);
      worker.removeEventListener("error", onError);
    };
    worker.addEventListener("message", onMessage);
    worker.addEventListener("error", onError);
  });
}

/**
 * Starts a Worker, which probes WebGPU and cross-origin isolation, picks a
 * backend and loads OpenCV, and resolves once it has. No model is fetched
 * until the first page. Rejects when the Worker cannot start; the app's
 * answer to that is the server.
 */
export async function createRecognizer(
  options: RecognizerOptions
): Promise<Recognizer> {
  const init: HostCommand = {
    kind: "init",
    settings: {
      baseUrl: absolute(options.baseUrl),
      prefer: options.prefer ?? "webgpu",
      wasmPaths:
        options.wasmPaths === undefined ? null : absolute(options.wasmPaths),
    },
  };
  const worker =
    options.createWorker?.() ??
    new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  try {
    const ready = started(worker);
    worker.postMessage(init);
    return new WorkerRecognizer(worker, await ready);
  } catch (cause) {
    worker.terminate();
    throw cause;
  }
}

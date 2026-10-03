/**
 * The main-thread half: `createRecognizer` starts the Worker and answers a
 * `Recognizer` once the Worker has chosen its backend; `recognizePage` posts
 * one page and settles with the Worker's result. Nothing here throws after
 * `createRecognizer` resolves.
 */

import {
  type Backend,
  failedResult,
  type Progress,
  type RecognizeError,
  type RecognizeResult,
} from "./result.js";
import {
  type HostCommand,
  type HostEvent,
  isPageInput,
  type PageInput,
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

export interface RecognizeOptions {
  readonly onProgress?: (progress: Progress) => void;
  /** Aborting settles the page at once as `cancelled`, or `timeout` when the reason is a TimeoutError. */
  readonly signal?: AbortSignal;
}

export interface Recognizer {
  /** Chosen by the Worker before any model byte moved; fixed for this recognizer's life. */
  readonly backend: Backend;
  /** Why that backend, in one line. */
  readonly backendReason: string;
  /** Idempotent. Terminates the Worker; a page being read settles as `cancelled`. */
  dispose: () => Promise<void>;
  /** Never rejects. One page at a time: a second call while one runs answers `busy`. */
  recognizePage: (
    page: PageInput,
    options?: RecognizeOptions
  ) => Promise<RecognizeResult>;
}

/** How long dispose waits for the Worker to release its sessions before terminating it. */
const CLOSE_GRACE_MS = 2000;

const pageBase = (): string | undefined =>
  (globalThis as { document?: { baseURI?: string } }).document?.baseURI ??
  (globalThis as { location?: { href?: string } }).location?.href;

const absolute = (url: string | URL): string => new URL(url, pageBase()).href;

const isTimeout = (signal: AbortSignal): boolean =>
  signal.reason instanceof Error && signal.reason.name === "TimeoutError";

interface LiveJob {
  readonly id: number;
  readonly onProgress: ((progress: Progress) => void) | undefined;
  readonly settle: (result: RecognizeResult) => void;
}

class WorkerRecognizer implements Recognizer {
  readonly backend: Backend;
  readonly backendReason: string;
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
    const { onProgress, signal } = options;
    const fail = (error: RecognizeError, log: string) =>
      Promise.resolve(failedResult(this.backend, error, log));
    if (this.#dead !== undefined) {
      return fail(
        this.#disposing === undefined ? "engine_failed" : "cancelled",
        this.#dead
      );
    }
    if (!isPageInput(page)) {
      return fail(
        "bad_input",
        "the page is not a Blob, ImageBitmap or ImageData"
      );
    }
    if (signal?.aborted) {
      return fail(
        isTimeout(signal) ? "timeout" : "cancelled",
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
        const timeout = signal !== undefined && isTimeout(signal);
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
      this.#workerIdle.then(() => this.#start(id, page)).catch(() => undefined);
    });
  }

  dispose(): Promise<void> {
    this.#disposing ??= this.#shutdown();
    return this.#disposing;
  }

  #start(id: number, page: PageInput): void {
    if (this.#live?.id !== id || this.#dead !== undefined) {
      return;
    }
    let done = (): void => undefined;
    this.#workerIdle = new Promise((resolve) => {
      done = resolve;
    });
    this.#inWorker = { done, id };
    this.#post({ id, kind: "recognize", page });
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
      this.#settle(live.id, failedResult(this.backend, "engine_failed", why));
    }
    this.#inWorker?.done();
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
      const closed = new Promise<void>((resolve) => {
        this.#closed = resolve;
        setTimeout(resolve, CLOSE_GRACE_MS);
      });
      this.#post({ kind: "close" });
      await closed;
    }
    this.#dead ??= "the recognizer was disposed";
    this.#worker.terminate();
  }
}

/** Waits for the Worker's answer to `init`; any other first answer is a failed start. */
function started(worker: Worker): Promise<RuntimeReport> {
  return new Promise((resolve, reject) => {
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
  const worker =
    options.createWorker?.() ??
    new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
  const ready = started(worker);
  worker.postMessage({
    kind: "init",
    settings: {
      baseUrl: absolute(options.baseUrl),
      prefer: options.prefer ?? "webgpu",
      wasmPaths:
        options.wasmPaths === undefined ? null : absolute(options.wasmPaths),
    },
  } satisfies HostCommand);
  try {
    return new WorkerRecognizer(worker, await ready);
  } catch (cause) {
    worker.terminate();
    throw cause;
  }
}

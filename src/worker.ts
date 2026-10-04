/**
 * The Worker entry (`homr-web/worker`): a `WorkerHost` serving one engine,
 * one page at a time, over the Worker's own port.
 *
 * The host owns the protocol and nothing else. The engine (runtime, OpenCV,
 * model store, image decoding, the pipeline) is started by an injected
 * function, so a Node test drives the host with a scripted engine and no
 * model.
 */

import { loadOpenCv } from "./cv/opencv.js";
import { decodePage, PageInputError } from "./image/decode.js";
import { startRuntime } from "./models/backend.js";
import { browserCache } from "./models/cache.js";
import { type ModelEvent, ModelStore } from "./models/store.js";
import {
  describeError,
  type PipelineOptions,
  recognizePage,
  recognizeTexts,
} from "./pipeline/recognize.js";
import { failedResult, type RecognizeResult } from "./result.js";
import {
  type HostEvent,
  type InitSettings,
  type PageInput,
  type PageTask,
  parseCommand,
  type RuntimeReport,
} from "./worker-protocol.js";

/** The models one page reads. */
const PAGE_ROLES = [
  "segnet",
  "encoder",
  "decoder",
  "ocrDetect",
  "ocrClassify",
  "ocrRecognize",
] as const;

export interface WorkerEngine {
  close: () => Promise<void>;
  /** Bytes per artifact of every model a page can need, for the `models` stage's total. */
  readonly modelBytes: ReadonlyMap<string, number>;
  /** Decodes and reads one page. Never throws. */
  read: (
    page: PageInput,
    task: PageTask,
    options: PipelineOptions
  ) => Promise<RecognizeResult>;
  readonly report: RuntimeReport;
}

export type StartEngine = (
  settings: InitSettings,
  onModel: (event: ModelEvent) => void
) => Promise<WorkerEngine>;

/** The engine a browser Worker runs: startRuntime, opencv.js, a ModelStore over the browser cache. */
export const startBrowserEngine: StartEngine = async (settings, onModel) => {
  const runtime = await startRuntime({
    maxBackend: settings.prefer,
    ...(settings.wasmPaths === null ? {} : { wasmPaths: settings.wasmPaths }),
  });
  const cv = await loadOpenCv();
  const store = new ModelStore({
    baseUrl: settings.baseUrl,
    cache: await browserCache(),
    onEvent: onModel,
    runtime,
  });
  const engine = {
    backend: runtime.backend,
    cv,
    open: async (
      role: Parameters<ModelStore["open"]>[0],
      batch?: number,
      signal?: AbortSignal
    ) => {
      // Bytes first, under the page's signal, so a cancel stops a download.
      await store.prefetch([role], signal);
      return await store.open(role, batch === undefined ? {} : { batch });
    },
  };
  return {
    close: () => store.close(),
    modelBytes: new Map(
      PAGE_ROLES.map((role) => {
        const { artifact, artifactId } = store.plan(role);
        return [artifactId, artifact.bytes] as const;
      })
    ),
    read: async (page, task, options) => {
      try {
        const image = await decodePage(page);
        const { minTextScore } = task;
        return task.kind === "page"
          ? await recognizePage(image, engine, {
              ...options,
              minTextScore,
              ocr: task.ocr,
            })
          : await recognizeTexts(image, task.staves, engine, {
              ...options,
              minTextScore,
            });
      } catch (cause) {
        return failedResult(
          runtime.backend,
          cause instanceof PageInputError ? "bad_input" : "engine_failed",
          describeError(cause)
        );
      }
    },
    report: {
      backend: runtime.backend,
      numThreads: runtime.numThreads,
      reason: runtime.reason,
    },
  };
};

interface Job {
  readonly controller: AbortController;
  readonly id: number;
  /** Bytes received per artifact, for the `models` stage. */
  readonly received: Map<string, number>;
}

type HostState =
  | { readonly kind: "idle" }
  | { readonly kind: "starting" }
  | { readonly engine: WorkerEngine; readonly kind: "ready" }
  | { readonly kind: "closed" };

export class WorkerHost {
  #job: Job | undefined;
  readonly #post: (event: HostEvent) => void;
  #running: Promise<void> = Promise.resolve();
  readonly #startEngine: StartEngine;
  #state: HostState = { kind: "idle" };

  constructor(post: (event: HostEvent) => void, startEngine: StartEngine) {
    this.#post = post;
    this.#startEngine = startEngine;
  }

  /** One message from the client. Never throws: a malformed message answers an `error` event. */
  handle(data: unknown): void {
    const parsed = parseCommand(data);
    if (!parsed.ok) {
      this.#post({
        id: null,
        kind: "error",
        message: `a homr-web worker cannot serve ${parsed.why}`,
      });
      return;
    }
    const command = parsed.value;
    switch (command.kind) {
      case "init":
        this.#init(command.settings).catch(() => undefined);
        return;
      case "recognize":
        this.#recognize(command.id, command.page, command.task);
        return;
      case "cancel":
        if (this.#job?.id === command.id) {
          this.#job.controller.abort(
            new DOMException(
              "the caller cancelled",
              command.timeout ? "TimeoutError" : "AbortError"
            )
          );
        }
        return;
      default:
        this.#close().catch(() => undefined);
    }
  }

  async #init(settings: InitSettings): Promise<void> {
    if (this.#state.kind !== "idle") {
      this.#post({
        id: null,
        kind: "error",
        message: `init while the worker is ${this.#state.kind}`,
      });
      return;
    }
    this.#state = { kind: "starting" };
    try {
      const engine = await this.#startEngine(settings, (event) =>
        this.#onModel(event)
      );
      if (this.#state.kind !== "starting") {
        await engine.close();
        return;
      }
      this.#state = { engine, kind: "ready" };
      this.#post({ kind: "ready", ...engine.report });
    } catch (cause) {
      this.#state = { kind: "idle" };
      this.#post({ kind: "init-failed", message: describeError(cause) });
    }
  }

  #recognize(id: number, page: PageInput, task: PageTask): void {
    const state = this.#state;
    if (state.kind !== "ready") {
      this.#post({
        id,
        kind: "error",
        message: `recognize while the worker is ${state.kind}`,
      });
      return;
    }
    if (this.#job !== undefined) {
      this.#post({
        id,
        kind: "result",
        result: failedResult(
          state.engine.report.backend,
          "busy",
          `page ${this.#job.id} is still being read`
        ),
      });
      return;
    }
    const job: Job = {
      controller: new AbortController(),
      id,
      received: new Map(),
    };
    this.#job = job;
    this.#running = state.engine
      .read(page, task, {
        onProgress: (progress) =>
          this.#post({ id, kind: "progress", progress }),
        signal: job.controller.signal,
      })
      .catch((cause: unknown) =>
        failedResult(
          state.engine.report.backend,
          "engine_failed",
          describeError(cause)
        )
      )
      .then((result) => {
        try {
          this.#post({ id, kind: "result", result });
        } catch (cause) {
          this.#post({
            id,
            kind: "result",
            result: failedResult(
              result.backend,
              "engine_failed",
              describeError(cause)
            ),
          });
        }
      })
      .finally(() => {
        this.#job = undefined;
      });
  }

  /** Store events are store-wide; with one job at a time, the current job is their only owner. */
  #onModel(event: ModelEvent): void {
    const job = this.#job;
    if (job === undefined || this.#state.kind !== "ready") {
      return;
    }
    const sizes = this.#state.engine.modelBytes;
    if (event.kind === "download") {
      job.received.set(event.artifact, event.received);
    } else if (event.kind === "cached" || event.kind === "verified") {
      job.received.set(event.artifact, sizes.get(event.artifact) ?? 0);
    } else {
      return;
    }
    let total = 0;
    for (const bytes of sizes.values()) {
      total += bytes;
    }
    let done = 0;
    for (const bytes of job.received.values()) {
      done += bytes;
    }
    this.#post({
      id: job.id,
      kind: "progress",
      progress: { done: Math.min(done, total), stage: "models", total },
    });
  }

  async #close(): Promise<void> {
    const state = this.#state;
    this.#state = { kind: "closed" };
    this.#job?.controller.abort(
      new DOMException("the recognizer was disposed", "AbortError")
    );
    await this.#running.catch(() => undefined);
    if (state.kind === "ready") {
      await state.engine.close();
    }
    this.#post({ kind: "closed" });
  }
}

interface WorkerScope {
  addEventListener: (
    type: "message",
    listener: (event: { readonly data: unknown }) => void
  ) => void;
  postMessage: (message: unknown) => void;
}

const scope = globalThis as unknown as Partial<WorkerScope> & {
  readonly WorkerGlobalScope?: unknown;
};

if (
  typeof scope.WorkerGlobalScope === "function" &&
  scope instanceof scope.WorkerGlobalScope &&
  scope.addEventListener !== undefined
) {
  const host = new WorkerHost(
    (event) => scope.postMessage?.(event),
    startBrowserEngine
  );
  scope.addEventListener("message", (event) => host.handle(event.data));
}

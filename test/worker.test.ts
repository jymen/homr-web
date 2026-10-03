/**
 * The client, the protocol and WorkerHost together on Node: createRecognizer
 * is given a stand-in Worker whose other end is a real WorkerHost serving a
 * scripted engine. Every message crosses structuredClone, as it would cross
 * postMessage, so the parsers see what a browser would deliver.
 */

import { describe, expect, it } from "vitest";
import { createRecognizer, type Recognizer } from "../src/client.js";
import type { ModelEvent } from "../src/models/store.js";
import type { PipelineOptions } from "../src/pipeline/recognize.js";
import {
  failedResult,
  type Progress,
  type RecognizeResult,
} from "../src/result.js";
import { type StartEngine, WorkerHost } from "../src/worker.js";
import type { InitSettings } from "../src/worker-protocol.js";

interface Read {
  readonly finish: (result: RecognizeResult) => void;
  readonly options: PipelineOptions;
}

const success = (log: string): RecognizeResult => ({
  backend: "wasm",
  durationMs: 1,
  engine: "browser",
  error: "",
  log,
  musicXml: "<score-partwise />",
  ok: true,
  staves: [{ cx: 0.5, cy: 0.5, h: 0.1, index: 0, w: 0.9 }],
  texts: [],
});

/** A WorkerHost behind something shaped like a Worker, and the engine's calls laid out for the test. */
function harness(fail?: string) {
  const reads: Read[] = [];
  const posted: unknown[] = [];
  const closes: string[] = [];
  let settings: InitSettings | undefined;
  let onModel: (event: ModelEvent) => void = () => undefined;
  let terminated = false;
  const startEngine: StartEngine = (given, listener) => {
    settings = given;
    onModel = listener;
    if (fail !== undefined) {
      return Promise.reject(new Error(fail));
    }
    return Promise.resolve({
      close: () => {
        closes.push("closed");
        return Promise.resolve();
      },
      modelBytes: new Map([
        ["segnet", 100],
        ["encoder", 300],
      ]),
      read: (_page, options) =>
        new Promise<RecognizeResult>((finish) =>
          reads.push({ finish, options })
        ),
      report: { backend: "wasm", numThreads: 1, reason: "test" },
    });
  };
  const worker = new EventTarget() as EventTarget & {
    postMessage: (data: unknown) => void;
    terminate: () => void;
  };
  const host = new WorkerHost((event) => {
    queueMicrotask(() =>
      worker.dispatchEvent(
        new MessageEvent("message", { data: structuredClone(event) })
      )
    );
  }, startEngine);
  worker.postMessage = (data) => {
    posted.push(data);
    queueMicrotask(() => host.handle(structuredClone(data)));
  };
  worker.terminate = () => {
    terminated = true;
  };
  return {
    closes,
    create: () =>
      createRecognizer({
        baseUrl: "http://studio.test/models",
        createWorker: () => worker as unknown as Worker,
        wasmPaths: "http://studio.test/ort/",
      }),
    host,
    model: (event: ModelEvent) => onModel(event),
    posted,
    reads,
    settings: () => settings,
    terminated: () => terminated,
  };
}

const page = () =>
  new Blob([new Uint8Array([137, 80, 78, 71])], { type: "image/png" });
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

const started = (h: ReturnType<typeof harness>): Promise<Recognizer> =>
  h.create();

describe("createRecognizer", () => {
  it("resolves with the Worker's backend and sends absolute URLs", async () => {
    const h = harness();
    const recognizer = await started(h);
    expect(recognizer.backend).toBe("wasm");
    expect(recognizer.backendReason).toBe("test");
    expect(h.settings()).toEqual({
      baseUrl: "http://studio.test/models",
      prefer: "webgpu",
      wasmPaths: "http://studio.test/ort/",
    });
  });

  it("rejects, and terminates the Worker, when the engine cannot start", async () => {
    const h = harness("no WebAssembly here");
    await expect(h.create()).rejects.toThrow("no WebAssembly here");
    expect(h.terminated()).toBe(true);
  });
});

describe("recognizePage", () => {
  it("settles with the engine's result and forwards its progress", async () => {
    const h = harness();
    const recognizer = await started(h);
    const progress: Progress[] = [];
    const pending = recognizer.recognizePage(page(), {
      onProgress: (p) => progress.push(p),
    });
    await tick();
    const [read] = h.reads;
    read?.options.onProgress?.({ done: 1, stage: "segment", total: 4 });
    read?.finish(success("read"));
    expect(await pending).toEqual(success("read"));
    expect(progress).toEqual([{ done: 1, stage: "segment", total: 4 }]);
  });

  it("reports model bytes as the models stage, against every model's size", async () => {
    const h = harness();
    const recognizer = await started(h);
    const progress: Progress[] = [];
    const pending = recognizer.recognizePage(page(), {
      onProgress: (p) => progress.push(p),
    });
    await tick();
    h.model({ artifact: "segnet", bytes: 100, from: "memory", kind: "cached" });
    h.model({
      artifact: "encoder",
      kind: "download",
      received: 120,
      total: 300,
    });
    h.reads[0]?.finish(success("read"));
    await pending;
    expect(progress).toEqual([
      { done: 100, stage: "models", total: 400 },
      { done: 220, stage: "models", total: 400 },
    ]);
  });

  it("answers busy to a second page while one runs, and bad_input to a non-image", async () => {
    const h = harness();
    const recognizer = await started(h);
    const first = recognizer.recognizePage(page());
    const second = await recognizer.recognizePage(page());
    expect(second).toMatchObject({ error: "busy", musicXml: "", ok: false });
    const odd = await recognizer.recognizePage("page.png" as unknown as Blob);
    expect(odd.error).toBe("bad_input");
    await tick();
    h.reads[0]?.finish(success("first"));
    expect((await first).ok).toBe(true);
  });

  it("settles a cancelled page at once, aborts the engine, and starts the next only after the Worker is free", async () => {
    const h = harness();
    const recognizer = await started(h);
    const controller = new AbortController();
    const first = recognizer.recognizePage(page(), {
      signal: controller.signal,
    });
    await tick();
    controller.abort();
    expect((await first).error).toBe("cancelled");
    await tick();
    expect(h.reads[0]?.options.signal?.aborted).toBe(true);

    const second = recognizer.recognizePage(page());
    await tick();
    expect(h.reads).toHaveLength(1);
    h.reads[0]?.finish(failedResult("wasm", "cancelled", "aborted"));
    await tick();
    expect(h.reads).toHaveLength(2);
    h.reads[1]?.finish(success("second"));
    expect((await second).log).toBe("second");
  });

  it("calls a TimeoutError abort a timeout, and an already aborted signal never reaches the Worker", async () => {
    const h = harness();
    const recognizer = await started(h);
    const late = await recognizer.recognizePage(page(), {
      signal: AbortSignal.abort(new DOMException("late", "TimeoutError")),
    });
    expect(late.error).toBe("timeout");
    const timed = new AbortController();
    const pending = recognizer.recognizePage(page(), { signal: timed.signal });
    await tick();
    timed.abort(new DOMException("late", "TimeoutError"));
    expect((await pending).error).toBe("timeout");
    await tick();
    expect(h.reads[0]?.options.signal?.reason).toMatchObject({
      name: "TimeoutError",
    });
  });
});

describe("dispose", () => {
  it("cancels the page in flight, closes the engine, terminates, and answers later calls at once", async () => {
    const h = harness();
    const recognizer = await started(h);
    const pending = recognizer.recognizePage(page());
    await tick();
    const disposed = recognizer.dispose();
    expect((await pending).error).toBe("cancelled");
    h.reads[0]?.finish(failedResult("wasm", "cancelled", "aborted"));
    await disposed;
    await recognizer.dispose();
    expect(h.closes).toEqual(["closed"]);
    expect(h.terminated()).toBe(true);
    expect((await recognizer.recognizePage(page())).error).toBe("cancelled");
  });
});

describe("WorkerHost", () => {
  it("answers a malformed message and a recognize before init with error events", async () => {
    const events: unknown[] = [];
    const host = new WorkerHost(
      (event) => events.push(event),
      () => Promise.reject(new Error("unused"))
    );
    host.handle({ kind: "fly" });
    host.handle({ id: 3, kind: "recognize", page: page() });
    host.handle({ id: 4, kind: "recognize", page: "page.png" });
    await tick();
    expect(events).toEqual([
      {
        id: null,
        kind: "error",
        message: 'a homr-web worker cannot serve the command "fly"',
      },
      { id: 3, kind: "error", message: "recognize while the worker is idle" },
      {
        id: null,
        kind: "error",
        message:
          "a homr-web worker cannot serve a recognize whose page is not a Blob, ImageBitmap or ImageData",
      },
    ]);
  });
});

# Phase 9 synthesis: the public API

Two candidates were sketched in parallel from the same brief (the brief, both
packages and this synthesis were produced with the `pstack:architect` skill;
the candidates are not kept in the repository). Candidate A: an async
`createRecognizer` resolving to a start-result union, a main-thread FIFO
queue, correlation ids, library-specific kebab-case failure codes and no
`engine` field. Candidate B: an async `createRecognizer` that rejects on a
failed start, one `MessageChannel` per job instead of ids, a `load()`
pre-warm, `busy` instead of a queue, Go-style snake_case codes and an
`engine` field. Both deleted `SegmentationWorker`, both decoded the image in
the Worker, both added a `models` progress stage, and both moved today's
`export *` to an unstable subpath.

## Decisions

| Question | Taken | From | Why |
|---|---|---|---|
| Result shape | Go's `OmrRecognizeResult` field for field (`engine`, `musicXml`, `log` as one string, `durationMs`, `ok`, `error`, `staves`, `texts`) plus `backend` | B, tightened | The phase file asks for the server route's shape so phase 10 reads it without an adapter. A union on `ok`, but each arm carries every Go field (`error: ""` on success, `musicXml: ""` and empty lists on failure), so a value is assignable to the app's existing type wherever the codes agree. |
| Error codes | Go's `bad_input`, `busy`, `engine_failed`, `engine_missing`, `timeout`, plus `not_music` and `cancelled` | both | `not_music` (a `DetectionError`) is the one failure where falling back to the server is pointless; `cancelled` is a user act, not an engine failure. `engine_missing` is a model that could not be downloaded or verified. A timeout is an abort whose reason is a `TimeoutError`, as `AbortSignal.timeout` gives. |
| Start | `createRecognizer` is async and **rejects** when the Worker cannot start | B | A recognizer whose runtime never started has no valid method, and the app's fallback is one `catch`. Recognition failures stay results. |
| Second call while one runs | resolves at once with `busy` | B, and Go | No hidden queue; one page at a time is the app's loop anyway. The Worker refuses a second job too, so the rule holds even if the client is wrong. |
| Correlation | an `id` per job on the Worker's own port | A | With `busy`, at most one job is ever live, so the id is only a guard against a stale `progress` or `result`. A port per job (B) adds a second lifetime for no case the id does not cover. |
| Model download | lazy, inside the first job, reported as the `models` stage in bytes | both | `load()` (B) is a non-breaking later addition. Models open lazily, so a page that is not music never downloads the transformer. |
| Options | `baseUrl`, `prefer`, `wasmPaths`, `createWorker` | A | `baseUrl` and `wasmPaths` are resolved to absolute URLs on the main thread, because inside the Worker a relative URL resolves against `worker.js`. `createWorker` covers bundlers that cannot resolve `new URL("./worker.js", import.meta.url)` and the tests. `numThreads`, the WebGPU decoder opt-in and the cache choice stay internal: the first two are measured defaults, and a cache object cannot cross `postMessage`. |
| Image input | `Blob`, `ImageBitmap` or `ImageData`, cloned to the Worker, decoded there with `createImageBitmap` and `OffscreenCanvas`, alpha dropped as `cv2.imread` drops it | A | Never detaches the caller's object; the main thread does no decoding. |
| Public surface | `.` exports `createRecognizer`, `HOMR_VERSION`, `HOMR_COMMIT` and the types; `./worker` is the Worker entry; `./internal` is today's `export *`, outside semver | both | Every internal rename would otherwise be a breaking change. The bench imports `./internal`; tests import `src/`. |
| Pipeline without a Worker | `recognizePage(page: ColorImage, engine, options)` in `src/pipeline/recognize.ts`, never throws | both | The Node golden test and the Worker call the same function. |
| `staves` | the numbers `formatStaffPositions` prints, through one `staffPositions` function, then the server's stable sort by `cy` | A | One computation for the file and the boxes, so they cannot drift; the findings explain why no string round trip is needed. |
| `SegmentationWorker` | deleted with its tests | both | Superseded by `WorkerHost`; two Worker protocols would be two things to keep correct. `DEFAULT_SEGNET_BATCH` moves to `segment.ts`. |

## The contract

```ts
// src/result.ts
export type Backend = "webgpu" | "wasm-threads" | "wasm";
export interface StaffBox { index; cx; cy; w; h }            // Go OmrStaff
export interface PageText { staff; text; score; x0; y0; x1; y1 } // Go OmrText; [] until phase 11
export type ProgressStage = "models" | "segment" | "detect" | "dewarp" | "staff" | "xml";
export interface Progress { stage: ProgressStage; done: number; total: number }
export const RECOGNIZE_ERRORS = ["bad_input", "busy", "cancelled", "engine_failed",
  "engine_missing", "not_music", "timeout"] as const;
export type RecognizeError = (typeof RECOGNIZE_ERRORS)[number];
interface ResultBase { engine: "browser"; backend: Backend; durationMs: number; log: string }
export type RecognizeResult =
  | ResultBase & { ok: true; error: ""; musicXml: string; staves: readonly StaffBox[]; texts: readonly PageText[] }
  | ResultBase & { ok: false; error: RecognizeError; musicXml: ""; staves: readonly []; texts: readonly [] };

// src/client.ts
export type PageInput = Blob | ImageBitmap | ImageData;
export interface RecognizerOptions { baseUrl: string | URL; prefer?: Backend; wasmPaths?: string | URL; createWorker?: () => Worker }
export interface RecognizeOptions { onProgress?: (progress: Progress) => void; signal?: AbortSignal }
export interface Recognizer {
  readonly backend: Backend;        // frozen by startRuntime inside the Worker
  readonly backendReason: string;
  recognizePage(page: PageInput, options?: RecognizeOptions): Promise<RecognizeResult>;  // never rejects
  dispose(): Promise<void>;         // idempotent; a running job resolves cancelled
}
export function createRecognizer(options: RecognizerOptions): Promise<Recognizer>;

// src/worker-protocol.ts, internal; both directions parsed at the boundary
type HostCommand =
  | { kind: "init"; settings: { baseUrl: string; prefer: Backend; wasmPaths: string | null } }
  | { kind: "recognize"; id: number; page: PageInput }
  | { kind: "cancel"; id: number }
  | { kind: "close" };
type HostEvent =
  | { kind: "ready"; backend: Backend; numThreads: number; reason: string }
  | { kind: "init-failed"; message: string }
  | { kind: "progress"; id: number; progress: Progress }
  | { kind: "result"; id: number; result: RecognizeResult }
  | { kind: "error"; id: number | null; message: string }   // a protocol defect, never a recognition outcome
  | { kind: "closed" };

// src/worker.ts: WorkerHost(port, startEngine) owns one engine and at most one job;
// startEngine is injected so a Node test drives the host over a MessageChannel.
// src/pipeline/recognize.ts
export interface RecognizeEngine { cv; backend; open(role): Promise<ModelSession>; onModel?(cb) }
export function recognizePage(page: ColorImage, engine: RecognizeEngine, options?: RecognizeOptions): Promise<RecognizeResult>;
```

## Rejected

- A FIFO queue (A): hides contention and lets a page the user navigated away
  from hold the GPU.
- A start-result union (A): one more branch at every call site to express a
  condition the app handles once.
- `load()` (B): deferred; adding it later breaks nothing.
- `engine`-less results with app-side code translation (A): the phase file
  asks for the server's shape.

## After the interrogate pass (2026-10-03)

Two reviewers (opus, sonnet) read the API diff. Changed in response:

- A page asked for while `dispose()` is pending answers `cancelled` at once;
  it used to reach a closed Worker and come back `engine_failed`. Both
  reviewers found it.
- `WorkerHost` turns an engine that throws, or a result that cannot be
  posted, into an `engine_failed` result; before, the page and every later
  one hung. Both found it.
- A new terminal code, `worker_lost`, for a crashed Worker, so the app can
  tell "dispose and recreate" from "this page failed".
- The page's signal reaches the model download (`open(role, batch,
  signal)`, through `store.prefetch`), so a cancel during the first 100 MB
  stops it.
- `createRecognizer` rejects after 30 s without an answer to `init`, builds
  its settings before creating the Worker, and terminates it on any failure.
- `Recognizer.numThreads`; the Worker's copy of the bitmap is always closed;
  dispose's timer is cleared and a crash ends its wait.
- Documented: `models` appears twice on a first page; a page after a cancel
  waits for the Worker.

Kept: images are cloned, not transferred (the caller keeps its object);
`parseEvent` trusts the payload of a result from its own package version;
no `signal` on `createRecognizer` (the 30 s deadline covers the hang).

Differences between the contract sketch above and the code: `cancel`
carries `timeout`; `RecognizeEngine` has no `onModel` and its `open` takes a
signal; the pipeline's options type is `PipelineOptions`; `WorkerHost` takes
`(post, startEngine)`; `Recognizer` has `numThreads`; `RECOGNIZE_ERRORS`
adds `worker_lost`.

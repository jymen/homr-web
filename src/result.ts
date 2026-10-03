/**
 * What the library answers with: the app server's `OmrRecognizeResult`
 * (AbcGoDb, abcsql/omr.go) field for field, plus `backend`, so the app's
 * transcriber page reads a browser result and a server result the same way.
 */

export const BACKENDS = ["webgpu", "wasm-threads", "wasm"] as const;
/** Chosen once per Worker by startRuntime and frozen before the first session. */
export type Backend = (typeof BACKENDS)[number];

/**
 * One line of homr's staff-positions file (save_staff_positions):
 * "<0|1> cx cy w h", the leading digit being is_grandstaff, the rest the
 * staff's axis-aligned extent normalised to the page.
 */
export interface StaffPosition {
  readonly cx: number;
  readonly cy: number;
  readonly h: number;
  readonly isGrandstaff: boolean;
  readonly w: number;
}

/** The server's `OmrStaff`: page-normalised 0..1, `index` in top-to-bottom order. */
export interface StaffBox {
  readonly cx: number;
  readonly cy: number;
  readonly h: number;
  readonly index: number;
  readonly w: number;
}

/** The server's `OmrText`: a chord or title read above staff `staff`, box page-normalised. Always empty until the browser has OCR. */
export interface PageText {
  readonly score: number;
  readonly staff: number;
  readonly text: string;
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

/**
 * `models` counts bytes of the three models, cached ones as done, and appears
 * on the first page a recognizer reads; `segment` counts segnet batches,
 * `dewarp` and `staff` count staffs, `detect` and `xml` go from 0/1 to 1/1.
 */
export const PROGRESS_STAGES = [
  "models",
  "segment",
  "detect",
  "dewarp",
  "staff",
  "xml",
] as const;
export type ProgressStage = (typeof PROGRESS_STAGES)[number];

export interface Progress {
  readonly done: number;
  readonly stage: ProgressStage;
  readonly total: number;
}

/**
 * The server's codes where the meaning is the same, and two of the library's
 * own. `engine_missing` is a model that could not be downloaded, verified or
 * opened; `not_music` is a page on which homr finds no staff or notehead,
 * where the server would fail too; `cancelled` is the caller's signal or
 * `dispose()`; `timeout` is a signal aborted with a TimeoutError, which is
 * what `AbortSignal.timeout` does; `busy` is a second page asked for while
 * one is running.
 */
export const RECOGNIZE_ERRORS = [
  "bad_input",
  "busy",
  "cancelled",
  "engine_failed",
  "engine_missing",
  "not_music",
  "timeout",
] as const;
export type RecognizeError = (typeof RECOGNIZE_ERRORS)[number];

interface ResultBase {
  readonly backend: Backend;
  readonly durationMs: number;
  readonly engine: "browser";
  /** homr's stderr lines and the port's own, joined by "\n"; on failure the reason is the last line. */
  readonly log: string;
}

export interface RecognizeSuccess extends ResultBase {
  readonly error: "";
  readonly musicXml: string;
  readonly ok: true;
  readonly staves: readonly StaffBox[];
  readonly texts: readonly PageText[];
}

export interface RecognizeFailure extends ResultBase {
  readonly error: RecognizeError;
  readonly musicXml: "";
  readonly ok: false;
  readonly staves: readonly [];
  readonly texts: readonly [];
}

/** Failure is a result, never a thrown error, as on the server: the page keeps one shape. */
export type RecognizeResult = RecognizeSuccess | RecognizeFailure;

export const failedResult = (
  backend: Backend,
  error: RecognizeError,
  log: string,
  durationMs = 0
): RecognizeFailure => ({
  backend,
  durationMs,
  engine: "browser",
  error,
  log,
  musicXml: "",
  ok: false,
  staves: [],
  texts: [],
});

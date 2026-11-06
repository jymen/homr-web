/**
 * What the library answers with: the app server's `OmrRecognizeResult`
 * (AbcGoDb, abcsql/omr.go) field for field, plus `backend`, so the app's
 * transcriber page reads a browser result and a server result the same way.
 * `tablature` is browser-only for now: the server has no tab guard, so its
 * result has no such field. Each of its systems carries the frets read on it.
 */

import type { ButtonTabReading } from "./tab/button-read.js";
import type { TabReading } from "./tab/read.js";

export type { ButtonTabLetter } from "./tab/button-letters.js";
export type { ButtonTabMark, ButtonTabReading } from "./tab/button-read.js";

export type { TabSystem } from "./tab/detect.js";
export type {
  TabAnnotation,
  TabEvent,
  TabNote,
  TabPlacedEvent,
  TabReading,
  TabSlur,
  TabSlurKind,
  TabTechnique,
  TabTimeSignature,
  TabUnreadFret,
} from "./tab/read.js";
export type { TabBarline, TabStem } from "./tab/rhythm.js";
export type {
  TabCapo,
  TabInstrumentKind,
  TabInstrumentName,
  TabTuning,
  TuningSource,
} from "./tab/tuning.js";

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

/**
 * The part of the page given that homr's reading saw, normalised by that page:
 * autocrop's paper rect, or the whole page `{0, 0, 1, 1}` when autocrop kept
 * it. `staves` and `noteColumns` are normalised by this part, as homr's
 * staff-positions file is; `tablature`, `buttonTabs` and `texts` by the page given.
 * `staffInPage` and `columnInPage` carry the first into the second.
 */
export interface PageCrop {
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

/** The server's `OmrText`: a chord or title read in the strip above staff `staff`, box page-normalised, score rounded to 3 digits. */
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
 * `models` counts bytes of the six models, cached ones as done, on the first
 * page a recognizer reads. The segmentation model opens before `segment`, the
 * transformer's two after `detect` and the three OCR models before `ocr`, so
 * `models` appears three times on a first page, and a page that is not
 * music, or read with `ocr: false`, stops short of the total. A page with
 * tablature adds two openings before the rest: the OCR recogniser for the
 * guard and the frets, then the OCR detector and classifier for the tuning
 * and capo text. `tab` counts the tab systems read plus one for that text,
 * and appears only on a page that has a tab. `segment` counts segnet
 * batches, `dewarp` and `staff` count staffs, `ocr` counts the chord strips
 * and the title band, `detect` and `xml` go from 0/1 to 1/1.
 */
export const PROGRESS_STAGES = [
  "models",
  "tab",
  "segment",
  "detect",
  "dewarp",
  "staff",
  "ocr",
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
 * one is running; `worker_lost` means the Worker crashed or broke the
 * protocol, and this recognizer answers nothing else from then on: dispose it
 * and create another. `tablature_only` is a page whose every system is line
 * tablature, which homr cannot read: the result's `tablature` lists them.
 */
export const RECOGNIZE_ERRORS = [
  "bad_input",
  "busy",
  "cancelled",
  "engine_failed",
  "engine_missing",
  "not_music",
  "tablature_only",
  "timeout",
  "worker_lost",
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
  /**
   * The melodeon button tabs (CADB) found under the staves, one per tab top
   * to bottom, each with its numbers read; empty on a page without any.
   * Their x is normalised by the page given, as `tablature`'s. Always set by
   * `recognizePage`; optional so that a caller's hand-written result still
   * compiles.
   */
  readonly buttonTabs?: readonly ButtonTabReading[];
  /**
   * Where `staves` and `noteColumns` sit on the page given. Always set by
   * this library (the whole page from `recognizeTexts`, whose staves are the
   * caller's); optional so that a caller's hand-written result still
   * compiles.
   */
  readonly crop?: PageCrop;
  readonly error: "";
  readonly musicXml: string;
  /**
   * One per `<note>` element of `musicXml`, in document order, chord tones
   * and rests included: the x of the notehead printed for that note,
   * page-normalised like `staves` (by the preprocessed page's width), or
   * null where none was tied to it (a rest, a notehead segnet missed, a
   * grand staff). Empty from `recognizeTexts`. Always set by this library;
   * optional only so that a caller's hand-written result still compiles.
   */
  readonly noteColumns?: readonly (number | null)[];
  readonly ok: true;
  readonly staves: readonly StaffBox[];
  /** The tab systems found, kept out of homr's reading and read on their own, empty on a page without any. Never in `staves`. */
  readonly tablature: readonly TabReading[];
  readonly texts: readonly PageText[];
}

interface FailureBase extends ResultBase {
  readonly musicXml: "";
  readonly ok: false;
  readonly staves: readonly [];
  readonly texts: readonly [];
}

/** Only a `tablature_only` failure carries systems, and it always carries at least one. */
export type RecognizeFailure =
  | (FailureBase & {
      readonly error: Exclude<RecognizeError, "tablature_only">;
      readonly tablature: readonly [];
    })
  | (FailureBase & {
      readonly error: "tablature_only";
      readonly tablature: readonly [TabReading, ...TabReading[]];
    });

/** Failure is a result, never a thrown error, as on the server: the page keeps one shape. */
export type RecognizeResult = RecognizeSuccess | RecognizeFailure;

export const failedResult = (
  backend: Backend,
  error: Exclude<RecognizeError, "tablature_only">,
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
  tablature: [],
  texts: [],
});

export const tablatureOnlyResult = (
  backend: Backend,
  tablature: readonly [TabReading, ...TabReading[]],
  log: string,
  durationMs: number
): RecognizeFailure => ({
  backend,
  durationMs,
  engine: "browser",
  error: "tablature_only",
  log,
  musicXml: "",
  ok: false,
  staves: [],
  tablature,
  texts: [],
});

/** An abort whose reason is a TimeoutError, as `AbortSignal.timeout()` gives. */
export const isTimeoutAbort = (signal: AbortSignal): boolean =>
  signal.reason instanceof Error && signal.reason.name === "TimeoutError";

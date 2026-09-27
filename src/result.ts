/**
 * What the library answers with, in the shape the app's Go route returns
 * today so that phase 10 reads it without an adapter. Phase 9 owns the
 * function that produces it; phase 1 only fixes the fields.
 */

export const BACKENDS = ["webgpu", "wasm-threads", "wasm"] as const;
/** Chosen once per page by phase 2 and frozen before the first session. */
export type Backend = (typeof BACKENDS)[number];

/**
 * One line of homr's staff-positions file (save_staff_positions):
 * "<0|1> cx cy w h", the leading digit being is_grandstaff, the rest the
 * staff's axis-aligned extent normalised to the page. This is the on-disk
 * form; StaffBox below is the app's.
 */
export interface StaffPosition {
  readonly cx: number;
  readonly cy: number;
  readonly h: number;
  readonly isGrandstaff: boolean;
  readonly w: number;
}

/** A staff's extent as the app draws it: page-normalised 0..1, indexed in reading order. */
export interface StaffBox {
  readonly cx: number;
  readonly cy: number;
  readonly h: number;
  readonly index: number;
  readonly w: number;
}

/** A chord or title read above staff `staff`, box page-normalised. Named as the app names it (`texts: PageText[]`). */
export interface PageText {
  readonly score: number;
  readonly staff: number;
  readonly text: string;
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

/** Phase 9's progress event; `stage` names the pipeline step in order. */
export interface Progress {
  readonly done: number;
  readonly stage: "segment" | "detect" | "dewarp" | "staff" | "xml";
  readonly total: number;
}

interface ResultBase {
  readonly backend: Backend;
  readonly durationMs: number;
  /** homr's eprint lines, one per entry; the app shows them verbatim. */
  readonly log: readonly string[];
}

/**
 * Failure is a result, never a thrown error (the Go route's contract): the
 * page keeps one shape and `ok` tells the two apart. A union rather than
 * one flat object so that `musicXml` exists exactly when `ok` is true.
 */
export type RecognizeResult =
  | (ResultBase & {
      readonly ok: true;
      readonly musicXml: string;
      readonly staves: readonly StaffBox[];
      readonly texts: readonly PageText[];
    })
  | (ResultBase & {
      readonly ok: false;
      readonly error: string;
    });

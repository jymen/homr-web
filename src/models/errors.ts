/**
 * The one error phase 2 throws. It carries a code a caller can act on and, for
 * a verification failure, the three values that make the failure actionable,
 * rather than prose the caller would have to parse.
 *
 * It sits in its own file because manifest.ts, backend.ts, session.ts and
 * store.ts all throw it while manifest.ts is the module the other three import:
 * defining the class in any one of them would close an import cycle.
 */

export const MODEL_ERROR_CODES = [
  /** handoff() was given two sessions the manifest does not pair. */
  "bad-handoff",
  /** A Placement the runtime cannot serve: the webgpu provider on a runtime that configured no GPU device. */
  "bad-placement",
  /** SessionTuning asked for something the artifact cannot be tuned with. */
  "bad-tuning",
  /** Bytes reached a session boundary whose digest is not the manifest's. */
  "digest-mismatch",
  /** The injected fetch could not produce the bytes. */
  "fetch-failed",
  /** The manifest contradicts itself: a role feeding itself, a missing input name. */
  "manifest",
  /** The store was closed while a load was in flight. */
  "store-closed",
  /** A second startRuntime whose options would apply different settings. */
  "runtime-frozen",
  /** A tensor name the opened session does not declare. */
  "unknown-tensor",
] as const;

export type ModelErrorCode = (typeof MODEL_ERROR_CODES)[number];

/**
 * What failed, in values rather than in a sentence: `id` is the artifact or the
 * role, and `expected`/`actual` are the two digests, backends or names that
 * disagreed. A caller that wants to self-heal reads these instead of matching
 * on the message.
 */
export interface ModelErrorDetail {
  readonly actual?: string;
  readonly expected?: string;
  readonly id?: string;
}

export class ModelError extends Error {
  readonly code: ModelErrorCode;
  readonly detail: ModelErrorDetail;

  constructor(
    code: ModelErrorCode,
    message: string,
    detail: ModelErrorDetail = {}
  ) {
    super(message);
    this.name = "ModelError";
    this.code = code;
    this.detail = detail;
  }
}

/**
 * The messages between `createRecognizer`'s client and the Worker's
 * `WorkerHost`, and the two parsers that are the only place either side
 * looks at an `unknown`.
 */

import {
  BACKENDS,
  type Backend,
  type Progress,
  type RecognizeResult,
  type StaffBox,
} from "./result.js";

export type PageInput = Blob | ImageBitmap | ImageData;

/** Absolute URLs only: the client resolves them, because inside the Worker a relative URL resolves against worker.js. */
export interface InitSettings {
  readonly baseUrl: string;
  readonly prefer: Backend;
  readonly wasmPaths: string | null;
}

/** What to read on a page: everything, or the chord strips above staves the caller has. */
export type PageTask =
  | {
      readonly kind: "page";
      readonly minTextScore: number;
      readonly ocr: boolean;
    }
  | {
      readonly kind: "texts";
      readonly minTextScore: number;
      readonly staves: readonly StaffBox[];
    };

export type HostCommand =
  | { readonly kind: "init"; readonly settings: InitSettings }
  | {
      readonly kind: "recognize";
      readonly id: number;
      readonly page: PageInput;
      readonly task: PageTask;
    }
  /** `timeout` when the caller's signal aborted with a TimeoutError, so the result says `timeout`. */
  | { readonly id: number; readonly kind: "cancel"; readonly timeout: boolean }
  | { readonly kind: "close" };

export interface RuntimeReport {
  readonly backend: Backend;
  readonly numThreads: number;
  readonly reason: string;
}

export type HostEvent =
  | ({ readonly kind: "ready" } & RuntimeReport)
  | { readonly kind: "init-failed"; readonly message: string }
  | {
      readonly id: number;
      readonly kind: "progress";
      readonly progress: Progress;
    }
  | {
      readonly id: number;
      readonly kind: "result";
      readonly result: RecognizeResult;
    }
  /** A protocol defect, never a recognition outcome: those are results with ok false. */
  | {
      readonly id: number | null;
      readonly kind: "error";
      readonly message: string;
    }
  | { readonly kind: "closed" };

export type Parsed<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly why: string };

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

const isId = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value);

const isBackend = (value: unknown): value is Backend =>
  BACKENDS.some((backend) => backend === value);

/** instanceof only where the class exists: a Worker has all three, Node has Blob alone. */
const isInstance = (value: unknown, name: string): boolean => {
  const type: unknown = (globalThis as Record<string, unknown>)[name];
  return typeof type === "function" && value instanceof type;
};

export const isPageInput = (value: unknown): value is PageInput =>
  isInstance(value, "Blob") ||
  isInstance(value, "ImageBitmap") ||
  isInstance(value, "ImageData");

const refuse = <T>(why: string): Parsed<T> => ({ ok: false, why });
const accept = <T>(value: T): Parsed<T> => ({ ok: true, value });

function parseSettings(value: unknown): InitSettings | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const { baseUrl, prefer, wasmPaths } = value;
  if (
    typeof baseUrl !== "string" ||
    !isBackend(prefer) ||
    !(wasmPaths === null || typeof wasmPaths === "string")
  ) {
    return undefined;
  }
  return { baseUrl, prefer, wasmPaths };
}

const isFiniteNumber = (value: unknown): value is number =>
  typeof value === "number" && Number.isFinite(value);

function parseStaffBox(value: unknown): StaffBox | undefined {
  if (!isRecord(value)) {
    return undefined;
  }
  const { cx, cy, h, index, w } = value;
  return isFiniteNumber(cx) &&
    isFiniteNumber(cy) &&
    isFiniteNumber(h) &&
    isFiniteNumber(w) &&
    isId(index)
    ? { cx, cy, h, index, w }
    : undefined;
}

function parseTask(value: unknown): PageTask | undefined {
  if (!(isRecord(value) && isFiniteNumber(value.minTextScore))) {
    return undefined;
  }
  const { kind, minTextScore } = value;
  if (kind === "page") {
    return typeof value.ocr === "boolean"
      ? { kind, minTextScore, ocr: value.ocr }
      : undefined;
  }
  if (kind !== "texts" || !Array.isArray(value.staves)) {
    return undefined;
  }
  const staves = value.staves
    .map(parseStaffBox)
    .filter((staff): staff is StaffBox => staff !== undefined);
  return staves.length === value.staves.length
    ? { kind, minTextScore, staves }
    : undefined;
}

export function parseCommand(data: unknown): Parsed<HostCommand> {
  if (!isRecord(data)) {
    return refuse(`a message that is ${data === null ? "null" : typeof data}`);
  }
  const { id, kind, page, settings, task, timeout } = data;
  if (kind === "close") {
    return accept({ kind });
  }
  if (kind === "init") {
    const parsed = parseSettings(settings);
    return parsed === undefined
      ? refuse("an init whose settings are not { baseUrl, prefer, wasmPaths }")
      : accept({ kind, settings: parsed });
  }
  if (kind !== "recognize" && kind !== "cancel") {
    return refuse(`the command ${JSON.stringify(kind) ?? typeof kind}`);
  }
  if (!isId(id)) {
    return refuse(`a ${kind} with no integer id`);
  }
  if (kind === "cancel") {
    return accept({ id, kind, timeout: timeout === true });
  }
  if (!isPageInput(page)) {
    return refuse(
      "a recognize whose page is not a Blob, ImageBitmap or ImageData"
    );
  }
  const parsedTask = parseTask(task);
  return parsedTask === undefined
    ? refuse("a recognize whose task is not a page or a texts task")
    : accept({ id, kind, page, task: parsedTask });
}

/**
 * The client trusts the payload of its own Worker once the envelope is right:
 * a result is built by `recognizePage` on the other side of this same
 * package version.
 */
export function parseEvent(data: unknown): Parsed<HostEvent> {
  if (!isRecord(data)) {
    return refuse(`a message that is ${data === null ? "null" : typeof data}`);
  }
  const { id, kind } = data;
  switch (kind) {
    case "ready":
      return isBackend(data.backend) &&
        typeof data.numThreads === "number" &&
        typeof data.reason === "string"
        ? accept({
            backend: data.backend,
            kind,
            numThreads: data.numThreads,
            reason: data.reason,
          })
        : refuse("a ready with no backend");
    case "init-failed":
      return accept({ kind, message: String(data.message) });
    case "progress":
      return isId(id) && isRecord(data.progress)
        ? accept({ id, kind, progress: data.progress as unknown as Progress })
        : refuse("a progress with no id");
    case "result":
      return isId(id) &&
        isRecord(data.result) &&
        typeof data.result.ok === "boolean"
        ? accept({
            id,
            kind,
            result: data.result as unknown as RecognizeResult,
          })
        : refuse("a result with no id or no ok");
    case "error":
      return accept({
        id: isId(id) ? id : null,
        kind,
        message: String(data.message),
      });
    case "closed":
      return accept({ kind });
    default:
      return refuse(`the event ${JSON.stringify(kind) ?? typeof kind}`);
  }
}

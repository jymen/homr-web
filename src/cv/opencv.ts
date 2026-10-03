/**
 * opencv.js, acquired once per realm and handed back as a typed surface, plus
 * the one thing a wasm library needs that a garbage-collected one does not:
 * somewhere that owns Mat lifetime.
 *
 * Nothing about how @techstark/opencv-js loads is visible from its types, and
 * four measured properties of it shape this module.
 *
 * It is a CommonJS emscripten bundle, so what a host hands back is
 * `module.exports`, or an ESM namespace wrapping it, rather than a ready
 * library.
 *
 * That object is thenable, and the shim behind it is
 * `Module.then = (func) => { if (calledRun) { func(Module) } ... }`. Once the
 * runtime is up it therefore calls back synchronously with the module itself, so
 * resolving a promise with the module enqueues the resolution of the same
 * thenable again, without end. It is an unbounded microtask loop rather than a
 * promise that fails to settle, which is why no timer runs afterwards and why a
 * test run hangs with no output and no timeout report. Before initialization the
 * same shim only records a callback, so the wait there merely never ends.
 *
 * So the module object must never be the value a promise resolves with. That is
 * why a source hands back an `AcquiredOpenCv` box rather than the module, why
 * the default source goes through a statically imported wrapper, and why this
 * loader clears `then` before returning: after that the module is an ordinary
 * object and travels through `Promise<OpenCv>` like anything else.
 *
 * Readiness is `onRuntimeInitialized`, and the readiness *test* is
 * `typeof Mat === "function"`, because `Mat` is undefined until the runtime is
 * up. The hook fires exactly once, so installing it on a module that has
 * already initialized waits forever too. Check first, hook second.
 *
 * The version is pinned at 4.12.0 because minAreaRect's angle convention
 * changed in 4.5.1 and homr's geometry is written against the 4.x behaviour.
 * That pin is why a wrong build earns a named error here instead of an
 * `undefined is not a function` somewhere in the middle of autocrop.
 */

export type OpenCv = typeof import("@techstark/opencv-js");

/** A build that is not the pinned one, or a source that produced no module at all. */
export class CvError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CvError";
  }
}

/**
 * The acquired module, in a box. The box is the whole point: `module.exports` is
 * thenable and calling that `then` hangs the realm, so it must not be what a
 * promise resolves with. An ESM `import()` boxes it for free, a module namespace
 * being an ordinary object; a `require` has to be boxed by hand.
 */
export interface AcquiredOpenCv {
  /** `module.exports`, or the namespace wrapping it: the loader takes either. */
  readonly module: unknown;
}

/**
 * How the module is acquired. The default is an `import()` of the package, which
 * is what a browser and a Worker take. It is injectable for exactly one reason:
 * under vitest that dynamic import never resolves, while
 * `createRequire(import.meta.url)` loads the same bundle in under 300 ms, so
 * test/support/opencv.ts passes a source of its own, the way
 * test/support/golden.ts injects node:fs and pngjs into the golden reader. The
 * consequence is that the default path is *not* covered by the Node tests, and
 * exercising it is the bench page's job.
 */
export type OpenCvSource = () => Promise<AcquiredOpenCv>;

/**
 * The members this port calls, in case-insensitive alphabetical order so that
 * the members an error names are deterministic. `createCLAHE` is deliberately
 * absent: the JS build has no such function, only the `CLAHE` constructor.
 */
const REQUIRED_MEMBERS = [
  "boundingRect",
  "calcHist",
  "CLAHE",
  "contourArea",
  "cvtColor",
  "dilate",
  "ellipse2Poly",
  "erode",
  "findContours",
  "fillConvexPoly",
  "fitEllipse",
  "getAffineTransform",
  "getStructuringElement",
  "Mat",
  "matFromArray",
  "MatVector",
  "minAreaRect",
  "morphologyEx",
  "pointPolygonTest",
  "resize",
  "PointVector",
  "rotatedRectangleIntersection",
  "Size",
  "subtract",
  "threshold",
  "warpAffine",
] as const;

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null;

/**
 * As much of a parse as an emscripten bundle admits: every member the port calls
 * is callable. Signatures are out of reach, which is why the tests assert the
 * version through getBuildInformation() as well.
 */
function isOpenCv(value: unknown): value is OpenCv {
  return (
    isRecord(value) &&
    REQUIRED_MEMBERS.every((name) => typeof value[name] === "function")
  );
}

function incomplete(surface: Record<string, unknown>, why: string): CvError {
  const missing = REQUIRED_MEMBERS.filter(
    (name) => typeof surface[name] !== "function"
  );
  return new CvError(
    `${why}: no ${missing.join(", ")}. @techstark/opencv-js is pinned at 4.12.0 and this is not that build`
  );
}

/**
 * A CommonJS bundle reaches an ESM importer as `{ default: module.exports }`,
 * through Node's own interop and through every bundler's, while a `require`
 * hands back `module.exports` itself. The wrapper is recognised by what it
 * wraps, never by itself: `Mat` and emscripten's `then` shim are the only
 * evidence either way, and before initialization only the shim is there.
 * Measured: a `require` gives an object whose own `default` is undefined, and
 * Node's namespace for this package is `cv`, `default` and `module.exports`.
 */
function surfaceOf(value: unknown): unknown {
  if (!isRecord(value)) {
    return value;
  }
  const inner = value.default;
  if (
    isRecord(inner) &&
    (typeof inner.Mat === "function" || typeof inner.then === "function")
  ) {
    return inner;
  }
  return value;
}

async function acquire(source: OpenCvSource): Promise<OpenCv> {
  const { module } = await source();
  const surface = surfaceOf(module);
  if (!isRecord(surface)) {
    throw new CvError(
      `the opencv.js source produced ${typeof surface}, not a module object`
    );
  }
  if (typeof surface.Mat !== "function") {
    if (typeof surface.then !== "function") {
      throw incomplete(
        surface,
        "the opencv.js source produced neither an initialized module nor an emscripten module to wait on"
      );
    }
    // Wrapped rather than handed `resolve` directly, because the callback is
    // invoked by the bundle: `run()` passes nothing today, and the `then` shim
    // passes `Module`. One upstream edit to the former would resolve this
    // `Promise<void>` with the thenable, in the one place a third party controls
    // the call.
    await new Promise<void>((resolve) => {
      surface.onRuntimeInitialized = () => {
        resolve();
      };
    });
  }
  // `then` outlives initialization, and while it is there every caller of this
  // loader is one `Promise.resolve` away from the microtask loop above. The
  // outer guard is what makes the write safe on a module namespace, which
  // refuses one; the inner check is because a silent refusal (an accessor, a
  // proxy) would otherwise surface as a frozen tab somewhere else entirely.
  if (typeof surface.then === "function") {
    surface.then = undefined;
    if (typeof surface.then === "function") {
      throw new CvError(
        "opencv.js would not give up its `then` shim, so it can never reach a promise: the module object refused the write"
      );
    }
  }
  if (!isOpenCv(surface)) {
    throw incomplete(surface, "opencv.js reported its runtime initialized");
  }
  return surface;
}

const importOpenCv: OpenCvSource = () =>
  import("./default-source.js").then((wrapper) => wrapper.acquiredOpenCv);

let loading: Promise<OpenCv> | undefined;

/**
 * Memoised per realm: concurrent callers share one initialization, and a call
 * after success never touches its source. A rejection clears the memo, so a
 * load that failed on a bad fetch or a bad build can be retried; only success
 * latches.
 */
export function loadOpenCv(
  source: OpenCvSource = importOpenCv
): Promise<OpenCv> {
  const existing = loading;
  if (existing !== undefined) {
    return existing;
  }
  const started = acquire(source);
  loading = started;
  return started.catch((cause: unknown) => {
    if (loading === started) {
      loading = undefined;
    }
    throw cause;
  });
}

/**
 * Anything opencv.js allocates in wasm memory and expects to be freed by hand:
 * Mat, MatVector, CLAHE. `isDeleted` is part of the contract because every
 * embind instance has it, and it is what makes release() safe to repeat.
 */
export interface Deletable {
  readonly delete: () => void;
  readonly isDeleted: () => boolean;
}

/**
 * Owns the lifetime of everything one operation allocates. A leaked Mat is wasm
 * heap that never comes back, so the port frees in one place rather than putting
 * a `.delete()` beside every call.
 */
export class MatScope {
  private readonly owned: Deletable[] = [];

  /** Returns `value`, so allocating it and giving it an owner is one expression. */
  keep<T extends Deletable>(value: T): T {
    this.owned.push(value);
    return value;
  }

  /**
   * Drains before freeing, so a second release frees nothing, and skips what
   * somebody already freed, because embind throws `Mat instance already deleted`
   * on a second delete.
   */
  release(): void {
    for (const value of this.owned.splice(0)) {
      try {
        if (!value.isDeleted()) {
          value.delete();
        }
      } catch {
        // A delete that throws has already lost that value's wasm memory, and
        // nothing outside this loop can still reach the rest of the scope, so
        // freeing them beats reporting the first failure. release() also runs
        // in a finally, where a throw would replace the body's own error.
      }
    }
  }
}

/**
 * Runs `body` with a scope and releases it however the body ends. Synchronous
 * because every opencv.js call is: an async body would have its Mats freed at
 * the first await it suspended on.
 */
export function withMatScope<T>(body: (scope: MatScope) => T): T {
  const scope = new MatScope();
  try {
    return body(scope);
  } finally {
    scope.release();
  }
}

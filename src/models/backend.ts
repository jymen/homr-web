/**
 * Which backend this realm got, why, and the one-shot configuration of
 * onnxruntime's global `env` that goes with it. This is the only module in the
 * library that reads a capability off the host or writes global state.
 *
 * onnxruntime reads `env.wasm.numThreads` and its GPU device only before the
 * first session is created and never again, so the configuration happens once
 * per realm and a ModelRuntime is the proof that it happened. You cannot
 * construct a ModelStore without one.
 */

import { env } from "onnxruntime-web";
import { BACKENDS, type Backend } from "../result.js";
import { ModelError } from "./errors.js";
import type { ExecutionProvider, Placement } from "./manifest.js";

/**
 * What the host offers, as values rather than as a set of `typeof` tests spread
 * through the decision. Fabricating one is how chooseBackend is tested
 * exhaustively with no browser and no models, which is why the adapter *handle*
 * is not in here: nothing but configure() needs it, and no test can build one.
 */
export interface RuntimeCapabilities {
  readonly adapterInfo: string | undefined;
  /** Browser-only evidence for why sharedMemory is what it is; undefined on Node, where there is no such gate. Reported, never tested against. */
  readonly crossOriginIsolated: boolean | undefined;
  readonly hardwareConcurrency: number;
  /** navigator.gpu granted an adapter. A host can have WebGPU and still refuse one, so this is not "navigator.gpu exists". */
  readonly hasAdapter: boolean;
  /** `typeof SharedArrayBuffer === "function"`: the actual capability behind WebAssembly threads, and the same expression on Node and in a browser, which is why this probe has no environment branch in it. */
  readonly sharedMemory: boolean;
}

export interface RuntimeProbe extends RuntimeCapabilities {
  /** The adapter itself, when there was one. Kept so startRuntime builds the device from it instead of letting onnxruntime request a second adapter. probeRuntime is the only producer, and it is what keeps this in step with hasAdapter. */
  readonly adapter: GPUAdapter | undefined;
}

export interface RuntimeOptions {
  /** Refuse anything above this. "wasm" pins the single-threaded CPU path, which is what a golden test wants even on a machine that has WebGPU. */
  readonly maxBackend?: Backend;
  readonly numThreads?: number;
  readonly powerPreference?: "high-performance" | "low-power";
  /** Prefix URL for onnxruntime's own .wasm/.mjs files. Omit on Node: the node build carries its own. Narrowed from ort's string-or-record type because the record form buys nothing here. */
  readonly wasmPaths?: string;
}

/**
 * Proof that this realm's onnxruntime env has been configured, and the record
 * of what was actually applied.
 */
export interface ModelRuntime {
  readonly backend: Backend;
  /** What was written to env.wasm.numThreads. Reported as applied, not as requested. */
  readonly numThreads: number;
  readonly probe: RuntimeProbe;
  readonly reason: string;
}

/**
 * Unmeasured, and deliberately not onnxruntime's own 0, which means "take what
 * you like": this runs in a Worker on a musician's laptop while the interface
 * has to stay responsive, and a segmentation pass that consumes every core to
 * finish half a second sooner is the wrong trade for an interactive
 * application. The bench reports the number so phase 3 can revise it.
 */
const DEFAULT_MAX_THREADS = 4;

const rankOf = (backend: Backend): number => BACKENDS.indexOf(backend);

/** vendor and architecture, which is all a reason string can usefully carry; `undefined` when the adapter reports neither. */
function describe(adapter: GPUAdapter): string | undefined {
  const parts = [adapter.info.vendor, adapter.info.architecture].filter(
    (part) => part.length > 0
  );
  return parts.length === 0 ? undefined : parts.join(" ");
}

/**
 * `options.maxBackend` is honoured here as well as in chooseBackend, so a
 * caller that has already refused WebGPU does not ask the host for an adapter
 * it will not use. chooseBackend clamps again, which keeps it a pure function
 * of the probe it is handed.
 */
export async function probeRuntime(
  options: RuntimeOptions = {}
): Promise<RuntimeProbe> {
  const wantsGpu =
    options.maxBackend === undefined || options.maxBackend === "webgpu";
  const gpu: GPU | undefined =
    typeof navigator === "undefined" ? undefined : navigator.gpu;
  let adapter: GPUAdapter | undefined;
  if (wantsGpu && gpu !== undefined) {
    // The power preference goes to our own requestAdapter because
    // env.webgpu.powerPreference is deprecated in 1.30.0 in favour of handing
    // onnxruntime a device we built ourselves.
    const requested =
      options.powerPreference === undefined
        ? await gpu.requestAdapter()
        : await gpu.requestAdapter({
            powerPreference: options.powerPreference,
          });
    adapter = requested ?? undefined;
  }
  return {
    adapter,
    adapterInfo: adapter === undefined ? undefined : describe(adapter),
    crossOriginIsolated:
      typeof crossOriginIsolated === "boolean"
        ? crossOriginIsolated
        : undefined,
    hardwareConcurrency:
      typeof navigator === "undefined" ? 1 : navigator.hardwareConcurrency,
    hasAdapter: adapter !== undefined,
    sharedMemory: typeof SharedArrayBuffer === "function",
  };
}

function capabilityOf(probe: RuntimeCapabilities): Backend {
  if (probe.hasAdapter) {
    return "webgpu";
  }
  return probe.sharedMemory ? "wasm-threads" : "wasm";
}

function reasonFor(
  probe: RuntimeCapabilities,
  chosen: Backend,
  capable: Backend
): string {
  const capped =
    chosen === capable ? "" : `, capped from ${capable} by request`;
  if (chosen === "webgpu") {
    const named =
      probe.adapterInfo === undefined ? "" : ` (${probe.adapterInfo})`;
    return `webgpu: navigator.gpu granted an adapter${named}${capped}`;
  }
  const gpu = probe.hasAdapter
    ? "a WebGPU adapter was available"
    : "navigator.gpu granted no adapter";
  if (chosen === "wasm-threads") {
    return `wasm-threads: ${gpu}, and SharedArrayBuffer is present on ${probe.hardwareConcurrency} logical cores${capped}`;
  }
  if (probe.sharedMemory) {
    return `wasm: ${gpu}, and SharedArrayBuffer is present${capped}`;
  }
  // The only fixable case, and the only one worth naming a remedy for.
  const isolation =
    probe.crossOriginIsolated === false
      ? " because the page is not cross-origin isolated (COOP same-origin with COEP require-corp would fix it)"
      : "";
  return `wasm: ${gpu}, and there is no SharedArrayBuffer${isolation}${capped}`;
}

export function chooseBackend(
  probe: RuntimeCapabilities,
  options: RuntimeOptions = {}
): { backend: Backend; reason: string } {
  const capable = capabilityOf(probe);
  const { maxBackend } = options;
  const backend =
    maxBackend !== undefined && rankOf(maxBackend) > rankOf(capable)
      ? maxBackend
      : capable;
  return { backend, reason: reasonFor(probe, backend, capable) };
}

/** The placement a runtime implies, and the store's default. "wasm-threads" and "wasm" give the same answer: they are the same execution provider, differing only in the realm's env.wasm.numThreads. */
export function placementOf(runtime: ModelRuntime): Placement {
  const provider: ExecutionProvider =
    runtime.backend === "webgpu" ? "webgpu" : "wasm";
  return { artifactsFor: runtime.backend, provider };
}

/** WebAssembly threads need SharedArrayBuffer, so the webgpu backend on a page without it still runs its decoder single-threaded. */
function threadsFor(
  backend: Backend,
  probe: RuntimeCapabilities,
  requested: number | undefined
): number {
  if (backend === "wasm" || !probe.sharedMemory) {
    return 1;
  }
  if (requested !== undefined) {
    return Math.max(1, Math.trunc(requested));
  }
  return Math.min(
    DEFAULT_MAX_THREADS,
    Math.max(1, probe.hardwareConcurrency - 1)
  );
}

/** The three settings that reach onnxruntime's env and can therefore never be changed afterwards. Compared field by field, so `{}` twice and the same numThreads twice are both the idempotent case. */
interface Applied {
  readonly backend: Backend;
  readonly numThreads: number;
  readonly wasmPaths: string | undefined;
}

let started:
  | { readonly applied: Applied; readonly runtime: Promise<ModelRuntime> }
  | undefined;

function conflictsBetween(frozen: Applied, wanted: Applied): string[] {
  const fields: string[] = [];
  if (frozen.backend !== wanted.backend) {
    fields.push(`backend ${frozen.backend} not ${wanted.backend}`);
  }
  if (frozen.numThreads !== wanted.numThreads) {
    fields.push(`numThreads ${frozen.numThreads} not ${wanted.numThreads}`);
  }
  if (frozen.wasmPaths !== wanted.wasmPaths) {
    fields.push(`wasmPaths ${frozen.wasmPaths} not ${wanted.wasmPaths}`);
  }
  return fields;
}

async function configure(
  probe: RuntimeProbe,
  applied: Applied,
  reason: string
): Promise<ModelRuntime> {
  env.wasm.numThreads = applied.numThreads;
  if (applied.wasmPaths !== undefined) {
    env.wasm.wasmPaths = applied.wasmPaths;
  }
  // Phases 3 and 7 own their Workers, and ort's proxy worker would nest one
  // inside another.
  env.wasm.proxy = false;
  if (applied.backend === "webgpu" && probe.adapter !== undefined) {
    env.webgpu.device = await probe.adapter.requestDevice();
  }
  return {
    backend: applied.backend,
    numThreads: applied.numThreads,
    probe,
    reason,
  };
}

/**
 * Probe, decide, configure, freeze. Memoised per realm, and idempotent on
 * equal input: a Worker re-entering its own init with the same options gets the
 * same runtime back. A second call whose options would apply a different
 * backend, thread count or wasmPaths throws instead of handing back a runtime
 * whose settings are somebody else's.
 *
 * A rejection clears the memo, so a failed configure can be retried. The
 * second call probes again rather than trusting the first probe, because the
 * thread-count default is derived from it; one extra requestAdapter on a
 * re-entered init is cheaper than a stale capability.
 */
export async function startRuntime(
  options: RuntimeOptions = {}
): Promise<ModelRuntime> {
  const probe = await probeRuntime(options);
  const { backend, reason } = chooseBackend(probe, options);
  const applied: Applied = {
    backend,
    numThreads: threadsFor(backend, probe, options.numThreads),
    wasmPaths: options.wasmPaths,
  };
  const existing = started;
  if (existing !== undefined) {
    const conflicts = conflictsBetween(existing.applied, applied);
    if (conflicts.length > 0) {
      throw new ModelError(
        "runtime-frozen",
        `this realm's onnxruntime env is already configured: ${conflicts.join(", ")}`,
        { actual: applied.backend, expected: existing.applied.backend }
      );
    }
    return await existing.runtime;
  }
  const runtime = configure(probe, applied, reason);
  started = { applied, runtime };
  return await runtime.catch((cause: unknown) => {
    if (started?.runtime === runtime) {
      started = undefined;
    }
    throw cause;
  });
}

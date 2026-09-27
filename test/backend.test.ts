import { describe, expect, it } from "vitest";
import {
  chooseBackend,
  type ModelRuntime,
  placementOf,
  probeRuntime,
  type RuntimeCapabilities,
  startRuntime,
} from "../src/models/backend.js";
import { ModelError } from "../src/models/errors.js";
import { BACKENDS, type Backend } from "../src/result.js";

const capabilities = (
  over: Partial<RuntimeCapabilities> = {}
): RuntimeCapabilities => ({
  adapterInfo: undefined,
  crossOriginIsolated: undefined,
  hardwareConcurrency: 8,
  hasAdapter: false,
  sharedMemory: false,
  ...over,
});

const MAX_BACKENDS: readonly (Backend | undefined)[] = [undefined, ...BACKENDS];
const rankOf = (backend: Backend): number => BACKENDS.indexOf(backend);
const BACKEND_AND_THREADS_CONFLICT =
  /backend wasm-threads not wasm, numThreads 2 not 1/;
const THREADS_CONFLICT = /numThreads 2 not 3/;
const WASM_PATHS_CONFLICT = /wasmPaths/;

/** maxBackend, hasAdapter, sharedMemory, the backend that must come out. */
const CAPPED: readonly [Backend, boolean, boolean, Backend][] = [
  ["webgpu", true, true, "webgpu"],
  ["wasm-threads", true, true, "wasm-threads"],
  ["wasm", true, true, "wasm"],
  ["wasm", false, true, "wasm"],
];

describe("chooseBackend", () => {
  it.each([
    ["an adapter and shared memory", true, true, "webgpu"],
    ["an adapter but no shared memory", true, false, "webgpu"],
    ["no adapter but shared memory", false, true, "wasm-threads"],
    ["neither", false, false, "wasm"],
  ])("takes the best of %s", (_label, hasAdapter, sharedMemory, expected) => {
    expect(
      chooseBackend(capabilities({ hasAdapter, sharedMemory })).backend
    ).toBe(expected);
  });

  it("never returns a backend better than maxBackend, and never worse than it needs to", () => {
    const seen: string[] = [];
    for (const hasAdapter of [false, true]) {
      for (const sharedMemory of [false, true]) {
        for (const maxBackend of MAX_BACKENDS) {
          const probe = capabilities({ hasAdapter, sharedMemory });
          const capable = chooseBackend(probe).backend;
          const chosen = chooseBackend(
            probe,
            maxBackend === undefined ? {} : { maxBackend }
          ).backend;
          expect(rankOf(chosen)).toBeGreaterThanOrEqual(rankOf(capable));
          if (maxBackend !== undefined) {
            expect(rankOf(chosen)).toBeGreaterThanOrEqual(rankOf(maxBackend));
          }
          seen.push(
            `adapter=${hasAdapter} sab=${sharedMemory} max=${maxBackend ?? "-"} -> ${chosen}`
          );
        }
      }
    }
    expect(seen).toHaveLength(16);
  });

  it.each(CAPPED)(
    "caps a capable host at %s",
    (maxBackend, hasAdapter, sharedMemory, expected) => {
      const probe = capabilities({ hasAdapter, sharedMemory });
      expect(chooseBackend(probe, { maxBackend }).backend).toBe(expected);
    }
  );

  it("does not raise a host above what it can do", () => {
    expect(
      chooseBackend(capabilities({ hasAdapter: false, sharedMemory: false }), {
        maxBackend: "webgpu",
      }).backend
    ).toBe("wasm");
  });
});

describe("the reason a backend was chosen", () => {
  it("names the adapter when there is one", () => {
    expect(
      chooseBackend(
        capabilities({ adapterInfo: "apple apple-m1", hasAdapter: true })
      ).reason
    ).toBe("webgpu: navigator.gpu granted an adapter (apple apple-m1)");
  });

  it("names the missing capability, not the chosen one", () => {
    expect(
      chooseBackend(capabilities({ sharedMemory: true })).reason
    ).toContain("navigator.gpu granted no adapter");
  });

  it("cites cross-origin isolation only where that is the fixable cause", () => {
    const inBrowser = chooseBackend(
      capabilities({ crossOriginIsolated: false })
    ).reason;
    expect(inBrowser).toContain("not cross-origin isolated");
    // Node has no such gate, so naming it would be advice nobody can act on.
    expect(chooseBackend(capabilities()).reason).not.toContain(
      "cross-origin isolated"
    );
    expect(
      chooseBackend(
        capabilities({ crossOriginIsolated: true, sharedMemory: true })
      ).reason
    ).not.toContain("cross-origin isolated");
  });

  it("says when the answer was capped rather than measured", () => {
    const capped = chooseBackend(
      capabilities({ hasAdapter: true, sharedMemory: true }),
      { maxBackend: "wasm" }
    ).reason;
    expect(capped).toContain("a WebGPU adapter was available");
    expect(capped).toContain("capped from webgpu by request");
    expect(chooseBackend(capabilities()).reason).not.toContain("capped");
  });
});

describe("placementOf", () => {
  const runtimeOn = (backend: Backend): ModelRuntime => ({
    backend,
    numThreads: 1,
    probe: { adapter: undefined, ...capabilities() },
    reason: "fabricated",
  });

  it("asks for the webgpu provider only on the webgpu backend", () => {
    expect(placementOf(runtimeOn("webgpu"))).toEqual({
      artifactsFor: "webgpu",
      provider: "webgpu",
    });
  });

  it("gives wasm-threads and wasm the same provider", () => {
    expect(placementOf(runtimeOn("wasm-threads")).provider).toBe("wasm");
    expect(placementOf(runtimeOn("wasm")).provider).toBe("wasm");
  });

  it("carries the backend through as the artifact side", () => {
    for (const backend of BACKENDS) {
      expect(placementOf(runtimeOn(backend)).artifactsFor).toBe(backend);
    }
  });
});

describe("probeRuntime under Node", () => {
  it("finds no adapter and no cross-origin gate, and does report SharedArrayBuffer", async () => {
    const probe = await probeRuntime();
    expect(probe.hasAdapter).toBe(false);
    expect(probe.adapter).toBeUndefined();
    expect(probe.crossOriginIsolated).toBeUndefined();
    expect(probe.sharedMemory).toBe(true);
    expect(probe.hardwareConcurrency).toBeGreaterThan(0);
  });
});

// These share one realm's frozen onnxruntime env, so the order is the test:
// the first call configures it and every later one is measured against that.
// numThreads is given explicitly, because the default is derived from the
// machine's core count and a conflict assertion must not depend on that.
describe("startRuntime freezes this realm", () => {
  const FROZEN = { maxBackend: "wasm-threads", numThreads: 2 } as const;

  it("applies what was asked for and says why that backend", async () => {
    const runtime = await startRuntime(FROZEN);
    expect(runtime.backend).toBe("wasm-threads");
    expect(runtime.numThreads).toBe(2);
    expect(runtime.reason).toContain("SharedArrayBuffer is present");
  });

  it("returns the same runtime for the same options", async () => {
    expect(await startRuntime(FROZEN)).toBe(await startRuntime(FROZEN));
  });

  it("names both fields when pinning the CPU path would change the thread count too", async () => {
    // maxBackend "wasm" is single-threaded by definition, so this one rejection
    // is also the proof that the "wasm" backend applies numThreads 1.
    await expect(startRuntime({ maxBackend: "wasm" })).rejects.toThrow(
      BACKEND_AND_THREADS_CONFLICT
    );
  });

  it("refuses a different thread count on the same backend", async () => {
    await expect(startRuntime({ ...FROZEN, numThreads: 3 })).rejects.toThrow(
      THREADS_CONFLICT
    );
  });

  it("refuses a different wasmPaths", async () => {
    await expect(
      startRuntime({ ...FROZEN, wasmPaths: "/ort/" })
    ).rejects.toThrow(WASM_PATHS_CONFLICT);
  });

  it("throws a ModelError, so a caller can act on the code", async () => {
    const thrown = await startRuntime({ maxBackend: "wasm" }).catch(
      (error: unknown) => error
    );
    expect(thrown).toBeInstanceOf(ModelError);
    expect(thrown).toMatchObject({ code: "runtime-frozen" });
  });
});

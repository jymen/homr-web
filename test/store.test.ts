import { readFileSync } from "node:fs";
import { afterEach, describe, expect, it, vi } from "vitest";
import { type ModelRuntime, startRuntime } from "../src/models/backend.js";
import {
  browserCache,
  cacheApiCache,
  type ModelCache,
  memoryCache,
  opfsCache,
} from "../src/models/cache.js";
import { ModelError } from "../src/models/errors.js";
import {
  type ArtifactRecord,
  DEFAULT_CATALOG,
  type ModelCatalog,
} from "../src/models/manifest.js";
import {
  type FetchBytes,
  httpFetch,
  type ModelEvent,
  ModelStore,
  sha256Hex,
} from "../src/models/store.js";
import {
  describeWithModels,
  FETCH_MODELS_HINT,
  MODELS_PRESENT,
  modelFileFor,
} from "./support/models.js";

const BASE_URL = "https://models.example/artifacts/";
/** Twelve bytes the test owns. Nothing here is a model: the store never opens one. */
const ARTIFACT_BYTES = new Uint8Array([
  8, 1, 18, 7, 104, 111, 109, 114, 45, 119, 101, 98,
]);
/** The same length and not the same bytes, so the digest refuses these rather than the length. */
const TAMPERED_BYTES = new Uint8Array([
  8, 1, 18, 7, 104, 111, 109, 114, 45, 119, 101, 99,
]);
const TRUNCATED_BYTES = ARTIFACT_BYTES.slice(0, 4);

const NO_CACHE_API = /no Cache API/;
const NO_OPFS = /origin-private file system/;
const WRONG_LENGTH = /holds 4 bytes, the manifest says 12 bytes/;
const WEBGPU_ON_WASM = /webgpu execution provider on a wasm runtime/;
const NOT_FOUND = /404 Not Found/;

/** Every store in this file shares one realm's frozen env, so every call must ask for the same thing. */
const wasmRuntime = (): Promise<ModelRuntime> =>
  startRuntime({ maxBackend: "wasm" });

const artifactOf = async (bytes: Uint8Array): Promise<ArtifactRecord> => {
  const sha256 = await sha256Hex(bytes);
  return {
    batchDim: null,
    bytes: bytes.length,
    inputs: [{ name: "input", shape: [1], type: "float32" }],
    outputs: [{ name: "output", shape: [1], type: "float32" }],
    sha256,
    urlPath: `${sha256}/fake.onnx`,
  };
};

/** segnet's row replaced, so plan("segnet") on the wasm placement resolves to bytes the test made. */
const catalogWith = (artifact: ArtifactRecord): ModelCatalog => ({
  artifacts: { ...DEFAULT_CATALOG.artifacts, "segnet-308-fp32": artifact },
  roles: DEFAULT_CATALOG.roles,
});

interface Gate<T> {
  readonly open: (value: T) => void;
  readonly settled: Promise<T>;
}

const gate = <T>(): Gate<T> => {
  let open: (value: T) => void = () => undefined;
  const settled = new Promise<T>((resolve) => {
    open = resolve;
  });
  return { open, settled };
};

interface Recorded {
  readonly calls: string[];
  readonly fetchBytes: FetchBytes;
  /** Resolves on the first call, so a test can close the store while a download is genuinely in flight. */
  readonly started: Promise<void>;
}

/** Rejects on abort the way a real fetch does, which is what makes close() observable. */
const fetchOf = (reply: () => Promise<Uint8Array>): Recorded => {
  const calls: string[] = [];
  const first = gate<void>();
  return {
    calls,
    fetchBytes: ({ signal, url }) => {
      calls.push(url);
      first.open(undefined);
      const answer = reply();
      if (signal === undefined) {
        return answer;
      }
      return Promise.race([
        answer,
        new Promise<Uint8Array>((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            reject(new Error("aborted"));
          });
        }),
      ]);
    },
    started: first.settled,
  };
};

interface Watched {
  readonly events: ModelEvent[];
  readonly onEvent: (event: ModelEvent) => void;
}

const watch = (): Watched => {
  const events: ModelEvent[] = [];
  return {
    events,
    onEvent: (event) => {
      events.push(event);
    },
  };
};

const traceOf = (events: readonly ModelEvent[]): string[] =>
  events.map((event) =>
    event.kind === "rejected" ? `${event.kind}/${event.source}` : event.kind
  );

const storeOf = async (parts: {
  cache: ModelCache;
  catalog: ModelCatalog;
  fetchBytes: FetchBytes;
  onEvent?: (event: ModelEvent) => void;
}): Promise<ModelStore> =>
  new ModelStore({
    baseUrl: BASE_URL,
    runtime: await wasmRuntime(),
    ...parts,
  });

describe("loading an artifact's bytes", () => {
  it("shares one download between two concurrent prefetches", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const held = gate<Uint8Array>();
    const fetch = fetchOf(() => held.settled);
    const store = await storeOf({
      cache: memoryCache(),
      catalog: catalogWith(artifact),
      fetchBytes: fetch.fetchBytes,
    });

    const first = store.prefetch(["segnet"]);
    const second = store.prefetch(["segnet"]);
    held.open(ARTIFACT_BYTES);
    await Promise.all([first, second]);

    expect(fetch.calls).toEqual([`${BASE_URL}${artifact.urlPath}`]);
  });

  it("verifies a cache hit rather than trusting it, and fetches nothing twice", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const fetch = fetchOf(() => Promise.resolve(ARTIFACT_BYTES));
    const { events, onEvent } = watch();
    const store = await storeOf({
      cache: memoryCache(),
      catalog: catalogWith(artifact),
      fetchBytes: fetch.fetchBytes,
      onEvent,
    });

    await store.prefetch(["segnet"]);
    const afterFirst = events.length;
    await store.prefetch(["segnet"]);

    expect(fetch.calls).toHaveLength(1);
    expect(traceOf(events.slice(afterFirst))).toEqual(["cached", "verified"]);
    expect(events[afterFirst]).toMatchObject({
      bytes: ARTIFACT_BYTES.length,
      from: "memory",
    });
  });

  it("drops a cache entry that does not verify and re-fetches it once", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const cache = memoryCache();
    await cache.write(artifact.sha256, TAMPERED_BYTES);
    const fetch = fetchOf(() => Promise.resolve(ARTIFACT_BYTES));
    const { events, onEvent } = watch();
    const store = await storeOf({
      cache,
      catalog: catalogWith(artifact),
      fetchBytes: fetch.fetchBytes,
      onEvent,
    });

    await store.prefetch(["segnet"]);

    expect(traceOf(events)).toEqual(["rejected/cache", "verified"]);
    expect(fetch.calls).toHaveLength(1);
    expect(await cache.read(artifact.sha256)).toEqual(ARTIFACT_BYTES);
  });

  it("gives up when the network is wrong too, without a second attempt", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const cache = memoryCache();
    await cache.write(artifact.sha256, TAMPERED_BYTES);
    const fetch = fetchOf(() => Promise.resolve(TAMPERED_BYTES));
    const { events, onEvent } = watch();
    const store = await storeOf({
      cache,
      catalog: catalogWith(artifact),
      fetchBytes: fetch.fetchBytes,
      onEvent,
    });

    const thrown = await store
      .prefetch(["segnet"])
      .catch((error: unknown) => error);

    expect(thrown).toBeInstanceOf(ModelError);
    expect(thrown).toMatchObject({
      code: "digest-mismatch",
      detail: {
        actual: await sha256Hex(TAMPERED_BYTES),
        id: "segnet-308-fp32",
      },
    });
    expect(fetch.calls).toHaveLength(1);
    expect(traceOf(events)).toEqual(["rejected/cache", "rejected/network"]);
  });

  it("writes nothing to the cache that did not verify", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const cache = memoryCache();
    const store = await storeOf({
      cache,
      catalog: catalogWith(artifact),
      fetchBytes: fetchOf(() => Promise.resolve(TAMPERED_BYTES)).fetchBytes,
    });

    await expect(store.prefetch(["segnet"])).rejects.toBeInstanceOf(ModelError);

    expect(await cache.read(artifact.sha256)).toBeUndefined();
  });

  it("refuses bytes of the wrong length by their length, before any digest", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const { events, onEvent } = watch();
    const store = await storeOf({
      cache: memoryCache(),
      catalog: catalogWith(artifact),
      fetchBytes: fetchOf(() => Promise.resolve(TRUNCATED_BYTES)).fetchBytes,
      onEvent,
    });

    await expect(store.prefetch(["segnet"])).rejects.toThrow(WRONG_LENGTH);

    expect(traceOf(events)).toEqual(["rejected/network"]);
    // Two byte counts where a digest comparison would have put two hashes:
    // the length is what refused these bytes, and nothing was hashed.
    expect(events[0]).toMatchObject({
      actual: "4 bytes",
      expected: "12 bytes",
    });
  });

  it("does not fail a load because the cache could not keep it", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const store = await storeOf({
      cache: {
        ...memoryCache(),
        write: () => Promise.reject(new Error("quota exceeded")),
      },
      catalog: catalogWith(artifact),
      fetchBytes: fetchOf(() => Promise.resolve(ARTIFACT_BYTES)).fetchBytes,
    });

    await expect(store.prefetch(["segnet"])).resolves.toBeUndefined();
  });

  it("makes a load in flight reject with store-closed, and closes twice happily", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const held = gate<Uint8Array>();
    const fetch = fetchOf(() => held.settled);
    const store = await storeOf({
      cache: memoryCache(),
      catalog: catalogWith(artifact),
      fetchBytes: fetch.fetchBytes,
    });

    const loading = store.prefetch(["segnet"]);
    await fetch.started;
    await store.close();
    await store.close();
    const thrown = await loading.catch((error: unknown) => error);

    expect(thrown).toBeInstanceOf(ModelError);
    expect(thrown).toMatchObject({ code: "store-closed" });
    held.open(ARTIFACT_BYTES);
  });
});

describe("planning a role", () => {
  it("fetches nothing", async () => {
    const artifact = await artifactOf(ARTIFACT_BYTES);
    const fetch = fetchOf(() => Promise.resolve(ARTIFACT_BYTES));
    const store = await storeOf({
      cache: memoryCache(),
      catalog: catalogWith(artifact),
      fetchBytes: fetch.fetchBytes,
    });

    const plan = store.plan("segnet");

    expect(plan.artifactId).toBe("segnet-308-fp32");
    expect(plan.artifact.sha256).toBe(artifact.sha256);
    expect(plan.provider).toBe("wasm");
    expect(fetch.calls).toEqual([]);
  });
});

describe("the placement a store accepts", () => {
  const optionsFor = async (): Promise<{
    baseUrl: string;
    cache: ModelCache;
    runtime: ModelRuntime;
  }> => ({
    baseUrl: BASE_URL,
    cache: memoryCache(),
    runtime: await wasmRuntime(),
  });

  it("refuses the webgpu provider on a runtime that configured no device", async () => {
    const options = await optionsFor();
    const build = () =>
      new ModelStore({
        ...options,
        placement: { artifactsFor: "wasm", provider: "webgpu" },
      });

    expect(build).toThrow(WEBGPU_ON_WASM);
    expect(build).toThrow(ModelError);
  });

  it("accepts the fp16 artifacts on the wasm provider, which is how CI covers them", async () => {
    const store = new ModelStore({
      ...(await optionsFor()),
      placement: { artifactsFor: "webgpu", provider: "wasm" },
    });

    expect(store.plan("segnet").artifactId).toBe("segnet-308-fp16");
    expect(store.plan("segnet").provider).toBe("wasm");
  });

  it("defaults to the placement the runtime implies", async () => {
    const store = new ModelStore(await optionsFor());

    expect(store.placement).toEqual({ artifactsFor: "wasm", provider: "wasm" });
  });
});

describe("the caches", () => {
  it("round-trips bytes in memory and forgets them on drop", async () => {
    const cache = memoryCache();

    expect(cache.kind).toBe("memory");
    expect(await cache.read("cafe")).toBeUndefined();
    await cache.write("cafe", ARTIFACT_BYTES);
    expect(await cache.read("cafe")).toEqual(ARTIFACT_BYTES);
    await cache.drop("cafe");
    expect(await cache.read("cafe")).toBeUndefined();
  });

  it("refuses a host that has neither store, which is what browserCache falls through", async () => {
    await expect(cacheApiCache()).rejects.toThrow(NO_CACHE_API);
    await expect(opfsCache()).rejects.toThrow(NO_OPFS);
    expect((await browserCache()).kind).toBe("memory");
  });
});

describe("httpFetch", () => {
  const URL_UNDER_TEST = "https://models.invalid/fake.onnx";

  const streamOf = (
    chunks: readonly Uint8Array[]
  ): ReadableStream<Uint8Array> =>
    new ReadableStream<Uint8Array>({
      start: (controller) => {
        for (const chunk of chunks) {
          controller.enqueue(chunk);
        }
        controller.close();
      },
    });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("streams the body and reports progress against the declared length", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        new Response(
          streamOf([ARTIFACT_BYTES.slice(0, 5), ARTIFACT_BYTES.slice(5)]),
          { headers: { "content-length": String(ARTIFACT_BYTES.length) } }
        )
      )
    );
    const seen: number[][] = [];

    const bytes = await httpFetch()({
      onProgress: (received, total) => {
        seen.push([received, total]);
      },
      signal: undefined,
      url: URL_UNDER_TEST,
    });

    expect(bytes).toEqual(ARTIFACT_BYTES);
    expect(seen).toEqual([
      [5, 12],
      [12, 12],
    ]);
  });

  it("reports a non-ok response as a fetch-failed naming the status", async () => {
    vi.stubGlobal("fetch", () =>
      Promise.resolve(
        new Response("not here", { status: 404, statusText: "Not Found" })
      )
    );

    const thrown = await httpFetch()({
      onProgress: undefined,
      signal: undefined,
      url: URL_UNDER_TEST,
    }).catch((error: unknown) => error);

    expect(thrown).toBeInstanceOf(ModelError);
    expect(thrown).toMatchObject({ code: "fetch-failed" });
    expect(thrown).toMatchObject({ message: expect.stringMatching(NOT_FOUND) });
  });
});

describe("the model bytes the real artifacts need", () => {
  it("are present whenever CI is set, so absent bytes cannot look like a pass", () => {
    expect(process.env.CI === undefined || MODELS_PRESENT).toBe(true);
  });
});

describeWithModels("sha256Hex against the generated manifest", () => {
  it("agrees with the hash tools/gen-manifest.mjs recorded", async () => {
    // The smallest artifact, because this proves an agreement between two
    // implementations and not a throughput figure.
    const artifact = DEFAULT_CATALOG.artifacts["ppocr-v2-cls-mobile"];
    const path = modelFileFor(artifact);
    if (path === undefined) {
      throw new Error(FETCH_MODELS_HINT);
    }
    const bytes = readFileSync(path);

    expect(bytes.length).toBe(artifact.bytes);
    expect(await sha256Hex(bytes)).toBe(artifact.sha256);
  });
});

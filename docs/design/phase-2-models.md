# Phase 2 design: the model manifest, store, backend and session

Synthesised 2026-09-27 from two parallel candidate designs. Written against
`docs/design/phase-1-types.md`'s house style; `src/result.ts`'s `Backend` and
`test/support/golden.ts`'s injection precedent are taken as given. Every number
quoted here was measured on 2026-09-27, and each is named where it is used.

## Problem

Three phases need model bytes on a session: phase 3 (segnet in a Worker, the
go/no-go gate), phase 7 (encoder once per staff, then a 608-step decoder loop
carrying 32 cache tensors), phase 11 (three PP-OCR models). They need it on the
best backend the browser has, and they need to be told which one that was and
why, because the bench page and `RecognizeResult.log` both report it. Node in CI
is a first-class consumer with no `caches`, no `navigator.gpu`, no OPFS and no
network access to 160 MB of artifacts.

What makes the shape non-obvious is that five independent decisions all key on
the same word "model" and none of them lines up with the others:

- **Which file.** homr runs fp16 segnet and fp16 encoder on its GPU path and
  keeps the decoder fp32 (`homr/onnx_providers.py:1-16`,
  `homr/main.py:411-413`); on CPU all three are fp32.
- **Which execution provider.** Three `Backend` values, two providers:
  `wasm-threads` and `wasm` are the same `wasm` EP differing only in
  `env.wasm.numThreads`, which is realm state, not a session option. And the
  shipped decoder has no `SkipLayerNormalization` kernel on the WebGPU EP until
  phase 8 re-exports it, so on the `webgpu` backend the decoder is still on
  `wasm` — for two unrelated reasons at once.
- **Which dtype, in both directions.** Measured: the fp16 artifacts declare
  `float16` tensors on the wire, and onnxruntime-web 1.30.0 has no
  `Float16Array` on either side of a run — an fp16 input is a `Uint16Array` of
  raw half bit patterns and so is an fp16 *output* (segnet fp16 returned 17630,
  18301, 18122 where the fp32 model returned 4.865, 7.488, 6.788).
- **Where the tensor lives.** `preferredOutputLocation` cannot be a blanket
  `"gpu-buffer"`: segnet's six logit planes go to a CPU argmax
  (`inference_segnet.py`), so only the decoder's 32 `cache_out*` want GPU
  residency, and only once the decoder is on the GPU at all.
- **Which artifact versus which provider.** Measured: both segnet artifacts run
  under Node on the wasm EP (159 ms fp32, 177 ms fp16 per tile, single thread —
  not the target surface, no WebGPU conclusion drawn). So "fp16" and "WebGPU"
  are two axes that merely travel together in production, and separating them
  is what makes the fp16 branch testable in CI with no GPU.

And the encoder emits float16 `[1,1280,512]` on the GPU path while the decoder
wants float32 `context`, so a mixed-precision pipeline is the *normal* case on
WebGPU. homr performs the same cast (`staff2score.py:43-49`).

Hard constraints from phase 0/1 and the grounding: `Backend` already exists in
`src/result.ts` and is not redefined; the library does not touch `node:fs` or
`caches` directly, it takes a port, as phase 1 took a `GoldenReader`;
`onnxruntime-web`'s global `env` is one-shot per realm (`numThreads` and the GPU
device are read only before the first session); the artifacts cannot be fetched
from GitHub releases (no `access-control-allow-origin`, re-checked today), so
they are self-hosted and `baseUrl` is public API; `noUncheckedIndexedAccess`,
`exactOptionalPropertyTypes`, no `enum`, no `any`, no `!`, no `console` in
`src/`.

## Usage (caller's view)

### The README section a consumer reads

> **Models.** Two objects: a `ModelRuntime`, which is the realm's onnxruntime
> configuration, and a `ModelStore`, which is a catalogue of artifacts over
> that runtime.
>
> ```ts
> import { startRuntime, ModelStore, browserCache } from "homr-web";
>
> const runtime = await startRuntime({ wasmPaths: "/ort/" });
> const models = new ModelStore({
>   runtime,
>   baseUrl: "/models/",
>   cache: await browserCache(),
>   onEvent: (event) => port.postMessage(event),
> });
>
> const segnet = await models.open("segnet", { batch: 16 });
> ```
>
> `runtime.backend` is one of `"webgpu"`, `"wasm-threads"`, `"wasm"` and
> `runtime.reason` is one sentence saying why that one and not the next one up.
> Call `startRuntime` once per realm: onnxruntime reads `env.wasm.numThreads`
> and its GPU device only before the first session is created, so the second
> call returns the same runtime, and a second call whose options would apply
> *different* settings throws instead of pretending they took effect.
>
> You ask for a **role** — `segnet`, `encoder`, `decoder`, `ocrDetect`,
> `ocrClassify`, `ocrRecognize` — never for a file and never for a precision.
> Which artifact a role uses, on which execution provider, with which outputs
> left in GPU buffers, is one table in `src/models/manifest.ts` plus one
> derivation, and no call site repeats it. `models.plan("decoder").reason`
> tells a human what the table decided.
>
> **Writing an input tensor**, read the element type off the session, never off
> the backend: `segnet.inputSpec("input").type` is `"float16"` on the WebGPU
> path and `"float32"` on WebAssembly, and `"float16"` means IEEE halves in a
> `Uint16Array` — `float16FromFloat32` encodes one, `encodeFloat16Array` a
> buffer.
>
> **Reading an output tensor**, call `await readFloat32(tensor)` and forget the
> question: it decodes halves through a lookup table when the artifact is fp16,
> hands back the tensor's own `Float32Array` when it is not, and reads a
> GPU-resident tensor back for you. Never touch `tensor.data`. When one role's
> output becomes another role's input, call `handoff(tensor, from, to)`: it
> casts on the WebGPU path and returns the tensor untouched everywhere else.
>
> Bytes are fetched from `baseUrl + urlPath`, verified against the manifest's
> SHA-256 **on every load, including cache hits** (82 ms for all three homr
> models, measured), and cached under their hash. Nothing unverified is ever
> written to the cache, so an interrupted download leaves nothing behind. A
> cached entry that fails verification is dropped and re-fetched once.
>
> Everything the library touches outside itself is injected: `cache` and
> `fetchBytes`. Node passes `memoryCache()` and a reader over the git-ignored
> `models/` directory; the browser passes `browserCache()` (Cache API, then
> OPFS, then memory) and the default `httpFetch()`. The store's own logic has
> no environment test in it.

### Call site 1 — phase 3, the segnet Worker (`src/segmentation/worker.ts`)

```ts
import { ModelStore, browserCache, startRuntime, readFloat32 } from "../index.js";
import type { ModelSession } from "../models/session.js";
import { Tensor } from "onnxruntime-web";
import { SEGNET_INPUT } from "../model/pipeline.js";

/** The Worker owns the runtime, the store and its one session; terminating it releases all three. */
export class SegmentationWorker {
  #store: ModelStore | undefined;
  #segnet: ModelSession | undefined;

  async start(config: WorkerConfig): Promise<StartedMessage> {
    const runtime = await startRuntime({ wasmPaths: config.wasmPaths });
    this.#store = new ModelStore({
      runtime,
      baseUrl: config.baseUrl,
      cache: await browserCache(),
      onEvent: (event) => self.postMessage(event),      // progress, and the bench's cache-hit proof
    });
    this.#segnet = await this.#store.open("segnet", { batch: config.batch });
    return {
      kind: "started",
      backend: runtime.backend,
      reason: runtime.reason,                            // the go/no-go gate reports this verbatim
      modelReason: this.#segnet.plan.reason,
    };
  }

  async segment(page: GrayImage): Promise<SegmentationResult> {
    const segnet = this.#segnet ?? raise("segment before start");
    const half = segnet.inputSpec("input").type === "float16";
    const dims = [this.#batch, SEGNET_INPUT.channels, SEGNET_INPUT.window, SEGNET_INPUT.window];

    for (const group of batchesOf(tilesOf(page, SEGNET_INPUT.window), this.#batch)) {
      // One branch on the input dtype, once, outside a 2.4 M-element loop: the
      // fp16 path writes IEEE halves straight from the pixel, which is why the
      // store *names* the input dtype instead of converting for us.
      const input = half
        ? new Tensor("float16", halfPlanesOf(group, dims), dims)
        : new Tensor("float32", floatPlanesOf(group, dims), dims);

      const { output } = await segnet.session.run({ input });
      // The output side has no branch at all: readFloat32 decodes the fp16
      // model's Uint16Array of half bit patterns through a table, and hands
      // the fp32 model's Float32Array straight back. argmaxPlanes keeps the
      // Float32Array signature phase 1 gave it.
      const logits = await readFloat32(output);
      mergePatches(classes, argmaxPlanes(logits, SEGNET_INPUT.classes, 320, 320), group);
    }
    return createSegmentationResult(classes);
  }
}
```

### Call site 2 — phase 7, the encoder-to-decoder handoff (`src/transformer/decoder.ts`)

```ts
import { handoff } from "../models/session.js";
import { DECODER_CACHE_IN, DECODER_CACHE_OUT } from "../models/manifest.js";

export async function readStaff(canvas: StaffCanvas, models: ModelStore): Promise<EncodedSymbol[]> {
  const encoder = await models.open("encoder");
  const decoder = await models.open("decoder");         // wasm fp32 on every backend until phase 8

  const { output } = await encoder.session.run({ input: normalizedCanvas(canvas, encoder) });
  // All this line knows is that the encoder's output becomes the decoder's
  // input. fp16-to-fp32 on WebGPU, a GPU read when the placements differ,
  // identity on WebAssembly; after phase 8 it becomes identity on WebGPU too
  // and this line does not change.
  const context = await handoff(output, encoder, decoder);

  const feeds: Record<string, Tensor> = { context, cache_len: cacheLen(0), ...startTokens() };
  for (const name of DECODER_CACHE_IN) {
    feeds[name] = emptyCache(decoder, name);
  }

  for (let step = 0; step < MAX_STEPS && !aborted(); step += 1) {
    const out = await decoder.session.run(feeds);
    // Tensors go straight back in as inputs, in whatever location they are in:
    // after phase 8 these 32 stay in GPU buffers and nothing here changes.
    for (const [i, name] of DECODER_CACHE_OUT.entries()) {
      feeds[DECODER_CACHE_IN[i] ?? raise("cache arity")] = out[name] ?? raise(name);
    }
    const heads = await Promise.all(DECODER_HEAD_OUTPUTS.map((h) => readFloat32(out[h] ?? raise(h))));
    ...
  }
}
```

### Call site 3 — phase 11, the three PP-OCR models (`src/ocr/readStrips.ts`)

```ts
export async function readTextStrips(page: GrayImage, staves: readonly Staff[], models: ModelStore) {
  // Three roles, no backend and no precision anywhere in this file. All three
  // resolve to wasm fp32 today; when phase 11 measures PP-OCR on the WebGPU EP
  // the change is one line of MODEL_ROLES, not a change here.
  const [detect, classify, recognize] = await Promise.all([
    models.open("ocrDetect"),
    models.open("ocrClassify"),
    models.open("ocrRecognize"),
  ]);
  ...
}
```

### Call site 4 — the Node golden tests (`test/models.test.ts`)

```ts
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { DEFAULT_CATALOG, ModelStore, memoryCache, resolveRole, startRuntime } from "../src/index.js";

/** models/ is git-ignored and filled by tools/fetch-models.sh; the library never reads it. */
const localModels = (): FetchBytes => async ({ url }) =>
  readFileSync(join(import.meta.dirname, "..", "models", url.slice(url.lastIndexOf("/") + 1)));

const storeOn = (runtime: ModelRuntime, placement?: Placement) =>
  new ModelStore({
    runtime,
    ...(placement === undefined ? {} : { placement }),
    baseUrl: "file:///models/",
    cache: memoryCache(),
    fetchBytes: localModels(),
  });

it("opens every role on the wasm backend", async () => {
  const runtime = await startRuntime({ maxBackend: "wasm" });   // pin the CPU path even on a WebGPU Mac
  const models = storeOn(runtime);
  for (const role of MODEL_ROLE_NAMES) {
    expect((await models.open(role)).plan.provider).toBe("wasm");
  }
  await models.close();
});

// The fp16 branch is CI-testable: both segnet artifacts run on the wasm EP
// (measured), so artifact choice and execution provider are asked for
// separately and the golden test covers fp16 with no GPU present.
it("the fp16 segnet agrees with the fp32 one on a golden tile", async () => {
  const runtime = await startRuntime({ maxBackend: "wasm" });
  const fp32 = await storeOn(runtime).open("segnet");
  const fp16 = await storeOn(runtime, { artifactsFor: "webgpu", provider: "wasm" }).open("segnet");
  expect(fp16.inputSpec("input").type).toBe("float16");
  expect(fp16.outputSpec("output").type).toBe("float16");
  const tile = goldenPageOf(fixture).preprocessed();
  const [a, b] = [await runTile(fp32, tile), await runTile(fp16, tile)];
  expect(agreementOf(argmaxOf(a), argmaxOf(b))).toBeGreaterThan(0.999);   // testing.md's mask tolerance
});

it("the golden fixtures were dumped from the artifacts the manifest names", () => {
  // meta.json records "<filename>:<sha256>" per model; the manifest records
  // the same hash. A re-pin that changes one and not the other fails here
  // instead of silently comparing the port against a different model.
  for (const fixture of listGoldenFixtures()) {
    for (const [role, entry] of Object.entries(goldenPageOf(fixture).meta().models)) {
      const plan = resolveRole(DEFAULT_CATALOG, role as ModelRole, { artifactsFor: "wasm", provider: "wasm" });
      expect(entry.split(":")[1]).toBe(plan.artifact.sha256);
    }
  }
});
```

## Shape

### Module map

Six files. The plan named four; `cache.ts` and `dtype.ts` are added because
each holds exactly the thing the other four must not contain — an environment
test, and bit-level arithmetic.

| file | holds | knows about |
|---|---|---|
| `src/models/manifest.ts` | `TensorSpec`, `ArtifactId`, `ArtifactRecord`, `ARTIFACTS` (**generated block**), `ModelRole`, `RolePolicy`, `MODEL_ROLES` (**hand-written policy**), `DEFAULT_CATALOG`, `Placement`, `resolveRole`, `ResolvedModel`, `Handoff`, `DECODER_CACHE_IN/OUT`, `DECODER_HEAD_OUTPUTS` (generated) | homr's source, the bytes on disk. No I/O. |
| `src/models/backend.ts` | `RuntimeProbe`, `RuntimeOptions`, `ModelRuntime`, `probeRuntime`, `chooseBackend` (pure), `placementOf` (pure), `startRuntime` | `navigator`, `SharedArrayBuffer`, ort's `env`. The only module that writes global state. |
| `src/models/cache.ts` | `ModelCacheKind`, `ModelCache`, `memoryCache`, `cacheApiCache`, `opfsCache`, `browserCache` | `caches`, OPFS. The only module with an environment test. |
| `src/models/dtype.ts` | `float16FromFloat32`, `float32FromFloat16`, `encodeFloat16Array`, `decodeFloat16Array` | IEEE 754 binary16. Pure; imports nothing. |
| `src/models/session.ts` | `ExecutionProvider`, `SessionTuning`, `ModelSession`, `sessionOptionsFor` (pure), `openModelSession`, `readFloat32`, `handoff` | onnxruntime's `InferenceSession` and `Tensor`. |
| `src/models/store.ts` | `FetchRequest`, `FetchBytes`, `httpFetch`, `ModelEvent`, `ModelError`, `ModelStoreOptions`, `ModelStore` | nothing external: it composes the five above over two injected ports. |

Trace from "phase 3 wants segnet" to a session: `store.open` → `resolveRole`
(manifest) → `#loadBytes` (store, over the injected cache and fetch) →
`sessionOptionsFor` + `openModelSession` (session) → ort. Three files.

### The load-bearing structure: artifact, role, placement

Two nouns and one pair of axes, deliberately not merged.

An **artifact** is a file. Its identity is its SHA-256, and that one string is
the in-flight key, the cache key and the first path segment of its URL, so
there is no second identity to keep in step.

```ts
/** The three ONNX element types the six artifacts use. Narrowed on purpose: a fourth means the generator saw something new and the design should be looked at. */
export type TensorElementType = "float16" | "float32" | "int64";

/** One model tensor as the model itself declares it (read from InferenceSession.inputMetadata, never transcribed). A string in `shape` is a free dimension. */
export interface TensorSpec {
  readonly name: string;
  readonly shape: readonly (number | string)[];
  readonly type: TensorElementType;
}

export interface ArtifactRecord {
  /** The one free dimension a caller may pin through SessionTuning.batch, or null. segnet: "batch_size". The decoder's `cache_exists` and `seq_len` are its own business and must never be overridden. */
  readonly batchDim: string | null;
  readonly bytes: number;
  readonly inputs: readonly TensorSpec[];
  readonly outputs: readonly TensorSpec[];
  readonly sha256: string;
  /** `{sha256}/{filename}`, appended to the store's baseUrl. Content-addressed, so the deployed file is immutable, `Cache-Control: immutable` is honest, and a manifest bump cannot read a stale copy. */
  readonly urlPath: string;
}
```

**There is no `precision` field.** The label lives in the `ArtifactId`
(`"segnet-308-fp16"`), which is what the role table names and what the reason
string prints; the load-bearing fact is the measured element type on each
`TensorSpec`. A `precision` field would answer "which file" and "what do I
write into the tensor" with one value, and the two are not the same question —
the fp16 artifacts declare float16 *on the wire in both directions*, which had
to be measured, not inferred from a filename suffix.

A **role** is what a caller names. Its policy is a hand-written table, one row
per role, each row citing homr:

```ts
export type ModelRole = "decoder" | "encoder" | "ocrClassify" | "ocrDetect" | "ocrRecognize" | "segnet";

export type GpuPlacement =
  | { readonly artifact: ArtifactId; readonly keepOutputsOnGpu: readonly string[]; readonly kind: "gpu" }
  /** `why` is a citation and, where one exists, the phase that removes it. It reaches the caller in ResolvedModel.reason. */
  | { readonly kind: "stay-on-cpu"; readonly why: string };

export interface RolePolicy {
  /** Used whenever placement.artifactsFor is not "webgpu", and when onWebgpu says stay-on-cpu. */
  readonly cpu: ArtifactId;
  /** The role this one's output feeds, and the input name it arrives as. The only place that coupling is written down; ResolvedModel.handoff is derived from it. */
  readonly feeds?: { readonly input: string; readonly role: ModelRole };
  /**
   * What this role does when placement.artifactsFor is "webgpu". Note the two
   * halves key on different axes, which is the thing that looked like one fact
   * until both segnet artifacts were measured running on the wasm EP: the
   * `artifact` is chosen by placement.artifactsFor, the `keepOutputsOnGpu`
   * request is granted by placement.provider.
   */
  readonly onWebgpu: GpuPlacement;
}

export const MODEL_ROLES = {
  decoder: {
    cpu: "decoder-396-fp32",
    onWebgpu: {
      kind: "stay-on-cpu",
      // Two independent reasons, and both must go before this row changes.
      // 1. The shipped decoder has no SkipLayerNormalization kernel on the
      //    WebGPU EP; phase 8 re-exports it from the public checkpoint.
      // 2. homr keeps the decoder on its CPU EP with the fp32 model because
      //    "the fp16 model ... is slower than the fp32 model on the CPU EP"
      //    (homr/onnx_providers.py:1-16). That reason survives phase 8 for the
      //    WebAssembly path, and is why ARTIFACTS has no decoder fp16 row.
      why: "the shipped decoder has no SkipLayerNormalization kernel on the WebGPU EP; phase 8 re-exports it",
    },
  },
  encoder: {
    cpu: "encoder-396-fp32",
    feeds: { input: "context", role: "decoder" },
    // homr's GPU path runs the fp16 encoder (homr/onnx_providers.py:1-16,
    // homr/transformer/configs.py:23-30). Keeping `output` on the GPU is a
    // request, not a decision: resolveRole grants it only when the decoder is
    // on the GPU too.
    onWebgpu: { artifact: "encoder-396-fp16", keepOutputsOnGpu: ["output"], kind: "gpu" },
  },
  segnet: {
    cpu: "segnet-308-fp32",
    // The six logit planes go to a CPU argmax (homr's inference_segnet.py does
    // the same on its GPU path), so keepOutputsOnGpu is empty: a gpu-buffer
    // output would only be copied straight back.
    onWebgpu: { artifact: "segnet-308-fp16", keepOutputsOnGpu: [], kind: "gpu" },
  },
  ocrClassify: { cpu: "ppocr-v2-cls-mobile", onWebgpu: { kind: "stay-on-cpu", why: "0.6 MB and one run per strip: the CPU EP is not the cost" } },
  ocrDetect: { cpu: "ppocr-v6-det-small", onWebgpu: { kind: "stay-on-cpu", why: "PP-OCR on the WebGPU EP is unmeasured; phase 11 measures it and this row is where the answer goes" } },
  ocrRecognize: { cpu: "ppocr-v6-rec-small", onWebgpu: { kind: "stay-on-cpu", why: "PP-OCR on the WebGPU EP is unmeasured; phase 11 measures it and this row is where the answer goes" } },
} as const satisfies Readonly<Record<ModelRole, RolePolicy>>;
```

`as const satisfies` is doing real work: a typo in an `ArtifactId` is a compile
error, and a new `ModelRole` nobody gave a policy is a compile error.

**A placement is the two axes, and resolution is one function.**

```ts
/**
 * The two things a Backend used to decide at once. They travel together in
 * production and are separable in fact: both segnet artifacts run on the wasm
 * EP (measured, 159 ms fp32 and 177 ms fp16 per tile under Node on one
 * thread), which is how CI covers the fp16 branch with no GPU and how the
 * bench A/Bs precision against provider on one machine.
 */
export interface Placement {
  /** Which side of the fp16/fp32 arrangement to take the artifacts from. */
  readonly artifactsFor: Backend;
  readonly provider: ExecutionProvider;
}

/** The placement a runtime implies, and the store's default. "wasm-threads" and "wasm" give the same answer: they are the same EP, differing only in the realm's env.wasm.numThreads. */
export function placementOf(runtime: ModelRuntime): Placement;

export interface ResolvedModel {
  readonly artifact: ArtifactRecord;
  readonly artifactId: ArtifactId;
  /** How this role's output becomes the next role's input. Absent when the role feeds no other role. */
  readonly handoff?: Handoff;
  /** Non-empty only when this role and its consumer both run on the WebGPU EP: the per-output-name form, never a blanket location. */
  readonly outputsOnGpu: readonly string[];
  readonly provider: ExecutionProvider;
  /** One sentence: the artifact, the provider, why the role did not go to the GPU when it could have, and — when the two axes disagree — that the placement was asked for rather than implied. The bench prints it and phase 9 folds it into RecognizeResult.log. */
  readonly reason: string;
  readonly role: ModelRole;
}

export interface Handoff {
  /** "none" when the two artifacts agree on the element type; otherwise the cast, derived by comparing two measured TensorSpecs. Never written down twice, and it disappears on its own the moment phase 8 makes the dtypes agree. */
  readonly cast: "none" | { readonly from: TensorElementType; readonly to: TensorElementType };
  readonly input: string;
  readonly location: "cpu" | "gpu-buffer";
  readonly to: ModelRole;
}

/**
 * The single place the fp16-on-GPU / fp32-on-CPU arrangement and the decoder's
 * forced stay on WebAssembly are expressed. No other function in the library
 * branches on Backend or on a precision, and no call site does.
 */
export function resolveRole(catalog: ModelCatalog, role: ModelRole, placement: Placement): ResolvedModel {
  // TODO artifact: placement.artifactsFor === "webgpu" && onWebgpu.kind ===
  //   "gpu" → onWebgpu.artifact; otherwise policy.cpu.
  // TODO provider: placement.provider, except that a stay-on-cpu role forces
  //   "wasm" — the decoder does not go to the WebGPU EP on any placement.
  // TODO outputsOnGpu: onWebgpu.keepOutputsOnGpu, kept only when this role's
  //   provider is "webgpu" AND the consumer named by policy.feeds resolves to
  //   "webgpu" as well. One level of recursion only: feeds is a two-node DAG
  //   and a cycle is a manifest bug, asserted.
  // TODO handoff: derived, not stored. `cast` from this artifact's output spec
  //   against the consumer's input spec; `location` from the two providers.
  // TODO reason: "<artifactId> on the <provider> EP", plus ", although WebGPU
  //   is available: <why>" when the role stayed on the CPU, plus ", placement
  //   requested" when the two axes disagree.
  throw new Error("not implemented");
}
```

Every consumer question is answered by a lookup or by a field on the session:
*which file* (`plan.artifactId`), *which EP* (`plan.provider`), *what do I
write* (`session.inputSpec(name).type`), *how do I read* (`readFloat32`), *what
does the next stage need* (`handoff`). None is an `if` over `Backend`.

### The runtime: one-shot global state, named as such

```ts
export interface RuntimeProbe {
  /** The adapter, when navigator.gpu granted one. Kept so startRuntime can build the device from it instead of letting onnxruntime request a second adapter. */
  readonly adapter: GPUAdapter | undefined;
  readonly adapterInfo: string | undefined;
  /** Browser-only evidence for why sharedMemory is what it is; undefined on Node, where there is no such gate. Reported, never tested against. */
  readonly crossOriginIsolated: boolean | undefined;
  readonly hardwareConcurrency: number;
  /** `typeof SharedArrayBuffer === "function"`: the actual capability behind WebAssembly threads, and the same expression on Node and in a browser — which is why this probe has no environment branch in it. */
  readonly sharedMemory: boolean;
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
 * of what was actually applied. You cannot construct a ModelStore without one,
 * which is how "configure env before the first session" becomes unmissable
 * rather than documented.
 */
export interface ModelRuntime {
  readonly backend: Backend;
  /** What was written to env.wasm.numThreads. 1 on the "wasm" backend. Reported as applied, not as requested. */
  readonly numThreads: number;
  readonly probe: RuntimeProbe;
  readonly reason: string;
}

export function chooseBackend(probe: RuntimeProbe, options?: RuntimeOptions): { backend: Backend; reason: string } {
  // TODO pure: adapter present → "webgpu"; else sharedMemory → "wasm-threads";
  //   else "wasm". Clamp to options.maxBackend. The reason names the first
  //   capability that was missing, and cites probe.crossOriginIsolated when
  //   sharedMemory is false in a browser, since that is the fixable case.
  throw new Error("not implemented");
}

/**
 * Probe, decide, configure, freeze. Memoised per realm: onnxruntime reads
 * env.wasm.numThreads and its GPU device only before the first session is
 * created and never again, so a second configuration is not a thing that can
 * happen.
 *
 * Idempotent on equal input, loud on conflicting input: a second call whose
 * options would apply a different backend, numThreads or wasmPaths throws
 * ModelError("runtime-frozen") naming the fields, rather than returning a
 * runtime whose settings are somebody else's. A Worker re-entering its own
 * init with the same options gets the same runtime back (make-operations-idempotent).
 */
export async function startRuntime(options?: RuntimeOptions): Promise<ModelRuntime> {
  // TODO module-level `let started: { applied: Applied; runtime: Promise<ModelRuntime> } | undefined`.
  // TODO compare the *applied* values (backend, numThreads, wasmPaths), not a
  //   deep equality of the option bag: {} twice and the same numThreads twice
  //   are both the idempotent case.
  // TODO env.wasm.numThreads = applied; env.wasm.wasmPaths = options.wasmPaths
  //   when given. env.wasm.proxy stays false: phases 3 and 7 own their Workers
  //   and ort's proxy worker would nest one inside another.
  // TODO on "webgpu", set env.webgpu.device from a device made off
  //   probe.adapter — not env.webgpu.powerPreference, which 1.30.0 marks
  //   @deprecated in favour of exactly this, and which would make ort request
  //   a second adapter we already hold.
  // TODO a rejection clears `started`, so a failed configure can be retried.
  throw new Error("not implemented");
}
```

### Where fetch and the cache enter

Two ports, mirroring `GoldenReader`. Neither has a default that differs between
Node and the browser, and the environment test lives in one factory the caller
picks — never inside the store.

```ts
// store.ts
export interface FetchRequest {
  readonly onProgress: ((received: number, total: number) => void) | undefined;
  readonly signal: AbortSignal | undefined;
  readonly url: string;
}
export type FetchBytes = (request: FetchRequest) => Promise<Uint8Array>;

/** globalThis.fetch, streamed so progress is reported. One implementation for Node 22+ and the browser; this is a global, not an environment branch. */
export function httpFetch(): FetchBytes;

// cache.ts
export type ModelCacheKind = "cache-api" | "memory" | "opfs";

/** Keyed on the artifact's SHA-256, never on a role or a URL, so a manifest bump can never read a stale entry and two homr-web versions share a byte-identical model. May throw; the store treats any throw as a miss, so that policy lives in one place. */
export interface ModelCache {
  readonly drop: (sha256: string) => Promise<void>;
  readonly kind: ModelCacheKind;
  readonly read: (sha256: string) => Promise<Uint8Array | undefined>;
  readonly write: (sha256: string, bytes: Uint8Array) => Promise<void>;
}

export function memoryCache(): ModelCache;
/** Cache API, then OPFS (Safari private mode), then memory. The one function in src/ allowed to test for a global, so nothing else has to. */
export async function browserCache(): Promise<ModelCache>;
```

### The store

```ts
export type ModelEvent =
  | { readonly artifact: string; readonly bytes: number; readonly from: ModelCacheKind; readonly kind: "cached" }
  | { readonly artifact: string; readonly kind: "download"; readonly received: number; readonly total: number }
  | { readonly artifact: string; readonly kind: "verified"; readonly ms: number }
  /** A hash mismatch. source "cache" is the self-heal path and is followed by one re-fetch; source "network" is the failure. */
  | { readonly actual: string; readonly artifact: string; readonly expected: string; readonly kind: "rejected"; readonly source: "cache" | "network" }
  | { readonly artifact: string; readonly kind: "opened"; readonly ms: number; readonly provider: ExecutionProvider; readonly role: ModelRole };

export interface ModelStoreOptions {
  /** Where the artifacts are served; urlPath is appended. Must be same-origin, or carry Cross-Origin-Resource-Policy: same-origin, when the page is COEP require-corp. */
  readonly baseUrl: string;
  readonly cache: ModelCache;
  readonly catalog?: ModelCatalog;
  readonly fetchBytes?: FetchBytes;
  readonly onEvent?: (event: ModelEvent) => void;
  /** Defaults to placementOf(runtime). Given explicitly by the fp16 golden test and by the bench's precision A/B. The constructor refuses provider "webgpu" on a runtime that is not on the webgpu backend; artifactsFor is free, because the fp16 artifacts run on the wasm EP. */
  readonly placement?: Placement;
  readonly runtime: ModelRuntime;
}

/** A class, not a factory: it owns mutable state (in-flight loads, open sessions) and lives in a Worker, per the project's Worker-state convention. */
export class ModelStore {
  constructor(options: ModelStoreOptions);

  readonly placement: Placement;
  readonly runtime: ModelRuntime;

  /** What this role would use, without fetching anything. The app asks before deciding whether to pull 160 MB over a phone connection. */
  plan(role: ModelRole): ResolvedModel;

  /** Verified bytes in the cache for these roles; no sessions. The transcriber page calls this while the musician is still choosing a file. Cheap and safe to call twice. */
  prefetch(roles: readonly ModelRole[], signal?: AbortSignal): Promise<void>;

  /** The session for this role. Opening the same role with the same tuning returns the same session; the store owns it until close(). */
  open(role: ModelRole, tuning?: SessionTuning): Promise<ModelSession>;

  /** Release every session and drop the byte references. Idempotent; safe during an open, which then rejects. */
  close(): Promise<void>;

  #bytes = new Map<string, Promise<Uint8Array>>();      // keyed on sha256
  #sessions = new Map<string, Promise<ModelSession>>(); // keyed on `${role}@${batch ?? 0}`
}
```

The concurrency and interruption rules, which is where the store earns its keep:

```ts
async #loadBytes(id: ArtifactId, artifact: ArtifactRecord, signal?: AbortSignal): Promise<Uint8Array> {
  // TODO single flight on artifact.sha256: two concurrent open("segnet") calls
  //   in one realm share one 57 MB download. Store the promise before the
  //   first await. Delete the entry on rejection and never on fulfilment, so a
  //   failed load is retried by the next caller and a good one is reused.
  // TODO across realms (phase 3's Worker and phase 7's Worker are separate
  //   realms and cannot share a Map) the content-keyed cache makes the
  //   duplicate harmless: two writers put identical bytes under one key.
  // TODO read: cache.read(sha256) inside a try; any throw or undefined is a
  //   miss. Check bytes.length first — a 404 HTML page is rejected without a
  //   digest and with a message that says which it was — then digest.
  // TODO verify always, including on a cache hit: 82 ms for all three homr
  //   models, measured, against a 16 s pipeline. On mismatch from the cache,
  //   emit rejected/cache, cache.drop, fall through to the network once. A
  //   second mismatch throws. There is no loop.
  // TODO fetch: nothing is written to the cache before the digest matches, so
  //   an interrupted or aborted download leaves no partial entry behind and a
  //   retry is a fresh full download. The cache only ever holds complete,
  //   verified artifacts, which is the whole idempotence argument.
  // TODO cache.write is best effort: a rejection (quota, private mode) is
  //   swallowed after an event. A full disk must not fail a load.
  throw new Error("not implemented");
}

async open(role: ModelRole, tuning: SessionTuning = {}): Promise<ModelSession> {
  // TODO single flight on `${role}@${tuning.batch ?? 0}`. Nothing else can
  //   vary: provider, output locations and graph options are policy derived
  //   from (role, placement), so there is no option bag to hash.
  // TODO drop the Uint8Array reference as soon as InferenceSession.create
  //   returns — ort has copied it into wasm memory, and 57 MB held for nothing
  //   is a real cost on iOS, where a tab dies near 1 to 1.5 GB.
  throw new Error("not implemented");
}
```

### The session

```ts
/** Two providers for three backends: "wasm-threads" and "wasm" are the same EP, differing only in the realm's env.wasm.numThreads. */
export type ExecutionProvider = "wasm" | "webgpu";

export interface SessionTuning {
  /** Runs per session call, for a role whose artifact declares a batch dimension — segnet only. freeDimensionOverrides *fixes* the dimension, so the caller pads the final short batch. Passing it for a role whose batchDim is null throws ModelError("bad-tuning"). */
  readonly batch?: number;
}

export interface ModelSession {
  readonly close: () => Promise<void>;                 // idempotent
  /** The spec by name, throwing ModelError("unknown-tensor") on a miss: the boundary check that absorbs noUncheckedIndexedAccess once instead of at every call site. `.type` is what the caller writes; "float16" means IEEE halves in a Uint16Array. */
  readonly inputSpec: (name: string) => TensorSpec;
  readonly outputSpec: (name: string) => TensorSpec;
  readonly plan: ResolvedModel;
  readonly role: ModelRole;
  /** onnxruntime's session, deliberately not wrapped: phases 3, 7 and 11 construct Tensors and call run() in hot loops, and a run() forwarder would be a pass-through method. */
  readonly session: InferenceSession;
}

/** Pure: (plan, tuning) to options, no I/O. The one exhaustive unit test in phase 2 — six roles by three backends by two providers — needs no model bytes and no network. */
export function sessionOptionsFor(plan: ResolvedModel, tuning: SessionTuning): InferenceSession.SessionOptions {
  // TODO executionProviders: [plan.provider].
  // TODO graphOptimizationLevel: "all". Settled by measurement, not left open:
  //   on the decoder's 39-in/39-out graph, open cost was 220 ms disabled,
  //   76 basic, 69 extended, 60 all, and every input and output name survived
  //   at every level — so ort's default is both the fastest and safe for the
  //   generated name tuples, and no artifact needs a special case. Written
  //   explicitly all the same, so that an ort default change cannot quietly
  //   move the numbers the golden tests pin.
  // TODO logSeverityLevel: 3 — the library must not write to the console
  //   (biome's noConsole states the intent; ort's own warnings otherwise do).
  // TODO preferredOutputLocation: omitted when plan.outputsOnGpu is empty;
  //   otherwise the per-output-name object form, {name: "gpu-buffer"}, never a
  //   blanket location — segnet's logits must come back for the CPU argmax.
  // TODO freeDimensionOverrides: only {[artifact.batchDim]: tuning.batch}, and
  //   only when both exist. The decoder's `cache_exists` and `seq_len` are
  //   never overridden; pinning them would break the token loop at step 1,
  //   which is why batchDim is a single named dimension and not a free record.
  throw new Error("not implemented");
}

/** Opens from bytes the caller already holds. Public for the bench page and for phase 8's A/B of a locally re-exported decoder, both of which want this options policy over a file that is in no manifest. Not the normal path: ModelStore.open is. */
export async function openModelSession(plan: ResolvedModel, bytes: Uint8Array, tuning?: SessionTuning): Promise<ModelSession>;

/**
 * Any output tensor as float32, wherever it lives and whatever it declares:
 * `getData()` first (so a GPU-resident tensor is read back and a CPU one is
 * not copied), then a table-driven half decode when the tensor's type is
 * float16. onnxruntime-web 1.30.0 has no Float16Array on either side of a
 * run, so an fp16 output arrives as a Uint16Array of raw half bit patterns
 * (measured: 17630, 18301, 18122 where the fp32 model gave 4.865, 7.488,
 * 6.788).
 *
 * This is the reason phase 3's argmax, phase 7's logit heads and the bench
 * page contain no dtype code: the decode every one of them needs is one pass
 * over the buffer, which they need anyway, so phase 2 performs it. The input
 * side is the opposite and is left to the caller — see the tradeoffs.
 */
export async function readFloat32(tensor: Tensor): Promise<Float32Array>;

/**
 * One role's output tensor as the next role's input wants it: the
 * fp16-to-fp32 cast on the WebGPU path (homr does the same,
 * staff2score.py:43-49), a GPU read when the placements differ, and the
 * identical tensor when both agree. Dims are carried through; slicing
 * `context` to [:, :1] after step 0 is phase 7's own business.
 *
 * This exists so that the mixed-precision arrangement lives in phase 2 as
 * data (ResolvedModel.handoff, derived from two measured TensorSpecs) and in
 * exactly one function, instead of as two lines of dtype-aware code in phase 7
 * that phase 8 would then have to find and delete.
 */
export async function handoff(tensor: Tensor, from: ModelSession, to: ModelSession): Promise<Tensor> {
  // TODO from.plan.handoff must name to.role, else ModelError("bad-handoff").
  // TODO cast "none" and locations agree → return the tensor untouched, so
  //   after phase 8 the 1280x512 context never leaves the GPU.
  // TODO otherwise readFloat32, then encodeFloat16Array when the target wants
  //   halves (no role does today; phase 8 could), then new Tensor with the
  //   source dims.
  throw new Error("not implemented");
}
```

### The half codec, and why phase 2 owns it

Phase 2 owns the **codec** and not the **casts**, and the distinction is the
whole answer to where half-float code lives.

A codec is a fact about a representation: there is one correct encoding of a
float as IEEE 754 binary16 and one correct decoding, and both are needed in at
least four places (phase 3's tile writer and its argmax, phase 7's encoder
handoff and its logit heads, the bench page's value display). Four copies of
bit twiddling, three of which would be written by someone debugging something
else, is the textbook case for one owner.

A cast is a policy about *when* a buffer changes representation, and it depends
on where the tensor is going — which is phase 2's knowledge only at a role
boundary. So the two directions come out asymmetric, and the asymmetry is
measured rather than aesthetic:

- **Output side, phase 2 performs it** (`readFloat32`). Every consumer of an
  fp16 output wants float32 — the argmax, the logit heads, the display — so a
  whole-buffer decode is a pass they all need. One table-driven pass over
  6×320×320 is ~1 ms and it leaves phase 1's `argmaxPlanes(logits:
  Float32Array, ...)` signature untouched.
- **Input side, the caller performs it** (`inputSpec(name).type` plus
  `float16FromFloat32`). Phase 3 writes halves straight from the pixel; a
  phase-2 converter would force an intermediate `Float32Array` and a second
  full pass over 2.4 M values per batch, to hide one `if` at the top of a
  function that already has two tight loops in it.

```ts
/** IEEE 754 binary16, round-to-nearest-even — what ONNX and numpy do. testing.md's determinism discipline applies: test/dtype.test.ts pins zero, subnormals, the largest finite half, overflow to Infinity and the ties. */
export function float16FromFloat32(value: number): number;
export function float32FromFloat16(half: number): number;
/** Buffer-wide, through a memoised 65536-entry Float32Array lookup table (256 KB, built in about a millisecond): a half decode becomes an array index, so a per-pixel decode in a hot loop costs no arithmetic. */
export function decodeFloat16Array(halves: Uint16Array, into?: Float32Array): Float32Array;
export function encodeFloat16Array(values: Float32Array, into?: Uint16Array): Uint16Array;
```

**Why `src/models/dtype.ts` and not `src/image/numeric.ts`.** numeric.ts's
charter is explicit in its own header: the 1-D numpy and Python arithmetic homr
relies on, with the rule "never call `Math.round` where the Python calls
`round()`". Every function in it mirrors a named Python call, and that is the
module's whole value — a reader can assume anything in there exists because
Python did it differently. Half-float encoding mirrors no Python call and
reproduces no homr semantic; it is an ONNX wire format that exists because
onnxruntime-web has no `Float16Array`. Putting it in numeric.ts would dilute
the one-sentence rule that makes numeric.ts worth having. What it does inherit
is the *discipline*: a rounding mode that must be pinned, a test file that pins
the boundary cases, and a row in `testing.md`'s determinism-trap list.

### The manifest is generated, and the split is measured-fact versus policy

`src/models/manifest.ts` follows the precedent already in the repository —
`tools/gen-vocabulary.mjs` rewriting a `// BEGIN GENERATED` block inside a
hand-written file — and the line it draws is not "data versus code":

- **Generated** (`tools/gen-manifest.mjs`): `ARTIFACTS`, and the
  `DECODER_CACHE_IN` / `DECODER_CACHE_OUT` / `DECODER_HEAD_OUTPUTS` tuples.
  Every value is a fact about a file: SHA-256, byte length, and I/O names,
  shapes and element types read from `InferenceSession.inputMetadata` /
  `outputMetadata` (present in 1.30.0).
- **Hand-written**: `MODEL_ROLES`, `resolveRole`, the types. Every value is a
  judgement citing homr's source or a measurement, and a generator has no
  business touching it.

Three reasons the generator is right, in order of weight:

1. **The 78 decoder tensor names.** `cache_in0..31`, `cache_out0..31`, six
   logit heads and `attention` are exactly the list a human transcribes once,
   off by one, with nobody noticing until step 2 of a 608-step loop produces
   plausible garbage. Generated from the model, phase 7 *iterates* them, and a
   re-pin that renames an output fails a test. The graph-optimisation
   measurement makes this safe: all 39 inputs and 39 outputs survive with
   names intact at every level, so the tuples are not invalidated by the
   optimiser.
2. **The hashes must be computed.** Eight artifacts, six from the venv and two
   downloaded and unzipped from homr's release; homr verifies no digest of any
   kind on download (`download_utils.py:11-78`) and the 40-hex string in each
   filename is of unstated provenance — SHA-1 length and git-commit length are
   the same, and nothing in homr says which. A reviewer must be able to rerun
   a command and read a diff, not audit eight 64-hex strings by eye.
3. **The element types had to be measured.** That the fp16 artifacts declare
   `float16` on the wire in *both* directions, rather than float32 IO over
   fp16 weights, is not derivable from a `_fp16` filename suffix. A
   hand-written manifest would have recorded a guess, and the guess would have
   been wrong on the output side, which is where three consumers meet it.

```bash
tools/fetch-models.sh          # once: homr's release assets + the venv models into models/ (git-ignored)
node tools/gen-manifest.mjs    # hashes them, opens each under Node's ort, rewrites the generated block
```

The generated block, as the generator will write it (fp32 values from the
2026-09-27 measurement, fp16 from today's download and probe):

```ts
// BEGIN GENERATED (tools/gen-manifest.mjs)
export const ARTIFACTS = {
  "segnet-308-fp32": {
    batchDim: "batch_size",
    bytes: 57311361,
    inputs: [{ name: "input", shape: ["batch_size", 3, 320, 320], type: "float32" }],
    outputs: [{ name: "output", shape: ["batch_size", 6, 320, 320], type: "float32" }],
    sha256: "6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d",
    urlPath: "6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
  },
  "segnet-308-fp16": {
    batchDim: "batch_size",
    bytes: 28667207,
    inputs: [{ name: "input", shape: ["batch_size", 3, 320, 320], type: "float16" }],
    outputs: [{ name: "output", shape: ["batch_size", 6, 320, 320], type: "float16" }],
    sha256: "60f495496cb41473c0521d0811d8f44b9d5cff892d287974a8aebb3eaee2fa83",
    urlPath: "60f4…fa83/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx",
  },
  "encoder-396-fp32": { batchDim: null, bytes: 52861122, inputs: [{ name: "input", shape: [1, 1, 256, 1280], type: "float32" }], outputs: [{ name: "output", shape: [1, 1280, 512], type: "float32" }], sha256: "4c16df85…5d5eb", urlPath: "4c16…/encoder_pytorch_model_396-f6feedb4….onnx" },
  "encoder-396-fp16": { batchDim: null, bytes: 26466256, inputs: [{ name: "input", shape: [1, 1, 256, 1280], type: "float16" }], outputs: [{ name: "output", shape: [1, 1280, 512], type: "float16" }], sha256: "9db62d5a…8087e", urlPath: "9db6…/encoder_pytorch_model_396-f6feedb4…_fp16.onnx" },
  "decoder-396-fp32": { batchDim: null, bytes: 47299551, inputs: [/* 39: rhythms pitchs lifts articulations slurs int64[1,1]; context float32[1,cache_exists,512]; cache_len int64[1]; cache_in0..31 float32[1,8,seq_len,64] */], outputs: [/* 39 */], sha256: "3e10fd5a…3e57d", urlPath: "3e10…/decoder_pytorch_model_396-f6feedb4….onnx" },
  "ppocr-v6-det-small": { batchDim: null, bytes: 9929594, sha256: "090f04ab…ff94f", /* … */ },
  "ppocr-v2-cls-mobile": { batchDim: null, bytes: 585532, sha256: "e47acedf…6215c", /* … */ },
  "ppocr-v6-rec-small": { batchDim: null, bytes: 21234383, sha256: "6f327246…c1488", /* … */ },
} as const satisfies Readonly<Record<ArtifactId, ArtifactRecord>>;

/** The 32 self- and cross-attention key/value caches in the order the model declares them: 8 layers x 4, matching homr's decoder_depth = 8. Phase 7 iterates these; it never builds a name with a template string. */
export const DECODER_CACHE_IN = ["cache_in0", /* … */ "cache_in31"] as const;
export const DECODER_CACHE_OUT = ["cache_out0", /* … */ "cache_out31"] as const;
export const DECODER_HEAD_OUTPUTS = ["out_rhythms", "out_pitchs", "out_lifts", "out_positions", "out_articulations", "out_slurs"] as const;
// END GENERATED
```

There is no `decoder-396-fp16` row. homr's release has one and uses it on CUDA
only; nothing in this port can reach it, and a row nothing can reach is a row
that goes stale silently. Its zip is also 86 MB against the fp32 decoder's
47 MB unzipped, which is unexplained — see the risks.

### What is validated where

Per `boundary-discipline`, four boundaries and nothing inside them:

1. **Bytes off the network or out of the cache**: length, then SHA-256, then
   `InferenceSession.create`. Inside the library a `Uint8Array` that reached a
   session is trusted.
2. **The opened session against the manifest**: `openModelSession` compares the
   session's own `inputNames`/`outputNames` against the artifact's specs and
   throws on a difference. A hash match already proves the file, so this fires
   only when the *generator* and the loader disagree — exactly the re-pin
   mistake worth catching.
3. **The placement against the runtime**: `provider: "webgpu"` on a runtime
   that is not on the webgpu backend is refused in the constructor;
   `artifactsFor` is free, because the fp16 artifacts run on the wasm EP. The
   useful half of the axis split is open and the dangerous half is impossible.
4. **Caller tuning**: `batch` on a role with no batch dimension, an unknown
   tensor name, a `handoff` between two roles the manifest does not pair.

Two free oracle checks come out of the shape, in the phase-1 spirit of having
the decoder verify what Python stored:

- The golden fixtures' `meta.json` already records `<filename>:<sha256>` per
  model. A test asserts it equals `resolveRole(...).artifact.sha256` on the
  CPU placement, so golden data dumped from one model and a port running
  another is a failing test rather than a mystery.
- `DECODER_HEAD_OUTPUTS` (generated from the model) is asserted to map onto
  phase 1's `DECODER_OUTPUT_HEADS` (generated from the vocabulary), so a
  renamed output head fails a test instead of producing wrong tokens.

### What this deliberately does not do

No tensor abstraction over ort's `Tensor`; no `run()` forwarder; no input-side
dtype hiding. No retry or backoff on a failed fetch (the caller knows whether a
retry is wanted; a library that retries silently hides an offline user). No
resumable download, no range requests, no partial cache entries. No eviction
policy — the Cache API evicts, and that is a miss. No cross-realm coordination
beyond the content-keyed cache. No per-artifact `graphOptimizationLevel`, which
the measurement settles.

### Interface depth

The public surface is two constructors (`startRuntime`, `new ModelStore`),
three store methods, six types a consumer reads, and five functions phase 7 or
the bench needs. Behind it sit: the backend probe and its reason, the one-shot
env write and its freeze, the artifact/role/dtype/EP/location arrangement over
two separable axes, verify-always with self-heal, single-flight, content-keyed
caching across three storage back ends, the per-output GPU location policy, the
half codec, and the mixed-precision handoff. Learning `open("segnet")` saves
the caller from learning all of it. What stays exposed is `InferenceSession`
and `Tensor` — a deliberate leak, argued below.

## Synthesis decision

Two candidates were designed in parallel against one measured grounding brief.
Candidate A is the base. The graft and the rejections are recorded here because
the losing shape was close enough that a future reader will wonder why.

**What made A the base.** Candidate B resolved role-to-artifact placement
through a `Tier = "cpu" | "gpu"` derived from the backend, which is the obvious
shape and is wrong in one specific way: it ties *which artifact* to *which
execution provider*, and those turned out to be separable. Both segnet
artifacts run on the WebAssembly execution provider (measured: 159 ms fp32 and
177 ms fp16 per tile under Node on one thread), so A's
`Placement { artifactsFor; provider }` can ask for the WebGPU *artifacts* on the
WebAssembly *provider*. That is not a refinement, it is the difference between
an fp16 path CI can cover on a Linux runner with no GPU and one that can only
ever be exercised by hand on a Mac. Everything else followed from taking that
axis split seriously: the role table's two halves key on different axes, the
bench can A/B precision against provider on one machine, and the fp16 golden
test in "Usage" exists at all.

A is also the shorter call chain. B threaded `backend` and a `RuntimeToken`
through every `openSession(store, role, backend, token)` call; A builds the
store *for* a runtime, so `open("segnet")` carries neither. The token stops
being a parameter every call site repeats and becomes a thing you cannot
construct a store without, which is the same invariant with none of the
threading (per `minimize-reader-load`).

**Grafted from B.** A structured verification failure carrying `id`, `expected`
and `actual` rather than a message string, so a caller can act on a mismatch
instead of parsing prose. B's explicit statement of why content-keying removes
the need for any separate version field is folded into `ModelCache`'s doc
comment. The `Record<ModelRole, Record<Backend, Plan>>` entry
under "Alternatives considered" now carries B's tier-shaped variant too, because
that is the shape somebody will propose again.

**Rejected from B, with reasons.** Its `precision: "fp16" | "fp32"` manifest
field, because after the element types were measured the field answers no
question the `TensorSpec`s do not answer better, and a label that merely
agrees with a filename suffix is the thing that invited the wrong guess in the
first place. Its default of `memoryCache()` when no cache is supplied, because
in a browser that silently costs the musician every re-download with no test
able to notice; a cache is required here instead. And two type assertions its
own sketch needed (`plan.gpu as ModelId`, `this.#segnet as OpenedSession`),
which the red-flag screen should have caught: A's resolution returns the
narrowed object directly and its worker returns the session from the call that
opens it.

**What neither candidate had, added during synthesis.** The measurement that
settles `graphOptimizationLevel` (B had correctly flagged it as an unmeasured
risk, A had it from a mid-flight correction). The decision on where CI gets
160 MB of model bytes, below. And the two free oracle checks A proposed are
promoted from a nice idea to required first tests, because they cost nothing
and they catch the one class of mistake this phase can make invisibly, which is
a re-pin that moves the model out from under the golden data.

## Tradeoffs accepted

- We accept exposing onnxruntime's `InferenceSession` and `Tensor` on the
  public surface in exchange for not writing a tensor framework: phases 3, 7
  and 11 build tensors of five shapes and dtypes and call `run` in hot loops,
  and a `run()` wrapper would be a pass-through method that hides nothing.
  `onnxruntime-web` is already a `dependencies` entry.
- We accept an asymmetry between the two dtype directions — phase 2 decodes
  outputs, callers encode inputs — in exchange for not adding a full extra
  pass over 2.4 M values per segnet batch. It looks like an inconsistency and
  is a measurement: every output consumer wants the whole buffer as float32,
  no input producer wants an intermediate float32 buffer.
- We accept two objects to construct (`startRuntime`, then `new ModelStore`)
  in exchange for honest lifetimes: the runtime is realm-global and frozen, a
  store is not, and the app wants to report the backend before committing to a
  160 MB download.
- We accept a `Placement` type whose two fields are equal in production, in
  exchange for a CI-testable fp16 path and a bench that can A/B precision
  against provider on one machine. Before the measurement this looked like one
  fact; it is two.
- We accept verifying 160 MB on *every* load including cache hits, in exchange
  for a cache that cannot silently serve a corrupt or foreign entry. Measured:
  82 ms for the three homr models against a 16 s pipeline.
- We accept re-downloading from scratch after an interrupted load, in exchange
  for the invariant that the cache only ever holds complete verified
  artifacts. A resume needs a partial entry, and a partial entry can be
  mistaken for a whole one.
- We accept `handoff()`, a function with one caller today, in exchange for
  phase 7 containing no dtype or location policy — the thing phase 8 would
  otherwise have to find and delete. If a reviewer wants it thinner, the
  load-bearing part is `ResolvedModel.handoff` as data; the function is three
  lines over it.
- We accept a single `batch` tuning knob rather than a general
  `freeDimensionOverrides` passthrough, in exchange for making it impossible
  to pin the decoder's `cache_exists`, which would break the token loop at
  step 1.
- We accept a 256 KB lookup table for half decoding, built lazily on first
  use, in exchange for a per-pixel decode that costs an array index. It is a
  real allocation on iOS, where a tab dies near 1 to 1.5 GB, and it is 0.02 %
  of one model.
- We accept six files where the plan named four: `cache.ts` holds the only
  environment test in `src/`, `dtype.ts` the only bit-level arithmetic, so the
  other four contain neither.
- We accept a generated block inside a hand-written module rather than a JSON
  file, in exchange for a closed `ArtifactId` union and literal tuple types —
  which is what makes a typo in `MODEL_ROLES` a compile error.
- We accept that `readFloat32` may return the tensor's own buffer rather than
  always copying, so a caller that mutated the result would corrupt a tensor it
  is about to feed back in. Always copying costs a 4.9 MB allocation per segnet
  batch output on the hot path, and every consumer today only reads. This is the
  phase's one accepted footgun, accepted on a measured cost rather than on a
  hope that it will not matter.
- We accept that the content-addressed `urlPath` fixes the deploy layout to
  `{baseUrl}/{sha256}/{filename}`. A consumer with a flat directory
  regenerates the manifest rather than being given a second way to resolve a
  URL.

## Alternatives considered

- **`openSession(id, backend)` as a free function beside the store**, as the
  plan's file list suggests. It forces the caller to fetch bytes, then open,
  which is the "callers coordinate several methods" red flag; it puts the
  session-option policy in a module that does not own the role table; and it
  leaves a 57 MB `Uint8Array` live in the caller's hands during
  `InferenceSession.create` with nobody owning its release. Rejected: less
  hidden, larger surface. `session.ts` survives as the *pure* policy module.
- **A `Record<ModelRole, Record<Backend, Plan>>` table**, 18 explicit rows, or
  the same idea collapsed onto a `Tier = "cpu" | "gpu"` derived from the
  backend, which is the shape the losing candidate chose.
  Flatter and lookup-only, but 12 of the 18 are identical (`wasm-threads` and
  `wasm` are one EP), the encoder's output location depends on the decoder's
  row, and the fp16-on-wasm combination is not in the table at all. Rejected
  per single-source-of-truth: one row per role, plus one derivation over two
  axes.
- **`precision: "fp16" | "fp32"` on the artifact**, as the plan specifies. It
  reads as the answer to "what do I write into the tensor" and is not — and
  the output-side measurement is what proves it, because a `precision` field
  would have told three consumers nothing about the `Uint16Array` they get
  back. Rejected in favour of measured `TensorSpec.type`, with the label
  living in the `ArtifactId`.
- **Half-float helpers in `src/image/numeric.ts`.** One fewer file, and the
  determinism discipline is already there. Rejected because numeric.ts's value
  is the rule that everything in it mirrors a named Python call; binary16
  mirrors none, it exists because onnxruntime-web has no `Float16Array`.
- **A dtype-aware `argmaxPlanes` taking `Float32Array | Uint16Array`.** Avoids
  the decode pass, at the cost of a second code path through phase 1's hottest
  helper and a widened signature on a function four other stages call.
  Rejected: `readFloat32` puts the branch in one place that no stage sees.
- **Lazy `startRuntime` inside the store's first `open`.** One fewer call, but
  the app cannot report the backend or size the download before committing to
  it, and it ties a realm-global freeze to whichever store happened to be used
  first.
- **A hand-written manifest.** Genuinely cheaper today, since all eight hashes
  are now measured. Rejected on the 78 decoder tensor names and on
  reviewability: nobody can check a 64-hex string, and anybody can rerun a
  generator and read a diff.
- **Verify only on first fetch**, the plan's implied policy. The measurement
  settles it: the saving is under 0.1 s and the cost is a cache that can serve
  a corrupt entry forever with no way to notice.
- **One injected `ModelSource` port combining fetch and cache.** Fewer
  parameters, but the Node test wants real cache semantics with a fake fetch
  and the browser wants a real fetch with a detected cache; one port makes
  each caller implement both halves.

## Decisions taken, and risks carried

The candidate raised nine open questions. Eight are technical and reversible, so
they are decided here rather than held for the owner (per
`never-block-on-the-human`); each records the reason, so a later reader can
overturn it on evidence rather than taste. The two that only a measurement can
answer are named as measurements the bench owes, not as questions.

**Where CI gets 160 MB of model bytes.** Both: model-running tests skip when
`models/` is absent, and CI fills `models/` from a cache keyed on the manifest's
hashes. A fresh clone's `npm test` must pass without a 160 MB download, because
a contributor who has to fetch the models before any test runs will not run the
tests. But a silently skipped test proves nothing and nobody notices, so the
skip is loud in the reporter *and* a test asserts that `models/` is present when
`process.env.CI` is set. Local convenience and strict CI, with no path where
absent bytes look like a pass. The manifest hash is the correct cache key by
construction, since it is the only thing that changes when the bytes change.

**`numThreads` default.** `min(4, max(1, hardwareConcurrency - 1))`, marked in
the code as unmeasured, and the bench reports it so phase 3 can revise it.
Explicitly not onnxruntime's own `0`, which means "take what you like": this
runs in a Worker on a musician's laptop while the interface has to stay
responsive, and a segmentation pass that consumes every core to finish half a
second sooner is the wrong trade for an interactive application (per
`experience-first`). The number is a guess; that it should be below the core
count is not.

**`env.webgpu.device`, not `powerPreference`.** The installed 1.30.0 typings
mark `powerPreference`, `forceFallbackAdapter` and `adapter` all `@deprecated`
in favour of building a `GPUDevice` yourself and assigning `env.webgpu.device`,
verified in `node_modules/onnxruntime-common/dist/esm/env.d.ts`. The probe
already holds an adapter, so this also stops onnxruntime requesting a second
one. We keep control of the power preference by passing it to our own
`requestAdapter()` call, which is the capability the deprecated field used to
provide.

**`Placement` stays public, documented as what it is.** It is the one store
option production never sets. Hiding it would be dishonest about how the fp16
path is tested, and phase 8 needs it to A/B a locally re-exported decoder
against the shipped one. The README names it as a test and bench affordance.
The dangerous half of the axis is closed instead: `provider: "webgpu"` on a
runtime that is not on the WebGPU backend is refused in the constructor, while
`artifactsFor` is left free, because that is the half the fp16 test needs and it
cannot produce an invalid session.

**`ModelEvent` stays separate from phase 9's `Progress`.** They answer different
questions today, model bytes against pipeline stages, and phase 9 owns how the
two are presented in one bar. Merging them now would couple this phase to a type
phase 9 has not finalised, and the merge is a rename when it comes.

**`readFloat32` does not copy.** It may hand back the tensor's own buffer on the
fp32 CPU path. Always copying would cost a 4.9 MB allocation per segnet batch
output on the hot path to remove a footgun no current consumer can trigger,
since the argmax, the logit heads and the bench all only read. The aliasing is
stated in the function's doc comment and listed in the tradeoffs. This is the
one accepted footgun in the phase, and it is accepted on a measured cost, not on
a guess that it will not matter.

**The service-worker exclusion pattern is stated in this phase's README.** One
line naming the `{baseUrl}/{sha256}/` path shape that the app's
`service-worker.ts` must leave untouched, so phase 10 copies a pattern instead
of re-deriving it. Phase 2 cannot enforce it and should not pretend to; what it
can do is make the requirement impossible to miss.

**`graphOptimizationLevel: "all"`, settled.** Measured across all four levels
against the decoder's 39-input, 39-output graph: every input and output name
survives at every level, and `"all"` is the fastest to open (60 ms against
220 ms for `"disabled"`, with basic at 76 and extended at 69). No artifact needs
a special case. Written explicitly anyway, so a future change to onnxruntime's
default cannot quietly move the numbers the golden tests pin.

Two risks are carried into implementation rather than resolved.

The fp16 decoder release asset is 86 MB zipped against the fp32 decoder's 47 MB
unzipped, which is unexplained. If it carries external weight data in a sidecar
file then `ArtifactRecord`'s assumption of one file per artifact is wrong and
phase 8 needs a second. Nothing in this phase can reach that artifact, so it is
deliberately absent from the manifest, and the anomaly is worth ten minutes
before phase 8 starts rather than now.

The browser figure for SHA-256 throughput is unconfirmed. The 82 ms for 157 MB
that justifies verify-always was measured under Node on an M-series Mac. The
same WebCrypto implementation class backs both, and the margin against a 16
second pipeline is roughly twenty-fold, so the conclusion is robust to being
several times wrong; it is still a number the bench page must report rather than
one this document should keep asserting.

## Next implementation step

Write `tools/fetch-models.sh` and `tools/gen-manifest.mjs`, run them, and
commit the generated `ARTIFACTS` block with the two cross-checks
(`meta.json` hashes against the manifest, `DECODER_HEAD_OUTPUTS` against phase
1's `DECODER_OUTPUT_HEADS`) as the first passing tests of phase 2 — before any
fetching, caching or session code exists.

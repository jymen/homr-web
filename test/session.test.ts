import { readFileSync } from "node:fs";
import { Tensor } from "onnxruntime-web";
import { describe, expect, it } from "vitest";
import {
  argmaxPlanes,
  maskOfClass,
  planeAgreement,
} from "../src/image/plane.js";
import { SEGNET_INPUT } from "../src/model/pipeline.js";
import { type ModelRuntime, startRuntime } from "../src/models/backend.js";
import { memoryCache } from "../src/models/cache.js";
import { encodeFloat16Array } from "../src/models/dtype.js";
import { ModelError } from "../src/models/errors.js";
import {
  DEFAULT_CATALOG,
  EXECUTION_PROVIDERS,
  MODEL_ROLE_NAMES,
  type ModelRole,
  type Placement,
  resolveRole,
} from "../src/models/manifest.js";
import {
  handoff,
  type ModelSession,
  readFloat32,
  sessionOptionsFor,
} from "../src/models/session.js";
import {
  type FetchBytes,
  type ModelEvent,
  ModelStore,
} from "../src/models/store.js";
import { BACKENDS } from "../src/result.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { describeWithModels, modelsDir } from "./support/models.js";

const NO_BATCH_DIMENSION = /declares no batch dimension/;
const NO_SUCH_INPUT = /declares no input named/;
const NOT_PAIRED = /does not pair/;

const CPU: Placement = { artifactsFor: "wasm", provider: "wasm" };
/** The fp16 artifacts on the WebAssembly provider: the axis split that lets CI cover the fp16 branch with no GPU present. */
const FP16_ON_WASM: Placement = { artifactsFor: "webgpu", provider: "wasm" };

const planOn = (role: ModelRole, placement: Placement) =>
  resolveRole(DEFAULT_CATALOG, role, placement);

function required<T>(value: T | undefined, what: string): T {
  if (value === undefined) {
    throw new Error(`the test needs ${what}`);
  }
  return value;
}

describe("sessionOptionsFor over every role, backend and provider", () => {
  const combinations: { placement: Placement; role: ModelRole }[] = [];
  for (const role of MODEL_ROLE_NAMES) {
    for (const artifactsFor of BACKENDS) {
      for (const provider of EXECUTION_PROVIDERS) {
        combinations.push({ placement: { artifactsFor, provider }, role });
      }
    }
  }

  it("covers the whole matrix", () => {
    expect(combinations).toHaveLength(
      MODEL_ROLE_NAMES.length * BACKENDS.length * EXECUTION_PROVIDERS.length
    );
  });

  it("asks for the provider the plan resolved to, at ort's fastest measured level, silently", () => {
    const providers: string[] = [];
    for (const { placement, role } of combinations) {
      const plan = planOn(role, placement);
      const options = sessionOptionsFor(plan);
      expect(options.executionProviders).toEqual([plan.provider]);
      expect(options.graphOptimizationLevel).toBe("all");
      expect(options.logSeverityLevel).toBe(3);
      expect(options.freeDimensionOverrides).toBeUndefined();
      providers.push(
        `${role} ${placement.artifactsFor}/${placement.provider} -> ${plan.provider}`
      );
    }
    expect(providers).toHaveLength(combinations.length);
  });

  it("never pins an output to a GPU buffer the plan did not ask for", () => {
    const pinned: string[] = [];
    for (const { placement, role } of combinations) {
      const plan = planOn(role, placement);
      const options = sessionOptionsFor(plan);
      if (options.preferredOutputLocation === undefined) {
        expect(plan.outputsOnGpu).toEqual([]);
      } else {
        pinned.push(
          `${role} ${placement.artifactsFor}/${placement.provider}: ${plan.outputsOnGpu.join(",")}`
        );
      }
    }
    // Nothing reaches a gpu-buffer today: segnet's logits go to a CPU argmax and
    // the encoder's request is refused while the decoder is on WebAssembly. The
    // first MODEL_ROLES row phase 8 changes turns this list non-empty.
    expect(pinned).toEqual([]);
  });

  it("pins only segnet's own batch dimension, and refuses a batch nothing can take", () => {
    expect(
      sessionOptionsFor(planOn("segnet", CPU), { batch: 16 })
        .freeDimensionOverrides
    ).toEqual({ batch_size: 16 });

    const refused: ModelRole[] = [];
    for (const role of MODEL_ROLE_NAMES) {
      const plan = planOn(role, CPU);
      if (plan.artifact.batchDim === null) {
        expect(() => sessionOptionsFor(plan, { batch: 4 })).toThrow(
          NO_BATCH_DIMENSION
        );
        refused.push(role);
      }
    }
    expect(refused).toContain("decoder");
    expect(refused).toContain("encoder");
  });

  it("leaves the decoder's cache_exists and seq_len alone", () => {
    const decoder = planOn("decoder", CPU);
    expect(decoder.artifact.batchDim).toBeNull();
    expect(() => sessionOptionsFor(decoder, { batch: 1 })).toThrow(ModelError);
    expect(sessionOptionsFor(decoder).freeDimensionOverrides).toBeUndefined();
  });
});

const wasmRuntime = (): Promise<ModelRuntime> =>
  startRuntime({ maxBackend: "wasm" });

/** models/ read here and not in the library: phase 1 set the precedent that node:fs lives behind an injected port. */
const localModels =
  (): FetchBytes =>
  ({ url }) =>
    Promise.resolve(
      new Uint8Array(
        readFileSync(`${modelsDir()}/${url.slice(url.lastIndexOf("/") + 1)}`)
      )
    );

const storeOn = async (
  placement: Placement,
  onEvent?: (event: ModelEvent) => void
): Promise<ModelStore> =>
  new ModelStore({
    baseUrl: "file:///models/",
    cache: memoryCache(),
    fetchBytes: localModels(),
    placement,
    runtime: await wasmRuntime(),
    ...(onEvent === undefined ? {} : { onEvent }),
  });

/** A short `context`: four positions of the encoder's 512 channels, which is all handoff() looks at. */
const CONTEXT_STEPS = 4;
const CONTEXT_CHANNELS = 512;

const contextFor = (session: ModelSession): Tensor => {
  const count = CONTEXT_STEPS * CONTEXT_CHANNELS;
  const values = Float32Array.from({ length: count }, (_v, i) => i / count);
  const dims = [1, CONTEXT_STEPS, CONTEXT_CHANNELS];
  return session.outputSpec("output").type === "float16"
    ? new Tensor("float16", encodeFloat16Array(values), dims)
    : new Tensor("float32", values, dims);
};

describeWithModels("opening the real artifacts", () => {
  it("opens every role on the wasm provider with the tensor names the manifest records", async () => {
    const events: ModelEvent[] = [];
    const store = await storeOn(CPU, (event) => {
      events.push(event);
    });
    try {
      const sessions = await Promise.all(
        MODEL_ROLE_NAMES.map((role) => store.open(role))
      );
      for (const session of sessions) {
        expect(`${session.role}: ${session.plan.provider}`).toBe(
          `${session.role}: wasm`
        );
        expect(session.session.inputNames).toEqual(
          session.plan.artifact.inputs.map((spec) => spec.name)
        );
        expect(session.session.outputNames).toEqual(
          session.plan.artifact.outputs.map((spec) => spec.name)
        );
      }
      expect(
        events
          .filter((event) => event.kind === "opened")
          .map((event) => event.role)
          .sort()
      ).toEqual([...MODEL_ROLE_NAMES].sort());
    } finally {
      await store.close();
    }
  });

  it("shares one download and one session between two concurrent opens", async () => {
    const urls: string[] = [];
    const store = new ModelStore({
      baseUrl: "file:///models/",
      cache: memoryCache(),
      fetchBytes: (request) => {
        urls.push(request.url);
        return localModels()(request);
      },
      placement: CPU,
      runtime: await wasmRuntime(),
    });
    try {
      const [first, second] = await Promise.all([
        store.open("segnet"),
        store.open("segnet"),
      ]);
      expect(second).toBe(first);
      expect(urls).toHaveLength(1);
    } finally {
      await store.close();
    }
  });

  it("returns one session per role and tuning, and a separate one per batch", async () => {
    const store = await storeOn(CPU);
    try {
      expect(await store.open("segnet")).toBe(await store.open("segnet"));
      expect(await store.open("segnet", { batch: 2 })).not.toBe(
        await store.open("segnet")
      );
    } finally {
      await store.close();
    }
  });

  it("reports the dtype a caller must write off the session, not off the backend", async () => {
    const fp32 = await storeOn(CPU);
    const fp16 = await storeOn(FP16_ON_WASM);
    try {
      const plain = await fp32.open("segnet");
      const half = await fp16.open("segnet");
      expect(plain.inputSpec("input").type).toBe("float32");
      expect(plain.outputSpec("output").type).toBe("float32");
      expect(half.inputSpec("input").type).toBe("float16");
      expect(half.outputSpec("output").type).toBe("float16");
      expect(() => plain.inputSpec("nope")).toThrow(NO_SUCH_INPUT);
      expect(() => plain.inputSpec("nope")).toThrow(ModelError);
    } finally {
      await Promise.all([fp32.close(), fp16.close()]);
    }
  });

  it("fixes the batch dimension it was tuned with", async () => {
    const store = await storeOn(CPU);
    try {
      const segnet = await store.open("segnet", { batch: 2 });
      const dims = [
        2,
        SEGNET_INPUT.channels,
        SEGNET_INPUT.window,
        SEGNET_INPUT.window,
      ];
      const { output } = await segnet.session.run({
        input: new Tensor(
          "float32",
          new Float32Array(dims.reduce((a, b) => a * b, 1)),
          dims
        ),
      });
      expect(required(output, "a segnet output").dims).toEqual([
        2,
        SEGNET_INPUT.classes,
        SEGNET_INPUT.window,
        SEGNET_INPUT.window,
      ]);
    } finally {
      await store.close();
    }
  });
});

describeWithModels("handoff from the encoder to the decoder", () => {
  it("hands the fp32 encoder's output straight through, untouched", async () => {
    const store = await storeOn(CPU);
    try {
      const encoder = await store.open("encoder");
      const decoder = await store.open("decoder");
      expect(encoder.plan.handoff).toEqual({
        cast: "none",
        input: "context",
        location: "cpu",
        to: "decoder",
      });
      const context = contextFor(encoder);
      expect(await handoff(context, encoder, decoder)).toBe(context);
    } finally {
      await store.close();
    }
  });

  it("casts the fp16 encoder's halves to the float32 the decoder declares", async () => {
    const store = await storeOn(FP16_ON_WASM);
    try {
      const encoder = await store.open("encoder");
      const decoder = await store.open("decoder");
      expect(encoder.plan.artifactId).toBe("encoder-396-fp16");
      expect(decoder.plan.artifactId).toBe("decoder-396-fp32");
      expect(encoder.plan.handoff).toMatchObject({
        cast: { from: "float16", to: "float32" },
        input: "context",
      });

      const context = contextFor(encoder);
      const cast = await handoff(context, encoder, decoder);

      expect(cast).not.toBe(context);
      expect(cast.type).toBe(decoder.inputSpec("context").type);
      expect(cast.dims).toEqual(context.dims);
      expect(await readFloat32(cast)).toEqual(await readFloat32(context));
    } finally {
      await store.close();
    }
  });

  it("refuses a pair the manifest does not name", async () => {
    const store = await storeOn(CPU);
    try {
      const encoder = await store.open("encoder");
      const segnet = await store.open("segnet");
      const context = contextFor(encoder);
      await expect(handoff(context, encoder, segnet)).rejects.toThrow(
        NOT_PAIRED
      );
      await expect(handoff(context, segnet, encoder)).rejects.toThrow(
        ModelError
      );
    } finally {
      await store.close();
    }
  });
});

/**
 * The tile is the ink-richest 320 square of the golden page, found with two 1-D
 * sliding windows, because a blank tile agrees perfectly and proves nothing: the
 * page's own middle is entirely background. The test asserts the tile is
 * contested before it compares anything.
 */
describeWithModels("the fp16 segnet against the fp32 one", () => {
  const TILE = SEGNET_INPUT.window;
  const INK = 128;
  const MIN_CLASSES = 3;

  const bestWindow = (counts: Float64Array, span: number): number => {
    let running = 0;
    for (let i = 0; i < span; i += 1) {
      running += counts[i] ?? 0;
    }
    let best = running;
    let at = 0;
    for (let i = span; i < counts.length; i += 1) {
      running += (counts[i] ?? 0) - (counts[i - span] ?? 0);
      if (running > best) {
        best = running;
        at = i - span + 1;
      }
    }
    return at;
  };

  const inkiestOrigin = (
    gray: Uint8Array,
    width: number,
    height: number
  ): { x: number; y: number } => {
    const rows = new Float64Array(height);
    const cols = new Float64Array(width);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        if ((gray[y * width + x] ?? 255) < INK) {
          rows[y] = (rows[y] ?? 0) + 1;
          cols[x] = (cols[x] ?? 0) + 1;
        }
      }
    }
    return { x: bestWindow(cols, TILE), y: bestWindow(rows, TILE) };
  };

  it("agrees on every class of a contested golden tile", async () => {
    const fixture = required(
      listGoldenFixtures()[0],
      "a golden fixture to take a tile from"
    );
    const page = goldenPageOf(fixture).preprocessed();
    const { x, y } = inkiestOrigin(page.data, page.width, page.height);

    // homr's inference_segnet does no normalisation: the gray page becomes three
    // identical channels of raw 0..255 floats (cv2.COLOR_GRAY2BGR, then astype).
    const pixels = TILE * TILE;
    const values = new Float32Array(SEGNET_INPUT.channels * pixels);
    for (let row = 0; row < TILE; row += 1) {
      for (let col = 0; col < TILE; col += 1) {
        const sx = Math.min(page.width - 1, x + col);
        const sy = Math.min(page.height - 1, y + row);
        const pixel = page.data[sy * page.width + sx] ?? 255;
        for (let channel = 0; channel < SEGNET_INPUT.channels; channel += 1) {
          values[channel * pixels + row * TILE + col] = pixel;
        }
      }
    }
    const dims = [1, SEGNET_INPUT.channels, TILE, TILE];

    const fp32Store = await storeOn(CPU);
    const fp16Store = await storeOn(FP16_ON_WASM);
    try {
      const fp32 = await fp32Store.open("segnet");
      const fp16 = await fp16Store.open("segnet");
      expect(fp16.inputSpec("input").type).toBe("float16");
      expect(fp16.outputSpec("output").type).toBe("float16");

      const plain = await fp32.session.run({
        input: new Tensor("float32", values, dims),
      });
      const half = await fp16.session.run({
        input: new Tensor("float16", encodeFloat16Array(values), dims),
      });

      // No branch on the output side: readFloat32 decodes the halves through the
      // table and hands the float32 buffer back as it is.
      const classesA = argmaxPlanes(
        await readFloat32(required(plain.output, "an fp32 segnet output")),
        SEGNET_INPUT.classes,
        TILE,
        TILE
      );
      const classesB = argmaxPlanes(
        await readFloat32(required(half.output, "an fp16 segnet output")),
        SEGNET_INPUT.classes,
        TILE,
        TILE
      );

      expect(new Set(classesA.data).size).toBeGreaterThanOrEqual(MIN_CLASSES);

      for (let klass = 0; klass < SEGNET_INPUT.classes; klass += 1) {
        // testing.md's mask criterion, per class, which is stricter than an
        // overall count. Measured 0.999961 on this tile, 2026-09-27.
        expect(
          planeAgreement(
            maskOfClass(classesA, klass),
            maskOfClass(classesB, klass)
          )
        ).toBeGreaterThan(0.999);
      }
    } finally {
      await Promise.all([fp32Store.close(), fp16Store.close()]);
    }
  });
});

describe("readFloat32", () => {
  it("decodes an fp16 tensor through the table", async () => {
    const values = Float32Array.from({ length: 64 }, (_v, i) => i * 8);
    const halves = new Tensor("float16", encodeFloat16Array(values), [64]);
    expect(await readFloat32(halves)).toEqual(values);
  });

  it("hands an fp32 tensor's own buffer back rather than copying it", async () => {
    const tensor = new Tensor("float32", new Float32Array([1, 2, 3]), [3]);
    expect(await readFloat32(tensor)).toBe(await tensor.getData());
  });

  it("refuses a tensor it cannot read as float32", async () => {
    const int64 = new Tensor("int64", new BigInt64Array([1n, 2n]), [2]);
    await expect(readFloat32(int64)).rejects.toThrow(ModelError);
  });
});

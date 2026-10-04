/**
 * One opened model, and the two tensor questions a caller must not have to
 * answer for itself: what dtype to write, and how to read what comes back.
 *
 * The session options are a pure function of the plan, which is what makes the
 * whole six-roles-by-backends-by-providers matrix testable with no bytes and no
 * network. onnxruntime's InferenceSession and Tensor are deliberately not
 * wrapped: phases 3, 7 and 11 build tensors and call run() in hot loops, and a
 * run() forwarder would be a pass-through method.
 */

import { InferenceSession, Tensor } from "onnxruntime-web";
import { decodeFloat16Array, encodeFloat16Array } from "./dtype.js";
import { ModelError } from "./errors.js";
import type {
  ArtifactRecord,
  ModelRole,
  ResolvedModel,
  TensorSpec,
} from "./manifest.js";
import { readOnnxMetadata } from "./onnx-metadata.js";

export interface SessionTuning {
  /**
   * Runs per session call, for a role whose artifact declares a batch
   * dimension. freeDimensionOverrides *fixes* the dimension, so the caller pads
   * the final short batch. Passing it for a role whose batchDim is null throws.
   */
  readonly batch?: number;
}

export interface ModelSession {
  /** Idempotent: onnxruntime's release() is not, and a Worker teardown races a close() from its own error path. */
  readonly close: () => Promise<void>;
  /** The spec by name, throwing on a miss: the boundary check that absorbs noUncheckedIndexedAccess once instead of at every call site. `.type` is what the caller writes, and "float16" means IEEE halves in a Uint16Array. */
  readonly inputSpec: (name: string) => TensorSpec;
  /** The file's metadata_props, which onnxruntime-web does not expose. */
  readonly metadata: ReadonlyMap<string, string>;
  readonly outputSpec: (name: string) => TensorSpec;
  readonly plan: ResolvedModel;
  readonly role: ModelRole;
  /** onnxruntime's session, unwrapped. */
  readonly session: InferenceSession;
}

/** onnxruntime's "error"; the library must not write to the console, and ort's own warnings otherwise do. */
const ORT_LOG_ERROR = 3;

const gpuLocationsFor = (
  names: readonly string[]
): Record<string, "gpu-buffer"> => {
  const locations: Record<string, "gpu-buffer"> = {};
  for (const name of names) {
    locations[name] = "gpu-buffer";
  }
  return locations;
};

function overridesFor(
  artifact: ArtifactRecord,
  role: ModelRole,
  batch: number | undefined
): Record<string, number> | undefined {
  if (batch === undefined) {
    return undefined;
  }
  if (artifact.batchDim === null) {
    throw new ModelError(
      "bad-tuning",
      `role ${role} declares no batch dimension, so a batch of ${batch} cannot be pinned`,
      { id: role }
    );
  }
  // Only the one named dimension. The decoder's `cache_exists` and `seq_len`
  // are never overridden: pinning either breaks the token loop at step 1, which
  // is why batchDim is a single name and not a free record.
  return { [artifact.batchDim]: batch };
}

/**
 * `graphOptimizationLevel: "all"` is settled by measurement rather than left to
 * ort's default: across all four levels on the decoder's 39-in/39-out graph,
 * every input and output name survived and "all" opened fastest (60 ms against
 * 220 for "disabled"). It is written explicitly so a change to ort's default
 * cannot quietly move the numbers the golden tests pin.
 */
export function sessionOptionsFor(
  plan: ResolvedModel,
  tuning: SessionTuning = {}
): InferenceSession.SessionOptions {
  const overrides = overridesFor(plan.artifact, plan.role, tuning.batch);
  return {
    executionProviders: [plan.provider],
    ...(overrides === undefined ? {} : { freeDimensionOverrides: overrides }),
    graphOptimizationLevel: "all",
    logSeverityLevel: ORT_LOG_ERROR,
    // Per output name, never a blanket location: segnet's six logit planes must
    // come back for the CPU argmax, so an empty list means the option is absent
    // rather than present and empty.
    ...(plan.outputsOnGpu.length === 0
      ? {}
      : {
          preferredOutputLocation: gpuLocationsFor(plan.outputsOnGpu),
        }),
  };
}

const specsByName = (
  specs: readonly TensorSpec[]
): ReadonlyMap<string, TensorSpec> =>
  new Map(specs.map((spec) => [spec.name, spec]));

function lookup(
  specs: ReadonlyMap<string, TensorSpec>,
  role: ModelRole,
  side: "input" | "output",
  name: string
): TensorSpec {
  const spec = specs.get(name);
  if (spec === undefined) {
    throw new ModelError(
      "unknown-tensor",
      `${role} declares no ${side} named ${JSON.stringify(name)}`,
      { expected: [...specs.keys()].join(", "), id: name }
    );
  }
  return spec;
}

/**
 * A hash match already proves the file, so this fires only when the *generator*
 * and the loader disagree, which is exactly the re-pin mistake worth catching:
 * phase 7 indexes DECODER_CACHE_IN positionally, so a renamed or reordered
 * tensor would otherwise feed the wrong cache into the wrong slot.
 */
function assertMatchesManifest(
  session: InferenceSession,
  plan: ResolvedModel
): void {
  const sides = [
    ["input", plan.artifact.inputs, session.inputNames],
    ["output", plan.artifact.outputs, session.outputNames],
  ] as const;
  for (const [side, specs, names] of sides) {
    const wanted = specs.map((spec) => spec.name).join(", ");
    const got = names.join(", ");
    if (wanted !== got) {
      throw new ModelError(
        "manifest",
        `${plan.artifactId} declares ${side}s the manifest does not: regenerate it with tools/gen-manifest.mjs`,
        { actual: got, expected: wanted, id: plan.artifactId }
      );
    }
  }
}

/**
 * Opens from bytes the caller already holds. Public for the bench page and for
 * phase 8's A/B of a locally re-exported decoder, both of which want this
 * options policy over a file that is in no manifest. Not the normal path:
 * ModelStore.open is.
 */
export async function openModelSession(
  plan: ResolvedModel,
  bytes: Uint8Array,
  tuning: SessionTuning = {}
): Promise<ModelSession> {
  const session = await InferenceSession.create(
    bytes,
    sessionOptionsFor(plan, tuning)
  );
  assertMatchesManifest(session, plan);
  const inputs = specsByName(plan.artifact.inputs);
  const outputs = specsByName(plan.artifact.outputs);
  let released = false;
  return {
    close: async () => {
      if (released) {
        return;
      }
      released = true;
      await session.release();
    },
    inputSpec: (name) => lookup(inputs, plan.role, "input", name),
    metadata: readOnnxMetadata(bytes),
    outputSpec: (name) => lookup(outputs, plan.role, "output", name),
    plan,
    role: plan.role,
    session,
  };
}

/**
 * A float16 tensor's data, when it is the elements themselves rather than their
 * bit patterns. `Tensor.DataTypeMap` maps float16 to Uint16Array and has no
 * Float16Array in it at all, so this is a parse of a value ort's own types say
 * cannot exist, not a narrowing of one they describe. It is recognised by the
 * constructor's name because Float16Array is a platform builtin: ort does not
 * define it and cannot rename it.
 */
const isFloat16Array = (data: unknown): data is ArrayLike<number> =>
  ArrayBuffer.isView(data) && data.constructor.name === "Float16Array";

/**
 * Any output tensor as float32, wherever it lives and whatever it declares.
 *
 * A float16 tensor's data arrives in one of two shapes, and which one is a
 * property of the host rather than of onnxruntime. A realm with no Float16Array
 * gets a Uint16Array of raw half bit patterns, which is what ort's own typings
 * declare and what Node gives (measured: 17893, 18611, 18476 on a golden tile
 * where the fp32 model gave 4.9, 7.5, 6.8). A realm that has one gets a real
 * Float16Array whose elements are already the numbers, and every browser this
 * port's WebGPU path runs on has had one since Chrome 135.
 *
 * Both are handled because the second is not optional: `placementOf` sends the
 * webgpu backend to the fp16 segnet artifact, so a browser on that path reads
 * every logit through here. Treating a Float16Array as bit patterns would not
 * throw, it would return a page of noise, which is why the two are told apart
 * rather than one being a fallback for the other.
 *
 * This is why phase 3's argmax, phase 7's logit heads and the bench page contain
 * no dtype code: the decode every one of them needs is one pass over the buffer,
 * which they need anyway.
 *
 * It may hand back the tensor's own buffer rather than a copy, so a caller that
 * wrote to the result would corrupt a tensor it is about to feed back in. Always
 * copying would cost a 4.9 MB allocation per segnet batch on the hot path to
 * remove a footgun no consumer today can reach: the argmax, the logit heads and
 * the bench all only read.
 */
export async function readFloat32(tensor: Tensor): Promise<Float32Array> {
  const data = await tensor.getData();
  if (tensor.type === "float16") {
    if (data instanceof Uint16Array) {
      return decodeFloat16Array(data);
    }
    if (isFloat16Array(data)) {
      return new Float32Array(data);
    }
    throw new ModelError(
      "unknown-tensor",
      `a float16 tensor gave ${data.constructor.name}, neither the Uint16Array of half patterns ort declares nor a Float16Array`
    );
  }
  if (data instanceof Float32Array) {
    return data;
  }
  throw new ModelError(
    "unknown-tensor",
    `readFloat32 cannot read a ${tensor.type} tensor`,
    { actual: tensor.type, expected: "float16 or float32" }
  );
}

/**
 * One role's output tensor as the next role's input wants it: the
 * fp16-to-fp32 cast on the WebGPU path (homr does the same,
 * staff2score.py:43-49), a read back from a GPU buffer when the placements
 * differ, and the identical tensor when both agree. Dims are carried through;
 * slicing `context` to [:, :1] after step 0 is phase 7's own business.
 *
 * This exists so the mixed-precision arrangement lives in phase 2 as data
 * (ResolvedModel.handoff, derived from two measured TensorSpecs) and in one
 * function, instead of as two lines of dtype-aware code in phase 7.
 */
export async function handoff(
  tensor: Tensor,
  from: ModelSession,
  to: ModelSession
): Promise<Tensor> {
  const plan = from.plan.handoff;
  if (plan === undefined || plan.to !== to.role) {
    throw new ModelError(
      "bad-handoff",
      `the manifest does not pair ${from.role} with ${to.role}`,
      { actual: to.role, expected: plan?.to ?? "no consumer", id: from.role }
    );
  }
  const wants = to.inputSpec(plan.input).type;
  // Both decoders take float32 at their edges, so on WebGPU the fp16 encoder's
  // context is cast below; were the dtypes to agree, it would stay on the GPU
  // without this line changing.
  if (
    plan.cast === "none" &&
    !(plan.location === "cpu" && tensor.location !== "cpu")
  ) {
    return tensor;
  }
  const values = await readFloat32(tensor);
  if (wants === "float16") {
    return new Tensor("float16", encodeFloat16Array(values), tensor.dims);
  }
  if (wants === "float32") {
    return new Tensor("float32", values, tensor.dims);
  }
  throw new ModelError(
    "bad-handoff",
    `${to.role} wants ${plan.input} as ${wants}, which a handoff cannot produce`,
    { actual: wants, id: to.role }
  );
}

/**
 * Which file each model role uses, on which execution provider, and which of
 * its outputs are left in GPU buffers. Two tables and one derivation: the
 * artifacts between the markers are measured facts about the nine files on
 * disk, written by tools/gen-manifest.mjs; MODEL_ROLES above them is
 * hand-written policy, one row per role, each citing homr's source.
 *
 * This is the only module in the library that expresses the fp16-on-GPU /
 * fp32-on-CPU arrangement and the decoder's stay on WebAssembly. No
 * other function and no call site branches on a backend or on a precision.
 * There is no I/O here.
 */

import type { Backend } from "../result.js";
import { ModelError } from "./errors.js";

/** Two providers for three backends: "wasm-threads" and "wasm" are the same execution provider, differing only in the realm's env.wasm.numThreads. */
export const EXECUTION_PROVIDERS = ["wasm", "webgpu"] as const;
export type ExecutionProvider = (typeof EXECUTION_PROVIDERS)[number];

/** The three ONNX element types the nine artifacts use. Narrowed on purpose: a fourth means the generator saw something new and the design should be looked at. */
export type TensorElementType = "float16" | "float32" | "int64";

/**
 * One model tensor as the model itself declares it, read from
 * InferenceSession.inputMetadata and never transcribed. A string in `shape` is
 * a free dimension. "float16" means IEEE halves in a Uint16Array:
 * onnxruntime-web 1.30.0 has no Float16Array on either side of a run.
 */
export interface TensorSpec {
  readonly name: string;
  readonly shape: readonly (number | string)[];
  readonly type: TensorElementType;
}

export interface ArtifactRecord {
  /**
   * The one free dimension a caller may pin through SessionTuning.batch, or
   * null. The generator derives it as the symbolic name every input carries at
   * index 0, so the decoder's `cache_exists` and `seq_len` cannot be mistaken
   * for it: pinning either would break the token loop at step 1.
   */
  readonly batchDim: string | null;
  readonly bytes: number;
  readonly inputs: readonly TensorSpec[];
  readonly outputs: readonly TensorSpec[];
  readonly sha256: string;
  /** `{sha256}/{filename}`, appended to the store's baseUrl. Content-addressed, so a deployed file is immutable, `Cache-Control: immutable` is honest, and a manifest bump cannot read a stale copy. */
  readonly urlPath: string;
}

/**
 * The nine files, by a label that names the model, its homr version and its
 * precision. The precision lives in the id and not in a field: the
 * load-bearing fact is the measured element type on each TensorSpec, and a
 * `precision` field would answer "which file" and "what do I write into the
 * tensor" with one value when the two are different questions.
 *
 * There is no decoder-396-fp16 row. homr's release has one and uses it on CUDA
 * only, fused, so the WebGPU EP cannot run it; a row nothing can reach goes
 * stale silently. decoder-396-web-fp16 is this port's own re-export, reached
 * through WEBGPU_DECODER_CATALOG.
 */
export const ARTIFACT_IDS = [
  "decoder-396-fp32",
  "decoder-396-web-fp16",
  "encoder-396-fp16",
  "encoder-396-fp32",
  "ppocr-v2-cls-mobile",
  "ppocr-v6-det-small",
  "ppocr-v6-rec-small",
  "segnet-308-fp16",
  "segnet-308-fp32",
] as const;

export type ArtifactId = (typeof ARTIFACT_IDS)[number];

/** What a caller names. Never a file and never a precision. */
export const MODEL_ROLE_NAMES = [
  "decoder",
  "encoder",
  "ocrClassify",
  "ocrDetect",
  "ocrRecognize",
  "segnet",
] as const;

export type ModelRole = (typeof MODEL_ROLE_NAMES)[number];

export type GpuPlacement =
  | {
      readonly artifact: ArtifactId;
      readonly keepOutputsOnGpu: readonly string[];
      readonly kind: "gpu";
    }
  /** `why` is a citation and, where one exists, the phase that removes it. It reaches the caller in ResolvedModel.reason. */
  | { readonly kind: "stay-on-cpu"; readonly why: string };

export interface RolePolicy {
  /** Used whenever placement.artifactsFor is not "webgpu", and when onWebgpu says stay-on-cpu. */
  readonly cpu: ArtifactId;
  /** The role this one's output feeds, and the input name it arrives as. The only place that coupling is written down; ResolvedModel.handoff is derived from it. */
  readonly feeds?: { readonly input: string; readonly role: ModelRole };
  /**
   * What this role does when placement.artifactsFor is "webgpu". The two halves
   * key on different axes, which is the thing that looked like one fact until
   * both segnet artifacts were measured running on the wasm execution provider:
   * the `artifact` is chosen by placement.artifactsFor, the `keepOutputsOnGpu`
   * request is granted by placement.provider.
   */
  readonly onWebgpu: GpuPlacement;
}

export const MODEL_ROLES = {
  decoder: {
    cpu: "decoder-396-fp32",
    onWebgpu: {
      kind: "stay-on-cpu",
      // The shipped decoder is fused (com.microsoft SkipLayerNormalization, no
      // WebGPU kernel). tools/export-decoder.py's unfused re-export runs on the
      // WebGPU EP with homr's tokens, but one token per run is all dispatch:
      // 77 to 85 ms a step with the caches left on the GPU, 130 ms without,
      // against 22 ms on four wasm threads in the same Chrome session
      // (docs/decisions.tsv, phase 8).
      // WEBGPU_DECODER_CATALOG keeps that placement measurable.
      why: "the decoder is 3 to 4 times slower a step on the WebGPU EP than on wasm threads",
    },
  },
  encoder: {
    cpu: "encoder-396-fp32",
    feeds: { input: "context", role: "decoder" },
    // homr's GPU path runs the fp16 encoder (homr/onnx_providers.py:1-16,
    // homr/transformer/configs.py:23-30). Keeping `output` on the GPU is a
    // request, not a decision: resolveRole grants it only when the decoder is
    // on the GPU too.
    onWebgpu: {
      artifact: "encoder-396-fp16",
      keepOutputsOnGpu: ["output"],
      kind: "gpu",
    },
  },
  ocrClassify: {
    cpu: "ppocr-v2-cls-mobile",
    onWebgpu: {
      kind: "stay-on-cpu",
      why: "0.6 MB and one run per strip: the CPU EP is not the cost",
    },
  },
  ocrDetect: {
    cpu: "ppocr-v6-det-small",
    onWebgpu: {
      kind: "stay-on-cpu",
      why: "PP-OCR on the WebGPU EP is unmeasured; phase 11 measures it and this row is where the answer goes",
    },
  },
  ocrRecognize: {
    cpu: "ppocr-v6-rec-small",
    onWebgpu: {
      kind: "stay-on-cpu",
      why: "PP-OCR on the WebGPU EP is unmeasured; phase 11 measures it and this row is where the answer goes",
    },
  },
  segnet: {
    cpu: "segnet-308-fp32",
    // The six logit planes go to a CPU argmax (homr's inference_segnet.py does
    // the same on its GPU path), so keepOutputsOnGpu is empty: a gpu-buffer
    // output would only be copied straight back.
    onWebgpu: {
      artifact: "segnet-308-fp16",
      keepOutputsOnGpu: [],
      kind: "gpu",
    },
  },
} as const satisfies Readonly<Record<ModelRole, RolePolicy>>;

/**
 * The two things a Backend used to decide at once. They travel together in
 * production and are separable in fact: both segnet artifacts run on the wasm
 * execution provider (measured; a single tile under Node on one wasm thread was
 * 159 ms fp32 and 177 ms fp16 on 2026-09-27, but phase 3 could not reproduce
 * those figures in any configuration and saw 878 to 1705 ms per tile on a page,
 * so treat them as unverified and take the bench's numbers instead
 * on one thread), which is how CI covers the fp16 branch with no GPU and how
 * the bench A/Bs precision against provider on one machine.
 */
export interface Placement {
  /** Which side of the fp16/fp32 arrangement to take the artifacts from. */
  readonly artifactsFor: Backend;
  readonly provider: ExecutionProvider;
}

export interface Handoff {
  /** "none" when the two artifacts agree on the element type; otherwise the cast, derived by comparing two measured TensorSpecs. Never written down twice, and it disappears on its own the day the two artifacts agree. */
  readonly cast:
    | "none"
    | { readonly from: TensorElementType; readonly to: TensorElementType };
  readonly input: string;
  readonly location: "cpu" | "gpu-buffer";
  readonly to: ModelRole;
}

export interface ResolvedModel {
  readonly artifact: ArtifactRecord;
  readonly artifactId: ArtifactId;
  /** How this role's output becomes the next role's input. Absent when the role feeds no other role. */
  readonly handoff?: Handoff;
  /** Non-empty only when this role and its consumer both run on the WebGPU execution provider: the per-output-name form, never a blanket location. */
  readonly outputsOnGpu: readonly string[];
  readonly provider: ExecutionProvider;
  /** One sentence: the artifact, the provider, why the role did not go to the GPU when it could have, and, when the two axes disagree, that the placement was asked for rather than implied. The bench prints it and phase 9 folds it into RecognizeResult.log. */
  readonly reason: string;
  readonly role: ModelRole;
}

/** The catalogue resolveRole reads. Substitutable, so the bench can A/B a locally exported artifact against a manifest one. */
export interface ModelCatalog {
  readonly artifacts: Readonly<Record<ArtifactId, ArtifactRecord>>;
  readonly roles: Readonly<Record<ModelRole, RolePolicy>>;
}

const artifactIdFor = (policy: RolePolicy, placement: Placement): ArtifactId =>
  policy.onWebgpu.kind === "gpu" && placement.artifactsFor === "webgpu"
    ? policy.onWebgpu.artifact
    : policy.cpu;

const providerFor = (
  policy: RolePolicy,
  placement: Placement
): ExecutionProvider =>
  policy.onWebgpu.kind === "stay-on-cpu" ? "wasm" : placement.provider;

const impliedProvider = (backend: Backend): ExecutionProvider =>
  backend === "webgpu" ? "webgpu" : "wasm";

/** A feeding role hands on one tensor, so its artifact must declare exactly one output for `cast` to be derivable at all. */
function soleOutput(artifact: ArtifactRecord, role: ModelRole): TensorSpec {
  const [only, ...rest] = artifact.outputs;
  if (only === undefined || rest.length > 0) {
    throw new ModelError(
      "manifest",
      `role ${role} feeds another role but its artifact declares ${artifact.outputs.length} outputs`,
      { id: role }
    );
  }
  return only;
}

function handoffFor(
  catalog: ModelCatalog,
  role: ModelRole,
  policy: RolePolicy,
  placement: Placement,
  artifact: ArtifactRecord
): Handoff | undefined {
  const { feeds } = policy;
  if (feeds === undefined) {
    return undefined;
  }
  if (feeds.role === role) {
    throw new ModelError("manifest", `role ${role} feeds itself`, { id: role });
  }
  const consumer = catalog.roles[feeds.role];
  const consumerArtifact =
    catalog.artifacts[artifactIdFor(consumer, placement)];
  const source = soleOutput(artifact, role);
  const sink = consumerArtifact.inputs.find(
    (spec) => spec.name === feeds.input
  );
  if (sink === undefined) {
    throw new ModelError(
      "manifest",
      `role ${role} feeds ${feeds.role} as ${feeds.input}, which that artifact does not declare`,
      { expected: feeds.input, id: role }
    );
  }
  return {
    cast:
      source.type === sink.type ? "none" : { from: source.type, to: sink.type },
    input: feeds.input,
    location:
      providerFor(policy, placement) === "webgpu" &&
      providerFor(consumer, placement) === "webgpu"
        ? "gpu-buffer"
        : "cpu",
    to: feeds.role,
  };
}

/**
 * The single place the fp16-on-GPU / fp32-on-CPU arrangement and the decoder's
 * stay on WebAssembly are expressed.
 */
export function resolveRole(
  catalog: ModelCatalog,
  role: ModelRole,
  placement: Placement
): ResolvedModel {
  const policy = catalog.roles[role];
  const { onWebgpu } = policy;
  const artifactId = artifactIdFor(policy, placement);
  const artifact = catalog.artifacts[artifactId];
  const provider = providerFor(policy, placement);
  const consumerOnGpu =
    policy.feeds === undefined ||
    providerFor(catalog.roles[policy.feeds.role], placement) === "webgpu";
  const outputsOnGpu =
    onWebgpu.kind === "gpu" && provider === "webgpu" && consumerOnGpu
      ? onWebgpu.keepOutputsOnGpu
      : [];
  const notes = [`${artifactId} on the ${provider} EP`];
  if (onWebgpu.kind === "stay-on-cpu" && placement.artifactsFor === "webgpu") {
    notes.push(`although WebGPU is available: ${onWebgpu.why}`);
  }
  if (placement.provider !== impliedProvider(placement.artifactsFor)) {
    notes.push("placement requested");
  }
  const reason = notes.join(", ");
  const handoff = handoffFor(catalog, role, policy, placement, artifact);
  if (handoff === undefined) {
    return { artifact, artifactId, outputsOnGpu, provider, reason, role };
  }
  return {
    artifact,
    artifactId,
    handoff,
    outputsOnGpu,
    provider,
    reason,
    role,
  };
}

// BEGIN GENERATED (tools/gen-manifest.mjs)
export const ARTIFACTS = {
  "decoder-396-fp32": {
    batchDim: null,
    bytes: 47_299_551,
    inputs: [
      { name: "rhythms", shape: [1, 1], type: "int64" },
      { name: "pitchs", shape: [1, 1], type: "int64" },
      { name: "lifts", shape: [1, 1], type: "int64" },
      { name: "articulations", shape: [1, 1], type: "int64" },
      { name: "slurs", shape: [1, 1], type: "int64" },
      { name: "context", shape: [1, "cache_exists", 512], type: "float32" },
      { name: "cache_len", shape: [1], type: "int64" },
      { name: "cache_in0", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in1", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in2", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in3", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in4", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in5", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in6", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in7", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in8", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in9", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in10", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in11", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in12", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in13", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in14", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in15", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in16", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in17", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in18", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in19", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in20", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in21", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in22", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in23", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in24", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in25", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in26", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in27", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in28", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in29", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in30", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in31", shape: [1, 8, "seq_len", 64], type: "float32" },
    ],
    outputs: [
      { name: "out_rhythms", shape: [1, 1, 259], type: "float32" },
      { name: "out_pitchs", shape: [1, 1, 72], type: "float32" },
      { name: "out_lifts", shape: [1, 1, 7], type: "float32" },
      { name: "out_positions", shape: [1, 1, 3], type: "float32" },
      { name: "out_articulations", shape: [1, 1, 54], type: "float32" },
      { name: "out_slurs", shape: [1, 1, 5], type: "float32" },
      { name: "attention", shape: [2], type: "float32" },
      { name: "cache_out0", shape: [1, 8, "seq_len + 1", 64], type: "float32" },
      { name: "cache_out1", shape: [1, 8, "seq_len + 1", 64], type: "float32" },
      {
        name: "cache_out2",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out3",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      { name: "cache_out4", shape: [1, 8, "seq_len + 1", 64], type: "float32" },
      { name: "cache_out5", shape: [1, 8, "seq_len + 1", 64], type: "float32" },
      {
        name: "cache_out6",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out7",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      { name: "cache_out8", shape: [1, 8, "seq_len + 1", 64], type: "float32" },
      { name: "cache_out9", shape: [1, 8, "seq_len + 1", 64], type: "float32" },
      {
        name: "cache_out10",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out11",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out12",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out13",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out14",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out15",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out16",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out17",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out18",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out19",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out20",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out21",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out22",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out23",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out24",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out25",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out26",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out27",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out28",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out29",
        shape: [1, 8, "seq_len + 1", 64],
        type: "float32",
      },
      {
        name: "cache_out30",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
      {
        name: "cache_out31",
        shape: [1, 8, "cache_exists + seq_len", 64],
        type: "float32",
      },
    ],
    sha256: "3e10fd5ae52d0b86792721922fcd954c283a7ed365de7446425bdabe38f3e57d",
    urlPath:
      "3e10fd5ae52d0b86792721922fcd954c283a7ed365de7446425bdabe38f3e57d/decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
  },
  "decoder-396-web-fp16": {
    batchDim: null,
    bytes: 94_133_299,
    inputs: [
      { name: "rhythms", shape: [1, 1], type: "int64" },
      { name: "pitchs", shape: [1, 1], type: "int64" },
      { name: "lifts", shape: [1, 1], type: "int64" },
      { name: "articulations", shape: [1, 1], type: "int64" },
      { name: "slurs", shape: [1, 1], type: "int64" },
      { name: "context", shape: [1, "cache_exists", 512], type: "float32" },
      { name: "cache_len", shape: [1], type: "int64" },
      { name: "cache_in0", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in1", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in2", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in3", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in4", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in5", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in6", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in7", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in8", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in9", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in10", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in11", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in12", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in13", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in14", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in15", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in16", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in17", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in18", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in19", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in20", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in21", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in22", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in23", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in24", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in25", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in26", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in27", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in28", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in29", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in30", shape: [1, 8, "seq_len", 64], type: "float32" },
      { name: "cache_in31", shape: [1, 8, "seq_len", 64], type: "float32" },
    ],
    outputs: [
      { name: "out_rhythms", shape: [1, 1, 259], type: "float32" },
      { name: "out_pitchs", shape: [1, 1, 72], type: "float32" },
      { name: "out_lifts", shape: [1, 1, 7], type: "float32" },
      { name: "out_positions", shape: [1, 1, 3], type: "float32" },
      { name: "out_articulations", shape: [1, 1, 54], type: "float32" },
      { name: "out_slurs", shape: [1, 1, 5], type: "float32" },
      { name: "attention", shape: [2], type: "float32" },
      {
        name: "cache_out0",
        shape: [1, 8, "Concatcache_out0_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out1",
        shape: [1, 8, "Concatcache_out1_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out2",
        shape: [1, 8, "Concatcache_out2_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out3",
        shape: [1, 8, "Concatcache_out3_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out4",
        shape: [1, 8, "Concatcache_out4_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out5",
        shape: [1, 8, "Concatcache_out5_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out6",
        shape: [1, 8, "Concatcache_out6_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out7",
        shape: [1, 8, "Concatcache_out7_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out8",
        shape: [1, 8, "Concatcache_out8_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out9",
        shape: [1, 8, "Concatcache_out9_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out10",
        shape: [1, 8, "Concatcache_out10_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out11",
        shape: [1, 8, "Concatcache_out11_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out12",
        shape: [1, 8, "Concatcache_out12_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out13",
        shape: [1, 8, "Concatcache_out13_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out14",
        shape: [1, 8, "Concatcache_out14_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out15",
        shape: [1, 8, "Concatcache_out15_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out16",
        shape: [1, 8, "Concatcache_out16_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out17",
        shape: [1, 8, "Concatcache_out17_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out18",
        shape: [1, 8, "Concatcache_out18_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out19",
        shape: [1, 8, "Concatcache_out19_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out20",
        shape: [1, 8, "Concatcache_out20_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out21",
        shape: [1, 8, "Concatcache_out21_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out22",
        shape: [1, 8, "Concatcache_out22_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out23",
        shape: [1, 8, "Concatcache_out23_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out24",
        shape: [1, 8, "Concatcache_out24_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out25",
        shape: [1, 8, "Concatcache_out25_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out26",
        shape: [1, 8, "Concatcache_out26_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out27",
        shape: [1, 8, "Concatcache_out27_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out28",
        shape: [1, 8, "Concatcache_out28_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out29",
        shape: [1, 8, "Concatcache_out29_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out30",
        shape: [1, 8, "Concatcache_out30_dim_2", 64],
        type: "float32",
      },
      {
        name: "cache_out31",
        shape: [1, 8, "Concatcache_out31_dim_2", 64],
        type: "float32",
      },
    ],
    sha256: "253172b02c50883e3f41beb6e842cf707596de8b2f7497a52ed7a01b3f76a27a",
    urlPath:
      "253172b02c50883e3f41beb6e842cf707596de8b2f7497a52ed7a01b3f76a27a/decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903_web_fp16.onnx",
  },
  "encoder-396-fp16": {
    batchDim: null,
    bytes: 26_466_256,
    inputs: [{ name: "input", shape: [1, 1, 256, 1280], type: "float16" }],
    outputs: [{ name: "output", shape: [1, 1280, 512], type: "float16" }],
    sha256: "9db62d5a6a13c8df2df321af3bcf72c7f81a95d4f876d4f0c202b28f8658087e",
    urlPath:
      "9db62d5a6a13c8df2df321af3bcf72c7f81a95d4f876d4f0c202b28f8658087e/encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903_fp16.onnx",
  },
  "encoder-396-fp32": {
    batchDim: null,
    bytes: 52_861_122,
    inputs: [{ name: "input", shape: [1, 1, 256, 1280], type: "float32" }],
    outputs: [{ name: "output", shape: [1, 1280, 512], type: "float32" }],
    sha256: "4c16df852b3789f2676b0d49f0545dab0740e4005f7b472c5252add642f5d5eb",
    urlPath:
      "4c16df852b3789f2676b0d49f0545dab0740e4005f7b472c5252add642f5d5eb/encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
  },
  "ppocr-v2-cls-mobile": {
    batchDim: null,
    bytes: 585_532,
    inputs: [{ name: "x", shape: ["?", 3, "?", "?"], type: "float32" }],
    outputs: [
      {
        name: "save_infer_model/scale_0.tmp_1",
        shape: ["?", 2],
        type: "float32",
      },
    ],
    sha256: "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c",
    urlPath:
      "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c/ch_ppocr_mobile_v2.0_cls_mobile.onnx",
  },
  "ppocr-v6-det-small": {
    batchDim: "DynamicDimension.0",
    bytes: 9_929_594,
    inputs: [
      {
        name: "x",
        shape: [
          "DynamicDimension.0",
          3,
          "DynamicDimension.1",
          "DynamicDimension.2",
        ],
        type: "float32",
      },
    ],
    outputs: [
      {
        name: "fetch_name_0",
        shape: [
          "ConvTranspose_459_o0__d0",
          1,
          "ConvTranspose_459_o0__d2",
          "ConvTranspose_459_o0__d3",
        ],
        type: "float32",
      },
    ],
    sha256: "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    urlPath:
      "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f/PP-OCRv6_det_small.onnx",
  },
  "ppocr-v6-rec-small": {
    batchDim: "DynamicDimension.0",
    bytes: 21_234_383,
    inputs: [
      {
        name: "x",
        shape: ["DynamicDimension.0", 3, 48, "DynamicDimension.1"],
        type: "float32",
      },
    ],
    outputs: [
      {
        name: "fetch_name_0",
        shape: ["DynamicDimension.0", "Reshape_470_o0__d2", 18_710],
        type: "float32",
      },
    ],
    sha256: "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    urlPath:
      "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884/PP-OCRv6_rec_small.onnx",
  },
  "segnet-308-fp16": {
    batchDim: "batch_size",
    bytes: 28_667_207,
    inputs: [
      { name: "input", shape: ["batch_size", 3, 320, 320], type: "float16" },
    ],
    outputs: [
      { name: "output", shape: ["batch_size", 6, 320, 320], type: "float16" },
    ],
    sha256: "60f495496cb41473c0521d0811d8f44b9d5cff892d287974a8aebb3eaee2fa83",
    urlPath:
      "60f495496cb41473c0521d0811d8f44b9d5cff892d287974a8aebb3eaee2fa83/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx",
  },
  "segnet-308-fp32": {
    batchDim: "batch_size",
    bytes: 57_311_361,
    inputs: [
      { name: "input", shape: ["batch_size", 3, 320, 320], type: "float32" },
    ],
    outputs: [
      { name: "output", shape: ["batch_size", 6, 320, 320], type: "float32" },
    ],
    sha256: "6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d",
    urlPath:
      "6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
  },
} as const satisfies Readonly<Record<ArtifactId, ArtifactRecord>>;

/** The 32 attention key/value caches in the order the decoder declares them (8 layers x 4, matching homr's decoder_depth = 8). Phase 7 iterates these; it never builds a name with a template string. */
export const DECODER_CACHE_IN = [
  "cache_in0",
  "cache_in1",
  "cache_in2",
  "cache_in3",
  "cache_in4",
  "cache_in5",
  "cache_in6",
  "cache_in7",
  "cache_in8",
  "cache_in9",
  "cache_in10",
  "cache_in11",
  "cache_in12",
  "cache_in13",
  "cache_in14",
  "cache_in15",
  "cache_in16",
  "cache_in17",
  "cache_in18",
  "cache_in19",
  "cache_in20",
  "cache_in21",
  "cache_in22",
  "cache_in23",
  "cache_in24",
  "cache_in25",
  "cache_in26",
  "cache_in27",
  "cache_in28",
  "cache_in29",
  "cache_in30",
  "cache_in31",
] as const;

/** The same caches on the output side, index for index with DECODER_CACHE_IN. */
export const DECODER_CACHE_OUT = [
  "cache_out0",
  "cache_out1",
  "cache_out2",
  "cache_out3",
  "cache_out4",
  "cache_out5",
  "cache_out6",
  "cache_out7",
  "cache_out8",
  "cache_out9",
  "cache_out10",
  "cache_out11",
  "cache_out12",
  "cache_out13",
  "cache_out14",
  "cache_out15",
  "cache_out16",
  "cache_out17",
  "cache_out18",
  "cache_out19",
  "cache_out20",
  "cache_out21",
  "cache_out22",
  "cache_out23",
  "cache_out24",
  "cache_out25",
  "cache_out26",
  "cache_out27",
  "cache_out28",
  "cache_out29",
  "cache_out30",
  "cache_out31",
] as const;

/** The logit outputs, in the order the decoder declares them. test/manifest.test.ts pins this against phase 1's DECODER_OUTPUT_HEADS. */
export const DECODER_HEAD_OUTPUTS = [
  "out_rhythms",
  "out_pitchs",
  "out_lifts",
  "out_positions",
  "out_articulations",
  "out_slurs",
] as const;
// END GENERATED

export const DEFAULT_CATALOG: ModelCatalog = {
  artifacts: ARTIFACTS,
  roles: MODEL_ROLES,
};

/**
 * DEFAULT_CATALOG with the decoder on WebGPU: the fp16 re-export, its 32 caches
 * left in GPU buffers between steps. Not the default because it is slower (see
 * MODEL_ROLES.decoder); the bench's `decoder=webgpu` and a golden test use it,
 * so the artifact cannot go stale unnoticed.
 */
export const WEBGPU_DECODER_CATALOG: ModelCatalog = {
  artifacts: ARTIFACTS,
  roles: {
    ...MODEL_ROLES,
    decoder: {
      cpu: "decoder-396-fp32",
      onWebgpu: {
        artifact: "decoder-396-web-fp16",
        keepOutputsOnGpu: DECODER_CACHE_OUT,
        kind: "gpu",
      },
    },
  },
};

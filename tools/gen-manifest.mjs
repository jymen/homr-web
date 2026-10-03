// Regenerates src/models/manifest.ts's ARTIFACTS block and decoder tensor-name
// tuples from the nine files in models/: eight from tools/fetch-models.sh, one from
// tools/export-decoder.py. Every
// value it writes is a fact about a file: the SHA-256, the byte length, and the
// input/output names, shapes and element types read from the opened session.
// Nothing it writes is a judgement; MODEL_ROLES and resolveRole are
// hand-written above the markers and this tool never touches them.
//
// Run after a re-pin. The diff is the model change, and a hash that moves under
// the golden data fails test/manifest.test.ts.
import { createHash } from "node:crypto";
import { readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { env, InferenceSession } from "onnxruntime-web";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const target = join(root, "src/models/manifest.ts");

/**
 * ArtifactId to filename. The ids are the hand-written ArtifactId union in
 * manifest.ts, so a name only present here is a compile error over there.
 */
const FILES = {
  "decoder-396-fp32":
    "decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
  "decoder-396-web-fp16":
    "decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903_web_fp16.onnx",
  "encoder-396-fp16":
    "encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903_fp16.onnx",
  "encoder-396-fp32":
    "encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx",
  "ppocr-v2-cls-mobile": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
  "ppocr-v6-det-small": "PP-OCRv6_det_small.onnx",
  "ppocr-v6-rec-small": "PP-OCRv6_rec_small.onnx",
  "segnet-308-fp16":
    "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx",
  "segnet-308-fp32": "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx",
};

/** The union manifest.ts narrows TensorElementType to; a fourth means the design needs another look. */
const ELEMENT_TYPES = new Set(["float16", "float32", "int64"]);

const CACHE_IN = /^cache_in(\d+)$/;
const CACHE_OUT = /^cache_out(\d+)$/;
const HEAD_OUTPUT = /^out_/;

/**
 * ONNX spells an unknown dimension either as a dim_param, which arrives as a
 * string, or as dim_value -1, which arrives through onnxruntime's metadata as
 * the unsigned 2^32 - 1. PP-OCRv2's classifier carries both spellings in one
 * input (`[-1, 3, ?, ?]`). Writing 4294967295 verbatim would put a dimension of
 * four billion in the manifest, so an unknown dimension becomes the "?" that
 * onnxruntime already uses for the unnamed case. TensorSpec.shape's rule then
 * stays true: a string is a free dimension.
 */
const UNKNOWN_DIM = "?";
const UNSIGNED_MINUS_ONE = 4_294_967_295;

const dimOf = (dim, artifact, tensor) => {
  if (typeof dim === "string") {
    return dim;
  }
  if (dim === UNSIGNED_MINUS_ONE || dim < 0) {
    return UNKNOWN_DIM;
  }
  if (!Number.isSafeInteger(dim) || dim === 0) {
    throw new Error(`${artifact}: ${tensor} has dimension ${dim}`);
  }
  return dim;
};

const specOf = (metadata, artifact) => {
  if (!metadata.isTensor) {
    throw new Error(`${artifact}: ${metadata.name} is not a tensor`);
  }
  if (!ELEMENT_TYPES.has(metadata.type)) {
    throw new Error(
      `${artifact}: ${metadata.name} is ${metadata.type}, which TensorElementType does not admit`
    );
  }
  return {
    name: metadata.name,
    shape: metadata.shape.map((dim) => dimOf(dim, artifact, metadata.name)),
    type: metadata.type,
  };
};

/**
 * The one free dimension SessionTuning.batch may pin: the named symbolic
 * dimension every input carries at index 0. Derived rather than declared, so
 * that the decoder's `cache_exists` and `seq_len`, which sit further in and must
 * never be pinned, cannot be mistaken for it. An unnamed free dimension is not
 * one either, because freeDimensionOverrides has no name to key it on.
 */
const batchDimOf = (inputs) => {
  const first = inputs[0]?.shape[0];
  if (typeof first !== "string" || first === UNKNOWN_DIM) {
    return null;
  }
  return inputs.every((input) => input.shape[0] === first) ? first : null;
};

const numbered = (names, pattern, artifact) => {
  const found = names
    .map((name) => [name, pattern.exec(name)])
    .filter(([, match]) => match !== null)
    .map(([name, match]) => [name, Number(match[1])])
    .sort((a, b) => a[1] - b[1]);
  for (const [index, [name, n]] of found.entries()) {
    if (index !== n) {
      throw new Error(`${artifact}: gap at ${name}, expected index ${index}`);
    }
  }
  return found.map(([name]) => name);
};

const quote = (value) => JSON.stringify(value);
const shapeLiteral = (shape) => `[${shape.map(quote).join(", ")}]`;
const specLiteral = (spec) =>
  `{ name: ${quote(spec.name)}, shape: ${shapeLiteral(spec.shape)}, type: ${quote(spec.type)} }`;
const specsLiteral = (specs) =>
  specs.length === 0
    ? "[]"
    : `[\n${specs.map((s) => `      ${specLiteral(s)},`).join("\n")}\n    ]`;

const tuple = (name, doc, names) =>
  `${doc}\nexport const ${name} = [\n${names.map((n) => `  ${quote(n)},`).join("\n")}\n] as const;\n`;

env.wasm.numThreads = 1;
env.logLevel = "error";

const records = [];
let decoder;

for (const [id, filename] of Object.entries(FILES)) {
  const path = join(root, "models", filename);
  const bytes = readFileSync(path);
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  const session = await InferenceSession.create(bytes, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,
  });
  const inputs = session.inputMetadata.map((m) => specOf(m, id));
  const outputs = session.outputMetadata.map((m) => specOf(m, id));
  records.push({
    batchDim: batchDimOf(inputs),
    bytes: statSync(path).size,
    id,
    inputs,
    outputs,
    sha256,
    urlPath: `${sha256}/${filename}`,
  });
  if (id === "decoder-396-fp32") {
    decoder = { inputs, outputs };
  }
  console.log(`${id}: ${sha256} ${inputs.length} in, ${outputs.length} out`);
  await session.release();
}

if (decoder === undefined) {
  throw new Error("no decoder artifact: the cache tuples cannot be generated");
}

// Phase 7 feeds every decoder through the one set of tuples generated below,
// so a re-export that renamed, reordered or retyped a tensor must stop here.
const contractOf = (specs) =>
  specs.map((s) => `${s.name}:${s.type}`).join(", ");
for (const record of records) {
  if (
    record.id.startsWith("decoder-") &&
    (contractOf(record.inputs) !== contractOf(decoder.inputs) ||
      contractOf(record.outputs) !== contractOf(decoder.outputs))
  ) {
    throw new Error(`${record.id} breaks decoder-396-fp32's tensor contract`);
  }
}

const artifacts = records
  .map(
    (r) => `  ${quote(r.id)}: {
    batchDim: ${r.batchDim === null ? "null" : quote(r.batchDim)},
    bytes: ${r.bytes},
    inputs: ${specsLiteral(r.inputs)},
    outputs: ${specsLiteral(r.outputs)},
    sha256: ${quote(r.sha256)},
    urlPath: ${quote(r.urlPath)},
  },`
  )
  .join("\n");

const inputNames = decoder.inputs.map((s) => s.name);
const outputNames = decoder.outputs.map((s) => s.name);
const cacheIn = numbered(inputNames, CACHE_IN, "decoder-396-fp32");
const cacheOut = numbered(outputNames, CACHE_OUT, "decoder-396-fp32");
const heads = outputNames.filter((name) => HEAD_OUTPUT.test(name));
if (cacheIn.length !== cacheOut.length) {
  throw new Error(
    `decoder cache arity: ${cacheIn.length} in, ${cacheOut.length} out`
  );
}

const block = `export const ARTIFACTS = {
${artifacts}
} as const satisfies Readonly<Record<ArtifactId, ArtifactRecord>>;

${tuple(
  "DECODER_CACHE_IN",
  `/** The ${cacheIn.length} attention key/value caches in the order the decoder declares them (8 layers x 4, matching homr's decoder_depth = 8). Phase 7 iterates these; it never builds a name with a template string. */`,
  cacheIn
)}
${tuple(
  "DECODER_CACHE_OUT",
  "/** The same caches on the output side, index for index with DECODER_CACHE_IN. */",
  cacheOut
)}
${tuple(
  "DECODER_HEAD_OUTPUTS",
  "/** The logit outputs, in the order the decoder declares them. test/manifest.test.ts pins this against phase 1's DECODER_OUTPUT_HEADS. */",
  heads
)}`;

const file = readFileSync(target, "utf8");
const begin = "// BEGIN GENERATED (tools/gen-manifest.mjs)\n";
const end = "// END GENERATED\n";
const start = file.indexOf(begin);
const stop = file.indexOf(end);
if (start < 0 || stop < 0) {
  throw new Error(`${target} lacks the generated-block markers`);
}
writeFileSync(
  target,
  `${file.slice(0, start + begin.length)}${block}${file.slice(stop)}`
);
console.log(
  `wrote ${records.length} artifacts, ${cacheIn.length} caches, ${heads.length} heads`
);

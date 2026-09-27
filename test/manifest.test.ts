import { describe, expect, it } from "vitest";
import {
  DECODER_CACHE_IN,
  DECODER_CACHE_OUT,
  DECODER_HEAD_OUTPUTS,
  DEFAULT_CATALOG,
  MODEL_ROLE_NAMES,
  type ModelRole,
  type Placement,
  resolveRole,
} from "../src/models/manifest.js";
import { DECODER_OUTPUT_HEADS } from "../src/transformer/vocabulary.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";

const CPU: Placement = { artifactsFor: "wasm", provider: "wasm" };
const THREADS: Placement = { artifactsFor: "wasm-threads", provider: "wasm" };
const GPU: Placement = { artifactsFor: "webgpu", provider: "webgpu" };
/** The fp16 artifacts on the WebAssembly provider: the combination CI covers with no GPU. */
const FP16_ON_WASM: Placement = { artifactsFor: "webgpu", provider: "wasm" };

/** The three models tools/dump-golden.py records in meta.json, as roles. */
const GOLDEN_MODEL_ROLES = [
  "decoder",
  "encoder",
  "segnet",
] as const satisfies readonly ModelRole[];

const filenameOf = (urlPath: string): string =>
  urlPath.slice(urlPath.lastIndexOf("/") + 1);

const planOn = (role: ModelRole, placement: Placement) =>
  resolveRole(DEFAULT_CATALOG, role, placement);

describe("the manifest against the golden data", () => {
  it("names the artifacts the fixtures were dumped from", () => {
    const fixtures = listGoldenFixtures();
    expect(fixtures.length).toBeGreaterThan(0);
    for (const fixture of fixtures) {
      const recorded = goldenPageOf(fixture).meta().models;
      for (const role of GOLDEN_MODEL_ROLES) {
        const { artifact } = planOn(role, CPU);
        expect(`${fixture.name} ${role}: ${recorded[role]}`).toBe(
          `${fixture.name} ${role}: ${filenameOf(artifact.urlPath)}:${artifact.sha256}`
        );
      }
    }
  });
});

describe("the decoder tuples against phase 1", () => {
  it("maps the generated logit heads onto DECODER_OUTPUT_HEADS, in order", () => {
    expect([...DECODER_HEAD_OUTPUTS]).toEqual(
      DECODER_OUTPUT_HEADS.map((head) => `out_${head}s`)
    );
  });

  it("pairs every cache input with one output", () => {
    expect(DECODER_CACHE_IN).toHaveLength(32);
    expect(DECODER_CACHE_OUT).toHaveLength(DECODER_CACHE_IN.length);
  });

  it("names only tensors the decoder artifact declares", () => {
    const { artifact } = planOn("decoder", CPU);
    const inputs = new Set(artifact.inputs.map((spec) => spec.name));
    const outputs = new Set(artifact.outputs.map((spec) => spec.name));
    expect(DECODER_CACHE_IN.filter((name) => !inputs.has(name))).toEqual([]);
    expect(
      [...DECODER_CACHE_OUT, ...DECODER_HEAD_OUTPUTS].filter(
        (name) => !outputs.has(name)
      )
    ).toEqual([]);
  });
});

describe("resolveRole", () => {
  it("puts every role on the wasm provider with its cpu artifact", () => {
    for (const role of MODEL_ROLE_NAMES) {
      const plan = planOn(role, CPU);
      expect(`${role}: ${plan.provider} ${plan.artifactId}`).toBe(
        `${role}: wasm ${DEFAULT_CATALOG.roles[role].cpu}`
      );
      expect(plan.outputsOnGpu).toEqual([]);
    }
  });

  it("treats wasm-threads and wasm as one placement", () => {
    for (const role of MODEL_ROLE_NAMES) {
      expect(planOn(role, THREADS)).toEqual(planOn(role, CPU));
    }
  });

  it("takes the fp16 artifacts on the webgpu placement", () => {
    expect(planOn("segnet", GPU).artifactId).toBe("segnet-308-fp16");
    expect(planOn("encoder", GPU).artifactId).toBe("encoder-396-fp16");
  });

  it("keeps the decoder on wasm fp32 whatever the placement asks for", () => {
    for (const placement of [CPU, GPU, FP16_ON_WASM]) {
      const plan = planOn("decoder", placement);
      expect(plan.artifactId).toBe("decoder-396-fp32");
      expect(plan.provider).toBe("wasm");
    }
    expect(planOn("decoder", GPU).reason).toContain(
      "although WebGPU is available: the shipped decoder has no SkipLayerNormalization kernel"
    );
  });

  it("refuses the encoder's gpu-buffer request while the decoder is on wasm", () => {
    expect(DEFAULT_CATALOG.roles.encoder.onWebgpu).toMatchObject({
      keepOutputsOnGpu: ["output"],
    });
    expect(planOn("encoder", GPU).outputsOnGpu).toEqual([]);
  });

  it("leaves segnet's logits on the cpu for the argmax", () => {
    expect(planOn("segnet", GPU).outputsOnGpu).toEqual([]);
  });

  it("derives the encoder-to-decoder cast from the two measured specs", () => {
    expect(planOn("encoder", CPU).handoff).toEqual({
      cast: "none",
      input: "context",
      location: "cpu",
      to: "decoder",
    });
    expect(planOn("encoder", GPU).handoff).toEqual({
      cast: { from: "float16", to: "float32" },
      input: "context",
      location: "cpu",
      to: "decoder",
    });
  });

  it("leaves a role that feeds nobody without a handoff", () => {
    expect(planOn("segnet", CPU).handoff).toBeUndefined();
    expect(planOn("decoder", CPU).handoff).toBeUndefined();
  });

  it("runs the fp16 artifacts on the wasm provider and says the placement was asked for", () => {
    const plan = planOn("segnet", FP16_ON_WASM);
    expect(plan.artifactId).toBe("segnet-308-fp16");
    expect(plan.provider).toBe("wasm");
    expect(plan.artifact.inputs[0]?.type).toBe("float16");
    expect(plan.artifact.outputs[0]?.type).toBe("float16");
    expect(plan.reason).toBe(
      "segnet-308-fp16 on the wasm EP, placement requested"
    );
  });
});

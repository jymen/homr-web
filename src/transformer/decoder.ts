/**
 * decoder_inference.py's ScoreDecoder.generate: the greedy loop with the KV
 * cache, up to 608 steps, one argmax per head, stopping at EOS.
 *
 * The feeds are one object reused across steps. Its five one-element id
 * tensors and `cache_len` wrap typed arrays the loop overwrites in place, which
 * is safe because onnxruntime copies every input into its own heap when a run
 * starts. Only `context` (after step 0) and the 32 caches are swapped.
 */

import { Tensor } from "onnxruntime-web";
import { argmax } from "../image/numeric.js";
import { ModelError } from "../models/errors.js";
import { DECODER_CACHE_IN, DECODER_CACHE_OUT } from "../models/manifest.js";
import { type ModelSession, readFloat32 } from "../models/session.js";
import type { DecodedSymbol } from "./symbol.js";
import { type Head, type TokenId, tokenAt, tokenIndex } from "./vocabulary.js";

/** One staff's symbols, straight out of the decoder (before the position filter). */
export type TokenSequence = readonly DecodedSymbol[];

/** configs.py: max_seq_len. */
export const MAX_SEQ_LEN = 608;
const EOS = tokenIndex("rhythm", "EOS");
const BOS = tokenIndex("rhythm", "BOS");
/** configs.py: decoder_heads and decoder_dim / decoder_heads. */
const CACHE_HEADS = 8;
const HEAD_DIM = 64;
const CONTEXT_WIDTH = 512;

/** The heads fed back, with their input names. The position head is read but never fed. */
const FED = [
  ["rhythm", "rhythms"],
  ["pitch", "pitchs"],
  ["lift", "lifts"],
  ["articulation", "articulations"],
  ["slur", "slurs"],
] as const satisfies readonly (readonly [Head, string])[];

function outputOf(outputs: Record<string, Tensor>, name: string): Tensor {
  const tensor = outputs[name];
  if (tensor === undefined) {
    throw new ModelError("unknown-tensor", `the decoder returned no ${name}`, {
      id: name,
    });
  }
  return tensor;
}

const idOf = async <H extends Head>(logits: Tensor): Promise<TokenId<H>> =>
  argmax(await readFloat32(logits)) as TokenId<H>;

/**
 * The state of one decode: the reused feeds and the 32 caches. It borrows the
 * session, which ModelStore owns and shares across staffs, and owns every
 * tensor it creates or receives; `dispose` releases them, on EOS, on the step
 * limit or on an abort alike.
 */
export class DecoderState {
  readonly #session: ModelSession;
  readonly #ids: Record<(typeof FED)[number][0], BigInt64Array>;
  readonly #cacheLen = new BigInt64Array(1);
  readonly #feeds: Record<string, Tensor> = {};
  readonly #reducedContext: Tensor;
  #step = 0;

  constructor(session: ModelSession, context: Tensor) {
    this.#session = session;
    const [, rows, width] = context.dims;
    if (
      context.type !== "float32" ||
      !(context.data instanceof Float32Array) ||
      rows === undefined ||
      width !== CONTEXT_WIDTH
    ) {
      throw new ModelError(
        "unknown-tensor",
        `the decoder context must be float32 [1, n, ${CONTEXT_WIDTH}] on the CPU, got ${context.type} [${context.dims.join(", ")}]`
      );
    }
    this.#ids = {
      articulation: new BigInt64Array(1),
      lift: new BigInt64Array(1),
      pitch: new BigInt64Array(1),
      rhythm: BigInt64Array.of(BigInt(BOS)),
      slur: new BigInt64Array(1),
    };
    for (const [head, input] of FED) {
      this.#feeds[input] = new Tensor("int64", this.#ids[head], [1, 1]);
    }
    this.#feeds.cache_len = new Tensor("int64", this.#cacheLen, [1]);
    this.#feeds.context = context;
    // context[:, :1], built once. Float32 on the CPU only: on every placement
    // today the decoder is the fp32 artifact on WebAssembly, and handoff has
    // already cast and downloaded the context. Phase 8 revisits this.
    this.#reducedContext = new Tensor(
      "float32",
      context.data.slice(0, CONTEXT_WIDTH),
      [1, 1, CONTEXT_WIDTH]
    );
    const empty = new Float32Array(0);
    for (const name of DECODER_CACHE_IN) {
      this.#feeds[name] = new Tensor("float32", empty, [
        1,
        CACHE_HEADS,
        0,
        HEAD_DIM,
      ]);
    }
  }

  get step(): number {
    return this.#step;
  }

  /** One run. Null when the rhythm head says EOS; the symbol otherwise. */
  async next(): Promise<DecodedSymbol | null> {
    this.#cacheLen[0] = BigInt(this.#step);
    const outputs = await this.#session.session.run(this.#feeds);
    for (const [i, name] of DECODER_CACHE_IN.entries()) {
      this.#feeds[name]?.dispose();
      this.#feeds[name] = outputOf(outputs, DECODER_CACHE_OUT[i] ?? "");
    }
    if (this.#step === 0) {
      this.#feeds.context = this.#reducedContext;
    }
    this.#step += 1;
    const rhythm = await idOf<"rhythm">(outputOf(outputs, "out_rhythms"));
    const pitch = await idOf<"pitch">(outputOf(outputs, "out_pitchs"));
    const lift = await idOf<"lift">(outputOf(outputs, "out_lifts"));
    const position = await idOf<"position">(outputOf(outputs, "out_positions"));
    const articulation = await idOf<"articulation">(
      outputOf(outputs, "out_articulations")
    );
    const slur = await idOf<"slur">(outputOf(outputs, "out_slurs"));
    const attention = await readFloat32(outputOf(outputs, "attention"));
    if (rhythm === EOS) {
      return null;
    }
    this.#ids.rhythm[0] = BigInt(rhythm);
    this.#ids.pitch[0] = BigInt(pitch);
    this.#ids.lift[0] = BigInt(lift);
    this.#ids.articulation[0] = BigInt(articulation);
    this.#ids.slur[0] = BigInt(slur);
    return {
      articulation: tokenAt("articulation", articulation),
      coordinates: {
        x: attention[0] ?? Number.NaN,
        y: attention[1] ?? Number.NaN,
      },
      lift: tokenAt("lift", lift),
      pitch: tokenAt("pitch", pitch),
      position: tokenAt("position", position),
      rhythm: tokenAt("rhythm", rhythm),
      slur: tokenAt("slur", slur),
    };
  }

  dispose(): void {
    for (const name of DECODER_CACHE_IN) {
      this.#feeds[name]?.dispose();
    }
  }
}

export interface DecodeOptions {
  /** Called after each step with the step count, for timing and progress. */
  readonly onStep?: (step: number) => void;
  /** Checked before every step; an abort rejects with the signal's reason after the state is released. */
  readonly signal?: AbortSignal;
}

/** ScoreDecoder.generate from BOS, for one encoded staff. */
export async function runDecoder(
  decoder: ModelSession,
  context: Tensor,
  options: DecodeOptions = {}
): Promise<TokenSequence> {
  const state = new DecoderState(decoder, context);
  const symbols: DecodedSymbol[] = [];
  try {
    while (state.step < MAX_SEQ_LEN) {
      options.signal?.throwIfAborted();
      // biome-ignore lint/performance/noAwaitInLoops: each step feeds the previous step's argmax
      const symbol = await state.next();
      options.onStep?.(state.step);
      if (symbol === null) {
        break;
      }
      symbols.push(symbol);
    }
  } finally {
    state.dispose();
  }
  return symbols;
}

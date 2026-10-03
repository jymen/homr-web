/**
 * The decoder loop against a scripted session, no model: what it feeds at
 * each step, when it stops, and that it releases its caches on every exit.
 * The golden test proves the tokens; this proves the contract phase 9's
 * Worker relies on (abort) and the feeds a token match could hide.
 */

import { type InferenceSession, Tensor } from "onnxruntime-web";
import { describe, expect, it, vi } from "vitest";
import {
  ARTIFACTS,
  DECODER_CACHE_IN,
  DECODER_CACHE_OUT,
} from "../src/models/manifest.js";
import type { ModelSession } from "../src/models/session.js";
import { MAX_SEQ_LEN, runDecoder } from "../src/transformer/decoder.js";
import { tokenIndex } from "../src/transformer/vocabulary.js";

const SIZES = {
  out_articulations: 54,
  out_lifts: 7,
  out_pitchs: 72,
  out_positions: 3,
  out_rhythms: 259,
  out_slurs: 5,
} as const;

type Step = Partial<Record<keyof typeof SIZES, number>>;

interface Seen {
  readonly cacheIn0: Tensor | undefined;
  readonly cacheLen: bigint;
  readonly contextRows: number;
  readonly fed: Record<string, bigint>;
}

const firstId = (tensor: Tensor | undefined): bigint =>
  tensor?.data instanceof BigInt64Array ? (tensor.data[0] ?? -1n) : -1n;

/** Each run answers the next scripted step (one-hot logits) and records what it was fed. */
function scriptedDecoder(script: readonly Step[]) {
  const seen: Seen[] = [];
  const caches: Tensor[] = [];
  const run = vi.fn((feeds: Record<string, Tensor>) => {
    const step = script[seen.length] ?? {};
    const fed: Record<string, bigint> = {};
    for (const name of [
      "rhythms",
      "pitchs",
      "lifts",
      "articulations",
      "slurs",
    ]) {
      fed[name] = firstId(feeds[name]);
    }
    seen.push({
      cacheIn0: feeds.cache_in0,
      cacheLen: firstId(feeds.cache_len),
      contextRows: feeds.context?.dims[1] ?? -1,
      fed,
    });
    const outputs: Record<string, Tensor> = {};
    for (const [name, size] of Object.entries(SIZES)) {
      const logits = new Float32Array(size);
      logits[step[name as keyof typeof SIZES] ?? 0] = 1;
      outputs[name] = new Tensor("float32", logits, [1, 1, size]);
    }
    outputs.attention = new Tensor(
      "float32",
      Float32Array.of(seen.length, 7),
      [2]
    );
    for (const name of DECODER_CACHE_OUT) {
      const cache = new Tensor(
        "float32",
        new Float32Array(8 * 64 * seen.length),
        [1, 8, seen.length, 64]
      );
      vi.spyOn(cache, "dispose");
      caches.push(cache);
      outputs[name] = cache;
    }
    return Promise.resolve(outputs);
  });
  const session = {
    close: () => Promise.resolve(),
    inputSpec: () => {
      throw new Error("unused");
    },
    outputSpec: () => {
      throw new Error("unused");
    },
    plan: undefined,
    role: "decoder",
    session: { run } as unknown as InferenceSession,
  } as unknown as ModelSession;
  return { caches, run, seen, session };
}

const context = () =>
  new Tensor("float32", new Float32Array(1280 * 512), [1, 1280, 512]);

const EOS = tokenIndex("rhythm", "EOS");

describe("runDecoder", () => {
  it("feeds BOS and nonote, then each step's argmax, the step count and context[:, :1]", async () => {
    const { seen, session } = scriptedDecoder([
      {
        out_articulations: 3,
        out_lifts: 2,
        out_pitchs: 40,
        out_rhythms: 69,
        out_slurs: 4,
      },
      { out_positions: 1, out_rhythms: 4 },
      { out_rhythms: EOS },
    ]);
    const symbols = await runDecoder(session, context());
    expect(
      symbols.map((s) => [
        s.rhythm,
        s.pitch,
        s.lift,
        s.articulation,
        s.slur,
        s.position,
      ])
    ).toEqual([
      ["note_2", "F4", "#", "accent_arpeggiate", "slurStop", "."],
      ["barline", ".", ".", ".", ".", "upper"],
    ]);
    expect(symbols.map((s) => s.coordinates)).toEqual([
      { x: 1, y: 7 },
      { x: 2, y: 7 },
    ]);
    expect(seen.map((s) => s.cacheLen)).toEqual([0n, 1n, 2n]);
    expect(seen.map((s) => s.contextRows)).toEqual([1280, 1, 1]);
    expect(seen.map((s) => s.fed)).toEqual([
      { articulations: 0n, lifts: 0n, pitchs: 0n, rhythms: 1n, slurs: 0n },
      { articulations: 3n, lifts: 2n, pitchs: 40n, rhythms: 69n, slurs: 4n },
      { articulations: 0n, lifts: 0n, pitchs: 0n, rhythms: 4n, slurs: 0n },
    ]);
    expect(seen[0]?.cacheIn0?.dims).toEqual([1, 8, 0, 64]);
  });

  it("feeds each cache_out back as the cache_in of the same index", async () => {
    const { caches, seen, session } = scriptedDecoder([
      {},
      { out_rhythms: EOS },
    ]);
    await runDecoder(session, context());
    expect(seen[1]?.cacheIn0).toBe(caches[0]);
    expect(DECODER_CACHE_IN.length).toBe(
      ARTIFACTS["decoder-396-fp32"].inputs.length - 7
    );
  });

  it("releases every cache on EOS", async () => {
    const { caches, session } = scriptedDecoder([{}, {}, { out_rhythms: EOS }]);
    await runDecoder(session, context());
    for (const cache of caches) {
      expect(cache.dispose).toHaveBeenCalled();
    }
  });

  it("stops after max_seq_len steps without EOS", async () => {
    const { run, session } = scriptedDecoder([]);
    const symbols = await runDecoder(session, context());
    expect(run).toHaveBeenCalledTimes(MAX_SEQ_LEN);
    expect(symbols).toHaveLength(MAX_SEQ_LEN);
  });

  it("rejects with the signal's reason before the next step, and releases the caches", async () => {
    const controller = new AbortController();
    const { caches, run, session } = scriptedDecoder([]);
    const reason = new Error("cancelled");
    const decoding = runDecoder(session, context(), {
      onStep: (step) => {
        if (step === 3) {
          controller.abort(reason);
        }
      },
      signal: controller.signal,
    });
    await expect(decoding).rejects.toBe(reason);
    expect(run).toHaveBeenCalledTimes(3);
    for (const cache of caches) {
      expect(cache.dispose).toHaveBeenCalled();
    }
  });

  it("refuses a context that is not float32 [1, n, 512]", async () => {
    const { session } = scriptedDecoder([]);
    await expect(
      runDecoder(
        session,
        new Tensor("float32", new Float32Array(10), [1, 1, 10])
      )
    ).rejects.toThrow("float32 [1, n, 512]");
  });
});

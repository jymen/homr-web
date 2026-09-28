/**
 * `inference_segnet.inference`: every tile of one preprocessed page through one
 * segnet session, and the per-tile argmax stitched back into one class map.
 *
 * A pure function of a session and a gray page, with no runtime, no store and no
 * opencv in it, so the golden test can drive it on the Python stage output and
 * compare masks. `SegmentationWorker` in worker.ts is the impure half: it owns
 * the session and runs preprocess first.
 *
 * Two allocations decide the shape of the loop. The input buffer is
 * `batch * 3 * 320 * 320` elements, 9.8 MB at a batch of 8, and it is allocated
 * once and refilled: onnxruntime copies a tensor's data into the wasm heap when
 * the run starts, so the buffer is free again the moment the previous run
 * resolved. And the output of a batch is `batch * 6 * 320 * 320` floats, 19.7 MB,
 * which is why the batches are run one after another: the page's seven batches
 * together would hold 207 MB of tensors in one wasm arena instead of 30.
 */

import { Tensor } from "onnxruntime-web";
import type { ClassMap, GrayImage } from "../image/plane.js";
import { argmaxPlanes, createGray } from "../image/plane.js";
import {
  createSegmentationResult,
  SEGNET_INPUT,
  type SegmentationResult,
} from "../model/pipeline.js";
import { ModelError } from "../models/errors.js";
import { type ModelSession, readFloat32 } from "../models/session.js";
import {
  extractTile,
  mergeTileClasses,
  tileGrid,
  writeTileInto,
} from "./tiles.js";

/** The one input and the one output segnet declares; `assertMatchesManifest` has already proved the opened session agrees with the manifest about both. */
const SEGNET_INPUT_NAME = "input";
const SEGNET_OUTPUT_NAME = "output";

export interface SegmentOptions {
  /** Tiles per session call. The session must have been opened with the same value, because freeDimensionOverrides pins the dimension. */
  readonly batch: number;
  readonly onProgress?: (done: number, total: number) => void;
}

/**
 * The batch buffer, in whatever the opened artifact wants. An unexpected dtype
 * is a `ModelError` rather than a float32 fallback, because writing float32 into
 * a half input is not a degraded result: it is 2.4 million garbage pixels that
 * still produce a plausible-looking mask.
 */
function inputBufferFor(
  session: ModelSession,
  elements: number
): Float32Array | Uint16Array {
  const { type } = session.inputSpec(SEGNET_INPUT_NAME);
  if (type === "float32") {
    return new Float32Array(elements);
  }
  if (type === "float16") {
    // onnxruntime-web has no Float16Array on either side of a run, so a half
    // tensor is raw IEEE bit patterns; writeTileInto branches on this type.
    return new Uint16Array(elements);
  }
  throw new ModelError(
    "unknown-tensor",
    `segnet declares its input as ${type}, which segmentPage cannot write`,
    {
      actual: type,
      expected: "float16 or float32",
      id: session.plan.artifactId,
    }
  );
}

/** Branching on the buffer's own type rather than on a flag, for writeTileInto's reason: the buffer the loop allocated is the only authority on what the model was given. */
const tensorOf = (
  data: Float32Array | Uint16Array,
  dims: readonly number[]
): Tensor =>
  data instanceof Uint16Array
    ? new Tensor("float16", data, dims)
    : new Tensor("float32", data, dims);

/**
 * Every tile of `page` through `session`, merged: `inference`'s two loops, its
 * batching and its `merge_patches` call.
 *
 * The final batch is short, and the batch dimension is pinned, so its unused
 * items are filled with a blank 255 tile and their outputs discarded. Leaving
 * the previous batch's bytes there would cost nothing and be wrong for a reason
 * that outlasts this function: the tensor handed to the model would then depend
 * on the page's tile order, so a run would not be reproducible from its inputs
 * and a dump of the last batch would show tiles that are not on the page. 255 is
 * what `extractTile` already pads an off-page tile with, and one blank tile is
 * written at most `batch - 1` times on the last batch of a page.
 */
export async function segmentPage(
  session: ModelSession,
  page: GrayImage,
  options: SegmentOptions
): Promise<SegmentationResult> {
  const { batch } = options;
  if (!Number.isInteger(batch) || batch < 1) {
    throw new ModelError(
      "bad-tuning",
      `a segnet batch is a positive integer, got ${batch}`,
      { actual: String(batch), id: session.plan.artifactId }
    );
  }
  const { channels, classes: classCount, window } = SEGNET_INPUT;
  const grid = tileGrid(page.width, page.height);
  const perInput = channels * window * window;
  const perOutput = classCount * window * window;
  const buffer = inputBufferFor(session, batch * perInput);
  const dims = [batch, channels, window, window];
  const wantedShape = [batch, classCount, window, window].join("x");
  const tiles: ClassMap[] = [];
  let blank: GrayImage | undefined;

  for (let first = 0; first < grid.length; first += batch) {
    // Iterated rather than indexed, because `entries()` yields the origin itself
    // and an index into `grid` would need a `| undefined` branch that the loop
    // bound already makes unreachable.
    const group = grid.slice(first, first + batch);
    for (const [item, origin] of group.entries()) {
      writeTileInto(buffer, item * perInput, extractTile(page, origin));
    }
    for (let item = group.length; item < batch; item += 1) {
      blank ??= createGray(window, window, 255);
      writeTileInto(buffer, item * perInput, blank);
    }

    // Sequential on purpose. The batch is what buys the parallelism here, and the
    // model's pinned batch dimension is what bounds it; running the seven batches
    // of a page together would only pile up their tensors. See biome.jsonc's
    // noAwaitInLoops override for this file.
    const outputs = await session.session.run({
      [SEGNET_INPUT_NAME]: tensorOf(buffer, dims),
    });
    const output = outputs[SEGNET_OUTPUT_NAME];
    if (output === undefined) {
      throw new ModelError(
        "unknown-tensor",
        `the segnet run produced ${Object.keys(outputs).join(", ") || "nothing"}, not ${SEGNET_OUTPUT_NAME}`,
        { expected: SEGNET_OUTPUT_NAME, id: session.plan.artifactId }
      );
    }
    // The shape, not just the element count: `assertMatchesManifest` compares
    // tensor *names* only, so a re-export that emitted NHWC would hand back the
    // right number of floats in the wrong order and the argmax below would read
    // six garbage planes into a mask that still looks like a mask.
    const shape = output.dims.join("x");
    if (shape !== wantedShape) {
      throw new ModelError(
        "unknown-tensor",
        `the segnet run gave ${shape} logits, not ${wantedShape}: the session was opened with a different batch, or the artifact is not NCHW`,
        { actual: shape, expected: wantedShape, id: session.plan.artifactId }
      );
    }
    const logits = await readFloat32(output);
    for (let item = 0; item < group.length; item += 1) {
      tiles.push(
        argmaxPlanes(
          logits.subarray(item * perOutput, (item + 1) * perOutput),
          classCount,
          window,
          window
        )
      );
    }
    options.onProgress?.(tiles.length, grid.length);
  }

  return createSegmentationResult(
    mergeTileClasses(tiles, page.width, page.height)
  );
}

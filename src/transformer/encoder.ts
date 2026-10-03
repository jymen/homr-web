/**
 * encoder_inference.py's Encoder.generate and the cast staff2score.py makes
 * before the decoder: one run, and the output handed to the decoder in the
 * element type and location the decoder wants (phase 2's `handoff`).
 */

import type { Tensor } from "onnxruntime-web";
import type { GrayImage } from "../image/plane.js";
import { handoff, type ModelSession } from "../models/session.js";
import { encoderInput } from "./normalize.js";

const INPUT = "input";
const OUTPUT = "output";

/** The decoder's `context`, [1, 1280, 512], for one staff canvas. */
export async function encodeCanvas(
  encoder: ModelSession,
  decoder: ModelSession,
  canvas: GrayImage
): Promise<Tensor> {
  const input = encoderInput(canvas, encoder.inputSpec(INPUT).type);
  const outputs = await encoder.session.run({ [INPUT]: input });
  const output = outputs[OUTPUT];
  if (output === undefined) {
    throw new Error(`the encoder returned no ${OUTPUT}`);
  }
  const context = await handoff(output, encoder, decoder);
  if (context !== output) {
    output.dispose();
  }
  return context;
}

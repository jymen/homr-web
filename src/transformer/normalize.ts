/**
 * staff2score.py's ConvertToArray: `(pixel / 255 - 0.7931) / 0.1738` in
 * float64, stored as float32, as the [1, 1, 256, 1280] encoder input. A pixel
 * takes one of 256 values, so the arithmetic is done once per value into a
 * table; JavaScript's float64 is IEEE, so each entry equals numpy's.
 *
 * For the fp16 encoder homr casts that float32 array to float16
 * (encoder_inference.py), a second rounding the half table reproduces.
 */

import { Tensor } from "onnxruntime-web";
import type { GrayImage } from "../image/plane.js";
import { ENCODER_CANVAS } from "../model/pipeline.js";
import { float16FromFloat32 } from "../models/dtype.js";
import { ModelError } from "../models/errors.js";
import type { TensorElementType } from "../models/manifest.js";

const MEAN = 0.7931;
const STD = 0.1738;

export const NORMALIZED_FLOAT32: Float32Array = Float32Array.from(
  { length: 256 },
  (_, v) => (v / 255 - MEAN) / STD
);

const NORMALIZED_FLOAT16: Uint16Array = Uint16Array.from(
  NORMALIZED_FLOAT32,
  float16FromFloat32
);

const DIMS = [1, 1, ENCODER_CANVAS.height, ENCODER_CANVAS.width] as const;

/** The encoder input tensor for a canvas, in the element type the opened encoder declares. */
export function encoderInput(
  canvas: GrayImage,
  type: TensorElementType
): Tensor {
  const pixels = canvas.data;
  if (type === "float32") {
    const out = new Float32Array(pixels.length);
    for (let i = 0; i < pixels.length; i += 1) {
      out[i] = NORMALIZED_FLOAT32[pixels[i] ?? 0] ?? 0;
    }
    return new Tensor("float32", out, DIMS);
  }
  if (type === "float16") {
    const out = new Uint16Array(pixels.length);
    for (let i = 0; i < pixels.length; i += 1) {
      out[i] = NORMALIZED_FLOAT16[pixels[i] ?? 0] ?? 0;
    }
    return new Tensor("float16", out, DIMS);
  }
  throw new ModelError(
    "unknown-tensor",
    `the encoder declares its input as ${type}, which encoderInput cannot write`,
    { actual: type, expected: "float16 or float32" }
  );
}

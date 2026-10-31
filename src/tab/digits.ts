/**
 * A small handwritten-digit classifier for the fret marks of a tab already
 * found: RapidOCR reads print well and a hand badly (a 4 as "u", a 0 as "9").
 * The model (the tabDigits role, tools/train-tab-digits.sh) takes MNIST's
 * form, the mark's tight ink box scaled so its longer side is 20 px and
 * centred in 28 x 28, ink 1 on 0, and answers ten digits and an eleventh
 * class, not a digit. digitInput is that form's one definition: the training
 * crops are made with it (tools/tab-digits-data.ts), and the trainer's own
 * MNIST augmentation follows it.
 */
import { Tensor } from "onnxruntime-web";
import type { ColorImage } from "../image/plane.js";
import { ModelError } from "../models/errors.js";
import { type ModelSession, readFloat32 } from "../models/session.js";

export const NOT_A_DIGIT = 10;

/** Values per crop in the classifier's input: one 28 x 28 plane. */
export const DIGIT_INPUT = 28 * 28;

export interface DigitGuess {
  /** 0 to 9, or NOT_A_DIGIT. */
  readonly digit: number;
  readonly probability: number;
}

/** One crop as the classifier's 28 x 28 input, written at `offset`. */
export function digitInput(
  crop: ColorImage,
  out: Float32Array,
  offset: number
): void {
  const { ink, box } = inkBoxOf(crop);
  if (box === undefined) {
    return;
  }
  const { width } = crop;
  const boxW = box.x1 - box.x0 + 1;
  const boxH = box.y1 - box.y0 + 1;
  const scale = 20 / Math.max(boxW, boxH);
  const outW = Math.max(1, Math.round(boxW * scale));
  const outH = Math.max(1, Math.round(boxH * scale));
  const left = Math.floor((28 - outW) / 2);
  const top = Math.floor((28 - outH) / 2);
  // Area resampling: each output pixel averages the source pixels it covers.
  const covered = (from: number, to: number) =>
    Array.from({ length: Math.ceil(to) - Math.floor(from) }, (_, k) => {
      const at = Math.floor(from) + k;
      return { at, share: Math.min(at + 1, to) - Math.max(at, from) };
    });
  for (let oy = 0; oy < outH; oy += 1) {
    const rows = covered(
      box.y0 + (oy * boxH) / outH,
      box.y0 + ((oy + 1) * boxH) / outH
    );
    for (let ox = 0; ox < outW; ox += 1) {
      const cols = covered(
        box.x0 + (ox * boxW) / outW,
        box.x0 + ((ox + 1) * boxW) / outW
      );
      let sum = 0;
      let area = 0;
      for (const row of rows) {
        for (const col of cols) {
          sum += (ink[row.at * width + col.at] ?? 0) * row.share * col.share;
          area += row.share * col.share;
        }
      }
      out[offset + (top + oy) * 28 + left + ox] = area > 0 ? sum / area : 0;
    }
  }
}

/** The crop's ink, 1 where darker than mid-gray, and the box around it (undefined on a blank crop). */
function inkBoxOf(crop: ColorImage): {
  readonly box?: { x0: number; x1: number; y0: number; y1: number };
  readonly ink: Float32Array;
} {
  const { width, height, data } = crop;
  const ink = new Float32Array(width * height);
  const box = { x0: width, x1: -1, y0: height, y1: -1 };
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const p = (y * width + x) * 3;
      const gray =
        ((data[p] ?? 255) + (data[p + 1] ?? 255) + (data[p + 2] ?? 255)) / 3;
      if (gray < 128) {
        ink[y * width + x] = 1;
        box.x0 = Math.min(box.x0, x);
        box.x1 = Math.max(box.x1, x);
        box.y0 = Math.min(box.y0, y);
        box.y1 = Math.max(box.y1, y);
      }
    }
  }
  return box.x1 < 0 ? { ink } : { box, ink };
}

/** The likeliest class of each row of logits, with its softmax probability. */
export function digitGuesses(logits: Float32Array, n: number): DigitGuess[] {
  const guesses: DigitGuess[] = [];
  for (let i = 0; i < n; i += 1) {
    const row = logits.subarray(i * 11, i * 11 + 11);
    const max = Math.max(...row);
    let total = 0;
    let best = 0;
    for (let c = 0; c < 11; c += 1) {
      total += Math.exp((row[c] ?? 0) - max);
      if ((row[c] ?? 0) > (row[best] ?? 0)) {
        best = c;
      }
    }
    guesses.push({ digit: best, probability: 1 / total });
  }
  return guesses;
}

/** The classifier (the tabDigits role) over a batch of crops, in order. */
export async function classifyDigits(
  session: ModelSession,
  crops: readonly ColorImage[]
): Promise<DigitGuess[]> {
  if (crops.length === 0) {
    return [];
  }
  const [input] = session.plan.artifact.inputs;
  const [output] = session.plan.artifact.outputs;
  if (input === undefined || output === undefined) {
    throw new ModelError(
      "manifest",
      `${session.role} declares no input or output`
    );
  }
  const data = new Float32Array(crops.length * DIGIT_INPUT);
  for (const [i, crop] of crops.entries()) {
    digitInput(crop, data, i * DIGIT_INPUT);
  }
  const outputs = await session.session.run({
    [input.name]: new Tensor("float32", data, [crops.length, 1, 28, 28]),
  });
  const logits = outputs[output.name];
  if (logits === undefined) {
    throw new ModelError(
      "unknown-tensor",
      `the ${session.role} run produced no ${output.name}`,
      { expected: output.name, id: session.plan.artifactId }
    );
  }
  try {
    return digitGuesses(await readFloat32(logits), crops.length);
  } finally {
    logits.dispose();
  }
}

/** The classifier over a batch of crops, in order. */
export type ClassifyDigits = (
  crops: readonly ColorImage[]
) => Promise<DigitGuess[]>;

/**
 * How sure the classifier must be for its digit to stand. At 0.95 it changed
 * no note on 16 printed pages (2189 notes) and gave Weeping Willow 94 of its
 * 129 notes against RapidOCR's 80; at 0.8 it added frets to two printed pages
 * (docs/design/tab-4-findings.md).
 */
export const DIGIT_SURE = 0.95;

/**
 * The text a fret mark is taken to say: RapidOCR's read, unless the classifier
 * is DIGIT_SURE of a digit and RapidOCR either read nothing the tab reader can
 * use (a hand's 4 as "u") or read another single digit (a hand's 0 as "9").
 * A two-digit fret, a technique letter or a fret with letters stuck to it is
 * never touched, and "not a digit" never removes a read: print stays read as
 * RapidOCR reads it.
 */
export function fuseDigit(
  ocr: string,
  usable: boolean,
  guess: DigitGuess
): string {
  if (guess.digit === NOT_A_DIGIT || guess.probability < DIGIT_SURE) {
    return ocr;
  }
  const said = String(guess.digit);
  if (!usable || SINGLE_DIGIT.test(ocr.trim())) {
    return said;
  }
  return ocr;
}

const SINGLE_DIGIT = /^\d$/;

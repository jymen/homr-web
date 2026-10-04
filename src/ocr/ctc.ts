/**
 * RapidOCR 3.9.2's CTCLabelDecode (rapidocr/ch_ppocr_rec/utils.py) for the
 * line-text case: greedy argmax, repeated indices merged, the blank dropped.
 */

import { mean, npRound, pyRound } from "../image/numeric.js";

/** str.splitlines' boundaries: \r\n first, then each of Python's single line breaks. */
// biome-ignore lint/suspicious/noControlCharactersInRegex: \x1c to \x1e are line boundaries to Python's splitlines.
const PYTHON_LINE_BREAKS = /\r\n|[\n\v\f\r\x1c\x1d\x1e\x85\u2028\u2029]/;

/**
 * The decoder's alphabet from the model's `character` metadata, as
 * get_character_list and get_character build it: `splitlines()`, a space
 * appended, "blank" at index 0.
 */
export function ctcCharacters(metadata: string): readonly string[] {
  const lines = metadata.split(PYTHON_LINE_BREAKS);
  if (lines.at(-1) === "") {
    lines.pop();
  }
  return ["blank", ...lines, " "];
}

export interface RecognizedText {
  readonly score: number;
  readonly text: string;
}

/**
 * One row of the recogniser's output, `steps` by `classes` probabilities.
 * Each kept character's probability is rounded to 5 digits (Python round),
 * and the score is numpy's mean of those, rounded to 5 digits again.
 */
export function ctcDecode(
  probabilities: Float32Array,
  offset: number,
  steps: number,
  classes: number,
  characters: readonly string[]
): RecognizedText {
  const kept: number[] = [];
  let text = "";
  let previous = -1;
  for (let t = 0; t < steps; t += 1) {
    const row = offset + t * classes;
    let best = 0;
    let bestValue = probabilities[row] ?? Number.NEGATIVE_INFINITY;
    for (let c = 1; c < classes; c += 1) {
      const value = probabilities[row + c] ?? Number.NEGATIVE_INFINITY;
      if (value > bestValue) {
        best = c;
        bestValue = value;
      }
    }
    if (best !== 0 && best !== previous) {
      text += characters[best] ?? "";
      kept.push(pyRound(bestValue, 5));
    }
    previous = best;
  }
  return { score: npRound(mean(kept.length === 0 ? [0] : kept), 5), text };
}

/**
 * The tab digit classifier (src/tab/digits.ts): the input it is given, how a
 * guess reads, the rule that fuses it with RapidOCR's read, and the model on
 * digits it never saw in training, drawn here in OpenCV's Hershey script face,
 * a pen-like hand.
 */

import { describe, expect, it } from "vitest";
import { withMatScope } from "../src/cv/opencv.js";
import { type ColorImage, createColor } from "../src/image/plane.js";
import {
  classifyDigits,
  DIGIT_INPUT,
  DIGIT_SURE,
  type DigitGuess,
  digitGuesses,
  digitInput,
  fuseDigit,
  NOT_A_DIGIT,
} from "../src/tab/digits.js";
import { CPU, describeWithModels, storeOn } from "./support/models.js";
import { testOpenCv } from "./support/opencv.js";

/** A white crop with a black rectangle. */
function bar(
  width: number,
  height: number,
  box: { x: number; y: number; w: number; h: number }
): ColorImage {
  const crop = createColor(width, height);
  crop.data.fill(255);
  const { h, w, x, y: top } = box;
  for (let y = top; y < top + h; y += 1) {
    crop.data.fill(0, (y * width + x) * 3, (y * width + x + w) * 3);
  }
  return crop;
}

describe("digitInput", () => {
  it("scales the ink box so its longer side is 20 pixels, centred in 28 x 28", () => {
    const out = new Float32Array(DIGIT_INPUT);
    digitInput(bar(30, 50, { h: 40, w: 10, x: 12, y: 5 }), out, 0);
    const inked = [...out.entries()].filter(([, v]) => v > 0).map(([i]) => i);
    const rows = inked.map((i) => Math.floor(i / 28));
    const cols = inked.map((i) => i % 28);
    expect([Math.min(...rows), Math.max(...rows)]).toEqual([4, 23]);
    expect([Math.min(...cols), Math.max(...cols)]).toEqual([11, 15]);
    expect(inked.every((i) => out[i] === 1)).toBe(true);
  });

  it("writes at the offset and leaves a blank crop's plane at zero", () => {
    const out = new Float32Array(2 * DIGIT_INPUT);
    digitInput(bar(20, 20, { h: 0, w: 0, x: 0, y: 0 }), out, 0);
    digitInput(bar(20, 20, { h: 10, w: 10, x: 5, y: 5 }), out, DIGIT_INPUT);
    expect(out.subarray(0, DIGIT_INPUT).every((v) => v === 0)).toBe(true);
    expect(out.subarray(DIGIT_INPUT).filter((v) => v === 1)).toHaveLength(400);
  });
});

describe("digitGuesses", () => {
  it("takes each row's likeliest class with its softmax probability", () => {
    const logits = new Float32Array(22);
    logits[4] = 3;
    logits[11 + NOT_A_DIGIT] = 1;
    const [four, none] = digitGuesses(logits, 2);
    expect(four?.digit).toBe(4);
    expect(four?.probability).toBeCloseTo(Math.exp(3) / (Math.exp(3) + 10), 6);
    expect(none?.digit).toBe(NOT_A_DIGIT);
  });
});

describe("fuseDigit", () => {
  const sure = (digit: number): DigitGuess => ({ digit, probability: 0.99 });
  it("takes the classifier's digit where RapidOCR read nothing usable, or another single digit", () => {
    expect(fuseDigit("u", false, sure(4))).toBe("4");
    expect(fuseDigit("", false, sure(1))).toBe("1");
    expect(fuseDigit("9", true, sure(0))).toBe("0");
  });

  it("never touches two digits, letters, or a fret with letters, and never removes a read", () => {
    expect(fuseDigit("12", true, sure(1))).toBe("12");
    expect(fuseDigit("H", true, sure(4))).toBe("H");
    expect(fuseDigit("S0", true, sure(5))).toBe("S0");
    expect(fuseDigit("1", true, sure(NOT_A_DIGIT))).toBe("1");
    expect(fuseDigit("u", false, sure(NOT_A_DIGIT))).toBe("u");
  });

  it("keeps RapidOCR's read when the classifier is less than DIGIT_SURE", () => {
    const unsure = { digit: 4, probability: DIGIT_SURE - 0.01 };
    expect(fuseDigit("u", false, unsure)).toBe("u");
    expect(fuseDigit("9", true, unsure)).toBe("9");
  });
});

/** A digit in the Hershey script face, black on white, as a mark's crop. */
function scriptDigit(
  cv: Awaited<ReturnType<typeof testOpenCv>>,
  digit: number,
  scale: number,
  thickness: number
): ColorImage {
  return withMatScope((scope) => {
    const size = Math.round(40 * scale);
    const mat = scope.keep(
      new cv.Mat(size, size, cv.CV_8UC3, new cv.Scalar(255, 255, 255))
    );
    cv.putText(
      mat,
      String(digit),
      new cv.Point(Math.round(size * 0.2), Math.round(size * 0.8)),
      cv.FONT_HERSHEY_SCRIPT_SIMPLEX,
      scale,
      new cv.Scalar(0, 0, 0),
      thickness
    );
    const crop = createColor(size, size);
    crop.data.set(mat.data as Uint8Array);
    return crop;
  });
}

describeWithModels("the tab digit classifier", () => {
  it("reads a pen-like hand it was not trained on", async () => {
    const store = await storeOn(CPU);
    try {
      const session = await store.open("tabDigits");
      const cv = await testOpenCv();
      const crops: ColorImage[] = [];
      const truth: number[] = [];
      for (const [scale, thickness] of [
        [0.8, 1],
        [1, 2],
        [1.4, 2],
      ] as const) {
        for (let digit = 0; digit < 10; digit += 1) {
          crops.push(scriptDigit(cv, digit, scale, thickness));
          truth.push(digit);
        }
      }
      const guesses = await classifyDigits(session, crops);
      const right = guesses.filter((g, k) => g.digit === truth[k]).length;
      process.stdout.write(
        `script digits: ${guesses.map((g) => (g.digit === NOT_A_DIGIT ? "X" : g.digit)).join("")}\n`
      );
      // 29 of 30 when trained (2026-10-31): the smallest 1 reads as not a digit.
      expect(right).toBeGreaterThanOrEqual(28);
    } finally {
      await store.close();
    }
  });
});

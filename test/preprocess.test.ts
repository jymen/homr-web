/**
 * autocrop.py and color_adjust.py against the golden, and the two measured
 * facts behind the gray-only shortcut preprocess.ts declines to take.
 *
 * Every stage is fed the Python output of the stage before it, never this
 * port's, and every claim is a differing-byte count against zero rather than a
 * tolerance: a CLAHE with the wrong tile grid, or a crop one column short, is
 * within a byte of the right answer nearly everywhere.
 */

import process from "node:process";
import { describe, expect, it } from "vitest";
import {
  type ColorImage,
  createColor,
  type Plane,
} from "../src/image/plane.js";
import {
  applyClahe,
  autocrop,
  findPaperRect,
  type PaperRect,
  preprocessPage,
} from "../src/segmentation/preprocess.js";
import {
  fixtureImageOf,
  goldenPageOf,
  listGoldenFixtures,
} from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";

const PAGE_TEST_TIMEOUT = 60_000;

function differingBytes(got: Uint8Array, want: Uint8Array): number {
  let differing = 0;
  for (let i = 0; i < want.length; i += 1) {
    if (got[i] !== want[i]) {
      differing += 1;
    }
  }
  return differing;
}

/** R === G === B at every pixel: how many pixels break it. */
function chromaticPixels(image: ColorImage): number {
  let chromatic = 0;
  for (let i = 0; i < image.width * image.height; i += 1) {
    const b = image.data[i * 3];
    if (b !== image.data[i * 3 + 1] || b !== image.data[i * 3 + 2]) {
      chromatic += 1;
    }
  }
  return chromatic;
}

const sizeOf = (plane: Plane): readonly number[] => [plane.width, plane.height];

// A photograph of a sheet of paper: 400 x 400, a black background, a white
// sheet, and a dark mark well inside the sheet so the cropped bytes are not
// uniform and a crop at the wrong offset cannot pass. RETR_EXTERNAL ignores the
// mark, so it does not move the rect.
const SYNTHETIC_SIZE = 400;
const SYNTHETIC_QUARTER = SYNTHETIC_SIZE * 0.25;
const MARK = { x1: 300, x2: 320, y1: 200, y2: 210 } as const;

function syntheticPhotograph(
  sheetLeft: number,
  sheetRight: number
): ColorImage {
  const page = createColor(SYNTHETIC_SIZE, SYNTHETIC_SIZE, 0);
  for (let y = sheetLeft; y < sheetRight; y += 1) {
    const from = (y * page.width + sheetLeft) * page.channels;
    page.data.fill(255, from, from + (sheetRight - sheetLeft) * page.channels);
  }
  for (let y = MARK.y1; y < MARK.y2; y += 1) {
    for (let x = MARK.x1; x < MARK.x2; x += 1) {
      const at = (y * page.width + x) * page.channels;
      page.data[at] = 10;
      page.data[at + 1] = 20;
      page.data[at + 2] = 30;
    }
  }
  return page;
}

/**
 * `img[y : y + h, x : x + w]`, per sample rather than per row, so it shares no
 * arithmetic with the implementation's row copy.
 */
function numpySlice(image: ColorImage, rect: PaperRect): Uint8Array {
  const out = new Uint8Array(rect.width * rect.height * image.channels);
  for (let y = 0; y < rect.height; y += 1) {
    for (let x = 0; x < rect.width; x += 1) {
      for (let c = 0; c < image.channels; c += 1) {
        out[(y * rect.width + x) * image.channels + c] =
          image.data[
            ((rect.y + y) * image.width + rect.x + x) * image.channels + c
          ] ?? 0;
      }
    }
  }
  return out;
}

/**
 * Both rects were measured from opencv.js and both are the erosion of a sheet
 * starting at 105: the 9x9 kernel takes four columns off each side it can, so
 * both start at 109 and the second ends at 391, four short of its sheet. The
 * first ends at 400 instead, because erode's +inf border value takes nothing off
 * an edge the paper runs to — which is what makes the crop's last column real.
 *
 * The sheet also has to cover more than half the photograph, or the background
 * wins the channel-0 histogram, the threshold goes to -30, and every pixel
 * passes it.
 */
describe("findPaperRect", () => {
  it("finds paper that runs off the edge of the photograph", async () => {
    const cv = await testOpenCv();
    expect(findPaperRect(syntheticPhotograph(105, SYNTHETIC_SIZE), cv)).toEqual(
      {
        height: 291,
        isFullPageView: false,
        width: 291,
        x: 109,
        y: 109,
      }
    );
  });

  it("finds paper with a margin all round", async () => {
    const cv = await testOpenCv();
    expect(findPaperRect(syntheticPhotograph(105, 395), cv)).toEqual({
      height: 282,
      isFullPageView: false,
      width: 282,
      x: 109,
      y: 109,
    });
  });

  /**
   * A page of pure blue: the histogram of channel 0 is all 255, so the threshold
   * is 225, while BGR2GRAY of (255, 0, 0) is 29 and nothing passes it. That is
   * the only way homr's `big_contour is None` is reached, since any achromatic
   * page has its dominant value 30 above the threshold by construction.
   */
  it("finds nothing on a page whose dominant blue outruns its gray", async () => {
    const cv = await testOpenCv();
    const blue = createColor(32, 32, 0);
    for (let i = 0; i < blue.width * blue.height; i += 1) {
      blue.data[i * 3] = 255;
    }
    expect(findPaperRect(blue, cv)).toBeNull();
    expect(autocrop(blue, cv)).toBe(blue);
  });
});

describe("autocrop", () => {
  it("cuts the rect it found, last column and all", async () => {
    const cv = await testOpenCv();
    const page = syntheticPhotograph(105, SYNTHETIC_SIZE);
    const rect = findPaperRect(page, cv);
    expect(rect).not.toBeNull();
    if (rect === null) {
      return;
    }
    const cropped = autocrop(page, cv);
    expect(cropped).not.toBe(page);
    expect(sizeOf(cropped)).toEqual([rect.width, rect.height]);
    // The trap this crop exists to avoid: the rect reaches the page's last
    // column, which a bound clamped to width - 1 would drop.
    expect(rect.x + rect.width).toBe(SYNTHETIC_SIZE);
    expect(differingBytes(cropped.data, numpySlice(page, rect))).toBe(0);
  });

  it("cuts a rect that has a margin all round", async () => {
    const cv = await testOpenCv();
    const page = syntheticPhotograph(105, 395);
    const rect = findPaperRect(page, cv);
    if (rect === null) {
      throw new Error("the synthetic photograph has paper on it");
    }
    const cropped = autocrop(page, cv);
    expect(sizeOf(cropped)).toEqual([282, 282]);
    expect(differingBytes(cropped.data, numpySlice(page, rect))).toBe(0);
  });

  it("hands back the page it was given when the rect starts near an edge", async () => {
    const cv = await testOpenCv();
    // The sheet starts inside the quarter margin, so the guard fires however
    // large the paper is.
    const page = syntheticPhotograph(
      Math.trunc(SYNTHETIC_QUARTER) - 20,
      SYNTHETIC_SIZE
    );
    expect(findPaperRect(page, cv)?.isFullPageView).toBe(true);
    expect(autocrop(page, cv)).toBe(page);
  });
});

/**
 * The load-bearing half of the gray-only shortcut written down in
 * preprocess.ts: cv2's BGR2GRAY weights are fixed-point and sum to 1 << 14, so
 * `(v * 16384 + 8192) >> 14` should be `v`. Asserted against opencv.js rather
 * than against that arithmetic, because the shortcut rests on the build's
 * behaviour and not on the derivation.
 */
describe("BGR2GRAY on achromatic pixels", () => {
  it("returns the channel value unchanged for all 256 values", async () => {
    const cv = await testOpenCv();
    const ramp = createColor(256, 1, 0);
    for (let value = 0; value < 256; value += 1) {
      ramp.data[value * 3] = value;
      ramp.data[value * 3 + 1] = value;
      ramp.data[value * 3 + 2] = value;
    }
    // applyClahe is the port's only BGR2GRAY, and CLAHE would rewrite what is
    // being measured, so this one calls cvtColor itself.
    const src = cv.matFromArray(1, 256, cv.CV_8UC3, Array.from(ramp.data));
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_BGR2GRAY);
    const got = Array.from(gray.data);
    src.delete();
    gray.delete();
    expect(got).toEqual(Array.from({ length: 256 }, (_, value) => value));
  });
});

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);
  const { stages } = page.meta();
  if (
    !["autocropped.png", "preprocessed.png", "resized.png"].every((stage) =>
      stages.includes(stage)
    )
  ) {
    continue;
  }

  describe(`preprocess ${fixture.name}`, () => {
    /**
     * On this page autocrop is a no-op: the largest contour's bounding rect is
     * the whole sheet, so the guard returns the input and an autocrop that did
     * nothing at all would pass the byte comparison. The rect and the guard are
     * asserted for that reason — the green tick below is not evidence that
     * cropping works, which only the synthetic pages above cover.
     */
    it(
      "reproduces Python's autocropped page byte for byte",
      async () => {
        const cv = await testOpenCv();
        const image = fixtureImageOf(fixture);
        const want = page.autocropped();
        const started = performance.now();
        const rect = findPaperRect(image, cv);
        const got = autocrop(image, cv);
        const ms = performance.now() - started;
        const differing = differingBytes(got.data, want.data);
        process.stdout.write(
          `autocrop ${fixture.name}: ${image.width}x${image.height} in ${ms.toFixed(0)} ms, ` +
            `rect ${JSON.stringify(rect)}, ${differing} differing bytes of ${want.data.length}\n`
        );
        expect(sizeOf(got)).toEqual(sizeOf(want));
        expect(differing).toBe(0);
        if (rect?.isFullPageView === true) {
          expect([rect.x, rect.y, rect.width, rect.height]).toEqual([
            0,
            0,
            image.width,
            image.height,
          ]);
          expect(got).toBe(image);
        }
      },
      PAGE_TEST_TIMEOUT
    );

    it(
      "reproduces Python's preprocessed page byte for byte",
      async () => {
        const cv = await testOpenCv();
        const want = page.preprocessed();
        const started = performance.now();
        const got = applyClahe(page.resized(), cv);
        const ms = performance.now() - started;
        const differing = differingBytes(got.data, want.data);
        process.stdout.write(
          `clahe ${fixture.name}: ${got.width}x${got.height} in ${ms.toFixed(0)} ms, ` +
            `${differing} differing bytes of ${want.data.length}\n`
        );
        expect([got.kind, ...sizeOf(got)]).toEqual(["gray", ...sizeOf(want)]);
        expect(differing).toBe(0);
      },
      PAGE_TEST_TIMEOUT
    );

    it(
      "runs the three stages end to end",
      async () => {
        const cv = await testOpenCv();
        const started = performance.now();
        const got = await preprocessPage(fixtureImageOf(fixture), cv);
        const ms = performance.now() - started;
        const resized = differingBytes(got.resized.data, page.resized().data);
        const preprocessed = differingBytes(
          got.preprocessed.data,
          page.preprocessed().data
        );
        process.stdout.write(
          `preprocessPage ${fixture.name}: ${ms.toFixed(0)} ms, ` +
            `${resized} differing bytes of ${page.resized().data.length} resized, ` +
            `${preprocessed} differing bytes of ${page.preprocessed().data.length} preprocessed\n`
        );
        expect(sizeOf(got.resized)).toEqual(sizeOf(page.resized()));
        expect(resized).toBe(0);
        expect(preprocessed).toBe(0);
      },
      PAGE_TEST_TIMEOUT
    );

    /**
     * The premise of the gray-only shortcut, on the record as measured: every
     * page this port has ever been fed is achromatic, so resizing one band
     * would be correct on it. One chromatic page makes the shortcut wrong, and
     * nothing in homr guarantees a scan is gray.
     */
    it(
      "is achromatic at every stage",
      () => {
        for (const image of [
          fixtureImageOf(fixture),
          page.autocropped(),
          page.resized(),
        ]) {
          expect(chromaticPixels(image)).toBe(0);
        }
      },
      PAGE_TEST_TIMEOUT
    );
  });
}

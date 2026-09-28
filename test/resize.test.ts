/**
 * resize.py's golden: `resizeToTargetWidth(page.autocropped())` must equal
 * `page.resized()` byte for byte, because that is what the segnet was fed. The
 * assertion is a differing-byte count against zero rather than a tolerance, on
 * purpose: the failure this unit exists to catch is a resample that is bicubic
 * but not PIL's, and every such resample is within a byte or two of the right
 * answer nearly everywhere.
 *
 * The input is the Python stage output, never this port's own autocrop.
 */

import process from "node:process";
import { describe, expect, it } from "vitest";
import { createGray } from "../src/image/plane.js";
import {
  RESIZE_TARGET_WIDTH,
  resizeToTargetWidth,
  targetImageSize,
} from "../src/segmentation/resize.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";

function differingBytes(got: Uint8Array, want: Uint8Array): number {
  let differing = 0;
  for (let i = 0; i < want.length; i += 1) {
    if (got[i] !== want[i]) {
      differing += 1;
    }
  }
  return differing;
}

describe("targetImageSize", () => {
  it("returns a page that is already 1920 wide unchanged", () => {
    expect(targetImageSize(RESIZE_TARGET_WIDTH, 4321)).toEqual({
      height: 4321,
      width: RESIZE_TARGET_WIDTH,
    });
  });

  it("holds the Kesh page's size", () => {
    expect(targetImageSize(2481, 3509)).toEqual({
      height: 2716,
      width: RESIZE_TARGET_WIDTH,
    });
  });

  /**
   * A width of 3840 or 7680 makes the ratio exactly 0.5 or 0.25, so an odd (or
   * 2-mod-4) height lands the scaled height exactly on .5 with no float slop to
   * argue about. Python's round() then goes to the even neighbour where
   * Math.round goes up, and the assertion pins both answers so a reader can see
   * they disagree rather than take it on trust.
   */
  it("rounds a height on .5 to even, as Python's round does", () => {
    for (const [width, height, expected] of [
      [3840, 5, 2],
      [3840, 13, 6],
      [7680, 10, 2],
    ] as const) {
      expect(targetImageSize(width, height).height).toBe(expected);
      expect(Math.round((height * RESIZE_TARGET_WIDTH) / width)).toBe(
        expected + 1
      );
    }
    // An odd neighbour still rounds up; half to even is not "always down".
    expect(targetImageSize(3840, 7).height).toBe(4);
  });
});

describe("resizeToTargetWidth", () => {
  it("hands back the plane it was given when it is already 1920 wide", () => {
    const image = createGray(RESIZE_TARGET_WIDTH, 7);
    expect(resizeToTargetWidth(image)).toBe(image);
  });

  /**
   * A 2:1 downscale of a step edge, which is the smallest case that shows the
   * whole fixed-point path move. The numbers were measured from this
   * implementation after it matched the page golden byte for byte, and they are
   * readable as PIL's arithmetic rather than as a dump: 17 and 238 sum to 255,
   * so the kernel's mass is conserved across the edge; the columns either side
   * stay exactly 0 and 255 because `_clip8` holds the bicubic's negative lobes
   * in range (at x = 958 the only tap that lands on a white pixel has weight
   * -0.0234375, so the accumulator is genuinely negative before the clamp, and
   * at x = 961 the white taps carry more than the full unit); and both output
   * rows are identical, which is the vertical kernel summing to one unit.
   */
  it("resamples a step edge the way PIL's fixed point does", () => {
    const image = createGray(3840, 4);
    for (let y = 0; y < image.height; y += 1) {
      image.data.fill(255, y * image.width + 1920, (y + 1) * image.width);
    }
    const out = resizeToTargetWidth(image);
    expect([out.width, out.height]).toEqual([RESIZE_TARGET_WIDTH, 2]);
    const row = (y: number): Uint8Array =>
      out.data.subarray(y * out.width, (y + 1) * out.width);
    expect(Array.from(row(0).subarray(956, 964))).toEqual([
      0, 0, 0, 17, 238, 255, 255, 255,
    ]);
    expect(
      row(0)
        .subarray(0, 959)
        .every((v) => v === 0)
    ).toBe(true);
    expect(
      row(0)
        .subarray(961)
        .every((v) => v === 255)
    ).toBe(true);
    expect(Array.from(row(1))).toEqual(Array.from(row(0)));
  });
});

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);
  const { stages } = page.meta();
  if (!(stages.includes("autocropped.png") && stages.includes("resized.png"))) {
    continue;
  }

  describe(`resize ${fixture.name}`, () => {
    it("reproduces Python's resized page byte for byte", () => {
      const autocropped = page.autocropped();
      const want = page.resized();
      const started = performance.now();
      const got = resizeToTargetWidth(autocropped);
      const ms = performance.now() - started;
      expect([got.kind, got.width, got.height]).toEqual([
        "bgr",
        want.width,
        want.height,
      ]);
      const differing = differingBytes(got.data, want.data);
      // Both numbers on the record: the byte count is the claim this unit
      // makes, and the time is the price the page pays for making it.
      process.stdout.write(
        `resize ${fixture.name}: ${autocropped.width}x${autocropped.height} -> ` +
          `${got.width}x${got.height} in ${ms.toFixed(0)} ms, ` +
          `${differing} differing bytes of ${want.data.length}\n`
      );
      expect(differing).toBe(0);
    });
  });
}

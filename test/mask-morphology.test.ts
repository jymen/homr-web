/**
 * The two masks detection reshapes before it reads them, against what Python
 * wrote for each fixture. Both are exact: they are morphology on 0/1 bytes.
 */

import { describe, expect, it } from "vitest";
import {
  makeLinesStronger,
  prepareBraceDotImage,
} from "../src/cv/mask-morphology.js";
import { createMask, type Mask } from "../src/image/plane.js";
import { goldenPageOf, listGoldenFixtures } from "./support/golden.js";
import { testOpenCv } from "./support/opencv.js";

const differing = (a: Mask, b: Mask): number => {
  if (a.width !== b.width || a.height !== b.height) {
    return Number.POSITIVE_INFINITY;
  }
  let count = 0;
  for (let i = 0; i < a.data.length; i += 1) {
    count += a.data[i] === b.data[i] ? 0 : 1;
  }
  return count;
};

const setPixels = (mask: Mask): string[] =>
  Array.from(mask.data.entries())
    .filter(([, value]) => value === 1)
    .map(([i]) => `${i % mask.width},${Math.floor(i / mask.width)}`);

describe("makeLinesStronger", () => {
  it("sets the pixel below every set pixel and no other", async () => {
    const cv = await testOpenCv();
    const mask = createMask(5, 5);
    mask.data[2 * 5 + 1] = 1;
    mask.data[4 * 5 + 3] = 1;
    expect(setPixels(makeLinesStronger(cv, mask))).toEqual([
      "1,2",
      "1,3",
      "3,4",
    ]);
  });
});

for (const fixture of listGoldenFixtures()) {
  const page = goldenPageOf(fixture);

  describe(`mask morphology ${fixture.name}`, () => {
    it("makeLinesStronger turns mask-denoised-staff.png into mask-filtered-staff.png", async () => {
      const cv = await testOpenCv();
      const strong = makeLinesStronger(cv, page.denoisedStaffMask());
      expect(differing(strong, page.mask("staff", true))).toBe(0);
    });

    it("prepareBraceDotImage turns the filtered symbols and staff masks into mask-brace_dot.png", async () => {
      const cv = await testOpenCv();
      const braceDot = prepareBraceDotImage(
        cv,
        page.mask("symbols", true),
        page.mask("staff", true)
      );
      expect(differing(braceDot, page.braceDotMask())).toBe(0);
    });
  });
}

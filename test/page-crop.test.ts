/**
 * The two frames a reading mixes: homr's staves and note columns are
 * normalised by the autocropped page, the tab readers' marks by the page
 * given. A photograph whose paper autocrop cuts out is where they part.
 */

import { describe, expect, it } from "vitest";
import {
  columnInPage,
  pageCrop,
  staffInPage,
  WHOLE_PAGE,
} from "../src/pipeline/page-crop.js";
import { type DetectedButtonTab, withStaves } from "../src/tab/button.js";

const page = { height: 2000, width: 1000 };
// Paper from x 100 to 900 and y 200 to 1800: the middle 80 % each way.
const paper = { height: 1600, isFullPageView: false, width: 800, x: 100, y: 200 };

describe("pageCrop", () => {
  it("is the whole page when autocrop kept it", () => {
    expect(pageCrop(null, page)).toEqual(WHOLE_PAGE);
  });

  it("is the paper rect as a fraction of the page", () => {
    expect(pageCrop(paper, page)).toEqual({ x0: 0.1, x1: 0.9, y0: 0.1, y1: 0.9 });
  });
});

describe("columnInPage and staffInPage", () => {
  const crop = pageCrop(paper, page);

  it("leave a reading of the whole page as it is", () => {
    const staff = { cx: 0.5, cy: 0.25, h: 0.04, index: 3, w: 0.8 };
    expect(staffInPage(staff, WHOLE_PAGE)).toEqual(staff);
    expect(columnInPage(0.37, WHOLE_PAGE)).toBe(0.37);
  });

  it("place the cropped page's edges on the paper's", () => {
    expect(columnInPage(0, crop)).toBeCloseTo(0.1, 12);
    expect(columnInPage(1, crop)).toBeCloseTo(0.9, 12);
    const staff = staffInPage({ cx: 0.5, cy: 0, h: 0.05, index: 0, w: 1 }, crop);
    expect(staff.cx).toBeCloseTo(0.5, 12);
    expect(staff.cy).toBeCloseTo(0.1, 12);
    expect(staff.w).toBeCloseTo(0.8, 12);
    expect(staff.h).toBeCloseTo(0.04, 12);
  });

  /**
   * The bug the crop fixes, in the lower half of a cropped photo where the
   * cropped frame pushes everything down: the staff is at 0.66 of the page
   * given (bottom 0.672, above the rule at 0.7), but its cropped numbers,
   * cy 0.7 and h 0.03, put its bottom at 0.715, below the rule.
   */
  it("let withStaves tie a band to the staff above it on a cropped photo", () => {
    const band: DetectedButtonTab = {
      band: { bandBottom: 0.71, bandTop: 0.69, rule: 0.7, x0: 0.15, x1: 0.85 },
      down: 10,
      letterLine: [],
      letters: {
        p: { h: 10, w: 8, x: 0, y: 0 },
        t: { h: 10, w: 8, x: 0, y: 20 },
      },
      marginX: 0,
      rule: { points: [{ x: 0, y: 0 }] },
      spacing: 5,
      up: 10,
      x0: 150,
      x1: 850,
    };
    const staff = { cx: 0.5, cy: 0.7, h: 0.03, index: 0, w: 0.9 };
    expect(withStaves([band], [staff])[0]?.band.staff).toBeUndefined();
    expect(
      withStaves([band], [staffInPage(staff, crop)])[0]?.band.staff
    ).toBe(0);
  });
});

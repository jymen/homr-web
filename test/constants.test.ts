/** constants.py's values, as the installed homr 0.7.0 holds them. */

import { describe, expect, it } from "vitest";
import {
  barLineMaxWidth,
  barLineMinHeight,
  GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR,
  GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR,
  IMAGE_NOISE_LIMIT,
  isShortConnectedLine,
  isShortLine,
  LINES_PER_STAFF,
  MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL,
  MAX_LEDGER_LINES,
  MIN_CONNECTIONS_TO_FORM_COMBINED_STAFF,
  maxLineGapSize,
  maxWidthForBraceRough,
  minHeightForBrace,
  minHeightForBraceRough,
  NOTEHEAD_SIZE_RATIO,
  STAFF_LINE_SEGMENT_X_TOLERANCE,
  STAFF_POSITION_TOLERANCE,
  toleranceForStaffLineDetection,
  toleranceForTouchingClefs,
} from "../src/model/constants.js";

describe("constants.py", () => {
  it("holds the module-level values", () => {
    expect({
      GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR,
      GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR,
      IMAGE_NOISE_LIMIT,
      LINES_PER_STAFF,
      MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL,
      MAX_LEDGER_LINES,
      MIN_CONNECTIONS_TO_FORM_COMBINED_STAFF,
      NOTEHEAD_SIZE_RATIO,
      STAFF_LINE_SEGMENT_X_TOLERANCE,
      STAFF_POSITION_TOLERANCE,
    }).toEqual({
      GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR: 5,
      GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR: 0.5,
      IMAGE_NOISE_LIMIT: 50,
      LINES_PER_STAFF: 5,
      MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL: 10,
      MAX_LEDGER_LINES: 4,
      MIN_CONNECTIONS_TO_FORM_COMBINED_STAFF: 1,
      NOTEHEAD_SIZE_RATIO: 1.285_714,
      STAFF_LINE_SEGMENT_X_TOLERANCE: 10,
      STAFF_POSITION_TOLERANCE: 50,
    });
  });

  it.each([
    ["tolerance_for_staff_line_detection", toleranceForStaffLineDetection, 6],
    ["max_line_gap_size", maxLineGapSize, 90],
    ["is_short_line", isShortLine, 3.6],
    ["is_short_connected_line", isShortConnectedLine, 36],
    ["min_height_for_brace_rough", minHeightForBraceRough, 36],
    ["max_width_for_brace_rough", maxWidthForBraceRough, 54],
    ["min_height_for_brace", minHeightForBrace, 72],
    ["bar_line_max_width", barLineMaxWidth, 36],
    ["bar_line_min_height", barLineMinHeight, 54],
  ] as const)("%s(18) is Python's", (_name, unitFunction, expected) => {
    expect(unitFunction(18)).toBe(expected);
  });

  it.each([
    [18, 36],
    [18.2, 36],
    [18.25, 36],
    [18.75, 38],
    [17.75, 36],
    [0.25, 0],
    [0.75, 2],
  ])(
    "tolerance_for_touching_clefs(%s) is %s: an int, halves to even",
    (unit, expected) => {
      expect(toleranceForTouchingClefs(unit)).toBe(expected);
    }
  );
});

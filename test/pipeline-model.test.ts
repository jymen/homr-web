import { describe, expect, it } from "vitest";
import { DETECTION_FAILURES, DetectionError } from "../src/model/pipeline.js";

describe("DetectionError", () => {
  it("carries homr's own message for the two exceptions homr raises", () => {
    expect(new DetectionError("no-noteheads").message).toBe(
      "No noteheads found"
    );
    expect(new DetectionError("no-staffs").message).toBe("No staffs found");
  });

  it("has exactly four codes, and each is an Error a caller can tell by code", () => {
    expect(Object.keys(DETECTION_FAILURES).sort()).toEqual([
      "no-noteheads",
      "no-staffs",
      "staff-without-points",
      "zone-without-lines",
    ]);
    const error = new DetectionError("zone-without-lines");
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe("DetectionError");
    expect(error.code).toBe("zone-without-lines");
  });
});

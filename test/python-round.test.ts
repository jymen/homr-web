import { describe, expect, it } from "vitest";
import { mean, npRound, pyRound } from "../src/image/numeric.js";
import { vectorSet } from "./support/vectors.js";

const { cases } = vectorSet("python-round");
const scalars = cases.filter(
  (c) => typeof c.x === "number"
) as unknown as readonly {
  readonly npRound5: number;
  readonly round3: number;
  readonly round4: number;
  readonly round5: number;
  readonly x: number;
}[];
const means = cases.filter((c) =>
  Array.isArray(c.conf)
) as unknown as readonly {
  readonly conf: readonly number[];
  readonly meanRound5: number;
}[];

describe("Python round(x, n) and numpy's round(5)", () => {
  it(`agree with Python on ${scalars.length} values, the exact ties among them`, () => {
    const wrong = scalars.flatMap((c) =>
      [
        [pyRound(c.x, 3), c.round3, "round3"],
        [pyRound(c.x, 4), c.round4, "round4"],
        [pyRound(c.x, 5), c.round5, "round5"],
        [npRound(c.x, 5), c.npRound5, "npRound5"],
      ]
        .filter(([got, want]) => got !== want)
        .map(([got, want, what]) => `${what}(${c.x}) = ${got}, Python ${want}`)
    );
    expect(wrong).toEqual([]);
  });

  it(`gives CTCLabelDecode's np.mean(conf).round(5) on ${means.length} lists`, () => {
    for (const c of means) {
      expect(npRound(mean(c.conf), 5)).toBe(c.meanRound5);
    }
  });
});

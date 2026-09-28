/**
 * The merge, with no opencv.js anywhere: that is the reason it is its own
 * module. The case that matters is the third one, where the union-find roots
 * come out non-ascending -- a plain object would still produce the same groups
 * and put them in the wrong order.
 */

import { describe, expect, it } from "vitest";
import { mergeOverlappingGroups } from "../src/geometry/box-merge.js";

const groupsOf = (
  count: number,
  edges: ReadonlyArray<readonly [number, number]>
): number[][] => {
  const overlaps = (a: number, b: number) =>
    edges.some(([i, j]) => (i === a && j === b) || (i === b && j === a));
  return mergeOverlappingGroups(
    Array.from({ length: count }, (_, i) => i),
    overlaps
  );
};

describe("mergeOverlappingGroups", () => {
  it("leaves nothing to group", () => {
    expect(groupsOf(0, [])).toEqual([]);
  });

  it("keeps items that touch nothing as singletons, in source order", () => {
    expect(groupsOf(3, [])).toEqual([[0], [1], [2]]);
  });

  /**
   * Checked against homr's own _merge_groups_optimized on 2026-09-28 with
   * _do_groups_overlap stubbed to these three edges: it answers
   * [[0, 2, 3, 4], [1]].
   *
   * The roots are 2 then 1, because (0, 4) and (2, 3) each make a rank-1 root
   * and (2, 4) then ties them, so 2 absorbs 0. A Record keyed by root would
   * iterate 1 before 2 and hand back [[1], [0, 2, 3, 4]].
   */
  it("follows insertion order when the roots are non-ascending", () => {
    expect(
      groupsOf(5, [
        [0, 4],
        [2, 3],
        [2, 4],
      ])
    ).toEqual([[0, 2, 3, 4], [1]]);
  });

  it("merges transitively through a chain", () => {
    expect(
      groupsOf(4, [
        [0, 1],
        [1, 2],
      ])
    ).toEqual([[0, 1, 2], [3]]);
  });

  it("orders members within a group by source index", () => {
    expect(
      groupsOf(4, [
        [3, 0],
        [1, 2],
      ])
    ).toEqual([
      [0, 3],
      [1, 2],
    ]);
  });
});

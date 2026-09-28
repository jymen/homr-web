/**
 * merge_overlaying_bounding_boxes and _merge_groups_optimized, collapsed into
 * one function, and the one fact in this phase that no tolerance can rescue:
 * the output order.
 *
 * Python wraps every box as a singleton group first, then unions groups where
 * any pair of members overlaps. The groups list is never mutated -- only the
 * union-find is -- so _do_groups_overlap always compares the original
 * singletons and reduces to one predicate call per pair. That is why this takes
 * items and a predicate rather than groups, and why it knows nothing about
 * boxes, cv or geometry: the merge is where the ordering contract lives, and
 * opencv.js under vitest is the most fragile part of this repo's test setup.
 *
 * The order is a consequence of two things together: the union-by-rank
 * tie-break, where equal ranks attach y under x and bump x's rank, and Python
 * dict insertion order. It sets the order a merged group's contours are
 * concatenated in, which sets what the refit minAreaRect sees. Reproducing the
 * grouping is not enough.
 */

/**
 * homr's UnionFind. Path compression to the root, as Python's recursive find
 * does, written iteratively so a long chain cannot overflow the stack; the
 * parent array it leaves behind is the same either way.
 */
class UnionFind {
  private readonly parent: number[];
  private readonly rank: number[];

  constructor(size: number) {
    this.parent = Array.from({ length: size }, (_, i) => i);
    this.rank = Array.from({ length: size }, () => 0);
  }

  find(x: number): number {
    let root = x;
    while ((this.parent[root] ?? root) !== root) {
      root = this.parent[root] ?? root;
    }
    let walk = x;
    while (walk !== root) {
      const next = this.parent[walk] ?? root;
      this.parent[walk] = root;
      walk = next;
    }
    return root;
  }

  union(x: number, y: number): void {
    const rootX = this.find(x);
    const rootY = this.find(y);
    if (rootX === rootY) {
      return;
    }
    const rankX = this.rank[rootX] ?? 0;
    const rankY = this.rank[rootY] ?? 0;
    if (rankX > rankY) {
      this.parent[rootY] = rootX;
    } else if (rankX < rankY) {
      this.parent[rootX] = rootY;
    } else {
      this.parent[rootY] = rootX;
      this.rank[rootX] = rankX + 1;
    }
  }
}

/**
 * Groups items that overlap, transitively. Members within a group are in
 * ascending source order and the groups are in first-member order.
 *
 * `Map<number, T[]>`, never a plain object: JavaScript iterates integer-like
 * keys in ascending numeric order where a Python dict iterates in insertion
 * order, and union-by-rank does not produce ascending roots. That substitution
 * compiles, type-checks, produces the same groups and silently reorders the
 * output.
 */
export function mergeOverlappingGroups<T>(
  items: readonly T[],
  overlaps: (a: T, b: T) => boolean
): T[][] {
  const sets = new UnionFind(items.length);
  for (const [i, left] of items.entries()) {
    for (const [j, right] of items.entries()) {
      if (j > i && overlaps(left, right)) {
        sets.union(i, j);
      }
    }
  }
  const groups = new Map<number, T[]>();
  for (const [i, item] of items.entries()) {
    const root = sets.find(i);
    const group = groups.get(root);
    if (group === undefined) {
      groups.set(root, [item]);
    } else {
      group.push(item);
    }
  }
  return [...groups.values()];
}

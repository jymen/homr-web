/**
 * `np.argsort` with numpy's default kind, for float64, with numpy's tie order.
 *
 * find_peaks orders candidate peaks with `np.argsort(heights)[::-1]`, the
 * heights are functions of integer row counts so ties are routine, and the
 * distance filter keeps whichever of two tied neighbours comes first. numpy's
 * default is its introsort, which is not stable: from 17 elements up, "stable
 * ascending then reversed" disagrees with numpy on tied input.
 *
 * So this is numpy's aquicksort, literally: median of three, the pivot parked
 * at right - 1, strict `<` on both scans, the larger side pushed and the
 * smaller continued, insertion sort up to 16 elements, heapsort for a range
 * popped after its depth budget of 2 * floor(log2(n)) is spent.
 *
 * Known limit: numpy ships SIMD argsorts for x86 whose tie order was not
 * compared. The expected orders in test/golden/vectors/argsort.json are data,
 * so the test is the same on any machine; what an x86 oracle could change is
 * the golden pages, and meta.json records the machine.
 */

const SMALL_RANGE = 16;

/** A range still to sort: left, right (inclusive) and the depth budget left. */
type Pending = [number, number, number];

class Order {
  readonly slots: Int32Array;
  private readonly values: ArrayLike<number>;

  constructor(values: ArrayLike<number>) {
    this.values = values;
    this.slots = Int32Array.from({ length: values.length }, (_, i) => i);
  }

  item(slot: number): number {
    return this.slots[slot] ?? 0;
  }

  valueOf(item: number): number {
    return this.values[item] ?? Number.NaN;
  }

  at(slot: number): number {
    return this.valueOf(this.item(slot));
  }

  swap(a: number, b: number): void {
    const held = this.item(a);
    this.slots[a] = this.item(b);
    this.slots[b] = held;
  }
}

/** Sorts left, middle and right by value and returns the pivot value, left at middle. */
function medianOfThree(
  order: Order,
  left: number,
  middle: number,
  right: number
): number {
  if (order.at(middle) < order.at(left)) {
    order.swap(middle, left);
  }
  if (order.at(right) < order.at(middle)) {
    order.swap(right, middle);
  }
  if (order.at(middle) < order.at(left)) {
    order.swap(middle, left);
  }
  return order.at(middle);
}

/** One partition of left..right; returns the slot the pivot ends in. */
function partition(order: Order, left: number, right: number): number {
  const middle = left + Math.floor((right - left) / 2);
  const pivot = medianOfThree(order, left, middle, right);
  let i = left;
  let j = right - 1;
  order.swap(middle, j);
  for (;;) {
    do {
      i += 1;
    } while (order.at(i) < pivot);
    do {
      j -= 1;
    } while (pivot < order.at(j));
    if (i >= j) {
      break;
    }
    order.swap(i, j);
  }
  order.swap(i, right - 1);
  return i;
}

function insertionSort(order: Order, left: number, right: number): void {
  for (let i = left + 1; i <= right; i += 1) {
    const held = order.item(i);
    const value = order.valueOf(held);
    let j = i;
    while (j > left && value < order.at(j - 1)) {
      order.slots[j] = order.item(j - 1);
      j -= 1;
    }
    order.slots[j] = held;
  }
}

/** numpy's aheapsort over `count` slots from `left`, with its one-based heap. */
function heapSort(order: Order, left: number, count: number): void {
  const base = left - 1;
  const sift = (held: number, from: number, size: number): void => {
    const value = order.valueOf(held);
    let parent = from;
    let child = from * 2;
    while (child <= size) {
      if (child < size && order.at(base + child) < order.at(base + child + 1)) {
        child += 1;
      }
      if (!(value < order.at(base + child))) {
        break;
      }
      order.slots[base + parent] = order.item(base + child);
      parent = child;
      child += child;
    }
    order.slots[base + parent] = held;
  };
  for (let start = Math.floor(count / 2); start > 0; start -= 1) {
    sift(order.item(base + start), start, count);
  }
  for (let size = count; size > 1; size -= 1) {
    const held = order.item(base + size);
    order.slots[base + size] = order.item(base + 1);
    sift(held, 1, size - 1);
  }
}

export function npArgsort(values: ArrayLike<number>): Int32Array {
  const n = values.length;
  const order = new Order(values);
  if (n < 2) {
    return order.slots;
  }
  const pending: Pending[] = [];
  let next: Pending | undefined = [0, n - 1, 2 * Math.floor(Math.log2(n))];
  while (next !== undefined) {
    let [left, right, depth] = next;
    if (depth < 0) {
      heapSort(order, left, right - left + 1);
    } else {
      while (right - left >= SMALL_RANGE) {
        const pivotSlot = partition(order, left, right);
        depth -= 1;
        if (pivotSlot - left < right - pivotSlot) {
          pending.push([pivotSlot + 1, right, depth]);
          right = pivotSlot - 1;
        } else {
          pending.push([left, pivotSlot - 1, depth]);
          left = pivotSlot + 1;
        }
      }
      insertionSort(order, left, right);
    }
    next = pending.pop();
  }
  return order.slots;
}

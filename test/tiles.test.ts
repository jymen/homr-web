import { describe, expect, it } from "vitest";
import type { ClassMap, GrayImage } from "../src/image/plane.js";
import {
  createClassMap,
  createGray,
  PlaneError,
  planeFromBytes,
} from "../src/image/plane.js";
import { float16FromFloat32, float32FromFloat16 } from "../src/models/dtype.js";
import {
  extractTile,
  mergeTileClasses,
  tileCoverage,
  tileGrid,
  writeTileInto,
} from "../src/segmentation/tiles.js";

/** The Kesh Jig fixture's preprocessed page, the size every grid figure below is measured on. */
const KESH = { height: 2716, width: 1920 } as const;

const grayPage = (
  width: number,
  height: number,
  value: (x: number, y: number) => number
): GrayImage => {
  const page = createGray(width, height);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      page.data[y * width + x] = value(x, y);
    }
  }
  return page;
};

const filledClassMap = (window: number, classIndex: number): ClassMap => {
  const tile = createClassMap(window, window);
  tile.data.fill(classIndex);
  return tile;
};

const histogramOf = (coverage: Uint8Array): Record<number, number> => {
  const counts: Record<number, number> = {};
  for (const weight of coverage) {
    counts[weight] = (counts[weight] ?? 0) + 1;
  }
  return counts;
};

const classAt = (map: ClassMap, x: number, y: number): number =>
  map.data[y * map.width + x] ?? -1;

const STEP_TOO_SMALL = /more than four times/;

describe("tileGrid (the two loops of inference_segnet.inference)", () => {
  it("covers the Kesh page in 54 tiles, row-major, the last row pulled back", () => {
    const grid = tileGrid(KESH.width, KESH.height);
    expect(grid).toHaveLength(54);
    expect([...new Set(grid.map((origin) => origin.y))]).toEqual([
      0, 320, 640, 960, 1280, 1600, 1920, 2240, 2396,
    ]);
    // 1920 is an exact multiple of 320, so no column ever pulls back.
    expect([...new Set(grid.map((origin) => origin.x))]).toEqual([
      0, 320, 640, 960, 1280, 1600,
    ]);
    expect(grid.slice(0, 7)).toEqual([
      { x: 0, y: 0 },
      { x: 320, y: 0 },
      { x: 640, y: 0 },
      { x: 960, y: 0 },
      { x: 1280, y: 0 },
      { x: 1600, y: 0 },
      { x: 0, y: 320 },
    ]);
    expect(grid.at(-1)).toEqual({ x: 1600, y: 2396 });
  });

  it("puts one negative origin on a page smaller than the window", () => {
    expect(tileGrid(100, 50)).toEqual([{ x: -220, y: -270 }]);
  });

  it("pulls the last column back only when the width is not a multiple of the step", () => {
    expect(tileGrid(960, 320).map((origin) => origin.x)).toEqual([0, 320, 640]);
    expect(histogramOf(tileCoverage(960, 320))).toEqual({ 1: 960 * 320 });
    expect(tileGrid(1000, 320).map((origin) => origin.x)).toEqual([
      0, 320, 640, 680,
    ]);
    // The fourth column pulls back by 280, and those 280 columns are covered twice.
    expect(histogramOf(tileCoverage(1000, 320))).toEqual({
      1: 230_400,
      2: 89_600,
    });
  });

  it("repeats an origin when the step is below the window, as min() clamps every late iteration to the same place", () => {
    expect(tileGrid(6, 4, 4, 3)).toEqual([
      { x: 0, y: 0 },
      { x: 2, y: 0 },
      { x: 0, y: 0 },
      { x: 2, y: 0 },
    ]);
  });

  it("refuses a tiling that is not a grid", () => {
    expect(() => tileGrid(10, 10, 4, 0)).toThrow(PlaneError);
    expect(() => tileGrid(10, 10, 0, 4)).toThrow(PlaneError);
    expect(() => tileGrid(10.5, 10)).toThrow(PlaneError);
    expect(() => tileGrid(-1, 10)).toThrow(PlaneError);
  });
});

describe("extractTile (extract_patch)", () => {
  // 1..200, so neither the padding value nor an empty byte can pass for page content.
  const pattern = (x: number, y: number) => 1 + ((x + 3 * y) % 200);

  it("pins a smaller-than-window page to the top-left corner and pads the rest with 255", () => {
    const page = grayPage(100, 50, pattern);
    const tile = extractTile(page, { x: -220, y: -270 });
    expect([tile.width, tile.height]).toEqual([320, 320]);
    let mismatches = 0;
    for (let y = 0; y < 50; y += 1) {
      for (let x = 0; x < 100; x += 1) {
        if (tile.data[y * 320 + x] !== page.data[y * 100 + x]) {
          mismatches += 1;
        }
      }
    }
    expect(mismatches).toBe(0);
    // The padding lands on the far side, not around the page: the negative
    // origin moved nothing.
    expect(tile.data[100]).toBe(255);
    expect(tile.data[50 * 320]).toBe(255);
    expect(tile.data[320 * 320 - 1]).toBe(255);
    expect(tile.data.filter((value) => value === 255)).toHaveLength(
      320 * 320 - 100 * 50
    );
  });

  it("copies a full window with no padding for a pulled-back column", () => {
    const page = grayPage(1000, 320, pattern);
    const tile = extractTile(page, { x: 680, y: 0 });
    expect(Array.from(tile.data.subarray(0, 320))).toEqual(
      Array.from(page.data.subarray(680, 1000))
    );
    expect(tile.data.includes(255)).toBe(false);
  });

  it("reaches the page's last row and column", () => {
    const page = grayPage(4, 4, pattern);
    const tile = extractTile(page, { x: -1, y: -1 }, 5);
    expect(Array.from(tile.data.subarray(0, 5))).toEqual([
      ...Array.from(page.data.subarray(0, 4)),
      255,
    ]);
    expect(tile.data[3 * 5 + 3]).toBe(page.data[15]);
  });
});

describe("writeTileInto (the NCHW batch item)", () => {
  // A 16x16 tile holds each of the 256 gray levels exactly once.
  const allBytes = (): GrayImage => {
    const tile = createGray(16, 16);
    for (let i = 0; i < tile.data.length; i += 1) {
      tile.data[i] = i;
    }
    return tile;
  };
  const ramp = Array.from({ length: 256 }, (_, i) => i);

  it("writes three identical float32 planes per batch item, unnormalised", () => {
    const tile = allBytes();
    const item = 3 * 256;
    const target = new Float32Array(2 * item);
    writeTileInto(target, 0, tile, 16);
    writeTileInto(target, item, tile, 16);
    for (const plane of [0, 1, 2, 3, 4, 5]) {
      expect(
        Array.from(target.subarray(plane * 256, plane * 256 + 256))
      ).toEqual(ramp);
    }
  });

  it("writes fp16 halves that decode back to the same 0..255 values", () => {
    const tile = allBytes();
    const target = new Uint16Array(3 * 256);
    writeTileInto(target, 0, tile, 16);
    expect(
      Array.from(target.subarray(0, 256), (half) => float32FromFloat16(half))
    ).toEqual(ramp);
    expect(Array.from(target.subarray(256, 512))).toEqual(
      Array.from(target.subarray(0, 256))
    );
    expect(Array.from(target.subarray(512, 768))).toEqual(
      Array.from(target.subarray(0, 256))
    );
    // The halves are the library's codec, not a cast made up here.
    expect(target[200]).toBe(float16FromFloat32(200));
    // And the codec is numpy's, measured with
    // np.arange(256, dtype=np.float32).astype(np.float16).view(np.uint16).
    expect(Array.from(target.subarray(0, 8))).toEqual([
      0, 15_360, 16_384, 16_896, 17_408, 17_664, 17_920, 18_176,
    ]);
    expect(Array.from(target.subarray(248, 256))).toEqual([
      23_488, 23_496, 23_504, 23_512, 23_520, 23_528, 23_536, 23_544,
    ]);
  });

  it("refuses a tile of the wrong size and a target that cannot hold the item", () => {
    const tile = allBytes();
    expect(() => writeTileInto(new Float32Array(3 * 256), 0, tile, 15)).toThrow(
      PlaneError
    );
    expect(() =>
      writeTileInto(new Float32Array(3 * 256 - 1), 0, tile, 16)
    ).toThrow(RangeError);
    expect(() => writeTileInto(new Float32Array(3 * 256), 1, tile, 16)).toThrow(
      RangeError
    );
  });
});

describe("mergeTileClasses (merge_patches)", () => {
  // A 6-wide page in 4px tiles: origins 0 and 2, so columns 2 and 3 are the
  // overlap band and the two tiles vote there.
  const twoTileRow = (left: number, right: number): ClassMap =>
    mergeTileClasses(
      [filledClassMap(4, left), filledClassMap(4, right)],
      6,
      4,
      4,
      4
    );

  it("averages the class indices and truncates, naming a class neither tile chose", () => {
    const fourFive = twoTileRow(4, 5);
    // trunc(9 / 2) = 4 in the band, the tiles' own choice outside it.
    expect(Array.from(fourFive.data.subarray(0, 6))).toEqual([
      4, 4, 4, 4, 5, 5,
    ]);
    const zeroFive = twoTileRow(0, 5);
    // trunc(5 / 2) = 2, a class from neither side.
    expect(Array.from(zeroFive.data.subarray(0, 6))).toEqual([
      0, 0, 2, 2, 5, 5,
    ]);
    for (let y = 1; y < 4; y += 1) {
      expect(Array.from(zeroFive.data.subarray(y * 6, y * 6 + 6))).toEqual([
        0, 0, 2, 2, 5, 5,
      ]);
    }
    expect(zeroFive.kind).toBe("classes");
  });

  it("divides by four where both axes pull back", () => {
    // 6x6 in 4px tiles: origins (0,0), (2,0), (0,2), (2,2), and the 2x2 square
    // at (2,2) is the only place all four meet.
    expect(histogramOf(tileCoverage(6, 6, 4, 4))).toEqual({
      1: 16,
      2: 16,
      4: 4,
    });
    const merged = mergeTileClasses(
      [
        filledClassMap(4, 5),
        filledClassMap(4, 4),
        filledClassMap(4, 3),
        filledClassMap(4, 0),
      ],
      6,
      6,
      4,
      4
    );
    expect(classAt(merged, 2, 2)).toBe(3); // trunc((5 + 4 + 3 + 0) / 4)
    expect(classAt(merged, 2, 0)).toBe(4); // trunc((5 + 4) / 2)
    expect(classAt(merged, 0, 2)).toBe(4); // trunc((5 + 3) / 2)
    expect(classAt(merged, 4, 2)).toBe(2); // trunc((4 + 0) / 2)
    expect(classAt(merged, 2, 4)).toBe(1); // trunc((3 + 0) / 2)
    expect(classAt(merged, 0, 0)).toBe(5);
    expect(classAt(merged, 4, 0)).toBe(4);
    expect(classAt(merged, 0, 4)).toBe(3);
    expect(classAt(merged, 4, 4)).toBe(0);
  });

  it("counts a repeated origin in both the sum and the weight", () => {
    const tiles = [4, 4, 4, 4].map((classIndex) =>
      filledClassMap(4, classIndex)
    );
    const merged = mergeTileClasses(tiles, 6, 4, 4, 3);
    expect(histogramOf(tileCoverage(6, 4, 4, 3))).toEqual({ 2: 16, 4: 8 });
    expect(Array.from(new Set(merged.data))).toEqual([4]);
  });

  it("refuses a tile list the grid does not match", () => {
    expect(() => mergeTileClasses([filledClassMap(4, 1)], 6, 4, 4, 4)).toThrow(
      PlaneError
    );
    expect(() =>
      mergeTileClasses([filledClassMap(4, 1), filledClassMap(3, 1)], 6, 4, 4, 4)
    ).toThrow(PlaneError);
  });
});

describe("the whole grid on the Kesh page", () => {
  it("covers every pixel once except the 164 rows the last row re-covers", () => {
    // 2716 - 320 = 2396 pulls the ninth row back over rows 2396..2559, and
    // 1920 / 320 is exact so no column pulls back: nothing reaches weight 4.
    expect(histogramOf(tileCoverage(KESH.width, KESH.height))).toEqual({
      1: 4_899_840,
      2: 314_880,
    });
    expect(164 * KESH.width).toBe(314_880);
  });

  it("splits and merges a page back to itself", () => {
    // 11x7 in 4px tiles: column 7 and row 3 are overlap bands. They are not
    // excluded from the comparison, because two tiles cut from the same page
    // carry the same class there: the sum is 2v over a weight of 2, and
    // trunc(2v / 2) is v. merge_patches only distorts where the model's tiles
    // disagree, which is what the truncation cases above pin.
    const page = grayPage(11, 7, (x, y) => (3 * x + 5 * y) % 6);
    const grid = tileGrid(11, 7, 4, 4);
    const tiles = grid.map((origin) => {
      const tile = extractTile(page, origin, 4);
      return planeFromBytes("classes", 4, 4, tile.data);
    });
    const merged = mergeTileClasses(tiles, 11, 7, 4, 4);
    expect(Array.from(merged.data)).toEqual(Array.from(page.data));
    expect(Array.from(new Set(tileCoverage(11, 7, 4, 4)))).toEqual([1, 2, 4]);
  });
});

describe("the step homr never takes", () => {
  it("refuses a step that would overflow the byte accumulators", () => {
    // A quarter-window step covers a pixel up to sixteen times, so a weight and
    // a sum of class indices both leave a byte. homr tiles at the window or at
    // half of it and nothing here needs to go lower, so this is refused rather
    // than paid for with wider accumulators.
    expect(() => tileGrid(64, 64, 16, 7)).toThrow(PlaneError);
    expect(() => tileCoverage(64, 64, 16, 7)).toThrow(STEP_TOO_SMALL);
    expect(() => tileGrid(64, 64, 16, 8)).not.toThrow();
  });
});

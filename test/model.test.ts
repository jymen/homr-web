import { describe, expect, it } from "vitest";
import {
  normalizeRotatedRect,
  pointListFromPairs,
  rotatedBoxFromParts,
} from "../src/geometry/boxes.js";
import { createClassMap } from "../src/image/plane.js";
import {
  createSegmentationResult,
  MASK_CLASSES,
} from "../src/model/pipeline.js";
import {
  createMultiStaff,
  createStaff,
  createStaffPoint,
  lastLineY,
  lineCount,
  StaffError,
  yTolerance,
} from "../src/model/staff.js";
import {
  createBarLine,
  symbolsOfKind,
  transformSymbol,
} from "../src/model/symbols.js";
import {
  createEncodedSymbol,
  hasPosition,
  isControlSymbol,
  isDecodedSymbol,
  NEWLINE,
  sameSymbol,
} from "../src/transformer/symbol.js";
import {
  isToken,
  tokenAt,
  tokenIndex,
  VOCABULARIES,
  VocabularyError,
} from "../src/transformer/vocabulary.js";

describe("staff points and staffs", () => {
  it("derives unit sizes, bounds and tolerance the way model.py does", () => {
    const p1 = createStaffPoint(10, [100, 110, 120, 130, 140], 0);
    const p2 = createStaffPoint(50, [102, 114, 126, 138, 150], 1);
    expect(p1.averageUnitSize).toBe(10);
    expect(p2.averageUnitSize).toBe(12);
    expect(lineCount(p1)).toBe(5);
    expect(lastLineY(p2)).toBe(150);
    const staff = createStaff([p1, p2]);
    expect(staff).toMatchObject({
      averageUnitSize: 11,
      isGrandstaff: false,
      maxX: 50,
      maxY: 150,
      minX: 10,
      minY: 100,
      space: "page",
    });
    expect(yTolerance(staff)).toBe(44);
    expect(() => createStaffPoint(0, [1, 2, 3], 0)).toThrow(StaffError);
    expect(() => createStaff([])).toThrow(StaffError);
  });
  it("sorts a multi staff by minY", () => {
    const low = createStaff([
      createStaffPoint(0, [300, 310, 320, 330, 340], 0),
    ]);
    const high = createStaff([
      createStaffPoint(0, [100, 110, 120, 130, 140], 0),
    ]);
    expect(createMultiStaff([low, high]).staffs.map((s) => s.minY)).toEqual([
      100, 300,
    ]);
    expect(() => createMultiStaff([])).toThrow(StaffError);
  });
});

describe("symbols", () => {
  it("creates symbols centred on their box and transforms only the centre", () => {
    const rect = normalizeRotatedRect({ angle: 0, cx: 5, cy: 7, h: 20, w: 2 });
    const box = rotatedBoxFromParts(
      rect,
      pointListFromPairs([]),
      pointListFromPairs([]),
      0
    );
    const bar = createBarLine(box);
    expect(bar.center).toEqual({ x: 5, y: 7 });
    const moved = transformSymbol(bar, (p) => ({ x: p.x + 1, y: p.y }));
    expect(moved.center).toEqual({ x: 6, y: 7 });
    expect(moved.box).toBe(box);
    expect(symbolsOfKind([bar, moved], "barLine")).toHaveLength(2);
    expect(symbolsOfKind([bar], "note")).toHaveLength(0);
  });
});

describe("segmentation table", () => {
  it("splits a class map into the five masks by channel", () => {
    const classes = createClassMap(3, 1);
    classes.data.set([
      MASK_CLASSES.staff.channel,
      MASK_CLASSES.notehead.channel,
      0,
    ]);
    const result = createSegmentationResult(classes);
    expect(Array.from(result.masks.staff.data)).toEqual([1, 0, 0]);
    expect(Array.from(result.masks.notehead.data)).toEqual([0, 1, 0]);
    expect(Array.from(result.masks.symbols.data)).toEqual([0, 0, 0]);
  });
});

describe("vocabulary and symbols", () => {
  it("indexes tokens like Python's build_dict", () => {
    expect(tokenIndex("rhythm", "EOS")).toBe(2);
    expect(tokenIndex("position", "lower")).toBe(2);
    expect(tokenAt("pitch", tokenIndex("pitch", "C4"))).toBe("C4");
    expect(VOCABULARIES.lift).toHaveLength(7);
    expect(isToken("rhythm", NEWLINE)).toBe(false);
    expect(() => tokenIndex("rhythm", "nope" as never)).toThrow(
      VocabularyError
    );
  });
  it("builds encoded symbols with homr's defaults and classifies them", () => {
    const s = createEncodedSymbol("clef_G2", { position: "upper" });
    expect(s).toMatchObject({
      articulation: ".",
      coordinates: null,
      lift: ".",
      pitch: ".",
      position: "upper",
      rhythm: "clef_G2",
      slur: ".",
    });
    expect(isControlSymbol(createEncodedSymbol("EOS"))).toBe(true);
    expect(hasPosition("rest_4")).toBe(true);
    expect(hasPosition("barline")).toBe(false);
    expect(isDecodedSymbol(s)).toBe(true);
    expect(isDecodedSymbol(createEncodedSymbol(NEWLINE))).toBe(false);
    expect(sameSymbol(s, { ...s, coordinates: { x: 1, y: 2 } })).toBe(true);
    expect(sameSymbol(s, { ...s, lift: "#" })).toBe(false);
  });
});

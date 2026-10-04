/**
 * RapidOCR 3.9.2's DBPostProcess and TextDetector.sorted_boxes
 * (rapidocr/ch_ppocr_det): the detection map in, the text quadrilaterals of
 * the image the detector was given out.
 *
 * numpy keeps cv2's float32 corners float32 through every step that follows,
 * a Python int being a weak scalar, so each step rounds to float32 here too.
 */

import type { Mat } from "@techstark/opencv-js";
import type { MatScope, OpenCv } from "../cv/opencv.js";
import { boxPointsFloat32, type Corner } from "../geometry/box-transforms.js";
import {
  rawMinAreaRectOf,
  toLegacyAngleConvention,
} from "../geometry/boxes.js";
import { roundHalfEven, toFloat32 } from "../image/numeric.js";
import { type IntPoint, offsetClosedPolygonRound } from "./clipper-offset.js";

/** Four corners: top-left, top-right, bottom-right, bottom-left once ordered. */
export type Quad = readonly [Corner, Corner, Corner, Corner];

export const DB_POSTPROCESS = {
  boxThresh: 0.5,
  maxCandidates: 1000,
  minSize: 3,
  thresh: 0.3,
  unclipRatio: 1.6,
} as const;

const BOX_SORT_Y_THRESHOLD = 10;

export interface DetectionMap {
  readonly data: Float32Array;
  readonly height: number;
  readonly width: number;
}

export interface MiniBox {
  readonly box: Quad;
  readonly sside: number;
}

function intPointsToMat(
  cv: OpenCv,
  scope: MatScope,
  points: readonly IntPoint[]
): Mat {
  const mat = scope.keep(new cv.Mat(points.length, 1, cv.CV_32SC2));
  mat.data32S.set(points.flat());
  return mat;
}

/**
 * get_mini_boxes: minAreaRect, boxPoints sorted by x (Python's stable
 * sorted), then the left pair and the right pair each ordered by y.
 * opencv.js reports the rect in the post-4.5.1 angle convention; the corners
 * are drawn from it in opencv-python 4.14's, which is what the oracle runs.
 */
export function miniBox(
  cv: OpenCv,
  scope: MatScope,
  points: readonly IntPoint[]
): MiniBox {
  const rect = toLegacyAngleConvention(
    rawMinAreaRectOf(cv.minAreaRect(intPointsToMat(cv, scope, points)))
  );
  const byX = [...boxPointsFloat32(rect)].sort((a, b) => a[0] - b[0]);
  const [p0, p1, p2, p3] = byX as unknown as Quad;
  const [first, fourth] = p1[1] > p0[1] ? [p0, p1] : [p1, p0];
  const [second, third] = p3[1] > p2[1] ? [p2, p3] : [p3, p2];
  return {
    box: [first, second, third, fourth],
    sside: Math.min(rect.w, rect.h),
  };
}

/** shapely's Polygon(box).area, GEOS's ring formula over the closed ring. */
function polygonArea(box: Quad): number {
  const ring = [...box, box[0]];
  const [[x0]] = box;
  let sum = 0;
  for (let i = 1; i < ring.length - 1; i += 1) {
    const x = (ring[i]?.[0] ?? 0) - x0;
    sum += x * ((ring[i - 1]?.[1] ?? 0) - (ring[i + 1]?.[1] ?? 0));
  }
  return Math.abs(sum / 2);
}

function polygonLength(box: Quad): number {
  let length = 0;
  for (let i = 0; i < box.length; i += 1) {
    const a = box[i] ?? box[0];
    const b = box[(i + 1) % box.length] ?? box[0];
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    length += Math.sqrt(dx * dx + dy * dy);
  }
  return length;
}

export const unclipDistance = (box: Quad): number =>
  (polygonArea(box) * DB_POSTPROCESS.unclipRatio) / polygonLength(box);

/** DBPostProcess.unclip: pyclipper truncates each float corner to an integer. */
export const unclip = (box: Quad): IntPoint[] =>
  offsetClosedPolygonRound(
    box.map(([x, y]) => [Math.trunc(x), Math.trunc(y)] as const),
    unclipDistance(box)
  );

/** box_score_fast: the map's mean under the box, rasterised by cv2.fillPoly. */
function boxScoreFast(
  cv: OpenCv,
  scope: MatScope,
  map: Mat,
  box: Quad
): number {
  const { cols: w, rows: h } = map;
  const xs = box.map(([x]) => x);
  const ys = box.map(([, y]) => y);
  const clip = (v: number, high: number) => Math.min(Math.max(v, 0), high);
  const xmin = clip(Math.floor(Math.min(...xs)), w - 1);
  const xmax = clip(Math.ceil(Math.max(...xs)), w - 1);
  const ymin = clip(Math.floor(Math.min(...ys)), h - 1);
  const ymax = clip(Math.ceil(Math.max(...ys)), h - 1);
  const mask = scope.keep(
    cv.Mat.zeros(ymax - ymin + 1, xmax - xmin + 1, cv.CV_8UC1)
  );
  const polygon = intPointsToMat(
    cv,
    scope,
    box.map(
      ([x, y]) =>
        [
          Math.trunc(toFloat32(x - xmin)),
          Math.trunc(toFloat32(y - ymin)),
        ] as const
    )
  );
  const polygons = scope.keep(new cv.MatVector());
  polygons.push_back(polygon);
  cv.fillPoly(mask, polygons, new cv.Scalar(1));
  const roi = scope.keep(
    map.roi(new cv.Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1))
  );
  return cv.mean(roi, mask)[0] ?? 0;
}

/** `np.round(v / width * dest)` in float32, then the clip to [0, dest]. */
function rescale(v: number, from: number, to: number): number {
  const scaled = roundHalfEven(toFloat32(toFloat32(v / from) * to));
  return Math.min(Math.max(scaled, 0), to);
}

function contoursOf(cv: OpenCv, scope: MatScope, bitmap: Mat): IntPoint[][] {
  const found = scope.keep(new cv.MatVector());
  const hierarchy = scope.keep(new cv.Mat());
  cv.findContours(
    bitmap,
    found,
    hierarchy,
    cv.RETR_LIST,
    cv.CHAIN_APPROX_SIMPLE
  );
  const contours: IntPoint[][] = [];
  for (let i = 0; i < found.size(); i += 1) {
    const contour = scope.keep(found.get(i));
    const points: IntPoint[] = [];
    for (let p = 0; p < contour.data32S.length; p += 2) {
      points.push([contour.data32S[p] ?? 0, contour.data32S[p + 1] ?? 0]);
    }
    contours.push(points);
  }
  return contours;
}

/** order_points_clockwise: by x, then each pair by y; numpy's argsort of four is stable. */
function orderClockwise(box: Quad): Quad {
  const byX = [...box].sort((a, b) => a[0] - b[0]);
  const left = byX.slice(0, 2).sort((a, b) => a[1] - b[1]);
  const right = byX.slice(2).sort((a, b) => a[1] - b[1]);
  const [tl, bl] = left as [Corner, Corner];
  const [tr, br] = right as [Corner, Corner];
  return [tl, tr, br, bl];
}

const norm = (a: Corner, b: Corner): number =>
  toFloat32(
    Math.sqrt(
      toFloat32(toFloat32((a[0] - b[0]) ** 2) + toFloat32((a[1] - b[1]) ** 2))
    )
  );

/** filter_det_res: ordered, clipped to the image, and at least 4 px each way. */
function filterBoxes(
  boxes: readonly Quad[],
  height: number,
  width: number
): Quad[] {
  const kept: Quad[] = [];
  for (const box of boxes) {
    const ordered = orderClockwise(box).map(
      ([x, y]) =>
        [
          Math.trunc(Math.min(Math.max(x, 0), width - 1)),
          Math.trunc(Math.min(Math.max(y, 0), height - 1)),
        ] as const
    ) as unknown as Quad;
    const rectWidth = Math.trunc(norm(ordered[0], ordered[1]));
    const rectHeight = Math.trunc(norm(ordered[0], ordered[3]));
    if (rectWidth > 3 && rectHeight > 3) {
      kept.push(ordered);
    }
  }
  return kept;
}

/** TextDetector.sorted_boxes: rows of first corners within 10 px, each row left to right. */
export function sortedBoxes(boxes: readonly Quad[]): Quad[] {
  const byY = [...boxes].sort((a, b) => a[0][1] - b[0][1]);
  const line: number[] = [];
  for (const [i, box] of byY.entries()) {
    const previous = byY[i - 1];
    const step =
      previous !== undefined &&
      box[0][1] - previous[0][1] >= BOX_SORT_Y_THRESHOLD
        ? 1
        : 0;
    line.push((line.at(-1) ?? 0) + step);
  }
  return byY
    .map((box, i) => ({ box, line: line[i] ?? 0 }))
    .sort((a, b) => a.line - b.line || a.box[0][0] - b.box[0][0])
    .map(({ box }) => box);
}

/**
 * DBPostProcess.__call__ and sorted_boxes: the probability map of an image
 * `destHeight` by `destWidth` in, its boxes out, in reading order.
 */
export function detectionBoxes(
  cv: OpenCv,
  scope: MatScope,
  map: DetectionMap,
  destHeight: number,
  destWidth: number
): Quad[] {
  const { height, width } = map;
  const threshold = toFloat32(DB_POSTPROCESS.thresh);
  const segmentation = scope.keep(new cv.Mat(height, width, cv.CV_8UC1));
  for (let i = 0; i < map.data.length; i += 1) {
    segmentation.data[i] = (map.data[i] ?? 0) > threshold ? 1 : 0;
  }
  const kernel = scope.keep(cv.Mat.ones(2, 2, cv.CV_8UC1));
  const dilated = scope.keep(new cv.Mat());
  cv.dilate(segmentation, dilated, kernel);
  const bitmap = scope.keep(new cv.Mat());
  dilated.convertTo(bitmap, cv.CV_8UC1, 255);
  const pred = scope.keep(new cv.Mat(height, width, cv.CV_32FC1));
  pred.data32F.set(map.data);

  const boxes: Quad[] = [];
  const contours = contoursOf(cv, scope, bitmap).slice(
    0,
    DB_POSTPROCESS.maxCandidates
  );
  for (const contour of contours) {
    const first = miniBox(cv, scope, contour);
    if (first.sside < DB_POSTPROCESS.minSize) {
      continue;
    }
    if (DB_POSTPROCESS.boxThresh > boxScoreFast(cv, scope, pred, first.box)) {
      continue;
    }
    const expanded = miniBox(cv, scope, unclip(first.box));
    if (expanded.sside < DB_POSTPROCESS.minSize + 2) {
      continue;
    }
    boxes.push(
      expanded.box.map(
        ([x, y]) =>
          [
            rescale(x, width, destWidth),
            rescale(y, height, destHeight),
          ] as const
      ) as unknown as Quad
    );
  }
  return sortedBoxes(filterBoxes(boxes, destHeight, destWidth));
}

/**
 * homr's staff_parsing.py up to the transformer: the regrouping of multi
 * staffs into voices, and prepare_staff_image, which cuts each staff out of
 * the page, dewarps it and centres it on the encoder canvas. The steps and
 * their rounding are in docs/design/phase-6-findings.md.
 */

import { planeToMat } from "../cv/mat-plane.js";
import { type MatScope, type OpenCv, withMatScope } from "../cv/opencv.js";
import { warpImage } from "../dewarp/piecewise-affine.js";
import { dewarpStaffImage } from "../dewarp/staff.js";
import type { Point } from "../geometry/boxes.js";
import { roundHalfEven } from "../image/numeric.js";
import {
  cropPlaneAndReturnNewTop,
  fillRectInPlace,
  type GrayImage,
  meanOfRegion,
  planeFromBytes,
} from "../image/plane.js";
import { blackSpotRemovalThreshold } from "../model/constants.js";
import {
  createStaffCanvas,
  ENCODER_CANVAS,
  type StaffCanvas,
} from "../model/pipeline.js";
import {
  createMultiStaff,
  type MultiStaff,
  type Staff,
  transformStaffCoordinates,
} from "../model/staff.js";

/** StaffRegions: (min_y, max_y) of every staff of every multi staff. */
export type StaffRegions = readonly (readonly [number, number])[];

export function staffRegions(multiStaffs: readonly MultiStaff[]): StaffRegions {
  return multiStaffs.flatMap((ms) =>
    ms.staffs.map((s): readonly [number, number] => [s.minY, s.maxY])
  );
}

/** get_start_of_closest_staff_above: the lowest max_y of the staffs that start above y, or 0. */
export function startOfClosestStaffAbove(
  regions: StaffRegions,
  y: number
): number {
  const above = regions.filter(([minY]) => minY < y).map(([, maxY]) => maxY);
  return above.length === 0 ? 0 : Math.max(...above);
}

/** get_start_of_closest_staff_below: the highest min_y of the staffs that end below y, or 1e12. */
export function startOfClosestStaffBelow(
  regions: StaffRegions,
  y: number
): number {
  const below = regions.filter(([, maxY]) => maxY > y).map(([minY]) => minY);
  return below.length === 0 ? 1e12 : Math.min(...below);
}

const sameStaffCount = (multiStaffs: readonly MultiStaff[]): boolean =>
  multiStaffs.every((ms) => ms.staffs.length === multiStaffs[0]?.staffs.length);

const EDGE_TOLERANCE = 50;

/**
 * `_is_close_to_image_top_or_bottom`. It compares each staff's min_x and
 * max_x with the image height: a homr bug, ported as it is.
 */
const isCloseToImageTopOrBottom = (
  ms: MultiStaff,
  imageHeight: number
): boolean =>
  Math.min(...ms.staffs.map((s) => Math.min(s.minX, imageHeight - s.maxX))) <
  EDGE_TOLERANCE;

/**
 * `_ensure_same_number_of_staffs`: unchanged when every multi staff has as
 * many staffs as the first; else drops an odd first or last system near an
 * edge; else breaks every system into single staffs, sorted by min_y.
 */
export function ensureSameNumberOfStaffs(
  multiStaffs: readonly MultiStaff[],
  imageHeight: number
): readonly MultiStaff[] {
  if (sameStaffCount(multiStaffs)) {
    return multiStaffs;
  }
  const [first] = multiStaffs;
  const last = multiStaffs.at(-1);
  if (multiStaffs.length > 2 && first !== undefined && last !== undefined) {
    if (
      isCloseToImageTopOrBottom(first, imageHeight) &&
      sameStaffCount(multiStaffs.slice(1))
    ) {
      return multiStaffs.slice(1);
    }
    if (
      isCloseToImageTopOrBottom(last, imageHeight) &&
      sameStaffCount(multiStaffs.slice(0, -1))
    ) {
      return multiStaffs.slice(0, -1);
    }
  }
  return multiStaffs
    .flatMap((ms) => ms.staffs.map((staff) => createMultiStaff([staff])))
    .sort((a, b) => a.staffs[0].minY - b.staffs[0].minY);
}

export interface CanvasSize {
  readonly height: number;
  readonly width: number;
}

/** get_tr_omr_canvas_size for an image of `height` by `width`, without margins. */
export function trOmrCanvasSize(height: number, width: number): CanvasSize {
  const { height: maxHeight, width: maxWidth } = ENCODER_CANVAS;
  if (height / width > maxHeight / maxWidth) {
    return {
      height: maxHeight,
      width: Math.trunc((width / height) * maxHeight),
    };
  }
  return { height: Math.trunc((height / width) * maxWidth), width: maxWidth };
}

/** x1, y1, x2, y2. */
export type Region = readonly [number, number, number, number];

/** `_calculate_region`: the staff with room around it, bounded by its neighbours, truncated to ints. */
export function staffRegion(staff: Staff, regions: StaffRegions): Region {
  const unit = staff.averageUnitSize;
  return [
    Math.trunc(staff.minX - 2 * unit),
    Math.trunc(
      Math.max(
        staff.minY - 4 * unit,
        startOfClosestStaffAbove(regions, staff.minY)
      )
    ),
    Math.trunc(staff.maxX + 2 * unit),
    Math.trunc(
      Math.min(
        staff.maxY + 4 * unit,
        startOfClosestStaffBelow(regions, staff.maxY)
      )
    ),
  ];
}

function grayFromMat(mat: {
  cols: number;
  data: Uint8Array;
  rows: number;
}): GrayImage {
  return planeFromBytes("gray", mat.cols, mat.rows, Uint8Array.from(mat.data));
}

/** cv2.resize(image, (width, height)) with its default INTER_LINEAR. */
export function resizeGray(
  cv: OpenCv,
  image: GrayImage,
  width: number,
  height: number
): GrayImage {
  return withMatScope((scope: MatScope) => {
    const out = scope.keep(new cv.Mat());
    cv.resize(
      planeToMat(cv, scope, image),
      out,
      new cv.Size(width, height),
      0,
      0,
      cv.INTER_LINEAR
    );
    return grayFromMat(out);
  });
}

/** center_image_on_canvas, gray and without margins: resized, then pasted at x = 0, centred in y, on white. */
export function centerImageOnCanvas(
  cv: OpenCv,
  image: GrayImage,
  size: CanvasSize
): GrayImage {
  const resized = resizeGray(cv, image, size.width, size.height);
  const { height, width } = ENCODER_CANVAS;
  const canvas = new Uint8Array(width * height).fill(255);
  const yOffset = Math.floor((height - resized.height) / 2);
  for (let y = 0; y < resized.height; y += 1) {
    const row = y + yOffset;
    if (row >= 0 && row < height) {
      canvas.set(
        resized.data
          .subarray(y * resized.width, (y + 1) * resized.width)
          .subarray(0, width),
        row * width
      );
    }
  }
  return planeFromBytes("gray", width, height, canvas);
}

const DARK_THRESHOLD = 97;
const AVERAGE_GRAY_INTENSITY = 127;

/**
 * remove_black_contours_at_edges_of_image, on a copy: whitens the bounding
 * rect of every large dark blob that touches the edge and is at least half
 * dark. (homr names that test `is_mostly_dark` and skips when it holds, but
 * it reads the inverted image, so the rects it clears are the dark ones.)
 */
export function removeBlackContoursAtEdges(
  cv: OpenCv,
  gray: GrayImage,
  unitSize: number
): GrayImage {
  const { height, width } = gray;
  const { inverted, rects } = withMatScope((scope) => {
    const thresh = scope.keep(new cv.Mat());
    // homr thresholds with THRESH_BINARY and then takes 255 - thresh, which is THRESH_BINARY_INV.
    cv.threshold(
      planeToMat(cv, scope, gray),
      thresh,
      DARK_THRESHOLD,
      255,
      cv.THRESH_BINARY_INV
    );
    const dark = grayFromMat(thresh);
    const contours = scope.keep(new cv.MatVector());
    cv.findContours(
      thresh,
      contours,
      scope.keep(new cv.Mat()),
      cv.RETR_TREE,
      cv.CHAIN_APPROX_SIMPLE
    );
    const found: CvRectangle[] = [];
    for (let i = 0; i < contours.size(); i += 1) {
      found.push(cv.boundingRect(scope.keep(contours.get(i))));
    }
    return { inverted: dark, rects: found };
  });
  const threshold = blackSpotRemovalThreshold(unitSize);
  const out = planeFromBytes("gray", width, height, Uint8Array.from(gray.data));
  for (const { height: h, width: w, x, y } of rects) {
    const large = w >= threshold && h >= threshold;
    const atEdge = x === 0 || y === 0 || x + w === width || y + h === height;
    if (
      large &&
      atEdge &&
      meanOfRegion(inverted, x, y, x + w, y + h) >= AVERAGE_GRAY_INTENSITY
    ) {
      fillRectInPlace(out, x, y, x + w, y + h, 255);
    }
  }
  return out;
}

interface CvRectangle {
  readonly height: number;
  readonly width: number;
  readonly x: number;
  readonly y: number;
}

/** `_dewarp_staff(staff, None, top_left, scaling)`: moved to `topLeft`, then scaled. */
export function staffInCrop(
  staff: Staff,
  topLeft: Point,
  scaling: number
): Staff {
  return transformStaffCoordinates(
    staff,
    (p) => ({ x: (p.x - topLeft.x) * scaling, y: (p.y - topLeft.y) * scaling }),
    "canvas"
  );
}

const mapRegion = (
  [x1, y1, x2, y2]: Region,
  f: (v: number, i: number) => number
): Region => [f(x1, 0), f(y1, 1), f(x2, 2), f(y2, 3)];

/** prepare_staff_image's `region + [-10, -50, 10, 50]`. */
const withMargin = ([x1, y1, x2, y2]: Region): Region => [
  x1 - 10,
  y1 - 50,
  x2 + 10,
  y2 + 50,
];

/**
 * prepare_staff_image without the debug drawing. The staff it returns is
 * the one homr returns: in the coordinates of the first, scaled crop,
 * undewarped, not those of the canvas.
 */
export function prepareStaffImage(
  cv: OpenCv,
  staff: Staff,
  page: GrayImage,
  regions: StaffRegions
): StaffCanvas {
  const region = staffRegion(staff, regions);
  const size = trOmrCanvasSize(region[3] - region[1], region[2] - region[0]);
  const scaling = size.height / (region[3] - region[1]);
  const resized = resizeGray(
    cv,
    page,
    Math.trunc(page.width * scaling),
    Math.trunc(page.height * scaling)
  );
  const scaled = mapRegion(region, (v) => roundHalfEven(v * scaling));
  const step1 = cropPlaneAndReturnNewTop(resized, ...withMargin(scaled));
  const topLeft = { x: step1.left / scaling, y: step1.top / scaling };
  const moved = staffInCrop(staff, topLeft, scaling);
  const transform = dewarpStaffImage(cv, step1.plane, moved);
  const warped =
    transform === null ? step1.plane : warpImage(cv, transform, step1.plane);
  const step2 = cropPlaneAndReturnNewTop(
    warped,
    ...mapRegion(scaled, (v, i) => v - (i % 2 === 0 ? step1.left : step1.top))
  );
  const cleaned = removeBlackContoursAtEdges(
    cv,
    step2.plane,
    moved.averageUnitSize
  );
  return createStaffCanvas(centerImageOnCanvas(cv, cleaned, size), moved);
}

/**
 * parse_staffs up to the transformer: the canvases in homr's parse order,
 * every staff of voice 0 top to bottom, then voice 1.
 */
export function staffCanvases(
  cv: OpenCv,
  multiStaffs: readonly MultiStaff[],
  page: GrayImage
): StaffCanvas[] {
  const systems = ensureSameNumberOfStaffs(multiStaffs, page.height);
  const regions = staffRegions(systems);
  const voices = systems[0]?.staffs.length ?? 0;
  const canvases: StaffCanvas[] = [];
  for (let voice = 0; voice < voices; voice += 1) {
    for (const system of systems) {
      const staff = system.staffs[voice];
      if (staff !== undefined) {
        canvases.push(prepareStaffImage(cv, staff, page, regions));
      }
    }
  }
  return canvases;
}

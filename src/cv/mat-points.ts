/**
 * Where a PointList and a Mat meet: a different byte layout from mat-plane.ts's
 * and a different direction of need, per contour rather than per page.
 *
 * cv2 wants a contour as CV_32SC2 with rows = n and cols = 1, which is numpy's
 * (n, 1, 2) and not (n, 2). Both shapes exist in homr -- `contours` is (n, 1, 2)
 * and `polygon` is (n, 2) -- and cv2 reads either as n points, so the port keeps
 * one flat representation and gives it cv2's own shape on the way in.
 */

import type { Mat } from "@techstark/opencv-js";
import {
  type PointList,
  pointCount,
  pointListFromInt32,
} from "../geometry/boxes.js";
import type { Plane1 } from "../image/plane.js";
import { planeToMat } from "./mat-plane.js";
import type { MatScope, OpenCv } from "./opencv.js";

export function pointListToMat(
  cv: OpenCv,
  scope: MatScope,
  points: PointList
): Mat {
  const mat = scope.keep(new cv.Mat(pointCount(points), 1, cv.CV_32SC2));
  mat.data32S.set(points);
  return mat;
}

export function pointListOfMat(mat: Mat): PointList {
  return pointListFromInt32(Int32Array.from(mat.data32S));
}

/**
 * `cv2.findContours(img, RETR_TREE, CHAIN_APPROX_SIMPLE)`, the one call shape
 * bounding_boxes.py has. No retrieval-mode parameter on purpose: there is
 * nothing to choose, and a different mode changes the contour *indices*, which
 * are homr's debug_id and phase 5's staff-line identity.
 *
 * Measured 2026-09-28 against opencv-python 4.14.0 on the four filtered Kesh
 * masks: identical contour count, order and points, 188, 387, 19 and 82
 * contours. `MatVector.get` allocates a wrapper per element, so each one is
 * handed to the scope rather than freed by hand.
 */
export function findContoursOf(
  cv: OpenCv,
  scope: MatScope,
  image: Plane1
): PointList[] {
  const source = planeToMat(cv, scope, image);
  const found = scope.keep(new cv.MatVector());
  const hierarchy = scope.keep(new cv.Mat());
  cv.findContours(
    source,
    found,
    hierarchy,
    cv.RETR_TREE,
    cv.CHAIN_APPROX_SIMPLE
  );
  const contours: PointList[] = [];
  for (let i = 0; i < found.size(); i += 1) {
    contours.push(pointListOfMat(scope.keep(found.get(i))));
  }
  return contours;
}

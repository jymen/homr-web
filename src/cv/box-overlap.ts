/**
 * When homr considers two shapes to touch, weaknesses included, and neither of
 * them may be improved.
 *
 * do_polygons_overlap only tests whether a *vertex* of one polygon lies inside
 * the other, so two quadrilaterals crossing like a plus sign are reported as
 * not overlapping. _can_shapes_possibly_touch compares the centre distance
 * against the sum of the two longer sides, which is not the circumscribing
 * radius. Both answers decide the merge grouping, and therefore the final list.
 */

import type { Mat } from "@techstark/opencv-js";
import {
  type AnyBox,
  canShapesPossiblyTouch,
  type Ellipse,
  type PointList,
  pointAt,
  pointCount,
  polygonOf,
  type RotatedBox,
} from "../geometry/boxes.js";
import { pointListToMat } from "./mat-points.js";
import type { MatScope, OpenCv } from "./opencv.js";

/**
 * `cv2.pointPolygonTest(polygon, point, False) >= 0` over every vertex. Zero
 * means "on the edge" and homr accepts it, which is why this stays a cv call: a
 * winding-number test in TypeScript would have to decide that case itself.
 */
function anyVertexInside(
  cv: OpenCv,
  vertices: PointList,
  polygon: Mat
): boolean {
  for (let i = 0; i < pointCount(vertices); i += 1) {
    const { x, y } = pointAt(vertices, i);
    if (cv.pointPolygonTest(polygon, { x, y }, false) >= 0) {
      return true;
    }
  }
  return false;
}

/**
 * do_polygons_overlap, vertex-only. The second Mat is built only if the first
 * half found nothing, matching Python's short circuit.
 */
export function doPolygonsOverlap(
  cv: OpenCv,
  scope: MatScope,
  poly1: PointList,
  poly2: PointList
): boolean {
  if (anyVertexInside(cv, poly1, pointListToMat(cv, scope, poly2))) {
    return true;
  }
  return anyVertexInside(cv, poly2, pointListToMat(cv, scope, poly1));
}

interface Outline {
  readonly mat: Mat;
  readonly vertices: PointList;
}

/**
 * is_overlapping and is_overlapping_with_any, with each box's polygon Mat built
 * once for the tester's life.
 *
 * Stateful because the state is the point: the merge asks O(n^2) questions and
 * a Mat per question is roughly 75 000 allocations for the 387 contours of one
 * Kesh page's stems_rest mask. The Mats go into the caller's scope, so their
 * lifetime is still the caller's operation, and the memo is keyed on box
 * identity rather than on the polygon because an AxisBox derives a fresh
 * outline on every read.
 *
 * The left operand is never an AxisBox: in homr is_overlapping is a method on
 * AngledBoundingBox and only `other` can be a plain BoundingBox.
 */
export class OverlapTester {
  private readonly cv: OpenCv;
  private readonly outlines = new WeakMap<AnyBox, Outline>();
  private readonly scope: MatScope;

  constructor(cv: OpenCv, scope: MatScope) {
    this.cv = cv;
    this.scope = scope;
  }

  overlaps(box: RotatedBox | Ellipse, other: AnyBox): boolean {
    if (!canShapesPossiblyTouch(box, other)) {
      return false;
    }
    const left = this.outlineOf(box);
    const right = this.outlineOf(other);
    return (
      anyVertexInside(this.cv, left.vertices, right.mat) ||
      anyVertexInside(this.cv, right.vertices, left.mat)
    );
  }

  overlapsAny(box: RotatedBox | Ellipse, others: readonly AnyBox[]): boolean {
    for (const other of others) {
      if (this.overlaps(box, other)) {
        return true;
      }
    }
    return false;
  }

  private outlineOf(box: AnyBox): Outline {
    const known = this.outlines.get(box);
    if (known !== undefined) {
      return known;
    }
    const vertices = polygonOf(box);
    const outline: Outline = {
      mat: pointListToMat(this.cv, this.scope, vertices),
      vertices,
    };
    this.outlines.set(box, outline);
    return outline;
  }
}

/**
 * The three constructors bounding_boxes.py exposes, and the filters that are
 * their only reason to exist.
 *
 * Order mirrors Python exactly: findContours, then per contour fit and filter,
 * then either return as-is or thicken, merge and refit. `skipMerging` returns
 * before the thickening too, which is why that is not a separate step.
 *
 * The two max-size predicates differ and are deliberately not shared.
 * create_rotated_bounding_boxes guards each dimension with `max_size[k] > 0`,
 * so a bound of zero or less disables that one dimension -- brace_dot's
 * `max_size=(100, -1)` is the only caller anywhere that uses it, and it is a
 * phase-5 call site, so the rule must not be trimmed as dead.
 * create_bounding_ellipses has no such guard at all, so the same `(100, -1)`
 * would reject every ellipse. Both reject when *either* dimension exceeds its
 * bound.
 */

import { mergeOverlappingGroups } from "../geometry/box-merge.js";
import type {
  AngledBox,
  Ellipse,
  PointList,
  RotatedBox,
  RotatedRectParams,
} from "../geometry/boxes.js";
import type { Mask } from "../image/plane.js";
import {
  fitEllipseForGating,
  fitRotatedRect,
  fitRotatedRectUnchecked,
  refitEllipseFromGroup,
  refitRotatedBoxFromGroup,
} from "./box-fitting.js";
import { OverlapTester } from "./box-overlap.js";
import { makeBoxThicker } from "./box-transforms.js";
import { findContoursOf } from "./mat-points.js";
import { type MatScope, type OpenCv, withMatScope } from "./opencv.js";

/** min_size and max_size: a (width, height) pair, read against the normalised size. */
export interface SizeBound {
  readonly h: number;
  readonly w: number;
}

export interface RotatedBoxesOptions {
  readonly maxSize?: SizeBound;
  readonly minSize?: SizeBound;
  readonly skipMerging?: boolean;
  /**
   * Never given a value anywhere in homr. Kept because whether it ran changes
   * the merge input, and therefore the grouping.
   */
  readonly thickenBoxes?: number;
}

export interface EllipsesOptions {
  readonly maxSize?: SizeBound;
  readonly minSize?: SizeBound;
  readonly skipMerging?: boolean;
}

const belowMinSize = (
  rect: RotatedRectParams,
  min: SizeBound | undefined
): boolean => min !== undefined && (rect.w < min.w || rect.h < min.h);

const exceedsBoxMaxSize = (
  rect: RotatedRectParams,
  max: SizeBound | undefined
): boolean =>
  max !== undefined &&
  ((max.w > 0 && rect.w > max.w) || (max.h > 0 && rect.h > max.h));

const exceedsEllipseMaxSize = (
  rect: RotatedRectParams,
  max: SizeBound | undefined
): boolean => max !== undefined && (rect.w > max.w || rect.h > max.h);

function mergedGroups<B extends AngledBox>(
  cv: OpenCv,
  scope: MatScope,
  boxes: readonly B[]
): B[][] {
  const tester = new OverlapTester(cv, scope);
  return mergeOverlappingGroups(boxes, (a, b) => tester.overlaps(a, b));
}

export function createRotatedBoundingBoxes(
  cv: OpenCv,
  image: Mask,
  options: RotatedBoxesOptions = {}
): RotatedBox[] {
  return withMatScope((scope) => {
    const boxes: RotatedBox[] = [];
    for (const [index, contour] of findContoursOf(cv, scope, image).entries()) {
      const box = fitRotatedRect(cv, scope, contour, index);
      if (
        box === null ||
        belowMinSize(box.rect, options.minSize) ||
        exceedsBoxMaxSize(box.rect, options.maxSize)
      ) {
        continue;
      }
      boxes.push(box);
    }
    if (options.skipMerging) {
      return boxes;
    }
    const thickness = options.thickenBoxes;
    const thickened =
      thickness === undefined
        ? boxes
        : boxes.map((box) => makeBoxThicker(cv, scope, box, thickness));
    return mergedGroups(cv, scope, thickened).map((group) =>
      refitRotatedBoxFromGroup(cv, scope, group)
    );
  });
}

/** create_rotated_bounding_box, singular, with no size check. */
export function createRotatedBoundingBox(
  cv: OpenCv,
  contour: PointList,
  debugId: number
): RotatedBox {
  return withMatScope((scope) =>
    fitRotatedRectUnchecked(cv, scope, contour, debugId)
  );
}

export function createBoundingEllipses(
  cv: OpenCv,
  image: Mask,
  options: EllipsesOptions = {}
): Ellipse[] {
  return withMatScope((scope) => {
    const ellipses: Ellipse[] = [];
    for (const [index, contour] of findContoursOf(cv, scope, image).entries()) {
      const ellipse = fitEllipseForGating(cv, scope, contour, index);
      if (
        ellipse === null ||
        belowMinSize(ellipse.rect, options.minSize) ||
        exceedsEllipseMaxSize(ellipse.rect, options.maxSize)
      ) {
        continue;
      }
      ellipses.push(ellipse);
    }
    if (options.skipMerging) {
      return ellipses;
    }
    return mergedGroups(cv, scope, ellipses).map((group) =>
      refitEllipseFromGroup(cv, scope, group)
    );
  });
}

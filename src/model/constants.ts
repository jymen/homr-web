/**
 * homr's constants.py, the part staff detection and the model read. One home,
 * so that no detection module defines a staff threshold of its own.
 *
 * Not here on purpose: StaffAnchor's local `max_number_of_ledger_lines = 5`,
 * which is not in constants.py and disagrees with MAX_LEDGER_LINES; it belongs
 * beside its one reader.
 */

import { roundHalfEven, truncToInt } from "../image/numeric.js";

/** constants.number_of_lines_on_a_staff: a staff point carries 5 line ordinates, or 10 for a merged grand staff. */
export const LINES_PER_STAFF = 5;

/** constants.max_number_of_ledger_lines; Staff._y_tolerance = this * average unit size. */
export const MAX_LEDGER_LINES = 4;

/** constants.tolerance_for_staff_line_detection: how far two extrapolated staff-line centres may sit apart. */
export function toleranceForStaffLineDetection(unitSize: number): number {
  return unitSize / 3;
}

/** constants.max_line_gap_size: the widest horizontal gap two fragments of one line may leave. */
export function maxLineGapSize(unitSize: number): number {
  return 5 * unitSize;
}

/** constants.is_short_line: a threshold despite the name; a fragment narrower than this is dropped. */
export function isShortLine(unitSize: number): number {
  return unitSize / 5;
}

/** constants.is_short_connected_line: also a threshold. */
export function isShortConnectedLine(unitSize: number): number {
  return 2 * unitSize;
}

export function minHeightForBraceRough(unitSize: number): number {
  return 2 * unitSize;
}

export function maxWidthForBraceRough(unitSize: number): number {
  return 3 * unitSize;
}

export function minHeightForBrace(unitSize: number): number {
  return 4 * unitSize;
}

/** constants.tolerance_for_touching_clefs: `int(round(unit_size * 2))`, the one unit function that returns an int. */
export function toleranceForTouchingClefs(unitSize: number): number {
  return truncToInt(roundHalfEven(unitSize * 2));
}

export function barLineMaxWidth(unitSize: number): number {
  return 2 * unitSize;
}

export function barLineMinHeight(unitSize: number): number {
  return 3 * unitSize;
}

/** constants.black_spot_removal_threshold: the smallest dark blob at a canvas edge that the canvas clean-up whitens. */
export function blackSpotRemovalThreshold(unitSize: number): number {
  return 2 * unitSize;
}

/** constants.staff_line_segment_x_tolerance: StaffLineSegment.get_at accepts an x this far outside a fragment. */
export const STAFF_LINE_SEGMENT_X_TOLERANCE = 10;

/** constants.minimum_connections_to_form_combined_staff. */
export const MIN_CONNECTIONS_TO_FORM_COMBINED_STAFF = 1;

/** constants.image_noise_limit: a noise-grid tile above this, with a neighbour above it too, is masked out. */
export const IMAGE_NOISE_LIMIT = 50;

/** constants.staff_position_tolerance: Staff.get_at answers None when the nearest grid point is further than this in x. */
export const STAFF_POSITION_TOLERANCE = 50;

/** constants.max_angle_for_lines_to_be_parallel, in degrees. */
export const MAX_ANGLE_FOR_LINES_TO_BE_PARALLEL = 10;

/** constants.NOTEHEAD_SIZE_RATIO, width over height: the literal constants.py holds, not 9 / 7. */
export const NOTEHEAD_SIZE_RATIO = 1.285_714;

/** constants.grandstaff_x_distance_threshold_factor. */
export const GRANDSTAFF_X_DISTANCE_THRESHOLD_FACTOR = 5;

/** constants.grandstaff_y_overlap_threshold_factor. */
export const GRANDSTAFF_Y_OVERLAP_THRESHOLD_FACTOR = 0.5;

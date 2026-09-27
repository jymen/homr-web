/**
 * The constants of homr's constants.py that the phase 1 types depend on.
 * Phase 5 ports the remaining unit-size functions into this file; nothing
 * else in the port defines a staff constant.
 */

/** constants.number_of_lines_on_a_staff: a staff point carries 5 line ordinates, or 10 for a merged grand staff. */
export const LINES_PER_STAFF = 5;

/** constants.max_number_of_ledger_lines; Staff._y_tolerance = this * average unit size. */
export const MAX_LEDGER_LINES = 4;

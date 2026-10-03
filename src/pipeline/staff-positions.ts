/**
 * Port of staff_position_save_load.save_staff_positions, and the app
 * server's reading of the file it writes.
 */

import { formatPythonFloat } from "../image/numeric.js";
import type { MultiStaff } from "../model/staff.js";
import type { StaffBox, StaffPosition } from "../result.js";

/**
 * One position per staff, in multi-staff then staff order, page-normalised
 * with homr's operation order (`x1 + width / 2`, then `/ imageWidth`).
 * `image` is the preprocessed page, whose shape homr passes.
 */
export function staffPositions(
  multiStaffs: readonly MultiStaff[],
  image: { readonly height: number; readonly width: number }
): StaffPosition[] {
  return multiStaffs
    .flatMap((multiStaff) => multiStaff.staffs)
    .map((staff) => {
      const width = staff.maxX - staff.minX;
      const height = staff.maxY - staff.minY;
      return {
        cx: (staff.minX + width / 2) / image.width,
        cy: (staff.minY + height / 2) / image.height,
        h: height / image.height,
        isGrandstaff: staff.isGrandstaff,
        w: width / image.width,
      };
    });
}

/**
 * The `--write-staff-positions` file, byte-identical to homr's: one line per
 * position, `<0|1> cx cy w h`, every number Python's str(float), each line
 * ending in "\n", no header. An empty list is an empty string.
 */
export const formatStaffPositions = (
  multiStaffs: readonly MultiStaff[],
  image: { readonly height: number; readonly width: number }
): string =>
  staffPositions(multiStaffs, image)
    .map(
      (p) =>
        `${p.isGrandstaff ? "1" : "0"} ${[p.cx, p.cy, p.w, p.h].map(formatPythonFloat).join(" ")}\n`
    )
    .join("");

/**
 * The server's `staves` (parseOmrStaves in the app's Go backend): stable
 * sorted top to bottom by `cy`, numbered after the sort. The file prints
 * shortest round-tripping floats and Go parses them exactly, so the numbers
 * need no trip through text to equal the server's.
 */
export const staffBoxes = (positions: readonly StaffPosition[]): StaffBox[] =>
  [...positions]
    .sort((a, b) => a.cy - b.cy)
    .map(({ cx, cy, h, w }, index) => ({ cx, cy, h, index, w }));

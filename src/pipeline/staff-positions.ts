/** Port of staff_position_save_load.save_staff_positions, as a string. */

import { formatPythonFloat } from "../image/numeric.js";
import type { MultiStaff } from "../model/staff.js";

/**
 * The `--write-staff-positions` file: one line per staff,
 * `<0|1> cx cy w h`, page-normalised, in multi-staff then staff order, each
 * ending in "\n", no header. An empty list is an empty string.
 *
 * Byte-identical to homr's file, so the operation order is homr's
 * (`x1 + width / 2`, then `/ imageWidth`) and every number is Python's
 * str(float). `image` is the preprocessed page, whose shape homr passes.
 */
export function formatStaffPositions(
  multiStaffs: readonly MultiStaff[],
  image: { readonly height: number; readonly width: number }
): string {
  return multiStaffs
    .flatMap((multiStaff) => multiStaff.staffs)
    .map((staff) => {
      const width = staff.maxX - staff.minX;
      const height = staff.maxY - staff.minY;
      const numbers = [
        (staff.minX + width / 2) / image.width,
        (staff.minY + height / 2) / image.height,
        width / image.width,
        height / image.height,
      ];
      return `${staff.isGrandstaff ? "1" : "0"} ${numbers.map(formatPythonFloat).join(" ")}\n`;
    })
    .join("");
}

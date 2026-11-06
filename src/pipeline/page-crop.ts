/**
 * homr normalises its staves by the autocropped page; the tab readers by the
 * page given. On a photograph whose paper autocrop cuts out the two differ, so
 * anything that compares them goes through the crop first.
 */

import type { PageCrop, StaffBox } from "../result.js";
import type { PaperRect } from "../segmentation/preprocess.js";

/** The whole page: what autocrop keeping the page amounts to. */
export const WHOLE_PAGE: PageCrop = { x0: 0, x1: 1, y0: 0, y1: 1 };

/** Autocrop's rect as a fraction of the page it was cut from. */
export function pageCrop(
  paper: PaperRect | null,
  page: { readonly height: number; readonly width: number }
): PageCrop {
  if (paper === null) {
    return WHOLE_PAGE;
  }
  return {
    x0: paper.x / page.width,
    x1: (paper.x + paper.width) / page.width,
    y0: paper.y / page.height,
    y1: (paper.y + paper.height) / page.height,
  };
}

/** An x normalised by the cropped page (a `noteColumns` entry), on the page given. */
export const columnInPage = (x: number, crop: PageCrop): number =>
  crop.x0 + x * (crop.x1 - crop.x0);

/** A staff box normalised by the cropped page, on the page given. */
export function staffInPage(staff: StaffBox, crop: PageCrop): StaffBox {
  const width = crop.x1 - crop.x0;
  const height = crop.y1 - crop.y0;
  return {
    cx: crop.x0 + staff.cx * width,
    cy: crop.y0 + staff.cy * height,
    h: staff.h * height,
    index: staff.index,
    w: staff.w * width,
  };
}

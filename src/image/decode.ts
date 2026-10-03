/**
 * A page as the caller hands it to the Worker, turned into the BGR image
 * `cv2.imread` gives homr. Browser-only: it needs `createImageBitmap` and
 * `OffscreenCanvas`, which a Worker has and Node does not.
 */

import type { PageInput } from "../worker-protocol.js";
import { type ColorImage, colorImageFromRgba } from "./plane.js";

export class PageInputError extends Error {
  override name = "PageInputError";
}

const isImageData = (page: PageInput): page is ImageData =>
  typeof ImageData === "function" && page instanceof ImageData;

async function bitmapOf(page: Blob | ImageBitmap): Promise<ImageBitmap> {
  if (!(page instanceof Blob)) {
    return page;
  }
  try {
    // No colour management, as cv2.imread applies none.
    return await createImageBitmap(page, {
      colorSpaceConversion: "none",
      premultiplyAlpha: "none",
    });
  } catch (cause) {
    throw new PageInputError(
      `the ${page.type || "untyped"} blob of ${page.size} bytes is not an image the browser can decode`,
      { cause }
    );
  }
}

/** The alpha channel is dropped, as cv2.imread with IMREAD_COLOR drops it. */
export async function decodePage(page: PageInput): Promise<ColorImage> {
  if (isImageData(page)) {
    return colorImageFromRgba(page.width, page.height, page.data);
  }
  const bitmap = await bitmapOf(page);
  const { height, width } = bitmap;
  if (width === 0 || height === 0) {
    throw new PageInputError(`the image is ${width}x${height}`);
  }
  const context = new OffscreenCanvas(width, height).getContext("2d");
  if (context === null) {
    throw new PageInputError("this Worker has no 2d OffscreenCanvas");
  }
  context.drawImage(bitmap, 0, 0);
  if (page instanceof Blob) {
    bitmap.close();
  }
  return colorImageFromRgba(
    width,
    height,
    context.getImageData(0, 0, width, height).data
  );
}

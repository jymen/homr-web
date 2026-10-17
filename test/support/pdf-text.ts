/**
 * A PDF page's text layer as homr-web's PageTextItem, read with pdf.js in
 * Node, the way an app would hand it to recognizePage: each run's box in
 * page-normalised coordinates, from its baseline up by its font size, or the
 * box around it when it is turned.
 */

import { readFileSync } from "node:fs";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import type { PageTextItem } from "../../src/tab/layer.js";

export async function textLayerOf(
  file: string,
  pageNumber = 1
): Promise<PageTextItem[]> {
  const doc = await getDocument({
    data: new Uint8Array(readFileSync(file)),
    verbosity: 0,
  }).promise;
  try {
    const page = await doc.getPage(pageNumber);
    const viewport = page.getViewport({ scale: 1 });
    const content = await page.getTextContent();
    return content.items.flatMap((item): PageTextItem[] => {
      if (!("str" in item) || item.str.trim() === "") {
        return [];
      }
      const [a, b, c, d, e, f] = item.transform as number[];
      const size = Math.hypot(c ?? 0, d ?? 0);
      const length = item.width;
      const along = Math.hypot(a ?? 0, b ?? 0) || 1;
      const ux = (a ?? 1) / along;
      const uy = (b ?? 0) / along;
      const corners = [
        [e ?? 0, f ?? 0],
        [(e ?? 0) + ux * length, (f ?? 0) + uy * length],
        [(e ?? 0) - uy * size, (f ?? 0) + ux * size],
        [
          (e ?? 0) + ux * length - uy * size,
          (f ?? 0) + uy * length + ux * size,
        ],
      ].map(([x, y]) => viewport.convertToViewportPoint(x ?? 0, y ?? 0));
      const xs = corners.map(([x]) => (x ?? 0) / viewport.width);
      const ys = corners.map(([, y]) => (y ?? 0) / viewport.height);
      return [
        {
          text: item.str,
          x0: Math.min(...xs),
          x1: Math.max(...xs),
          y0: Math.min(...ys),
          y1: Math.max(...ys),
        },
      ];
    });
  } finally {
    await doc.destroy();
  }
}

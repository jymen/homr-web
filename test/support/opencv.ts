/**
 * The opencv.js source the Node tests use. Under vitest a plain
 * `await import("@techstark/opencv-js")` never resolves at all, and neither
 * `server.deps.external` nor `interopDefault: false` changes that, while
 * `createRequire(import.meta.url)` loads the same CommonJS bundle in under
 * 300 ms with no vitest configuration. So the tests inject their own source,
 * the way this directory's golden.ts injects node:fs and pngjs into the golden
 * reader, and the library's default browser path stays uncovered here.
 */

import { createRequire } from "node:module";
import {
  type AcquiredOpenCv,
  loadOpenCv,
  type OpenCv,
  type OpenCvSource,
} from "../../src/cv/opencv.js";

const requireCjs = createRequire(import.meta.url);

export function nodeOpenCvSource(): OpenCvSource {
  return () => {
    // Boxed rather than returned, because resolving a promise with
    // module.exports calls its `then` and that hangs the whole run.
    const acquired: AcquiredOpenCv = {
      module: requireCjs("@techstark/opencv-js"),
    };
    return Promise.resolve(acquired);
  };
}

/** The one call every test that needs opencv.js makes. */
export function testOpenCv(): Promise<OpenCv> {
  return loadOpenCv(nodeOpenCvSource());
}

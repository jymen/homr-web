/**
 * The default acquisition, in a module of its own for one reason. `import()`
 * resolves its promise *with* the namespace it produced, so if any bundler's
 * CommonJS interop ever emitted a `then` export for opencv.js, that resolution
 * would call the shim described in opencv.ts and hang the realm before a line of
 * this port ran. A static import cannot do that, and this wrapper's own
 * namespace exports one plain object, so a dynamic import of *this* file is
 * safe. The 10.9 MB still arrives in a chunk of its own, which is why opencv.ts
 * imports this lazily and src/index.ts does not re-export it.
 */

import * as opencv from "@techstark/opencv-js";
import type { AcquiredOpenCv } from "./opencv.js";

export const acquiredOpenCv: AcquiredOpenCv = { module: opencv };

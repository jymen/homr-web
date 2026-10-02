/**
 * homr-web public surface. Phase 1: the data model every later phase reads
 * and writes, and the golden-fixture boundary. Phase 2 adds the model
 * manifest, the runtime, the store and the session. No algorithm yet.
 */

export const HOMR_VERSION = "0.7.0" as const;
export const HOMR_COMMIT = "8b5dcf7d7bdd1a47911dc0c661c573b957271eab" as const;

export * from "./cv/box-fitting.js";
export * from "./cv/box-overlap.js";
export * from "./cv/box-transforms.js";
export * from "./cv/create-boxes.js";
export * from "./cv/mask-morphology.js";
export * from "./cv/mat-plane.js";
export * from "./cv/mat-points.js";
export * from "./cv/opencv.js";
export * from "./geometry/box-merge.js";
export * from "./geometry/box-transforms.js";
export * from "./geometry/boxes.js";
export * from "./golden/box-tolerance.js";
export * from "./golden/decode.js";
export * from "./golden/page.js";
export * from "./image/argsort.js";
export * from "./image/find-peaks.js";
export * from "./image/numeric.js";
export * from "./image/plane.js";
export * from "./model/constants.js";
export * from "./model/pipeline.js";
export * from "./model/staff.js";
export * from "./model/symbols.js";
export * from "./models/backend.js";
export * from "./models/cache.js";
export * from "./models/dtype.js";
export * from "./models/errors.js";
export * from "./models/manifest.js";
export * from "./models/session.js";
export * from "./models/store.js";
export * from "./pipeline/predict-symbols.js";
export * from "./result.js";
export * from "./segmentation/preprocess.js";
export * from "./segmentation/resize.js";
export * from "./segmentation/segment.js";
export * from "./segmentation/tiles.js";
export * from "./segmentation/worker.js";
export * from "./transformer/symbol.js";
export * from "./transformer/vocabulary.js";

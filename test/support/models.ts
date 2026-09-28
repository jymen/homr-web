/**
 * Test-side access to models/, the git-ignored directory tools/fetch-models.sh
 * fills. The library never reads it, and a fresh clone's `npm test` must pass
 * without a 160 MB download: a contributor who has to fetch the models before
 * any test runs will not run the tests.
 *
 * So the tests that need real bytes skip, and the skip is loud. A silently
 * skipped test proves nothing and nobody notices it. CI gets the strict half:
 * test/store.test.ts asserts models/ is present whenever process.env.CI is set,
 * so absent bytes fail there instead of looking like a pass.
 *
 * The store wiring below it lives here rather than in one test file because
 * test/session.test.ts and test/segment.test.ts both open the real artifacts off
 * disk, and a second copy of `storeOn` is how the two would drift into opening
 * them under different placements.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe } from "vitest";
import { type ModelRuntime, startRuntime } from "../../src/models/backend.js";
import { memoryCache } from "../../src/models/cache.js";
import {
  ARTIFACT_IDS,
  ARTIFACTS,
  type ArtifactRecord,
  type Placement,
} from "../../src/models/manifest.js";
import {
  type FetchBytes,
  type ModelEvent,
  ModelStore,
} from "../../src/models/store.js";

export const FETCH_MODELS_HINT = "models/ absent, run tools/fetch-models.sh";

export function modelsDir(): string {
  return join(import.meta.dirname, "..", "..", "models");
}

const fileNameOf = (artifact: ArtifactRecord): string =>
  artifact.urlPath.slice(artifact.urlPath.lastIndexOf("/") + 1);

/**
 * Every one of the eight artifacts, not merely the directory: fetch-models.sh
 * creates models/ before it copies anything, so an interrupted first run leaves
 * a directory that exists and serves nothing. A partial fetch must read as
 * absent, or CI's presence assertion passes on bytes it does not have.
 */
export const MODELS_PRESENT = ARTIFACT_IDS.every((id) =>
  existsSync(join(modelsDir(), fileNameOf(ARTIFACTS[id])))
);

/**
 * Where this artifact's file is on disk, or undefined when it has not been
 * fetched. The name is the last segment of the manifest's content-addressed
 * urlPath, which is how fetch-models.sh lays the directory out.
 */
export function modelFileFor(artifact: ArtifactRecord): string | undefined {
  const path = join(modelsDir(), fileNameOf(artifact));
  return existsSync(path) ? path : undefined;
}

/** describe, or a describe.skip whose title says why, so the reporter prints the reason. */
export function describeWithModels(title: string, suite: () => void): void {
  if (MODELS_PRESENT) {
    describe(title, suite);
    return;
  }
  describe.skip(`${title} (${FETCH_MODELS_HINT})`, suite);
}

export const CPU: Placement = { artifactsFor: "wasm", provider: "wasm" };
/** The fp16 artifacts on the WebAssembly provider: the axis split that lets CI cover the fp16 branch with no GPU present. */
export const FP16_ON_WASM: Placement = {
  artifactsFor: "webgpu",
  provider: "wasm",
};

export const wasmRuntime = (): Promise<ModelRuntime> =>
  startRuntime({ maxBackend: "wasm" });

/** models/ read here and not in the library: phase 1 set the precedent that node:fs lives behind an injected port. */
export const localModels =
  (): FetchBytes =>
  ({ url }) =>
    Promise.resolve(
      new Uint8Array(
        readFileSync(`${modelsDir()}/${url.slice(url.lastIndexOf("/") + 1)}`)
      )
    );

export const storeOn = async (
  placement: Placement,
  onEvent?: (event: ModelEvent) => void
): Promise<ModelStore> =>
  new ModelStore({
    baseUrl: "file:///models/",
    cache: memoryCache(),
    fetchBytes: localModels(),
    placement,
    runtime: await wasmRuntime(),
    ...(onEvent === undefined ? {} : { onEvent }),
  });

/** The one-liner every models test needs against noUncheckedIndexedAccess: a missing fixture or output is the test's own failure, not a narrowing to carry around. */
export function required<T>(value: T | undefined, what: string): T {
  if (value === undefined) {
    throw new Error(`the test needs ${what}`);
  }
  return value;
}

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
 */

import { existsSync } from "node:fs";
import { join } from "node:path";
import { describe } from "vitest";
import {
  ARTIFACT_IDS,
  ARTIFACTS,
  type ArtifactRecord,
} from "../../src/models/manifest.js";

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

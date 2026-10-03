/**
 * `@techstark/opencv-js` as an ES module, which the package itself is not.
 *
 * It ships one file, `dist/opencv.js`, a UMD emscripten bundle with no ESM
 * entry. Imported as a module its `this` is undefined and its `root.cv =
 * factory()` throws, so it is fetched and run through an indirect eval, whose
 * top-level `this` is the global object in a page and in a module Worker
 * alike (a module Worker has no script tag and no working importScripts).
 * The server rewrites dist/'s `@techstark/opencv-js` to this file, and the
 * page's import map does the same on the main thread.
 *
 * `export default` is the shape `acquire()` in `src/cv/opencv.ts` unwraps.
 * Waiting for `onRuntimeInitialized` is its job, not this file's. The bundle
 * carries its wasm inline as a data URI, so there is no `locateFile` to supply.
 */

const SOURCE = "/node_modules/@techstark/opencv-js/dist/opencv.js";

const response = await fetch(SOURCE);
if (!response.ok) {
  throw new Error(`${SOURCE} answered ${response.status}`);
}
// biome-ignore lint/security/noGlobalEval: the indirect eval is the point, see above; the source is this repository's own node_modules
const runGlobally = eval;
runGlobally(await response.text());

if (globalThis.cv === undefined) {
  throw new Error(`${SOURCE} ran and set no globalThis.cv`);
}

export default globalThis.cv;

/**
 * `@techstark/opencv-js` as an ES module, which the package itself is not.
 *
 * It ships one file, `dist/opencv.js`, a UMD emscripten bundle with no ESM
 * entry and no `export` statement anywhere in it. Parsed as a module it exports
 * nothing, and it cannot even install its global: the UMD is invoked as
 * `}(this, function () {...}))`, `this` is `undefined` at the top level of an ES
 * module, and `root.cv = factory()` throws on undefined. In a classic script
 * `this` is `window`, the bundle's `typeof window === "object"` branch runs and
 * `globalThis.cv` is the module. So the script tag is the load and this file is
 * the export, and the bench page's import map sends `@techstark/opencv-js` here.
 *
 * `export default` is the shape the library already expects. `default-source.ts`
 * imports the package as a namespace, so `acquire()` in `src/cv/opencv.ts` is
 * handed `{ default: <the module> }`, and `surfaceOf` unwraps `.default` on the
 * evidence of `Mat` or emscripten's `then` shim. At the moment this file
 * resolves only the shim is there, which is the case that unwrap was written for.
 *
 * Waiting for `onRuntimeInitialized` is deliberately not done here.
 * `loadOpenCv`'s `acquire()` owns that wait and the clearing of the `then` shim
 * that would otherwise make the module unresolvable as a promise, and a second
 * implementation of either is a second chance to get it wrong.
 *
 * The bundle carries its wasm inline as a data URI, so there is no sibling file
 * to locate and no `locateFile` to supply.
 */

const SOURCE = "/node_modules/@techstark/opencv-js/dist/opencv.js";

function loadClassicScript(src) {
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    script.src = src;
    script.addEventListener("load", () => resolve());
    script.addEventListener("error", () =>
      reject(new Error(`${src} did not load`))
    );
    document.head.append(script);
  });
}

await loadClassicScript(SOURCE);

if (globalThis.cv === undefined) {
  throw new Error(`${SOURCE} loaded and set no globalThis.cv`);
}

export default globalThis.cv;

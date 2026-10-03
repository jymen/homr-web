# homr-web

A port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition engine, to TypeScript running in the browser. A
page of sheet music goes in, MusicXML comes out, and no server is involved:
the segmentation and transformer models run on the musician's own machine
through onnxruntime-web, on WebGPU where the browser has it and on
WebAssembly elsewhere.

On both public test pages the MusicXML equals the file homr 0.7.0 writes for
the same page, and the staff rectangles equal what the AbcMusicStudio server
reads from homr's staff-positions file.

## Usage

```bash
npm install homr-web
```

```ts
import { createRecognizer } from "homr-web";

// Starts a module Worker, picks WebGPU or WebAssembly and loads OpenCV. No model yet.
const recognizer = await createRecognizer({ baseUrl: "/models/" });
console.log(recognizer.backend); // "webgpu" | "wasm-threads" | "wasm"

const controller = new AbortController();
const result = await recognizer.recognizePage(pngBlob, {
  onProgress: ({ stage, done, total }) => console.log(stage, done, total),
  signal: controller.signal,
});

if (result.ok) {
  render(result.musicXml); // MusicXML 4.0, one part per voice
  console.log(result.staves); // [{ index, cx, cy, w, h }], page-normalised, top to bottom
} else {
  console.warn(result.error, result.log);
}

await recognizer.dispose();
```

`recognizePage` takes a `Blob` (any image the browser decodes), an
`ImageBitmap` or an `ImageData`, and never rejects. A failure is a result
with `ok: false` and one of these codes in `error`:

| `error` | meaning |
|---|---|
| `bad_input` | the image could not be decoded |
| `not_music` | homr found no staff or no notehead on the page |
| `engine_missing` | a model could not be downloaded or did not match its hash |
| `engine_failed` | anything else went wrong inside the engine |
| `busy` | this recognizer is still reading another page |
| `cancelled` | the signal was aborted, or the recognizer disposed |
| `timeout` | the signal was aborted with a `TimeoutError`, as `AbortSignal.timeout()` does |
| `worker_lost` | the Worker crashed; this recognizer answers only this from now on, so dispose it and create another |

The result has the shape of the AbcMusicStudio server's homr route (`engine`,
`ok`, `error`, `musicXml`, `log`, `durationMs`, `staves`, `texts`) plus
`backend`. `texts` is always empty: chord and title OCR is not ported yet.

Progress arrives in stages: `models` (bytes of the three models, on the first
page only), `segment` (tiles), `detect`, `dewarp` and `staff` (one each per
staff), and `xml`. On a first page `models` appears twice, before `segment`
for the segmentation model and after `detect` for the transformer, and a
page that is not music never downloads the transformer. On an Apple M-series laptop with WebGPU a page takes 5 to
7 seconds once the models are cached. Under Node on one WebAssembly thread it
takes about 40 seconds; WebAssembly threads in a browser were not timed. The models are about 100 MB on WebGPU and 160 MB on WebAssembly,
downloaded on the first page and cached by the browser after that.

`createRecognizer` rejects when the Worker cannot start, for example when
the browser has no module Workers, or when it has not answered within 30
seconds. One recognizer reads one page at a time; read the pages of a PDF one
after another. A page asked for right after a cancel is accepted and starts
once the Worker has finished the cancelled page's current step.

Options: `baseUrl` (required), `prefer` (the best backend to try, default
`"webgpu"`), `wasmPaths` (where onnxruntime-web's `.wasm` and `.mjs` files
are served, when your bundler does not place them itself), and
`createWorker` (see below).

### Bundlers

The library starts its Worker with
`new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`,
which Vite, Rollup and webpack 5 resolve. OpenCV.js is a UMD bundle, and
Vite's dev server does not pre-bundle a dependency reached only from a
Worker, so Vite needs it named:

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: { include: ["homr-web > @techstark/opencv-js"] },
});
```

That configuration was checked with a fresh `npm create vite` project (Vite
8.3), in `vite dev`, where Vite pre-bundled homr-web itself and the Worker
still resolved, and in `vite build` with `vite preview`. Not yet checked
under SvelteKit. For a bundler that cannot
resolve the Worker URL, pass your own Worker:
`createRecognizer({ baseUrl, createWorker: () => new Worker(url, { type: "module" }) })`,
where `url` serves the `homr-web/worker` entry.

### Hosting the models

Browsers cannot download the models from homr's GitHub release, because the
redirect carries no CORS header, so you serve them yourself, unchanged,
under `${baseUrl}/${sha256}/${filename}`. Every file is checked against its
SHA-256 before use.

| path under `baseUrl` | used on | bytes |
|---|---|---|
| `60f495496cb41473c0521d0811d8f44b9d5cff892d287974a8aebb3eaee2fa83/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16.onnx` | WebGPU | 28 667 207 |
| `9db62d5a6a13c8df2df321af3bcf72c7f81a95d4f876d4f0c202b28f8658087e/encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903_fp16.onnx` | WebGPU | 26 466 256 |
| `6ed36640db4ef5d223098b6d5efe4eda97c66b24a2c72faab8a018c749003a8d/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx` | WebAssembly | 57 311 361 |
| `4c16df852b3789f2676b0d49f0545dab0740e4005f7b472c5252add642f5d5eb/encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx` | WebAssembly | 52 861 122 |
| `3e10fd5ae52d0b86792721922fcd954c283a7ed365de7446425bdabe38f3e57d/decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx` | both | 47 299 551 |

The fp32 files are the ones homr 0.7.0 installs; the two fp16 files are
homr's own `onnx_checkpoints` release assets. `tools/fetch-models.sh`
collects all of them into `models/`.

### Cross-origin isolation

WebAssembly threads need `SharedArrayBuffer`, which a browser grants only to
a cross-origin isolated page. Serve the page that uses the library with

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

and serve the models and onnxruntime's files from the same origin, or with
`Cross-Origin-Resource-Policy: cross-origin`. Without isolation the library
still works: WebGPU needs none, and WebAssembly falls back to one thread,
which is slower. `recognizer.backendReason` says which case applies.

## What this reproduces

| | |
|---|---|
| homr release | 0.7.0, commit `8b5dcf7d7bdd1a47911dc0c661c573b957271eab` |
| segmentation model | `segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f` |
| transformer | `pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903` (encoder and decoder) |

`HOMR_VERSION` and `HOMR_COMMIT` are exported. Three behaviours differ from
running homr on the command line. The title is not detected, so
`<work-title>` is empty. A multi-page input is one call per page; homr 0.7.0
does not join pages either. And the image is decoded by the browser without
colour management and with its alpha channel dropped, as `cv2.imread` does,
so a transparent pixel reads as its stored colour: flatten a transparent
image onto white before passing it.

Two differences inside the pipeline are known and pinned by tests. opencv.js
fits another rectangle than opencv-python to two noteheads of the piano test
page (`docs/design/phase-5-minarearect.md`). The staff canvases differ from
homr's by 0.009 to 0.022 gray levels on average, from the resize of the arm64
OpenCV build. Neither changes a token on the test pages.

## How the port is tested

`tools/dump-golden.py` runs the pinned homr on each page in
`test/fixtures/` and writes the output of every stage to
`test/golden/<fixture>/`. Each TypeScript stage is tested against the
Python output of the stage before it, so a tolerance accepted in one stage
cannot hide a defect in the next, and `test/recognize-golden.test.ts` runs
each page from its PNG through the whole port against homr's
`page.musicxml`. `tools/dump-vectors.py` covers the branches no page reaches,
and `tools/dump-staves.sh` runs the AbcMusicStudio server's own parser on
homr's staff-positions file.

Only pages the AbcMusicStudio app typeset itself are committed as fixtures.
Scans, published pages and photographs stay in the git-ignored
`test/fixtures/local/`.

```bash
PYTHON=python3.12 ./tools/venv.sh   # once: homr 0.7.0 in .venv
./tools/fetch-models.sh              # the models, into models/
npm run golden && npm run vectors    # regenerate the oracle
npm ci && npm run check && npm run lint && npm test
npm run bench                        # the browser bench, with COOP and COEP
```

vitest is pinned to the 3.x line: npm 11.5 crashes in its peer-dependency
resolver on vitest 4 and 5.

`homr-web/internal` exports every module of the port, for the bench and for
anyone porting the next homr release. It is outside semver. The design
records are under `docs/design/`, and `docs/decisions.tsv` holds every
decision with its evidence.

## Licence

homr is licensed under the GNU Affero General Public License version 3, and
so is this port: [LICENSE](LICENSE), with the attributions in
[NOTICE](NOTICE). An application that serves this library to browsers is
conveying it and must offer its source; this repository,
https://github.com/jymen/homr-web, is that offer for the unmodified library.
The MusicXML a musician produces with it is theirs.

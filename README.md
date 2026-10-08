# homr-web

A port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition engine, to TypeScript running in the browser. A
page of sheet music goes in, MusicXML and the chord symbols above each staff
come out, and no server is involved: the segmentation, transformer and OCR
models run on the musician's own machine through onnxruntime-web, on WebGPU where the browser has it and on
WebAssembly elsewhere.

On the three public test pages the MusicXML, title included, equals the file
homr 0.7.0 writes for the same page, the staff rectangles equal what the
AbcMusicStudio server reads from homr's staff-positions file, and the texts
equal the ones the server's chord OCR reads with RapidOCR 3.9.2.

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
  render(result.musicXml); // MusicXML 4.0, one part per voice, homr's title in work-title
  console.log(result.staves); // [{ index, cx, cy, w, h }], page-normalised, top to bottom
  console.log(result.texts); // [{ staff, text, score, x0, y0, x1, y1 }], the chords above each staff
  console.log(result.tablature); // [{ index, lines, cx, cy, w, h }], tab systems kept out of the reading
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
| `tablature_only` | every system on the page is line tablature, which homr cannot read; `tablature` lists them |
| `engine_missing` | a model could not be downloaded or did not match its hash |
| `engine_failed` | anything else went wrong inside the engine |
| `busy` | this recognizer is still reading another page |
| `cancelled` | the signal was aborted, or the recognizer disposed |
| `timeout` | the signal was aborted with a `TimeoutError`, as `AbortSignal.timeout()` does |
| `worker_lost` | the Worker crashed; this recognizer answers only this from now on, so dispose it and create another |

The result has the shape of the AbcMusicStudio server's homr route (`engine`,
`ok`, `error`, `musicXml`, `log`, `durationMs`, `staves`, `texts`) plus
`backend` and `tablature`, which the server does not have yet. `texts` holds what RapidOCR reads in the strip from 1.9 to 0.05
staff heights above each staff, 3 % of the page wider on the left and 2 % on
the right: one entry per line of text, its box normalised to the page and
rounded to 4 digits, its score to 3, sorted by staff and then from left to
right, as the server answers. Lines scoring under 0.5 are dropped, as the
server drops them; `minTextScore` changes that threshold (the AbcMusicStudio
app keeps texts at 0.6 and above). Chord symbols come out as the recogniser
reads them, which is not always as typeset: on the chord test page `F♯m` reads
`7#m` and `Em` is missed twice, on the server as here. A failure of the OCR alone never fails the page: `texts` is
empty, `work-title` blank, and the reason is a line of `log`.

### Tablature

homr has no notion of tablature. Given a tab, it reads the tab lines as a
staff and invents notes on them, and a staff with its tab underneath comes
out as a piano grand staff whose bass staff is the tab. So before homr
reads the page, a guard finds the tab systems and paints them white:
`staves` and the MusicXML then hold the standard staves alone, and
`tablature` lists the tab systems (`lines`, 4 to 6, and the extent of the
lines, page-normalised like `staves`, `index` from the top). A page with
nothing but tabs answers `tablature_only` with its systems, rather than
notes that are not there.

A system is a run of four to six evenly spaced horizontal lines. Four or six
lines is a tab; five lines is a tab when the marks sitting on its lines read
as numbers, which RapidOCR's recogniser decides from a sample of at most
twelve of them. A page with no five-line group carrying such marks never
loads the recogniser. Nothing is read from the tab yet: no frets, strings,
tuning or rhythm. Detection is measured on 15 pages at 300 dpi (banjo,
mandolin and guitar PDFs, one of them a scan), 56 tab systems out of 56; it
finds nothing on a low-resolution phone photograph (lines 5 px apart), where
homr reads the tab as before.

To read the strips alone above staves you already have, for example the
server's:

```ts
const strips = await recognizer.readTextStrips(pngBlob, staves, { signal });
console.log(strips.texts); // musicXml is "", staves are the ones given
```

Progress arrives in stages: `models` (bytes of the six models, on the first
page only), `segment` (tiles), `detect`, `dewarp` and `staff` (one each per
staff), `ocr` (one per staff, and one for the title), and `xml`. On a first
page `models` appears three times, before `segment` for the segmentation
model, after `detect` for the transformer and before `ocr` for the three OCR
models, and a fourth time first of all when the tab guard needs the OCR
recogniser; a page that is not music never downloads the transformer, and
`{ ocr: false }` never downloads the OCR models, except the recogniser on a
page with a five-line tablature candidate. On an Apple M-series laptop with WebGPU a page takes 7 to
9 seconds once the models are cached, of which the OCR is about 3 seconds
(about 5 on WebAssembly threads); `readTextStrips` alone takes about 2.5
seconds. Under Node on one WebAssembly thread a page takes about 55 seconds,
about 14 of them OCR. The models are about 134 MB on WebGPU and 189 MB on
WebAssembly, the OCR's 32 MB included, downloaded on the first page and
cached by the browser after that.

`createRecognizer` rejects when the Worker cannot start, for example when
the browser has no module Workers, or when it has not answered within 30
seconds. One recognizer reads one page at a time; read the pages of a PDF one
after another. A page asked for right after a cancel is accepted and starts
once the Worker has finished the cancelled page's current step.

Options of `createRecognizer`: `baseUrl` (required), `prefer` (the best
backend to try, default `"webgpu"`), `wasmPaths` (where onnxruntime-web's
`.wasm` and `.mjs` files are served, when your bundler does not place them
itself), and `createWorker` (see below). Options of `recognizePage`:
`onProgress`, `signal`, `ocr` (default `true`) and `minTextScore` (default
`0.5`); `readTextStrips` takes the same less `ocr`.

### Bundlers

The library starts its Worker with
`new Worker(new URL("./worker.js", import.meta.url), { type: "module" })`,
which Vite, Rollup and webpack 5 resolve in a production build. Vite's dev
server needs three things named. OpenCV.js is a UMD bundle, and Vite does not
pre-bundle a dependency reached only from a Worker, so it must be included.
homr-web and onnxruntime-web must be excluded: pre-bundled, homr-web's
Worker URL is rewritten to `/node_modules/.vite/deps/worker.js`, which
answers 404, and onnxruntime-web looks for its `.wasm` under `.vite/deps`.

```ts
// vite.config.ts
export default defineConfig({
  optimizeDeps: {
    exclude: ["homr-web", "onnxruntime-web"],
    include: ["homr-web > @techstark/opencv-js"],
  },
});
```

That configuration was checked in the AbcMusicStudio app (SvelteKit, Vite
dev and production build) on 2026-10-04 with homr-web 0.1.0. With only the
`include`, `createRecognizer` rejects in dev with "homr-web's worker did not
load"; production builds work either way. For a bundler that cannot
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
| `090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f/PP-OCRv6_det_small.onnx` | both, OCR | 9 929 594 |
| `e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c/ch_ppocr_mobile_v2.0_cls_mobile.onnx` | both, OCR | 585 532 |
| `6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884/PP-OCRv6_rec_small.onnx` | both, OCR | 21 234 383 |

The fp32 files are the ones homr 0.7.0 installs; the two fp16 files are
homr's own `onnx_checkpoints` release assets; the three OCR files are the ones
rapidocr 3.9.2 ships in its wheel and loads by default (PP-OCRv6 small
detection and recognition, the PP-OCRv4 line of the v2.0 mobile direction
classifier), Apache 2.0. `tools/fetch-models.sh`
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
| chord and title OCR | RapidOCR 3.9.2's default pipeline, and the AbcMusicStudio server's strip geometry (`omr_chord_ocr.py`) |

`HOMR_VERSION` and `HOMR_COMMIT` are exported. Three behaviours differ from
running homr on the command line. A multi-page input is one call per page; homr 0.7.0
does not join pages either. The image is decoded by the browser without
colour management and with its alpha channel dropped, as `cv2.imread` does,
so a transparent pixel reads as its stored colour: flatten a transparent
image onto white before passing it. And the title keeps letters of any
alphabet, apostrophes and hyphens, where homr keeps only `a` to `z` and
digits: homr turns "Marche des élèves" into "Marche des l ves", and drops a
title in another alphabet as a tempo marking. A title in `a` to `z` comes
out exactly as homr's (`test/title-text.test.ts`).

Two differences inside the pipeline are known and pinned by tests. opencv.js
fits another rectangle than opencv-python to two noteheads of the piano test
page (`docs/design/phase-5-minarearect.md`). The staff canvases differ from
homr's by 0.009 to 0.022 gray levels on average, from the resize of the arm64
OpenCV build. Neither changes a token on the test pages.

## How the port is tested

`tools/dump-golden.py` runs the pinned homr on each page in
`test/fixtures/` and writes the output of every stage to
`test/golden/<fixture>/`, homr's title included. `tools/dump-texts.sh` runs
the AbcMusicStudio server's own chord OCR code on each page into
`texts.json`, and `tools/dump-ocr.py` writes every RapidOCR stage of each
strip into `ocr.json`. Each TypeScript stage is tested against the
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
./tools/fetch-models.sh              # the models, into models/ (from the venv)
# or, with no venv, as CI does: download and hash-check all nine
HOMR_WEB_MODELS_RELEASE=models-homr0.7.0 ./tools/download-models.sh
npm run golden && npm run vectors    # regenerate the oracle
npm run ocr                          # RapidOCR's stages per strip
./tools/dump-texts.sh <AbcGoDb checkout> <server homr venv>   # the server's texts
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

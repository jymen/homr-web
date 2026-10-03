# homr-web

A port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition engine, to TypeScript running in the browser.
A photographed or scanned page of sheet music goes in, MusicXML comes out,
and no server is involved: the segmentation and transformer models run on
the musician's own machine through onnxruntime-web, on WebGPU where the
browser has it and on WebAssembly elsewhere.

**Status: phase 6, staff canvases.** A page goes from pixels to homr's staffs,
and each staff to the 1280 by 256 canvas the transformer reads. It
passes through autocrop, PIL's bicubic resize and CLAHE to byte-identical
output, is tiled the way homr tiles it, and comes back out of the segmentation
model as the five masks, which agree with homr's own pixel for pixel on the
test page. From the masks the port fits homr's boxes, finds the staff lines,
pairs noteheads with stems, puts the notes on their staffs and joins braced
staffs into grand staffs. `detectStaffsInImage` reproduces homr's
`multistaffs.json` and `notes.json` on both public pages, a single-staff tune
and a piano page of four braced systems, and `formatStaffPositions` writes
homr's staff-positions file byte for byte.

Two differences from homr are known and pinned by tests rather than hidden.
opencv.js fits another rectangle than opencv-python to two noteheads of the
piano page, because the native arm64 build uses a fused multiply-add that
WebAssembly does not have (`docs/design/phase-5-minarearect.md`). And
noteheads whose centres are one float32 step apart in height can come out in
another order. The notes found, their positions and the staffs are the same.

`staffCanvases` cuts each staff out of the page, dewarps it with homr's
piecewise affine transform and centres it on the encoder canvas. On both
public pages its canvases differ from homr's by a mean of 0.009 to 0.022 gray
levels, nearly all of it cv2.resize: the arm64 build of OpenCV resizes through
the KleidiCV HAL, which lands up to two levels from the opencv.js result.
opencv.js has no `Subdiv2D`, so the triangulation is delaunator over the same
three outer vertices Subdiv2D adds; given homr's own input, the warp is exact
on seven of the eight canvases and three pixels off on the eighth. Both public
pages are typeset and nearly flat; a photograph is the case the dewarp exists
for, and no photograph is among the public fixtures.

Underneath that: phase 1's data model (planes, rotated boxes with homr's angle
normalisation, staffs and symbols, the six decoder vocabularies, and the
decoders that turn a golden dump into those types while checking every derived
value Python stored), and phase 2's model layer (a generated manifest of the
eight artifacts with their hashes, the runtime probe that picks WebGPU or
WebAssembly, the verifying store and cache, and the sessions).

`npm run bench` serves a page that runs the chain in the browser and draws the
staffs it found over the page. Nothing after detection is ported: no dewarp,
no transformer, so there is no MusicXML yet.

The design records are under `docs/design/`, one or two files per phase, and
`docs/decisions.tsv` holds every decision with its evidence. The phase plan is
in the AbcMusicStudio repository under `docs/homr-web-plan/`.

## What this reproduces

| | |
|---|---|
| homr release | 0.7.0, commit `8b5dcf7d7bdd1a47911dc0c661c573b957271eab` |
| segmentation model | `segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f` |
| transformer | `pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903` (encoder and decoder) |

SHA-256 of every model file is recorded per fixture in
`test/golden/<fixture>/meta.json` by the golden dumper. The models are
homr's, downloaded unchanged; this repository never contains them.

## Fixtures: public pages only

Only pages the AbcMusicStudio app typeset itself are committed; they carry
no third-party rights. Every other page (scans, published collections,
photographs) stays private in `test/fixtures/local/`, which git ignores,
with its golden data in `test/golden/local/`. The dumper and the tests
handle both directories the same way, so CI runs on the public pages and a
developer's machine on all of them. Details in `test/fixtures/SOURCE.md`.

## Layout

```
src/image/        plane.ts (Mask, GrayImage, ClassMap, ColorImage and the numpy-shaped helpers), numeric.ts (Python round, //, str(float), numpy median/std)
src/geometry/     boxes.ts (PointList, RotatedRect, RotatedBox, Ellipse, AxisBox)
src/model/        staff.ts, symbols.ts, pipeline.ts, constants.ts (model.py's data, the stage contracts)
src/models/       manifest.ts (generated), backend.ts (the runtime probe), store.ts, cache.ts, session.ts, dtype.ts (the fp16 codec)
src/cv/           opencv.ts (the loader and Mat lifetime), default-source.ts (opencv.js in its own chunk)
src/dewarp/       delaunay.ts (Subdiv2D's triangulation over delaunator), piecewise-affine.ts, staff.ts (the control points)
src/pipeline/     detect.ts (masks to multi staffs), staff-image.ts (multi staffs to canvases), staff-positions.ts
src/segmentation/ preprocess.ts (autocrop, resize, CLAHE), resize.ts (PIL's bicubic in fixed point), tiles.ts (the grid and the merge), segment.ts, worker.ts
src/transformer/  vocabulary.ts (generated tables), symbol.ts (EncodedSymbol)
src/golden/       decode.ts (Python dump to domain types, with derivation checks), page.ts (one fixture behind one object)
src/result.ts     what recognizePage will answer with
test/             one test file per module, golden.test.ts over every fixture
tools/            venv.sh, dump-golden.py, dump-vectors.py, gen-vocabulary.mjs, gen-manifest.mjs, fetch-models.sh
```

## How the port is tested

Every stage of homr's pipeline is a function from arrays to arrays.
`tools/dump-golden.py` runs the pinned homr on each page in
`test/fixtures/` and writes the output of every stage to
`test/golden/<fixture>/`. Each TypeScript stage is tested against the
Python output of the stage before it, never against the TypeScript output,
so a tolerance accepted in one stage cannot hide a defect in the next.
`tools/dump-vectors.py` runs small hand-built inputs through the same Python
and writes them to `test/golden/vectors/`, for the branches no page reaches.

```bash
PYTHON=python3.12 ./tools/venv.sh   # once: homr 0.7.0 in .venv, fp32 models
npm run golden                       # regenerate test/golden from test/fixtures
npm run vectors                      # regenerate test/golden/vectors
node tools/gen-vocabulary.mjs        # then refresh the token tables in src/transformer/vocabulary.ts
npm ci && npm run check && npm run lint && npm test
```

vitest is pinned to the 3.x line: npm 11.5 crashes in its peer-dependency
resolver (`Cannot read properties of null (reading 'edgesOut')`) on vitest 4
and 5, reproduced 2026-09-25 in an empty directory, so the pin is npm's, not
ours.

The dumper is deterministic: running it twice produces byte-identical
files, and the test suite fails if a golden directory does not match the
fixture it was produced from.

## Licence

homr is licensed under the GNU Affero General Public License version 3, and
so is this port: [LICENSE](LICENSE), with the attributions in
[NOTICE](NOTICE). An application that serves this library to browsers is
conveying it and must offer its source; this repository is that offer for
the unmodified library. The MusicXML a musician produces with it is theirs.

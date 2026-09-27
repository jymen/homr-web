# homr-web

A port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition engine, to TypeScript running in the browser.
A photographed or scanned page of sheet music goes in, MusicXML comes out,
and no server is involved: the segmentation and transformer models run on
the musician's own machine through onnxruntime-web, on WebGPU where the
browser has it and on WebAssembly elsewhere.

**Status: phase 1.** The repository holds the scaffold, the pinned Python
oracle, the golden fixtures, and the data model every later phase reads and
writes: planes (the byte-per-pixel image layout opencv.js wraps without a
copy), rotated boxes with homr's angle normalisation, staffs and symbols,
the six decoder vocabularies, and the decoders that turn a golden dump into
those types while checking every derived value Python stored. No algorithm
is ported yet. The design record is `docs/design/phase-1-types.md`; the
phase plan is in the AbcMusicStudio repository under `docs/homr-web-plan/`.

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
src/transformer/  vocabulary.ts (generated tables), symbol.ts (EncodedSymbol)
src/golden/       decode.ts (Python dump to domain types, with derivation checks), page.ts (one fixture behind one object)
src/result.ts     what recognizePage will answer with
test/             one test file per module, golden.test.ts over every fixture
tools/            venv.sh, dump-golden.py, gen-vocabulary.mjs
```

## How the port is tested

Every stage of homr's pipeline is a function from arrays to arrays.
`tools/dump-golden.py` runs the pinned homr on each page in
`test/fixtures/` and writes the output of every stage to
`test/golden/<fixture>/`. Each TypeScript stage is tested against the
Python output of the stage before it, never against the TypeScript output,
so a tolerance accepted in one stage cannot hide a defect in the next.

```bash
PYTHON=python3.12 ./tools/venv.sh   # once: homr 0.7.0 in .venv, fp32 models
npm run golden                       # regenerate test/golden from test/fixtures
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

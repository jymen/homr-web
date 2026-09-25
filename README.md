# homr-web

A port of [homr](https://github.com/liebharc/homr), Christian Liebhardt's
optical music recognition engine, to TypeScript running in the browser.
A photographed or scanned page of sheet music goes in, MusicXML comes out,
and no server is involved: the segmentation and transformer models run on
the musician's own machine through onnxruntime-web, on WebGPU where the
browser has it and on WebAssembly elsewhere.

**Status: phase 0.** The repository holds the scaffold, the pinned Python
oracle and the golden fixtures; nothing of homr is ported yet. The plan is
in the AbcMusicStudio repository under `docs/homr-web-plan/` and moves here
with phase 1.

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

# Phase 3 design: preprocess, the tiling, and segnet in a Worker

Back to [phase 2](phase-2-models.md). The phase plan is in the AbcMusicStudio
repository under `docs/homr-web-plan/phase-3-segnet.md`; `testing.md` beside it
fixes the tolerances and the Python-to-TypeScript determinism traps.

This record covers the **library half** only: the opencv.js loader, the three
preprocess stages, the tiling and merge, `segmentPage` and the Worker, each with
golden tests on Node. `bench/`, the browser, WebGPU and the timings the go/no-go
gate needs are deliberately not here; an earlier attempt at this phase stalled
because it carried them together.

## Problem

The five segmentation masks are the first real algorithm in the port and the
phase the plan calls its go/no-go gate. Everything downstream consumes them, and
they come out of a chain where each link can be wrong in a way the next link
hides: autocrop decides what page is even being read, PIL's resize decides what
pixels the model was trained to see, CLAHE decides their contrast, the tiling
decides what the model is shown, and only then does a model run.

Three of those five links have no tolerance at all. The golden dump fixes
`autocropped.png`, `resized.png` and `preprocessed.png` as exact byte arrays,
because they are deterministic integer image processing and any drift in them is
a defect, not a numeric accident. The masks get 99.9 % per-class pixel agreement,
because onnxruntime's kernels are not Python's and an argmax over near-ties flips
a pixel now and then.

This phase's scope was cut deliberately: the library half only. No `bench/`, no
browser, no WebGPU. The previous attempt bundled them and stalled.

## Usage (caller's view)

```ts
import {
  loadOpenCv, preprocessPage, segmentPage, SegmentationWorker,
  MASK_CLASS_NAMES, planeAgreement,
} from "homr-web";

// The library half, as a bench page or a test drives it.
const { preprocessed, resized } = await preprocessPage(pageAsBgr);
const session = await store.open("segnet", { batch: 8 });
const result = await segmentPage(session, preprocessed, { batch: 8 });
result.masks.staff;      // a Mask, 1 where segnet said staff
result.classes;          // the class map the five masks are split from

// The Worker, which owns the runtime, the store, the session and both halves.
const worker = new SegmentationWorker({ baseUrl, cache, batch: 8 });
const started = await worker.start();
started.backend; started.reason; started.modelReason;  // the go/no-go gate quotes these
started.numThreads;     // what the runtime *applied*, which is what a timing is read against
const page = await worker.segment(pageAsBgr);
page.preprocessMs; page.segmentMs;
await worker.close();

// In the application's own worker script, and this is the whole entry:
serveSegmentation(self, worker);
```

Three things a caller does not do. It does not choose a dtype: the tile writer
reads `session.inputSpec("input").type` and writes float32 or IEEE halves from
that. It does not pad the last batch: `segmentPage` does, because
`freeDimensionOverrides` pins the batch dimension. And it does not free anything:
every Mat the preprocess allocates belongs to a scope that releases it.

## Shape

### Module map

```
src/cv/opencv.ts            loadOpenCv, CvError, MatScope, withMatScope, the boundary parse
src/cv/default-source.ts    the static namespace import, in its own chunk, imported lazily
src/segmentation/resize.ts  PIL's bicubic resample in fixed point, and calc_target_image_size
src/segmentation/preprocess.ts  autocrop, applyClahe, preprocessPage, findPaperRect
src/segmentation/tiles.ts   tileGrid, extractTile, writeTileInto, tileCoverage, mergeTileClasses
src/segmentation/segment.ts segmentPage: the batch loop, the argmax, the merge
src/segmentation/worker.ts  SegmentationWorker and the MessagePort transport
test/support/opencv.ts      the createRequire source the Node tests inject
```

Phase 1 already owned more of this than it looked like. `argmaxPlanes`,
`maskOfClass`, `createSegmentationResult`, `planeAgreement`, `SEGNET_INPUT` and
`MASK_CLASSES` are all its work, so this phase writes no argmax, no mask split and
no agreement metric. The one thing it had wrong was reachable rather than absent:
`GoldenPage.mask(name, filtered)` could not open a filtered file.


## The two questions the plan left open, answered from the Python

### The sixth mask is not a segnet class

`SegmentationResult` names five masks while the golden set holds six raw
`mask-*.png` files and segnet emits six classes. Those are three different sixes
and none of them contradicts the others.

`inference_segnet.inference` argmaxes six channels and maps five of them:
`merged == 1` is `stems_rests`, 2 `notehead`, 3 `clefs_keys`, 4 `staff`,
5 `symbols`. **Class 0 is background and has no mask** — that is the sixth class.
Phase 1's `MASK_CLASSES` table already records exactly this, with the channel
numbers, and it is right.

The sixth golden file, `mask-brace_dot.png`, is not segnet output at all.
`tools/dump-golden.py` writes it at line 186, immediately before
`boxes-brace_dot.json`, from `brace_dot_detection.prepare_brace_dot_image(symbols,
staff)`: `cv2.subtract(symbols, staff)`, an erode with a 1x5 ellipse and a dilate
with a 5x35 one. It is a derived image belonging to the brace and grand-staff
detection of phase 5, and `MASK_CLASS_NAMES` correctly excludes it. Nothing in
phase 3 produces it and nothing here should.

### `mask-filtered-*.png` is phase 4's input, not a segnet variant

`dump-golden.py` dumps the five masks twice. The raw set comes straight off
`get_predictions`. Then it runs `noise_filtering.filter_predictions` and
`staff_detection.make_lines_stronger(staff, (1, 2))` and dumps the same five
names again as `mask-filtered-*`. That second set is the masks **as
`predict_symbols` receives them**, which makes it the input of phase 4's box
extraction and not a second opinion about phase 3's output.

So phase 3 tests against the raw five. Measured on the Kesh page, which confirms
the reading: four of the five filtered files are byte-identical to their raw
counterparts, because `filter_predictions` found fewer than the threshold of
noisy cells and returned the prediction unchanged; only `staff` differs, at
0.99401 pixel agreement, 135,163 set pixels against 166,393, which is
`make_lines_stronger` dilating. There is no `mask-filtered-brace_dot.png`,
consistent with brace_dot not being one of the five.

A side effect of establishing this: `GoldenPage.mask(name, filtered)` could not
reach the filtered files at all. Its memo was keyed on one argument and bound
past the second, so every `mask(name, true)` in the repository silently returned
the raw mask, and `test/golden.test.ts`'s "loads the five raw and five filtered
masks" test passed on it because it only checked the size. Fixed here, and the
test now asserts the two are different data.

## PIL's resize is a fixed-point filter, and calling a bicubic resize is not enough

The plan carried a contradiction, bilinear in one line and bicubic in another.
It is bicubic: `resize.py` calls `Image.resize` with no `resample`, and Pillow's
default has been `BICUBIC` since Pillow 10. Resolving it does not get the port
very far, because `cv.resize` with `INTER_CUBIC` is a different function from
PIL's and the segnet was trained on PIL's output.

PIL resamples in fixed point. The reproduction needs all of it: `PRECISION_BITS
= 22`; the bicubic kernel with `a = -0.5` and support 2.0; per-output-pixel
coefficient precomputation with `xmin`/`xmax` bounds derived from
`center = (xx + 0.5) * scale`; the coefficients normalised to integers by
`trunc(±0.5 + w * (1 << 22))`; two passes, the horizontal one writing a temp
holding only the rows the vertical pass will read, with every vertical bound
shifted by the first row's; each accumulator seeded with `1 << 21` and closed
with `>> 22` and a clamp to 0..255.

That is not an optimisation of a float algorithm, it is the algorithm. A float
version that rounds at the end differs in the last bit on a meaningful fraction
of 15.6 million bytes, and the golden is byte-exact, so the difference is visible
immediately. The shift is a 32-bit shift because PIL accumulates in a C `int`.

`calc_target_image_size` rounds with Python's `round`, so `roundHalfEven` and
never `Math.round`; `testing.md` lists `resize.py` among the sites for that
reason. Target width 1920, and the Kesh page's 2481 x 3509 becomes 1920 x 2716.

## The optimisation this phase did not take, and the measurement that killed it

The plan flags half a second in the resize as a real line item against a 10 second
budget, and it is right to. An earlier draft of this phase acted on it: since every
page the port will see is a scan or a render whose three BGR channels are equal,
and since `COLOR_BGR2GRAY` of an achromatic pixel returns the channel value
exactly, the three-band resize is three copies of one computation and could be one.
Measured on the Kesh fixture, its `autocropped.png` and its `resized.png`: all
three are achromatic, R equals G equals B at every pixel, so the premise holds.

Then the numbers arrived on a quiet machine and the optimisation stopped being
worth it. The resize is **397 ms**, not the 1.1 s an earlier spike measured while
a subagent had a core; segnet is **525 ms a tile, 28 s for the page's 54 tiles**,
on Node with one wasm thread. A 3x on the resize therefore saves about 260 ms of
28,000, and it buys that with a branch, a full pass to detect achromacy, a
replicate pass, and a correctness argument about cv2's fixed-point grayscale
weights. That does not earn its place, so the port is the faithful one: three
bands, no branch.

The option is recorded rather than discarded. The two facts it rests on are
asserted in the tests, so they are measured rather than remembered, and
`preprocessPage` carries a comment saying that the colour page is computed for a
reader that does not exist yet. homr uses `InputPredictions.original` only for its
debug overlays, segnet is fed the CLAHE gray, and the resized colour page is
therefore 397 ms of work whose only current consumer is the golden test. If a
later phase finds the preprocess on the critical path, the gray-only route and its
premise are already worked out.

Two notes on the numbers themselves. The first is that **525 ms a tile does not
match what is written down**: `src/models/manifest.ts`'s comment records "159 ms
fp32 and 177 ms fp16 per tile under Node on one thread", and that figure did not
reproduce here, at batch 8 with `freeDimensionOverrides` pinning the dimension and
`numThreads` at 1. It is the one measured claim in phase 2 this phase could not
confirm, and it matters because the go/no-go gate is a timing gate. The second is
that every timing in this record is a single machine under no particular control,
and the load average moved a page's resize by a factor of three within one session.
They are orders of magnitude, not benchmarks.

## opencv.js, and the traps in loading it

`@techstark/opencv-js` 4.12.0 is pinned and stays pinned: `minAreaRect`'s angle
convention changed in OpenCV 4.5.1 and homr's geometry depends on the 4.x
behaviour. The loader's test asserts `getBuildInformation()` reports 4.12.0, so a
silent bump fails there rather than in a phase 5 tolerance.

Loading it cost more than it looks like it should, and the reasons are worth
writing down because two of the three present as a hang with no output rather than
as an error.

1. **`module.exports` is thenable, its `then` never settles, and calling it
   starves the event loop.** Measured: a `then` callback had not fired 8 s after
   the module was ready, while `onRuntimeInitialized` fired at 220 ms. The
   starvation is the part that makes this expensive to diagnose, because
   **vitest's own test timeout does not fire either**: a run sits there with no
   output and no failure, which is not a shape anyone reads as "a thenable was
   adopted". Every hang in this investigation was this.

   It is not only `await theModule` that does it. Resolving *any* promise with the
   module calls its `then`, so a source typed `() => Promise<unknown>` hangs the
   moment its `createRequire` implementation resolves with `module.exports`, and an
   `async function acquire(): Promise<OpenCv>` cannot `return` the module either.
   Two things follow, and both are in the types rather than in a comment asking
   callers to be careful. A source hands back `AcquiredOpenCv`, a `{ module }` box,
   which an ESM `import()` provides for free because a namespace is an ordinary
   object and a `require` must do by hand. And the loader **clears `then`** before
   returning, after which the module travels through `Promise<OpenCv>` like
   anything else; the write is guarded on `typeof then === "function"`, which is
   what makes it safe on the browser path, since a module namespace refuses a write
   and a namespace can never be the object carrying `then` or the `import()` would
   not have resolved. `then` is emscripten's legacy "the module is a promise" shim
   and nothing wants it.

   The readiness signal is the hook; the readiness *test* is
   `typeof m.Mat === "function"`, because Emscripten fires the hook once and a
   loader that installs it on an already-initialized module waits for a callback
   that will never come. `then` survives initialization, so it cannot be used as
   the readiness test.
2. **`await import("@techstark/opencv-js")` never resolves under vitest.** Under
   plain Node it resolves in 138 ms and the runtime is ready 80 ms later; inside a
   vitest test the same line produces no output and no timeout.
   `server.deps.external` and `server.deps.interopDefault: false` both change
   nothing. `createRequire(import.meta.url)` works: 88 ms to require, 161 ms to
   ready, with no config change at all. So the module acquisition is an injectable
   `OpenCvSource`, defaulting to the dynamic import for the browser and the Worker,
   and `test/support/opencv.ts` passes the `createRequire` one. This mirrors the
   golden reader, which already takes node:fs and pngjs by injection.
   The consequence is honest and worth stating: **the default browser path is not
   covered by any Node test.** The bench page is its first exercise, and that is a
   later piece of work.
3. **There is no `createCLAHE` on the JS build.** The constructor is
   `new cv.CLAHE(1.0, new cv.Size(8, 8))`. `calcHist`, `findContours`,
   `contourArea`, `boundingRect`, `threshold`, `morphologyEx`, `cvtColor`,
   `matFromArray`, `Mat.ones` and `MatVector` all exist.

The acquired value is untyped, so it is parsed at the boundary by a predicate
that checks the members this port actually uses and throws naming the first
missing one. That is not ceremony: it is what catches a wrong or partial build,
which is the failure the pin exists to prevent.

A `Mat` is freed by hand and a leak is wasm memory that never returns, so
lifetime is owned in one place rather than by a `.delete()` at every call site.

## Autocrop's coverage gap, stated plainly

On the Kesh fixture **autocrop does nothing, and a function that did nothing at
all would pass an equality check against `autocropped.png`.** Measured: the
dominant channel-0 value is 255, so the threshold is 225; `findContours` returns
exactly one contour, of area 8,699,840; its bounding rect is
`{x: 0, y: 0, width: 2481, height: 3509}`, the whole page. Two independent things
then make the output the input: the rect is the full page, and
`is_full_page_view` (`x < width * 0.25 || y < height * 0.25`) is true because x is
0.

So the golden comparison for this stage is vacuous and must not be read as
coverage. The tests answer it three ways. They assert the *decision* rather than
only the output, pinning the contour count, the winning area, the rect and which
guard returned the input; they carry a one-sentence comment on the test saying the
green tick is not evidence that cropping works; and the branch that actually crops
is covered by a synthetic page, a white sheet on a dark border placed so both x
and y clear a quarter of the page.

No fixture with a real margin exists in this repository today. The public set is
one typeset page, and `test/fixtures/local/` is git-ignored and absent here. The
first photographed page added will exercise it; until then the synthetic case is
the only coverage of the cropping branch, and that is a known gap rather than a
silent one.

## The tiling, and the merge that invents classes

`main.py` passes `step_size = 320` with `win_size = 320`, so interior tiles do not
overlap. The last row and the last column are pulled back inside the page by
`y = min(y_loop, h - win_size)`, which is the only source of overlap. On the
1920 x 2716 page that is nine rows by six columns, **54 tiles**, and I measured
the overlap: 4,899,840 pixels at weight 1 and 314,880 at weight 2, the 164 rows
from 2396 to 2559 that the last row re-covers. No pixel has weight 4 on this page,
because 1920 is an exact multiple of 320 and the last column never pulls back.

`merge_patches` accumulates the **argmax class indices**, not the logits, divides
by the count and casts back to an integer. So two overlapping tiles that disagree
produce `trunc((a + b) / 2)`, which can name a class neither tile chose: 4 and 5
give 4, and 0 and 5 give 2. That is homr's behaviour on 314,880 pixels of every
page and the port reproduces it rather than improving on it. Sums reach at most 20
and weights at most 4, so `Uint8Array` accumulators and integer arithmetic are
exact and no float is involved.

`extract_patch` pads with **255**, and it copies into the patch's top-left corner
whatever the clamping did, so a tile with a negative origin is pinned to the
corner with the padding on the far side. `merge_patches` reads `patch[:ph, :pw]`,
from the same corner.


## What this phase found and did not fix

`npm run build` has been broken since phase 2, and no test notices because
`npm run check` uses `tsconfig.json` while the build uses `tsconfig.build.json`,
which sets `"types": []`. That drops `@webgpu/types`, and `src/models/backend.ts`
names `GPUAdapter`, `GPU` and `navigator.gpu` in its public types, so the build
fails with five `TS2304`/`TS2339` errors. Confirmed pre-existing by building with
phase 3's export line removed, and confirmed as the cause by building once with
`"types": ["@webgpu/types"]`, which succeeds.

It is left alone deliberately, because the one-line change is not obviously the
right fix and the choice is phase 2's. `@webgpu/types` is a devDependency, so a
published `.d.ts` that names `GPUAdapter` asks every consumer to have those types.
Either it moves to `dependencies`, or `RuntimeProbe` stops exposing a `GPUAdapter`
in the library's public surface. That is a decision about what the package
promises, not a tsconfig tweak.

## Three sharp edges the preprocess port had to work around

**`cropPlane` cannot return a plane's last column, and autocrop's crop needs it.**
`cropPlane` (`src/image/plane.ts`) is the faithful port of homr's own
`crop_image`, whose `_limit_x` clamps every bound through `Math.min(size - 1, ...)`.
So `cropPlane(page, x, y, x + w, y + h)` with `x + w === page.width` silently
yields `w - 1` columns. autocrop's crop is a plain numpy slice `img[y:y+h, x:x+w]`
with no clamping, and the rect it is handed genuinely does reach the page edge
whenever the paper does, because `MORPH_ERODE`'s default border value is `+inf`
(measured): eroding an all-white Mat with a 9x9 kernel leaves every border pixel
at 255, so the contour includes the edge. autocrop therefore does its own row
copy, and the synthetic test whose rect ends exactly at the page width pins the
trap rather than merely avoiding it.

`cropPlane` is correct for its own callers and stays as it is. It is worth knowing
that its name is wider than its contract.

**The dominant-value histogram must be cv2's, not a counting loop.** `calcHist`
accumulates into float32, so beyond 2^24 samples of one value its counter stalls
while an exact integer count keeps going, and the two can then pick different
winners on a large uniform page. The port calls `calcHist`, because reproducing
homr means reproducing that. They agree on this page (both 255, at 8,431,774
samples, which is under the threshold).

**A Mat is filled by writing `mat.data`, not by `matFromArray`.** `matFromArray`
wants a JS array, which for a 26 MB page is 8.7 million boxed numbers.

## The Worker, and one thing it cannot do that the plan assumed

`docs/design/phase-2-models.md` already sketched this class, at
`src/segmentation/worker.ts`, and that sketch won over the phase-3 plan's bare
`src/worker.ts`. Two departures from it, both with a reason that outlives them.

`segment` takes the **BGR page**, not an already-preprocessed `GrayImage`: the
point of a Worker is that the whole CPU-heavy job leaves the main thread, and
preprocess is a second of it. `segmentPage` is the pure function on a gray page
that the sketch was describing, and it is what the golden test drives with the
Python `preprocessed.png`. And `start()` takes no argument, reading its
configuration from the constructor, so a second `start()` cannot ask for settings
the first already froze; `startRuntime` refuses that disagreement anyway and there
is no reason to offer a caller the chance to trip it.

**There is no `self.addEventListener` auto-install, and the reason is the first
departure's consequence.** Since `baseUrl` and the cache reach the class only
through its constructor, only the application's own worker script can build one, so
a module that installed a listener on import would have no configuration to start
from. The documented entry is one line of `serveSegmentation(self, worker)` in that
script, and importing the module does nothing anywhere.

**The transport's types were wrong in a way the Node tests could not see.** A
browser `MessagePort` and a `DedicatedWorkerGlobalScope` were not assignable to the
port interface, because the DOM's `postMessage` overload declares
`transfer: Transferable[]` with no `undefined` in it while `@types/node`'s takes an
optional `transferList`. So the Node transport test was green over a documented
browser entry that could not compile. `transfer` is now required and always passed
as an array, and `test/segment.test.ts` carries a compile-time assignability
assertion over both browser types, so the next divergence fails the typecheck
rather than the bench page.

The short final batch is padded with **blank 255 tiles** rather than left holding
the previous batch's bytes. Both are correct, since the extra outputs are
discarded, but the tensor handed to the model should not depend on the page's tile
order: a run must be reproducible from its inputs, and a dump of the last batch
should show tiles that are on the page. 255 is also what `extractTile` pads an
off-page tile with, so the padding reads as empty paper in both dtypes instead of
as ink in the margin.

Nine defects in this unit were found and fixed after its first green run, and the
ones worth remembering are all lifecycle. `close()` during an in-flight `start()`
could not reach the store, so it could neither abort the 40 MB download nor return
until it landed. `close()` could release the `InferenceSession` under a running
`segmentPage`, which frees a handle onnxruntime is reading rather than cancelling a
job, and it was reachable from the transport with no misuse at all by posting
segment then close; segments now chain on one queue that `close()` drains, which
also serialises two concurrent segment commands. A second concurrent `close()`
resolved before the first had released anything. And `segmentPage` validated its
output tensor by element count alone, so a re-exported model emitting NHWC would
have handed back the right number of floats in the wrong order and produced six
garbage planes in a mask that still looks like a mask; the dims are compared now.

## The result: the masks are exact, not merely inside the tolerance

`testing.md` allows the masks 99.9 % per-class pixel agreement, because
onnxruntime's kernels are not Python's and an argmax over near-ties can flip.
It was not needed. Fed the Python `preprocessed.png`, the fp32 segnet on
onnxruntime-web's WebAssembly backend reproduces all five Python masks at
**1.000000000 agreement, zero differing pixels out of 5,214,720 each**,
reproduced across five runs.

That is worth more than a passing test. It means the whole chain from the tile
grid through the tensor layout, the argmax tie-break and the merge's truncation is
right in every detail, because any one of them being wrong would show as a band or
a scattering of pixels rather than as nothing. Phase 4 consumes the Python masks
anyway, per the Python-input rule, so this cannot cascade either way; but it does
retire the worry the tolerance was written for, on the wasm path.

The fp16 artifact, measured on the same wasm provider so that precision is
separated from the execution provider (which is what `Placement` splits
`artifactsFor` from `provider` for), does flip pixels, and stays far inside the
bar: page-wide class maps agree with fp32 at **0.999990220**, 51 pixels of
5,214,720. Against the Python masks it gives staff 0.999991562 (44 px), symbols
1.000000000, stemsRest 0.999999425 (3 px), notehead 0.999999041 (5 px) and
clefsKeys 0.999999616 (2 px). Phase 2 had measured 0.999961 on the inkiest single
tile of this page; the page as a whole is better than that, so that tile is the
worst case rather than a typical one. The test is opt-in behind `HOMR_FP16_PAGE=1`
because it is a second full-page run, and its skip title says so.

## Measurements, and why they are not benchmarks

**This machine cannot produce a trustworthy timing number.** Measuring the same
page's 54 tiles four times over one session gave 525, 1380, 1582 and 1705 ms per
tile, a factor of three, and the cause was visible in `ps`: Path Finder at 85 %,
Backblaze's `bztransmit` at 80 %, pCloud Drive at 69 % and `mediaanalysisd` at
64 %, with load averages between 7 and 23. Every figure below is an order of
magnitude. The go/no-go gate needs the bench page on a controlled run, and it
should not be settled from a Node number at all.

| stage | time | against the golden |
|---|---|---|
| autocrop | 245 ms to 1.1 s | 0 differing bytes of 26,117,487 |
| resize (PIL bicubic, three bands) | 397 ms to 3.0 s | 0 differing bytes of 15,644,160 |
| CLAHE | 95 ms to 550 ms | 0 differing bytes of 5,214,720 |
| segnet, 54 tiles, batch 8, one wasm thread | 28 s to 105 s, 63 s on the quietest run | 1.000000000 per class |
| segnet, the same page at nine wasm threads | 27 s | |
| segnet session open | 188 ms to 650 ms |  |

So a page is **half a minute at best and under two minutes at worst on Node**, and
segnet is about 97 % of it whichever end you take. One thread's own spread across
four runs was 525, 1167, 1380 and 1705 ms per tile, which is the clearest statement
of how much this machine can be trusted with a stopwatch.

**One structural finding does move the number, and it is not machine load.** The
golden test asks `startRuntime({ maxBackend: "wasm" })`, and `threadsFor` returns
1 for the `wasm` backend by definition. But Node 23 has both
`navigator.hardwareConcurrency` (10 here) and `SharedArrayBuffer` (the Worker's own
started report says so), so an unconstrained `startRuntime()` under Node would
choose `wasm-threads` instead. Run back to back on the quietest window this machine
offered, the page took **63.0 s at one thread and 27.1 s at nine**, and under load
85.4 s against 33.6 s at four: a factor of 2.3 to 2.5 either way, with little
between four threads and nine.

So the CI figure is the single-threaded floor **by choice**, not the platform's
limit, and the number the browser has to beat is the threaded one. `StartedReport`
carries `numThreads` as applied for exactly this reason: a timing is meaningless
without it.

**The 159 ms a tile recorded in `src/models/manifest.ts` did not reproduce**, in
any configuration tried, and it is the one measured claim from phase 2 this phase
could not confirm. It is flagged in the test header as unverified. That matters
because phase 3 is a timing gate: the plan's rule is 10 s on WebGPU, and an
unverified baseline is a bad thing to measure a gate against.

## Tradeoffs accepted

**The browser path of the opencv loader has no test.** The Node tests inject a
`createRequire` source because the dynamic import hangs under vitest, so the
default `import()` is exercised only by a manual run under plain Node and,
eventually, by the bench page. The alternative was to make the library's default
the Node-shaped one, which would have been a library shaped by its test harness.

**The `then` clearing mutates a third-party module singleton.** The alternative,
boxing the loader's output as well as its input, would push an unwrap onto every
call site for the whole life of the port to avoid deleting a legacy shim nothing
uses. The write is guarded so it cannot fail on a sealed namespace.

**One test pushes a whole page through segnet, and it takes about half a minute.**
Three page runs would have been five minutes of `npm test` against thirteen
seconds today. So the correctness test is the one full run, on the Python
`preprocessed.png`, and the Worker's wiring is proven on a small crop, with the
page-wide fp16 comparison behind an environment variable. The risk accepted is
that the Worker's own composition of preprocess and segment is only checked at
640 x 640; both halves are checked exactly at full size separately.

**`resizeToTargetWidth` returns the input plane itself when no resize is needed,
and `autocrop` returns the input when it does not crop.** A caller that mutated a
result would be writing into the page it passed in. Nothing in the port mutates a
plane it did not create, and copying 26 MB to protect against a caller that does
not exist is not worth it.

## Alternatives considered

**`cv.resize` with `INTER_CUBIC`.** Rejected before it was tried, because PIL's
filter is not OpenCV's and the golden is byte-exact; the phase plan already said
so and the byte-exact result confirms it. A float reimplementation of PIL's own
kernel was also rejected: the fixed-point truncation is where the bytes are
decided.

**Staging the tiles as a single big tensor.** One `[54, 3, 320, 320]` tensor is
66 MB of float32 and the session is pinned to its batch dimension anyway. The
buffer is allocated once at `batch * 3 * 320 * 320` and refilled.

**A vitest config change to make the opencv dynamic import resolve.**
`server.deps.external` and `server.deps.interopDefault: false` were both measured
and neither helped, which was the clue that the problem was the thenable rather
than the transform.

**Widening the mask tolerance.** Never on the table. `testing.md` says a widening
to make a test pass is a bug hidden, and the fp16 figures are reported rather than
absorbed.

## Open questions and risks

- **The manifest's 159 ms per tile did not reproduce.** 525 ms is what this
  machine measures at batch 8 on one wasm thread. Since the phase is a timing
  gate, the discrepancy needs settling on the bench, on a browser with threads and
  a GPU, before anyone decides go or no-go from a Node number.
- **No fixture crops.** Autocrop's real branch is covered only synthetically. The
  first photographed page in `test/fixtures/local/` closes it.
- **`npm run build` is fixed in the working tree by a one-line `tsconfig.build.json`
  change a delegate made, and the fix is incomplete.** See the section above: a
  published `.d.ts` naming `GPUAdapter` still asks consumers for types that are a
  devDependency here. The line is easy to drop if phase 2 would rather solve it the
  other way.
- **`cropPlane`'s name is wider than its contract**, and two independent ports of
  two different Python functions both had to avoid it. It is faithful to
  `image_utils.crop_image` and correct for its own callers; a third caller will
  make the same mistake.
- **`InputPredictions.original` has no reader yet**, so 397 ms a page is spent
  producing the resized colour image for a debug overlay that does not exist.
- **`openModelSession` leaks a session when `assertMatchesManifest` throws**
  (`src/models/session.ts`): the `InferenceSession.create` has resolved by then and
  nothing releases it. Pre-existing, outside this phase's diff, and now reachable
  from `SegmentationWorker.start()`. It only fires on a re-pin mistake, which is
  exactly when somebody is already looking.
- **`isRecord` is byte-identical in `src/cv/opencv.ts` and
  `src/segmentation/worker.ts`.** Two boundary parses that happen to need the same
  two-line predicate. Left duplicated rather than given a module of its own; worth
  revisiting when a third appears.

## Next implementation step

`bench/`, the browser, and the WebGPU numbers: the half of phase 3 this work
deliberately excluded. It has the whole library to drive, and the two figures it
owes the decision log are the WebGPU page time on an M1 and the batch-size A/B at
8, 16 and 32 that the plan asks for. Phase 4 then adds the box extraction on top
of the `mask-filtered-*.png` goldens, whose provenance this phase established.

# Phase 11 findings: chord-strip OCR and the title

What the AbcMusicStudio server reads above each staff, what homr 0.7.0
reads as the title, and how the port reproduces both on onnxruntime-web.
Measured 2026-10-04 on arm64 macOS, Node 23.5.0, onnxruntime-web 1.30.0,
@techstark/opencv-js 4.12.0; oracle rapidocr 3.9.2, onnxruntime 1.30.0,
numpy 2.5.3. Every decision below has a row in `docs/decisions.tsv`.

## The oracle

- **The chord texts come from two programs.** `omr_chord_ocr.py` (embedded in
  the Go binary, run with the homr virtualenv's `python`) cuts one strip per
  staff from the decoded page, from 1.9 to 0.05 staff heights above the
  staff's top, 3 % of the page wider on the left and 2 % on the right, and
  calls `RapidOCR()` with its default configuration. `runOmrChordOcr`
  (`abcsql/omr.go`) then remaps the script's line numbers to the sorted
  staves and stable-sorts by staff and `x0`. It filters nothing: the only
  threshold on the route is RapidOCR's own `text_score` 0.5.
  `MIN_OCR_SCORE = 0.6` is the app's, in
  `src/utils/converters/pdfToAbc/analyzer.ts`, applied to server and browser
  texts alike.
- **`tools/dump-texts.sh` runs that Go code itself** (a throwaway test in
  `abcsql`, as `dump-staves.sh` does), so `texts.json` is exactly what the
  route answers. `tools/dump-ocr.py` writes every RapidOCR stage of every
  strip into `ocr.json` from the repository's `.venv`, unrolling
  `RapidOCR.__call__` and checking the unrolled result against the real call.
- **RapidOCR 3.9.2 loads three files from its wheel**, all three already in
  `models/` and in the manifest since phase 2, hashes equal to
  `default_models.yaml`: `PP-OCRv6_det_small.onnx` (sha256 `090f04ab…`,
  9 929 594 bytes), `PP-OCRv6_rec_small.onnx` (`6f327246…`, 21 234 383) and
  `ch_ppocr_mobile_v2.0_cls_mobile.onnx` (`e47acedf…`, 585 532). Defaults from
  `config.yaml`: `Global` text_score 0.5, min_side_len 30, max_side_len 2000,
  vertical padding with min_height 30 and width_height_ratio 8; `Det`
  limit_type min, limit_side_len 736, mean and std 0.5, thresh 0.3,
  box_thresh 0.5, max_candidates 1000, unclip_ratio 1.6, dilation on,
  score_mode fast; `Cls` image shape 3×48×192, batch 6, thresh 0.9, labels
  "0" and "180"; `Rec` image shape 3×48×320, batch 6, the alphabet from the
  model's own `character` metadata (18 708 characters, plus blank and space:
  18 710 classes).
- **The title is homr's, not the server's.** `title_detection.py` runs its
  own `RapidOCR()` (same defaults) on a crop 15 unit sizes above the first
  staff of `detect_staff`, 50 px wider on each side, taken from homr's
  autocropped and resized page; it drops lines under four characters or
  four Latin letters ("tempo markings"), keeps the line with the largest
  box height per character, and reduces it to letters, digits and single
  spaces. homr's CLI writes it into `work-title`, so the server's MusicXML
  has it. `dump-golden.py` passed `""` until this phase; it now unrolls the
  task into `title.json` and the golden `page.musicxml` carries the title.
  The server virtualenv's CLI gives the same three titles: The Kesh, Grand
  Staff Study, CHORD STUDY.
- **The oracle depends on the OpenCV build.** The local server virtualenv
  has both opencv-python 5.0.0.93 and opencv-python-headless 4.14 installed,
  and loads 5.0; production pins headless < 5. Run on the three pages with
  both: every text and box equal, three scores apart by 0.001 to 0.002.
- **The local server virtualenv's `bin/homr` has a stale shebang**
  (`/Users/jymen/development/AbcGoDb/static/tools/homr-venv/bin/python`), so
  the Go route on this machine cannot start homr; `bin/python` works, which
  is all the chord OCR and these dumps need. Not a homr-web matter.

## The fixtures

The Kesh page has no chord symbol: its strips read "Jtg" (the word *jig*,
0.595) and "小=85" (the tempo, 0.96). The grand-staff page reads only its
title, twice ("Grand Staff Study" over staff 0 and a partial "Grand Stan
Study" over staff 1) and a stray "上". So a third public page was added,
`chord-study-300dpi.png`: sixteen bars with chord symbols, typeset by abcjs
6.6.4 with the app's own header directives (Cinzel title, Satisfy chords)
through `tools/typeset-fixture.html` and headless Chrome, which re-renders it
byte-identical. Its oracle has 19 texts over four strips and is a useful
test of a realistic, imperfect reading: Em is missed on strips 0 and 3, Em and
A7 on strip 2 read "w3" (0.618) and "L4" (0.599), F♯m reads "7#m".

On that page detect-golden pins one grid point: from the port's own staff
fragments, `multistaffs[1].staffs[0]` repeats x 1770 at index 176. From
Python's fragments `detect_staff` is exact, so the cause is phase 5's
opencv.js minAreaRect drift on five fragment polygons.

## @gutenye/ocr-browser against a direct port

Measured by running `@gutenye/ocr-node` 1.4.9 (the same `ocr-common` code as
the browser package) on the 12 oracle strips:

| | strips equal | oracle texts found |
|---|---|---|
| ocr-common, its own PP-OCRv4 models | 3 of 12 | 1 of 24 |
| ocr-common, homr-web's PP-OCRv6 det/rec and v6 alphabet | 2 of 12 | 1 of 24 |
| direct port | 12 of 12 | 24 of 24 |

ocr-common is another pipeline on the same model family: detection
threshold 0.03, no box score, unclip 1.5, no 736 limit, no padding, no
classifier, one recognition run per line. It merges a row of chords into one
box ("G Em Am7 D7"), so no chord gets its own x, which is what the app places
chords by. It also brings js-clipper (408 kB), its own 15 MB of models unless
overridden, and @techstark/opencv-js 4.9 beside the port's 4.12.

## The port

- `src/ocr/rapid-ocr.ts`: `RapidOcr` with the three sessions, and each
  step exported: `preprocessImage` (resize_image_within_bounds),
  `verticalPadding`, `detectionInput`, `cropText` (perspective warp, bicubic,
  replicated edges, a quarter turn for tall crops), `classifyCrops`,
  `recognizeCrops`, then the empty-text and `text_score` filters and
  `map_boxes_to_original`. Batches follow numpy's own argsort (`npArgsort`),
  so equal aspect ratios batch as in Python.
- `src/ocr/db-postprocess.ts`: DBPostProcess and `sorted_boxes`. numpy keeps
  cv2's float32 corners float32 through every operation with a Python int or
  float (NEP 50 weak scalars), so the port rounds to float32 at each step,
  including the cast of a Python-float ratio to float32 before a multiply.
- `src/ocr/clipper-offset.ts`: Clipper 6.4.2's `ClipperOffset` for one
  closed polygon with round joins, without the closing union, which cannot
  change the convex hull minAreaRect reads. pyclipper truncates float corners.
- `src/ocr/ctc.ts`: greedy CTC with Python's `round(conf, 5)` per character
  and numpy's `mean(...).round(5)`; the two roundings differ (correctly
  rounded decimal against scale-rint-unscale), both in `image/numeric.ts`.
- `src/models/onnx-metadata.ts`: onnxruntime-web has no custom metadata
  accessor, so the alphabet is read from the verified model bytes.
- `src/ocr/strips.ts` and `src/ocr/page.ts`: the server's strip geometry and
  normalisation, and homr's title task.

## Measurements

- **Stages** (`test/ocr-golden.test.ts`, Node wasm, against `ocr.json`): every
  strip's preprocess size, ratios and padding exact; every detection box
  exact; every classifier label and recognised text exact; recognition scores
  within 0.0037.
- **Route** (against `texts.json`): 24 of 24 texts with the same staff and
  text, boxes 0 drift, scores within 0.006 (largest "w3", 0.612 against
  0.618; 0.616 under opencv 4.14). At the app's 0.6 every text falls on the
  server's side. Test tolerances: score 0.01, box 5e-4 page units.
- **Unclip vectors**: 240 rectangles, pyclipper's polygon reproduced, the
  expanded rectangle exact on 59 and within 4.9e-4 elsewhere.
- **Title**: equal on the three pages, crop rectangle exact.
- **The event loop.** onnxruntime-web's wasm runs settle as microtasks, so
  the OCR's hundred or so runs chained without a macrotask: vitest's worker
  RPC timed out after 60 s (with every test passing), and a Worker could not
  have received a cancel during OCR. Each OCR run now yields a
  `setTimeout(0)` turn.
- **End to end** (`test/recognize-golden.test.ts`, Node wasm, one thread): the
  three pages from their PNG give homr's MusicXML with its title, the
  server's staves and the server's texts.
- **Timings**, Chrome on apple metal-3 through the public API in the bench's
  Worker, warm, each placement in a fresh tab:

| page | ocr stage, OCR on WebGPU | ocr stage, OCR on 4 wasm threads | readTextStrips, WebGPU / wasm | whole page, OCR on WebGPU |
|---|---|---|---|---|
| chord study | 3.30 s | 5.29 s | 2.58 / 3.88 s | 7.3 s |
| The Kesh | 3.12 s | 4.71 s | 2.35 / 3.45 s | 7.7 s |
| grand staff | 3.17 s | 4.70 s | 2.35 / 3.44 s | 8.7 s |

  The ocr stage includes the title (about a fifth of it). The OCR models now
  run on WebGPU by default on a WebGPU runtime. Under Node on one wasm
  thread: the strips 8.8 to 10.2 s a page, the title 3.7 to 4.0 s, a whole
  page 51 to 57 s against 39 to 45 s without OCR. A browser tab that had run
  the bench's main-thread pipeline first ran everything 5 to 10 times
  slower, wasm and WebGPU alike; those runs were discarded. Bundle: the OCR
  adds 38.5 kB of JavaScript to `dist/` (npm unpacked size 789 kB to 855 kB)
  and 31.7 MB of models.

## The ocr progress stage is a compile break for the app

`ProgressStage` gains `"ocr"`. The app keys a `Record<ReadingStep, string>`
on `Exclude<ProgressStage, "models">`, so 0.2.0 fails its type check until it
adds one `ocr` label. Adding the stage anyway was chosen over hiding OCR
inside `staff` or `xml`: OCR is a few seconds a page, and a progress bar that
stalls on "writing MusicXML" for that long is worse than one more key. The
version is 0.2.0 for that reason.

## Corrections to the plan file

- The plan's `readTextStrips(image, staves)` as a second export of
  `index.ts` cannot be a free function: it needs the Worker's models and
  OpenCV. It is a method of `Recognizer`, sharing the one-page rule, and
  answers the result shape with `musicXml` empty and `staves` as given. The
  pure function is `readStripTexts` in `homr-web/internal`.
- "`MIN_OCR_SCORE` filter ... becomes a documented option": the route has no
  0.6 filter; the app does. The option is `minTextScore`, RapidOCR's
  `text_score`, default 0.5, so default output equals the route's.
- The plan's model sizes (detect 10 MB, recognise 21 MB, classifier 0.6 MB)
  hold. The three files were already in the manifest and the store; the
  Worker's list of roles a page reads gained them, and their placement moved
  from "stay on CPU" to the WebGPU EP on a WebGPU runtime, measured above.
- The classifier is the PP-OCRv4 line's `ch_ppocr_mobile_v2.0_cls_mobile`,
  not a v6 model; "PP-OCR v6 detect and recognise" is right for the other two.
- "`♯` and `♭` are not [in PP-OCR's character set]": the PP-OCRv6 alphabet
  holds `♯`, `♭`, `#` and `b`. What limits sharps and flats is the reading,
  not the alphabet: the Satisfy `F♯m` of the chord page reads `7#m`, on the
  server as here.
- Neither public page from earlier phases has a chord, so the plan's
  runtime check ("chords over the staves for the Kesh page") cannot show a
  chord. The chord page is the one to look at.

## Not verified

- A Worker in a background tab: the per-run `setTimeout(0)` yield was not
  timed there. Chrome throttles timers of hidden pages; a dedicated
  Worker's were not measured.
- WebAssembly-only browsers (no WebGPU) were not timed with OCR; the
  wasm-threads figures above come from a WebGPU runtime with the OCR roles
  placed on wasm.
- Pages with ♯ or ♭ chord symbols beyond the one F♯m, lyrics, or text under
  the staff: the strip geometry is the server's and reads only above.

# Tab 2 findings: the fret reader

Step 2 of the line-tablature plan in `.scratch/tab-sketch/REPORT.md`: the
prototype's connected-component reader as library code in `src/tab/`, beside
the step-1 guard and apart from homr's port. Measured 2026-10-09 on arm64
macOS, Node 23.5.0, onnxruntime-web 1.30.0 (wasm, one thread),
@techstark/opencv-js 4.12.0. Every decision below has a `tab-2` row in
`docs/decisions.tsv`.

## What it reads

- **Input is what the guard already has.** `readTablature(cv, page, tabs,
  read)` takes the page and the guard's `DetectedTab`s, whose pixel line
  geometry it needs for erasing lines, and the same `ReadCrops` the guard
  uses (RapidOCR's recogniser, opened once). No new dependency.
- **One mark finder for guard and reader.** `marksOf` in `detect.ts` erases
  the lines over a system's band, takes 4-connected components of digit
  size within 0.3 spacing of a line, merges touching ones on a line ("10",
  "12") and splits off the marks under 0.75 of the median height. The guard
  samples the tall marks of a five-line group; the reader reads all of them.
- **Each mark is cropped from its own ink and read alone.** A fret is a
  number from 0 to 24, an O read as 0, with any letters stuck to it kept as
  its technique ("S0" is a 0 with Sl). A mark that reads as a technique
  letter is an annotation; anything else counts in `unread`.
- **Events are frets left to right.** A fret within 0.45 spacing of the open
  event's first fret joins it if its string is free, so a chord is one event
  with one fret per string, ordered by string.
- **Pitch is outside the reader.** `pitchTab(reading, { strings, capo })` is
  open string + capo + fret, strings top line first. A banjo's fifth string
  is the bottom line and takes the capo like the others, the rule the
  prototype's Cripple Creek pitches matched against the staff above.

## The result type

```ts
interface TabReading extends TabSystem {   // index, lines, cx, cy, w, h
  readonly events: readonly TabEvent[];    // { x, notes: [{ string, fret }, ...] }
  readonly annotations: readonly TabAnnotation[]; // { x, string, technique }
  readonly unread: number;
}
```

`RecognizeSuccess.tablature` and the `tablature_only` failure carry
`TabReading`, which extends `TabSystem`, so code typed on `TabSystem`
compiles unchanged. `string` 1 is the top line and `x` is page-normalised.
`technique` is one of `Sl`, `Po`, `H`, `R`, `p`, `h`, `x`, `Harm.`
(`TAB_TECHNIQUES`). `PROGRESS_STAGES` gains `tab`, one step per system.

## Fixtures

- **Typeset by us, committed as PNG.** `tools/typeset-tab.mjs` builds three A4
  pages at 300 dpi as SVG from column specs, rasterises them with
  `rsvg-convert` and writes `test/fixtures/tab/truth.json` in the same run.
  Banjo (5 lines, Roboto, knock-outs, the four TablEdit letters, a slur, a
  TAB clef), guitar (6 lines, TeX Gyre Bonum, h, p, x, Harm.), mandolin
  (4 lines, one system without knock-outs). 36 to 48 KB each, 122 events.
- **Glyphs are placed by measurement.** Each label is rasterised alone first
  and its ink box read back, so truth `x` is the drawn ink's centre to the
  half pixel, and the two digits of a two-digit fret share one ink column.
- **`test/tab-read.test.ts`** checks every event's frets and strings exactly,
  `x` within one spacing, every technique letter on its string, and pitch
  through `pitchTab`. `test/tab-guard.test.ts` checks that `recognizePage`'s
  `tablature_only` result carries the frets of the guard's Hershey page and
  reports the `tab` stage.

## What the fixtures found

The prototype's reader, ported as is, read every event of the knocked-out
systems and failed four ways. Each fix was then re-measured on the local
pages, and two first attempts were reverted for costing Cripple Creek events.

| fixture case | failure | fix |
|---|---|---|
| H inside a knock-out | its bar, on the line, erased as line: read "I" "I" | erase a short run only if one of its ends is free |
| lowercase x (Bonum) | 0.68 of digit height: dropped with the arcs | read short marks, accept only a technique |
| Harm. | H tall, "arm" short: read as H | join short letters to a tall letter within 0.35 spacing |
| 12 with the line through it | read "2" | crop from line-erased ink when the line runs into both sides |

## Accuracy on the local pages

`tools/tab-accuracy.ts` (local only, needs `test/fixtures/local/tablature/`
and a git-ignored `tab-truth.json` beside it) on the prototype's own rasters,
which are byte-identical to `test/tab-local.test.ts`'s:

| page | prototype | library |
|---|---|---|
| Cold Frosty Morning, hand truth system 1 | 15/15 | 15/15 |
| Chicken reel, hand truth system 1 | 35/35 | 35/35 |
| Sourwood Mountain, hand truth system 1 | 33/33 | 33/33 |
| Arkansas Traveler vs homr's staff | 115/115 | 115/115 |
| Cripple Creek tab-only vs with-score | 161/169 | 161/169 |
| Cripple Creek with score vs homr's staff | 129/169 | 117/169 |
| Kildare (scan), hand truth system 1 | 0/37 | 0/37 |

Events per page equal the prototype's on every born-digital page (Cripple
Creek 169 on both); notes too, but for Cripple Creek tab-only, 202 for 204,
whose events still all match. Kildare, a scan, has 17 events for 21.
Cripple Creek against the staff is lower because homr now reads the painted
page: the guard's first row in `docs/decisions.tsv` measured the same 117 for
homr after painting, and the tab reading itself has not moved.

## Speed

Node, one wasm thread, recogniser already open (it opens in 0.25 to 0.3 s):

| page | detect | read | total |
|---|---|---|---|
| Arkansas Traveler (4 lines) | 0.04 s | 7.9 s | 7.9 s |
| Kildare (4 lines, scan) | 0.04 s | 2.3 s | 2.3 s |
| Cluck Old Hen | 0.9 s | 4.2 s | 5.1 s |
| Chicken reel | 1.3 s | 6.2 s | 7.5 s |
| Cold Frosty Morning | 2.7 s | 8.1 s | 10.8 s |
| Sourwood Mountain | 1.3 s | 10.0 s | 11.3 s |
| Cripple Creek | 1.7 s | 15.9 to 16.8 s | 17.6 to 18.5 s |

2.3 to 18.5 s a page, against the prototype's 10 to 43 s, which also ran the
strip reader on every system. Reading is one recogniser call per system on
every mark, about 0.07 s a mark; WebGPU in a browser will be faster.

## What remains

- **Step 3, tuning and capo.** Nine of the ten sample pages print them as
  text ("gDGBD", "Capo 2", "Sawmill"). `pitchTab` takes them as input today;
  the OCR and a parser to `Tuning` are the step. The app needs a way to
  correct them, since a wrong tuning moves every pitch.
- **Step 4, rhythm.** No stems, beams or flags are read; `events` are ordered
  but have no duration. On a staff-plus-tab page homr's rhythm can carry the
  frets (events align 115/115 on Arkansas Traveler); a tab-only page needs
  its own stem and beam reading.
- **Scans.** Kildare reads 0/37: digits struck through by a scanned line split
  into halves the reader drops. The prototype's strip reader (68 %) is the
  planned fallback (step 5).
- **Techniques are positional only.** An annotation has a string and an `x`;
  which note it belongs to (the slide from where to where) is not decided.

# Phase 5 synthesis: staffs to multi-staffs

Two candidates were sketched in parallel from the same grounding
(`phase-5-findings.md`). Candidate A is the base. This file records what was
taken from each, what was rejected, and the build order. Where this file and
candidate A disagree, this file wins.

Neither candidate is committed. Each was a design package of three parts: a
rationale, a module map naming every file with its exports and the Python it
ports, and a type-level sketch of those files with empty bodies.

- **Candidate A** kept the detection algorithms in `src/geometry/` behind a
  five-question `BoxOps` interface, so that ordering and tie rules can be
  tested without opencv.js. Its sketch type-checked over a copy of this
  repository, and it came with probes that measured its numeric helpers
  against numpy. Its oracle is the extended `tools/dump-golden.py` plus the
  vector files of `tools/dump-vectors.py`.
- **Candidate B** passed `(cv, scope)` to every detection function, which
  moved seven algorithm files into `src/cv/`. It interned fragments as
  `Fragment{id, box}`, degraded and logged at homr's two crash sites, and
  took its extra coverage from synthetic mask pages run through the dumper.

What stage 1 of the build order produced from candidate A's module map is in
the tree: the dumper additions, `tools/dump-vectors.py`, the `Golden*`
decoders in `src/golden/decode.ts` and the accessors in `src/golden/page.ts`.
The files later stages add are listed under "Modules" at the end.

## Base: candidate A

A is the base for three reasons. Its sketch type-checks over a copy of the
repository under the repository's own strict settings. Its numeric bodies were
measured against numpy 2.5.3 (`sum`/`mean`/`std` 880 of 880, `npArgsort` 2750
of 2750, `floorDiv` 4000 of 4000). And `BoxOps` keeps the ordering and tie
logic, which is most of this phase, in `src/geometry/` with no `cv` handle,
where candidate B moves seven algorithm files into `src/cv/` only because they
ask an overlap question.

## The seven forks

| Fork | Decision | From |
|---|---|---|
| 1. Staff-line types | Plain readonly objects in `src/geometry/`, each beside its algorithm (`staff-lines.ts`, `staff-anchors.ts`, `raw-staffs.ts`). `FiveLines` is a 5-tuple. `RawStaff` has a `box`. Fragment identity is the `rectKey` set, because Python compares fragments by value. | A |
| 2. Numerics | `sum`, `mean`, `std`, `floorDiv` change in place. `npArgsort` is a literal `aquicksort` in `src/image/argsort.ts`. | A and B agree |
| 3. Errors | One `DetectionError` with four codes, thrown. The two upstream `IndexError` sites throw where Python crashes. Phase 9 turns it into `{ ok: false }`. | A |
| 4. `preprocessed` | The masked page, as upstream. `PageDetection.noise` is a three-way `NoiseOutcome`. | A |
| 5. Oracle | `dump-golden.py` unrolls `detect_staff` and the brace function with a self-check, and adds the six files in A's module map. `dump-vectors.py` writes the eleven vector files. A typeset grand-staff page is attempted as a second public fixture. | A |
| 6. Mutation | None. `addNotesToStaffs` returns `{ notes, staffs }`; `Staff.symbols` becomes readonly. | A and B agree |
| 7. Modules | A's module map. `src/index.ts` keeps one `export *` per module. | A |

## Grafted from candidate B

- **The heapsort fallback is never shipped unverified.** A reaches it with an
  adversarial input in `argsort.json`. If that input cannot be built, the
  fallback throws a `NumericError` (B's choice) and the decision log says so.
- **`get_staff_for_anchor` is a subset test per line index.** B caught that the
  findings described it loosely. `staff_detection.py:172-178` returns the first
  staff where, for some line index `i`, the anchor line's fragment set is a
  subset of the staff's line `i`. A's `holdsAllFragmentsOf` already has this
  shape; the findings file is corrected when it is copied into the repository.
- **The fixture gate.** The piano page is committed only if its dump shows a
  non-empty `boxes-brace_dot.json` and `is_grandstaff: true` (B's gate, A's
  four-system page).

## Rejected

- **B's degrade-and-log at the crash sites.** A dropped staff produces a page
  no golden can check, and the browser would answer where the server fails.
- **B's synthetic mask pages** (a generator, a dumper branch, eight golden
  directories). A's vectors call the same Python functions on hand-built
  inputs at a fraction of the weight. The cost is that vectors skip the
  box-fitting path, which phase 4's goldens already cover.
- **B's `Fragment{id, box}` interning.** An integer id is identity, and Python's
  sets use value equality. The two differ on two fragments with equal rects.
- **B's `PageDetection.log` and `log-detect.txt`.** Whether `RecognizeResult.log`
  carries homr's stderr is phase 9's question. Phase 5 adds no log.
- **B's restricted `src/index.ts`.** It breaks the convention of phases 1 to 4
  for no caller.
- **B's `staffLinePolylines` helper.** The bench reads `staff.grid` directly, as
  A's call site does.

## Constraints on the implementation

- The move of seven phase 4 functions from `src/cv/` to `src/geometry/`, and of
  `prepareBarLineImage` the other way, is its own unit with the suite unchanged
  as its check. It is not mixed with new code.
- The one-Mat-scope-per-stage tradeoff rests on an estimate of about 1500 Mats
  on Kesh. Measure it when `detect.ts` runs and record the number.
- `fromPythonInputs` (1e-9) is reasoned, not measured. A stage row that needs
  more gets a decision-log row naming the numeric cause before the bound moves.
- Every existing golden file must come out of the extended dumper
  byte-identical. That is the first check of the phase.
- Nothing is committed or pushed. The owner reviews the working tree.

## Throughput checkpoint

- **Blocking first steps.** The oracle extension (dumper, vectors, decoders,
  the fixture attempt), then the numeric helpers, then the move and `BoxOps`.
  Nothing else can be checked before these exist.
- **Independent workstreams.** After the blocking steps, the leaf stages touch
  disjoint files: `noise-filter.ts`, `makeLinesStronger`, `break-fragments.ts`,
  `combineNoteheadsWithStems` with `barlines.ts`, `find-peaks.ts`.
- **Shared mutable state.** `src/golden/decode.ts`, `src/golden/page.ts`,
  `src/index.ts`, `src/model/staff.ts` and `docs/decisions.tsv` are written by
  every unit. One owner writes them in sequence; no parallel writers.
- **Smallest safe decomposition.** One owner per stage, four stages, each
  ending on a green suite. The staff chain (anchors, raw staffs, resampling) is
  one worker because each row's input is the previous row's type.

## Build order

1. Oracle: `dump-golden.py` additions, `dump-vectors.py`, the fixture attempt,
   decoders and `GoldenPage` accessors. Check: existing goldens byte-identical,
   every new file decodes.
2. `numeric.ts` in place, `argsort.ts`, `find-peaks.ts`. Check: `vectors.test.ts`,
   the existing suite unchanged.
3. The move and the biome override. Check: suite unchanged, lint passes.
4. `BoxOps`, `cv/box-ops.ts`, `REQUIRED_MEMBERS`. Check: `intersections.json`.
5. `constants.ts`, the `staff.ts` functions, `staff-tolerance.ts`. Check: golden
   staffs against themselves and a perturbed copy, `staff-merge.json`.
6. Leaf stages: noise filter, `makeLinesStronger`, `breakWideFragments`,
   `combineNoteheadsWithStems`, bar lines.
7. The staff chain, one golden row at a time, ending on `staffs.json`.
8. Notes, braces, staff positions.
9. `detect.ts`, the end-to-end golden, the bench overlay.

## Modules

Candidate A's module map, amended by the grafts above. Paths are relative to
the repository root. "Ports" names the Python function with its line in the
installed 0.7.0 file. Stage 1 built the rows marked so, `test/support/vectors.ts`
and the two tools; every other row is a later stage's.

### New files

#### `src/geometry/` (pure; after this phase nothing here imports from `src/cv/`)

| File | Exports | Ports |
|---|---|---|
| `box-ops.ts` | `BoxOps` (`ellipseFromRect`, `fitRotatedBox`, `intersects`, `overlaps`, `thicker`), `overlapsAny` | the five OpenCV-backed questions detection asks about a box; `is_overlapping_with_any` (`bounding_boxes.py`) |
| `box-transforms.ts` | `polygonViaBoxPoints`, `makeBoxTaller`, `makeBoxTallerKeepCenter`, `moveToXHorizontalBy`, `ensureMinDimension`, `getCenterExtrapolated`, `isOverlappingExtrapolated` (all **moved**, bodies unchanged), `rotatedBoxFromRect` (new) | the `RotatedBoundingBox` methods that never needed a Mat; `rotatedBoxFromRect` is the constructor call at `staff_detection.py:656` |
| `staff-lines.ts` | `StaffLineSegment` (no `debugId`: only homr's drawing reads it), `FiveLines`, `createStaffLineSegment`, `asFiveLines`, `mapFiveLines`, `mergeSegments`, `fragmentAt`, `holdsAllFragmentsOf`, `segmentsOverlap`, `connectStaffLines`, `areLinesCrossing`, `areLinesParallel`, `beginsOrEndsOnOneStaffLine` | `StaffLineSegment` (36-90), `connect_staff_lines` (239), `are_lines_crossing` (287), `are_lines_parallel` (295), `begins_or_ends_on_one_staff_line` (317) |
| `staff-anchors.ts` | `StaffAnchor`, `AnchorZone`, `ANCHOR_ZONE_LEDGER_LINES`, `AnchorSymbolKind`, `ANCHOR_SHIFTS`, `createStaffAnchor`, `findStaffAnchors`, `filterUnusualAnchors` | `StaffAnchor` (93-131), `find_staff_anchors` (330), `filter_unusual_anchors` (516) |
| `other-clefs.ts` | `ColumnZone`, `initZones`, `zoneColumns`, `groupLinePeaks`, `findHorizontalLines`, `predictOtherAnchorsFromClefs` | `init_zone` (530), `filter_line_peaks` (555, `groups` only), `find_horizontal_lines` (608), `predict_other_anchors_from_clefs` (638) |
| `raw-staffs.ts` | `RawStaff`, `rawStaffContour`, `rawStaffFromParts`, `createRawStaff`, `mergeRawStaffs`, `staffForAnchor`, `findRawStaffsByConnectingLineFragments`, `removeDuplicateStaffs` | `RawStaff` (143-169), `_get_all_contours` (133), `get_staff_for_anchor` (172), `find_raw_staffs_by_connecting_line_fragments` (181), `remove_duplicate_staffs` (217) |
| `resample.ts` | `STAFF_DENSITY`, `resampleStaffSegment`, `resampleStaff`, `resampleStaffs`, `filterEdgeOfVision`, `sortStaffsTopToBottom` | `resample_staff_segment` (391), `resample_staff` (440), `resample_staffs` (473), `filter_edge_of_vision` (486), `sort_staffs_top_to_bottom` (512) |
| `break-fragments.ts` | `WIDE_FRAGMENT_LIMIT`, `breakWideFragments` | `break_wide_fragments` (660) |
| `noise-filter.ts` | `NOISE_GRID_DIVISIONS`, `NoiseGrid`, `estimateNoise`, `createNoiseGrid`, `noiseOutcomeOf`, `filterPredictions` | `noise_filtering.py`: `estimate_noise` (11), `create_grid` (34), `apply_noise_filter` (48), `get_neighbors` (81), `handle_filter_results` (94), `create_noise_grid` (18), `filter_predictions` (108) |
| `notes.ts` | `STEM_SEARCH_THICKNESS`, `combineNoteheadsWithStems`, `averageNoteheadHeight`, `PixelBox`, `adjustBbox`, `checkBboxSize`, `splitClumpsOfNoteheads`, `addNotesToStaffs` | `note_detection.py`: `combine_noteheads_with_stems` (120), `adjust_bbox` (28), `get_center` (39, private), `check_bbox_size` (45), `split_clumps_of_noteheads` (90), `add_notes_to_staffs` (149); `main.py:250` median |
| `braces.ts` | `filterForTallElements`, `connectionsBetweenStaffs`, `mergeMultiStaffsSharingAStaff`, `scoreBraceWithStaffPair`, `createGrandstaffs`, `findBracesBracketsAndGrandStaffLines` | `brace_dot_detection.py`: `_filter_for_tall_elements` (22), `_get_connections_between_staffs_at_lines` (86), `_merge_multi_staff_if_they_share_a_staff` (116), `find_braces_brackets_and_grand_staff_lines` (142); `model.py`: `MultiStaff._score_brace_with_staff_pair` (430), `_select_grandstaffs` (468), `_merge_selected_pairs` (498), `create_grandstaffs` (511) |

#### `src/cv/` (needs a `cv` handle)

| File | Exports | Ports |
|---|---|---|
| `box-ops.ts` | `isIntersecting`, `createCvBoxOps`, `withBoxOps` | `RotatedBoundingBox.is_intersecting` (`bounding_boxes.py:207`); the opencv.js implementation of `BoxOps` over `OverlapTester`, `fitRotatedRectUnchecked`, `polygonViaEllipse2Poly`, `makeBoxThicker` |
| `mask-morphology.ts` | `prepareBarLineImage` (**moved** from `src/geometry/barlines.ts`), `makeLinesStronger`, `prepareBraceDotImage` | `prepare_bar_line_image` (`bar_line_detection.py:9`), `make_lines_stronger` (`staff_detection.py:29`), `prepare_brace_dot_image` (`brace_dot_detection.py:11`) |

#### `src/image/`

| File | Exports | Ports |
|---|---|---|
| `argsort.ts` | `npArgsort` | numpy's `aquicksort` (the default `np.argsort` on float64), for `find_peaks.py:119` |
| `find-peaks.ts` | `FindPeaksOptions`, `findPeaks` | `find_peaks.py:33` |

#### `src/pipeline/`

| File | Exports | Ports |
|---|---|---|
| `detect-staff.ts` | `detectStaff` | `detect_staff` (`staff_detection.py:694`) |
| `detect.ts` | `detectStaffsInImage`, `predictBraceDot` | `detect_staffs_in_image` (`main.py:232`) from `filter_predictions` (line 121) on; the brace_dot boxes of `main.py:278-280` |
| `staff-positions.ts` | `formatStaffPositions` | `save_staff_positions` (`staff_position_save_load.py:16`) |

#### `src/golden/`

| File | Exports |
|---|---|
| `staff-tolerance.ts` | `StaffFloatTolerance`, `STAFF_TOLERANCES` (`fromPythonInputs`, `fromOwnBoxes`), `compareStaffLists`, `compareNoteLists`, `compareNoteheadLists`, `compareMultiStaffLists`, `compareAnchorLists`, `compareRawStaffLists`, their report types, `assertGoldenMatches`, `describeStaffComparison`, `StaffGoldenMismatch` |

#### Tests and tooling

| File | Holds |
|---|---|
| `test/staffs-golden.test.ts` | one row per `detect_staff` stage, each fed Python's value of the stage before |
| `test/detect-golden.test.ts` | `detectStaffsInImage` from raw masks to `multistaffs.json`, `notes.json` and `staff-positions.txt` under `fromOwnBoxes`, with the reordered and the refitted notes pinned by index; the `no-noteheads` and `no-staffs` failures |
| `test/notes-golden.test.ts`, `test/braces-golden.test.ts` | the same row pattern for notes, bar lines and braces (same shape as `staffs-golden`), and the vector files of those modules: `bbox-split`, `notehead-clumps`, `braces`, `multi-staff-merge`, `grand-staffs` |
| `test/staff-positions.test.ts` | `formatStaffPositions` from Python's multi staffs to `staff-positions.txt`, byte for byte |
| `test/raw-staffs.test.ts` | the three `remove_duplicate_staffs` branches, the line-index rule of `get_staff_for_anchor` and the order of a merge, with a literal `BoxOps`, no opencv.js |
| `test/staff-chain.test.ts` | hand cases of the staff chain no page and no vector reaches, with homr's own answers as literals (see the decision log) |
| `test/noise-filter.test.ts` | `noise.json` of each page, and the seven planes of a masked page |
| `test/vectors.test.ts` | `sum`/`mean`/`std`, `floorDiv`, `npArgsort`, `findPeaks`, `findHorizontalLines`, noise grid, `isIntersecting`, `checkBboxSize`, `mergeStaffs`, braces, `connectStaffLines` against `test/golden/vectors/*.json` |
| `test/support/vectors.ts` | `VECTOR_FILES`, `readVectors` |
| `tools/dump-vectors.py` | writes `test/golden/vectors/*.json`; see the oracle section |

### Changed files

| File | Change |
|---|---|
| `src/image/numeric.ts` | `sum` becomes numpy's pairwise sum **in place**; `mean` and `std` are built on it; `floorDiv` becomes CPython's float floor division **in place**; new `pySliceBounds`. No left-fold `sum` remains exported. |
| `src/image/plane.ts` | new `rgbaFromPlane` |
| `src/model/constants.ts` | 16 constants and unit-size functions of `constants.py` (the ones phase 5 reads) |
| `src/model/staff.ts` | `Staff.symbols` and `StaffOptions.symbols` become `readonly SymbolOnStaff[]`; new `staffPointAt` (`Staff.get_at`, 316), `yDistanceTo` (322), `isOnStaffZone` (286), `findPositionInUnitSizes` (247), `staffPointToAxisBox` (`to_bounding_box`, 262), `mergeStaffPoints` (238), `mergeStaffs` (297), `withSymbols` (replaces `add_symbol`, 313), `mergeMultiStaffs` (411) |
| `src/model/pipeline.ts` | new `NoiseOutcome`, `DETECTION_FAILURES`, `DetectionFailure`, `DetectionError`; `PageDetection` gains `noise` and documents that `preprocessed` is the masked page |
| `src/geometry/barlines.ts` | loses `prepareBarLineImage`; gains `barLineCandidates` (`main.py:255-262`) and `detectBarLines` (`bar_line_detection.py:15`) |
| `src/cv/box-transforms.ts` | keeps `makeBoxThicker` only |
| `src/cv/box-fitting.ts` | loses `polygonViaBoxPoints` and its private `rotatedBoxOf` (builds its boxes with `rotatedBoxFromRect` of `../geometry/box-transforms.js`) |
| `src/cv/opencv.ts` | `REQUIRED_MEMBERS` gains `erode`, `getStructuringElement`, `rotatedRectangleIntersection`, `subtract` |
| `src/pipeline/predict-symbols.ts` | imports `prepareBarLineImage` from `../cv/mask-morphology.js`; its header comment drops the claim that the brace mask needs `detect_staff` |
| `src/golden/decode.ts` | new `decodeStaffAnchors`, `decodeRawStaffs`, `decodeNoteheadSplits`, `decodeBraces`, `decodeNoise` and their `Golden*` types (built in stage 1) |
| `src/golden/page.ts` | `GoldenPage` gains `staffAnchors`, `rawStaffs`, `noteheadSplits`, `braces`, `braceDotMask`, `staffPositionsText`, `denoisedStaffMask`, `noise`, `noiseMask`, `staffsWithNotes`; `GoldenMeta` gains optional `oracle` (built in stage 1) |
| `src/index.ts` | 19 new `export *` lines |
| `biome.jsonc` | one override: `noRestrictedImports` with pattern `**/cv/**` for `src/geometry/**` (the option exists in the installed 2.5.14 schema) |
| `test/box-transforms.test.ts` | import paths only (the one existing file that imports the moved functions; its header already says their only production callers are phase 5's) |
| `test/numeric.test.ts` | `floorDiv` near-multiple cases; the old cases pass unchanged |
| `bench/bench.js`, `bench/index.html` | continue the chain past `segmentPage` with `detectStaffsInImage`; one `<canvas id="overlay">` |
| `tools/dump-golden.py` | see the oracle section |
| `docs/decisions.tsv` | one row per fork, plus the arm64 dependency of `argsort` and of the noise-grid cast |

Every name phases 1 to 4 exported is still exported from `src/index.ts`; only module paths inside the repository move.

## The oracle as stage 1 built it

`tools/dump-golden.py` keeps its rule: it calls homr's functions in homr's
order and saves what they return. Four functions are unrolled into the calls
they make: `create_noise_grid`, `detect_staff`, `add_notes_to_staffs` and
`find_braces_brackets_and_grand_staff_lines`. After each unrolled block the
script calls the real function and exits non-zero unless the two results
serialise identically. Candidate A's map unrolled two; the noise grid and the
notehead splits cannot be read without the other two.

Fragments are written as indices into `boxes-staff_fragments-broken.json`,
found by `id()`, so a decoded anchor holds the same box objects as the decoded
fragment list.

| File | Content |
|---|---|
| `meta.json` | gains `"oracle": {"numpy", "opencv", "python", "machine"}` |
| `noise.json` | `{"tile": [height, width], "grid": [[uint8]], "filtered", "total", "outcome"}`, the outcome one of `clean`, `skipped`, `masked` |
| `mask-noise.png` | the 0/255 mask of kept tiles, written only when the outcome is `masked` |
| `mask-denoised-staff.png` | the staff mask after `filter_predictions`, before `make_lines_stronger` |
| `staff-anchors.json` | `{"clefs": [Anchor], "zones": [{"start", "stop", "lines": [[y x5]]}], "otherClefSymbols": [RotatedBoundingBox], "otherClefs": [Anchor], "barLines": [Anchor], "kept": [int]}`. `kept` indexes `clefs + otherClefs + barLines`. `Anchor = {"symbol": RotatedBoundingBox, "lines": [[int] x5], "averageUnitSize", "minY", "maxY", "zone": [start, stop]}` |
| `raw-staffs.json` | `{"connected": [RawStaff], "deduplicated": [int], "resampledFrom": [int], "droppedAtEdge": [int]}`. `deduplicated` indexes `connected`; the other two index `deduplicated`. `RawStaff = {"box", "polygon", "staffId", "lines": [[int] x5], "anchors": [int]}` with `anchors` as positions in `kept` |
| `notehead-splits.json` | `[{"staff", "notehead", "pieces": [BoundingEllipse]}]` for every pair `add_notes_to_staffs` reaches where `split_clumps_of_noteheads` returns more than one piece |
| `braces.json` | `{"notesPerStaff": [int], "tall": [int], "connections": [{"staff", "neighbour", "symbols": [int]}], "merged": [[int]]}` |

Stage 1's decoders returned plain `Golden*` shapes for anchors and raw staffs,
because those types belonged to the stage that ported their constructors.
Stage 3 ported them, and `decodeStaffAnchors` and `decodeRawStaffs` now build
`StaffAnchor` and `RawStaff` through their factories and assert the derived
values Python stored. The other decoders still return `Golden*` shapes.

`tools/dump-vectors.py` writes eleven files under `test/golden/vectors/`, each
`{"meta": {numpy, opencv, python, machine, ...}, "cases": [...]}` with one case
per line. `meta` also describes any field whose encoding is not obvious.

| File | Cases |
|---|---|
| `pairwise.json` | arrays of 0, 1, 4, 7, 8, 9, 10, 127 to 130, 255 to 257, 1000 and 2718 floats with `np.sum`, `np.mean`, `np.std` |
| `floor-div.json` | `(a, b, a // b)` with `a` one ulp either side of a multiple of `b`, the multiple itself, and 200 random pairs |
| `argsort.json` | tied arrays of 5 to 1000 with `np.argsort`, and two inputs built by McIlroy's adversary that reach numpy's heapsort fallback |
| `find-peaks.json` | plateaus, shoulders, tied peaks closer than `distance`, and random tied arrays with up to 38 peaks |
| `line-groups.json` | row-count images through `find_horizontal_lines`: one and two staffs, four and six lines, a sixth line either side of the gap limit, a full page column, an all-zero zone (`IndexError`) |
| `noise.json` | small staff masks through the noise grid: clean, masked, skipped, exactly half the tiles noisy, a tile estimate above 255, short last tiles |
| `intersections.json` | 313 rotated rect pairs with `cv2.rotatedRectangleIntersection`'s code and `is_intersecting`, touching pairs included, every value a float32 |
| `bbox-split.json` | boxes and small masks through `adjust_bbox` and `check_bbox_size`: width split, height split, the second pass dropping a box, a box on row 0 |
| `connect-lines.json` | fragment rects through `connect_staff_lines`, `are_lines_parallel`, `begins_or_ends_on_one_staff_line`: a fragment in two chains, a non-parallel set, equal `bottom_left` x in both input orders |
| `staff-merge.json` | pairs of hand-built staffs through `Staff.merge`: sorted y, partly shared x, half-even keys, duplicate keys, the two error cases |
| `braces.json` | hand-built staffs and brace boxes through `find_braces_brackets_and_grand_staff_lines`, with the intermediates of a page's `braces.json` |

The second public fixture is `test/fixtures/grand-staff-300dpi.png`; its
origin is in `test/fixtures/SOURCE.md`.

## The oracle as stage 4 left it

Stages 3 and 4 added files and changed none: every file above is byte-identical
to what stage 1 wrote, except that `meta.json` of each page lists one more
stage name.

| File | Content |
|---|---|
| `other-clefs.json` (per page) | the boxes `predict_other_anchors_from_clefs` builds before it drops those overlapping a clef anchor's symbol: 1 on Kesh, 3 on the grand-staff page, all dropped by the filter |
| `vectors/connect-lines-cleanup.json` | `connect_staff_lines` with and without a short fragment that runs the clean-up |
| `vectors/find-anchors.json` | `find_staff_anchors` with a unit of 2.5 and a sixth line of four widths, and a bar line one unit off its lines |
| `vectors/resample.json` | `resample_staff` with a missing top line beside a too-close pair, and on a sloped staff |
| `vectors/edge-of-vision.json` | `filter_edge_of_vision` on ten staffs, one per condition |
| `vectors/raw-staff-merge.json` | `RawStaff.merge` with two fragments at one centre x |
| `vectors/line-peak-groups.json` | `filter_line_peaks`' groups for four peak lists |
| `vectors/notehead-clumps.json` | hand-built staffs, noteheads and masks through `add_notes_to_staffs`, with the splits as in a page's `notehead-splits.json`: a width split, a height split, both, a staff that steps under a clump, a clump above row 0, a note on two staffs, noteheads no staff takes |
| `vectors/multi-staff-merge.json` | `_merge_multi_staff_if_they_share_a_staff` on entries given out of order, as indices |
| `vectors/grand-staffs.json` | `MultiStaff._score_brace_with_staff_pair` for every pair and brace, and `create_grandstaffs`, either side of each threshold |

`notehead-splits.json` is empty on both pages, so `notehead-clumps.json` is the
only oracle for a split notehead. `braces.json` in the vectors goes through the
whole brace function, where a merged entry is always the last one, so
`multi-staff-merge.json` is the only oracle for the move to the end of the
list.

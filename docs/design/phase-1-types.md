# Phase 1 design: core types and the array model

## Problem

Phases 3 to 9 of the homr port read and write one set of data shapes: a
page as bytes, five masks, a few thousand fitted boxes, four to nine staffs
with their notes, a token sequence per staff, and the result the app already
consumes. The Python side settles most of it (numpy (H, W) uint8 row-major,
cv2's RotatedRect triple with homr's angle normalisation, an object graph
dumped to JSON with its snake_case names), and the constraints are firm:
strict TypeScript with `noUncheckedIndexedAccess` and
`exactOptionalPropertyTypes`, no `enum`, no ndarray library, values that
run in a Worker and cross `postMessage`, and golden JSON that must parse
into the types with no `any`. What makes the shape non-obvious is which of
homr's stored fields are facts and which are copies (`center`, `size`,
`angle` copy `box`; the four corners ignore the angle; `polygon` is built
from the raw cv2 rect, not the stored one; `Note.center` diverges from
`box.center` after dewarping), and which are vestigial (`has_dot`,
`circle_of_fifth`, `beams`, `flags`, `NoteHeadType` are never read or
written after construction in 0.7.0).

## Usage (caller's view)

The library's own README paragraph for a contributor:

> Images are `Plane`s: `{ kind, width, height, channels, data: Uint8Array }`,
> row-major, one byte per sample, the layout `cv.matFromArray` reads and
> `cv.Mat.data` exposes. A `Mask` is 0/1, a `GrayImage` 0..255, a `ClassMap`
> a segnet class index, a `ColorImage` BGR. Shapes fitted to masks are
> `RotatedBox`, `Ellipse` and `AxisBox`, plain readonly objects holding a
> normalised `rect`, the `polygon` outline and the source `contour` as flat
> `Int32Array` point lists. Staffs, notes and tokens are plain objects too;
> nothing has a prototype, so everything survives structured clone. The
> golden decoders in `src/golden/decode.ts` are the only code that knows
> the Python dump's shape.

Call site 1, phase 3, the segnet stage producing masks (`src/segmentation/tiles.ts`):

```ts
import { argmaxPlanes, createGray, blitInPlace } from "../image/plane.js";
import { createSegmentationResult, SEGNET_INPUT } from "../model/pipeline.js";

export async function segmentPage(page: GrayImage, run: SegnetRunner): Promise<SegmentationResult> {
  const classes = createClassMap(page.width, page.height);
  for (const tile of tilesOf(page, SEGNET_INPUT.window)) {          // extract_patch, padded white
    const logits = await run(tile.input);                            // Float32Array, (6, 320, 320) planar
    const tileClasses = argmaxPlanes(logits, SEGNET_INPUT.classes, tile.width, tile.height);
    blitInPlace(classes, tileClasses, tile.x, tile.y);               // merge_patches with step == window
  }
  return createSegmentationResult(classes);                          // masks split by SEGNET_CLASSES
}
```

Call site 2, phase 4, a box stage consuming a mask and producing boxes (`src/geometry/boxes.ts`, the port of `create_rotated_bounding_boxes`):

```ts
export function createRotatedBoundingBoxes(mask: Mask, options: BoxOptions = {}): RotatedBox[] {
  return withMats((mats) => {
    const src = mats.fromPlane(mask);                                 // cv.matFromArray(rows, cols, CV_8UC1, mask.data)
    const contours = mats.findContours(src);                          // cv.MatVector, deleted on scope exit
    const boxes: RotatedBox[] = [];
    for (const [i, contourMat] of contours.entries()) {
      const raw = cvRotatedRectOf(cv.minAreaRect(contourMat));       // CvRotatedRect, unnormalised
      if (!hasValidSize(raw)) continue;
      const box = createRotatedBox(raw, pointListFromContourMat(contourMat), i);
      if (withinSize(box.rect, options)) boxes.push(box);
    }
    return options.skipMerging ? boxes : boxForWholeGroup(mergeOverlayingBoundingBoxes(boxes));
  });
}
```

Call site 3, the phase 1 test decoding a golden file (`test/golden.test.ts`, written in full in this package):

```ts
const notes = decodeNotes(readGoldenJson(fixture, "notes.json"));     // Note[] or a GoldenError naming the path
for (const note of notes) {
  if (note.stem !== null) expect(["UP", "DOWN"]).toContain(note.stem.direction);
}
const tokens = decodeTokens(readGoldenJson(fixture, "tokens-0.json")); // DecodedSymbol[]: every field a table token
```

## Shape

Module map, chosen so each type lives in the file its phase-4-to-7 functions
will join (the plan's `src/types/*.ts` would put a type and its methods in
different directories, and the runner discipline asks for short traces):

| file | holds |
|---|---|
| `src/image/plane.ts` | `Mask`, `GrayImage`, `ClassMap`, `ColorImage`; factories; the 2-D helpers (`cropPlane`, `blitInPlace`, `fillRectInPlace`, `rowNonzeroCounts`, `nonzeroRowBounds`, `meanOfRegion`, `applyMask`, `maskOfClass`, `argmaxPlanes`, `planeAgreement`) |
| `src/image/numeric.ts` | `mean`, `median`, `std`, `diff`, `sum`, `argmin`, `argmax`, `roundHalfEven`, `floorDiv`, `truncToInt`, `formatPythonFloat` |
| `src/geometry/boxes.ts` | `Point`, `PointList`, `CvRotatedRect`, `RotatedRect`, `Corners`, `RotatedBox`, `Ellipse`, `AxisBox`, `AnyBox`, constructors and accessors |
| `src/model/staff.ts`, `symbols.ts`, `pipeline.ts`, `constants.ts` | `StaffPoint`, `Staff`, `MultiStaff`; the `SymbolOnStaff` union, `Stem`, `NoteheadWithStem`; `SEGNET_CLASSES`, `SegmentationResult`, `InputPredictions`, `PredictedSymbols`, `PageDetection`, `StaffCanvas` |
| `src/transformer/vocabulary.ts`, `symbol.ts` | the six tables, `TokenId<Head>`, `DECODER_OUTPUT_HEADS`; `DecodedSymbol`, `EncodedSymbol` |
| `src/result.ts` | `Backend`, `StaffPosition`, `StaffBox`, `PageText`, `Progress`, `RecognizeResult` |
| `src/golden/decode.ts`, `src/cv/convert.ts` | the two boundaries: Python dumps in, opencv.js Mats in and out |

Data structures, in flow order:

**Planes.** One layout for every image (per `foundational-thinking`: the
layout numpy, cv2 and opencv.js already share). `kind` is a runtime
discriminant so a mask and a gray image are distinct types with the same
bytes; `channels` is 1 or 3, never inferred. A `ClassMap` is the single
source of truth for segmentation; the five masks are derived from it by
`maskOfClass` and then owned independently, because homr filters and
dilates masks in place afterwards. The dominant access patterns (crop a
region, count non-zeros per row, blit a tile, mask a plane) each map to one
helper that names its numpy expression; there is no general slicing, and no
helper allocates a view.

**Point lists.** `PointList = Int32Array` (branded, even length) is the
byte layout of a CV_32SC2 Mat, so contours and polygons go into
`pointPolygonTest`, `minAreaRect` and `fitEllipse` with one
`matFromArray` and come out with one copy. Both homr's `contours` and
`polygon` use it; they differ by field name, not representation.

**Boxes.** `RotatedRect` is homr's normalised triple, branded type-only so a
raw `CvRotatedRect` cannot be stored without passing
`normalizeRotatedRect` (idempotent, as homr relies on). `RotatedBox` and
`Ellipse` are one generic shape over `kind`: `rect`, `polygon`, `contour`,
`debugId`. `polygon` is stored (see the field's comment: it is computed
from the rect handed to the constructor, raw or normalised, and the point
order is not recoverable from the stored rect); `center`, `size`, `angle`
and the four corners are not stored, they are `rect` and `cornersOf(rect)`
(`single source of truth`, derive instead of sync). `AxisBox` keeps its
integer corners and derives everything else. `AnyBox` plus `rotatedRectOf`
and `polygonOf` is what the overlap tests take, replacing the
`isinstance` branch in `_can_shapes_possibly_touch`. `sameRect` and
`rectKey` replace `__eq__` and `__hash__`, the only identity semantics
phase 5 needs.

**Staffs and symbols.** Plain readonly objects with factories, not classes,
because they cross the Worker boundary by structured clone and because the
golden decoder and the detector must produce the same type. Derived
fields that homr computes in constructors and reads in hot loops (`minX`,
`averageUnitSize` as a median) are stored, written only by the factory,
and checked by the decoder against the derivation. `_y_tolerance` is a
constant times a stored field and becomes `yTolerance(staff)`.
`StaffLineYs` is a union of a 5-tuple and a 10-tuple so `y[0]..y[4]` are
numbers under `noUncheckedIndexedAccess`; `Staff.grid` and
`MultiStaff.staffs` are non-empty tuples because the Python constructors
index `[0]` unconditionally. `SymbolOnStaff` is a discriminated union;
`Stem` folds homr's always-paired `stem` and `stem_direction` into one
nullable object. `Note.center` is stored because dewarping moves it while
`box` stays in page space. The four vestigial `Note` fields are dropped,
and the decoder asserts they are still at their defaults.

**Tokens.** The six vocabularies are `as const` tables printed from the
installed homr, giving literal-union token types. `TokenId<H>` is a
per-head brand for the one place ids exist, the decoder's int64 inputs
and logits. `DecodedSymbol` is closed over the tables (what `tokens-n.json`
holds, and the proof the tables contain what the model emits);
`EncodedSymbol` is open in `rhythm`, `articulation`, `slur` because homr
rewrites those into strings outside the tables (`newline`, `note_14`,
re-joined lists), and it is a supertype so the decoder's output flows into
post-processing without a cast.

**Result.** `RecognizeResult` is a union on `ok`, so `musicXml` exists
exactly when the run succeeded, with the Go route's field names.

Interface depth: the public surface is types, factories and about twenty
helpers; behind it sits every representation decision (row-major bytes,
flat int32 points, normalised angles, stored-versus-derived) and every
Python-compatibility rule (banker's rounding, numpy's even median,
population std, `str(float)`). Callers never see a Mat, a JSON field name
or a raw cv2 angle (per `boundary-discipline`: the two decoders validate,
everything inside trusts the types). Nothing here is a pass-through: each
helper either hides a numpy semantic or a copy.

What the design deliberately does not do: no strided views, no generic
`slice`, no class hierarchy, no `DebugDrawable`, no retained raw cv2 rect,
no `NoteHeadType`, no `has_dot`.

## Synthesis decision

Three candidates were written in parallel on 2026-09-25 (Opus, Fable,
Sonnet) from one grounding, scored by an independent cross-judge against a
six-criterion rubric, and read end to end by the orchestrator. This text is
the Fable candidate, chosen as the base because it won or tied every
criterion (18 of 18 against 14 and 11) and because its three load-bearing
choices are the ones a phase 5 implementer could not re-derive cheaply:
`polygon` stored (the judge measured that deriving it from the normalised
rect reproduces only 19 of 377 golden lists), `rect` branded so an
un-normalised cv2 triple cannot be stored, and the open `EncodedSymbol`
beside the closed `DecodedSymbol` (voices.json holds `newline`, which is in
no vocabulary).

Grafted from Opus: `clampToIndex` (the exact `_limit_x` composition) and
`toFloat32` with its "only where cv2 produced the value" rule in
`image/numeric.ts`; a reader-injected golden facade in `src/golden/page.ts`
so that phases 4 to 9, which take golden data as *input*, never import from
`test/`; and `space: "page" | "canvas"` on `Staff`, justified by the
measured divergence of a symbol's `center` from its `box` after dewarping.
Grafted from Sonnet: the observation that a vocabulary table should re-pin
as a diff; done here by generating the tables from the dumper's
`vocabulary.json` (`tools/gen-vocabulary.mjs`) rather than by porting
`build_rhythm`, which keeps one construction of the tables (Python's).

Rejected: Opus's claim that `boxPoints`/`ellipse2Poly` can be derived
(measured false); its rhythm brand parsed at the golden boundary (breaks
voices.json); its execution-ordered `pipeline/stages.ts`. Sonnet's
`src/types/` layout, its non-recomputing `decodeStaff` (discards a free
oracle check), its `(0, 90]`/`(-45, 45]` angle bounds (34 golden boxes sit
at exactly -45) and its sentinel rule that would reject the first Kesh
token. Fable's `StaffLineYs` 5-or-10 tuple narrows Python's "any multiple
of five"; kept, noted. All three candidates overlooked that
`merge_patches` averages overlapping edge tiles; phase 3 must reproduce
that, not overwrite.

Deviations made while implementing: the cv2-dependent constructors
(`createRotatedBox`, `createEllipse`, `cv/convert.ts`) are left to phase 4
so the library ships no `not implemented` body; `rotatedBoxFromParts` and
`ellipseFromParts` assemble boxes from values homr already computed, which
is what the decoder and, later, phase 4 both need.

## Tradeoffs accepted

- We accept storing `polygon` (tens of ints per ellipse, thousands of
  ellipses on a dense page) in exchange for exact equality with the golden
  point lists and no recomputation inside the O(n²) overlap loop.
- We accept storing `minX`/`maxX`/`minY`/`maxY`/`averageUnitSize` on a
  `Staff`, a derived-field duplication, in exchange for not recomputing a
  median per `is_on_staff_zone` call; the factory is the only writer and
  the decoder verifies the derivation on every golden file.
- We accept `EncodedSymbol.rhythm: string` (losing the literal union after
  decoding) in exchange for not lying to the compiler about `newline` and
  the tuplet rewrites; `DecodedSymbol` keeps the strict type where it is
  true.
- We accept dropping four `Note` fields homr carries, in exchange for a
  smaller type; the decoder's constant-check is the tripwire if upstream
  starts using them.
- We accept a runtime `kind` byte on every plane and box, in exchange for
  discriminants that survive structured clone and can be asserted at the
  boundary; a phantom brand would cost nothing and check nothing.
- We accept `Uint8Array` for colour images instead of the plan's
  `Uint8ClampedArray`, so every plane shares one element type and one
  `data` signature; clamping never applies to bytes that are already in
  range, and `colorImageFromRgba` is the single conversion from the
  canvas.
- We accept twelve files instead of the plan's five so that each type sits
  in the file its methods will join in phases 4 to 7; the trace from mask
  to token still crosses four files, one per domain.
- We accept a type-check-only sketch: bodies throw, the golden test will
  fail until phase 1's implementation lands, which is the intended order.

## Alternatives considered

- **Classes for boxes and staffs** (a `RotatedBoundingBox` class with
  `isOverlapping`, a `Staff` class with `getAt`). Deeper-looking surface,
  but every instance loses its prototype at `postMessage`, so phase 9 would
  need a rehydration layer for exactly the values it returns, and the
  golden decoder would have to construct through the class's own
  invariants or bypass them. Plain objects with factories hide the same
  complexity (the factory is the constructor) and clone for free.
- **A strided `NdArray<T>` wrapper** (or `ndarray`/`numjs`). Would let
  `crop` return a view and port numpy slices literally, but every opencv.js
  call needs contiguous bytes, so views become hidden copies at each Mat
  boundary, and homr uses at most a dozen array idioms. The explicit
  helpers expose exactly one decision to callers (a crop is a copy) and
  hide all the index arithmetic.
- **Deriving `polygon` from the stored normalised rect** (no `polygon`
  field). Cleaner single-source, but reproduces the golden lists only up to
  point order and int truncation, and the merging loop would recompute a
  polygon per pair test. Rejected on both counts.
- **Keeping `stem` and `stemDirection` as two nullable fields** as homr
  does, easing the line-by-line port. It leaves "both or neither" as prose
  every consumer must re-check; the paired object costs one field access
  (`note.stem.direction`) and makes the illegal state unrepresentable.
- **Flat `RecognizeResult` with `ok: boolean` and always-present `error`
  and `musicXml`**, byte-identical to the Go JSON. Serialises the same, but
  types `musicXml` as present on failure; the union is what the app's
  `if (!result.ok)` branch already assumes.

## Open questions and risks

- Should `RecognizeResult` be the union proposed here, or must the object
  literally carry `engine`, `error: ""` and `musicXml: ""` on every answer
  so the app's existing `OmrRecognize` type-checks it unchanged? Phase 10
  can wrap either, but the answer decides whether phase 9's public type is
  the app's or the library's.
- `noUncheckedIndexedAccess` makes every `data[i]` on a typed array
  `number | undefined`; the helpers absorb it in `src/image`, but phases 4
  to 6 will index arrays in their own loops. Is a `?? 0` idiom acceptable
  in geometry code, or should the geometry files relax that flag locally?
- Is `cv.ellipse2Poly` in the opencv.js build the plan pins? The plan notes
  `boxPoints` is absent and reimplements it; if `ellipse2Poly` is absent
  too, `createEllipse` needs a port that reproduces OpenCV's integer
  stepping exactly, since the point list is compared against the golden.
- `staff-positions.txt` must be byte-exact and is written with Python's
  `str(float)`; `formatPythonFloat` handles the `1.0`/`1` and exponent
  differences, but does the owner want the byte-exact rule kept, or is a
  parsed comparison (five floats per line) acceptable, which would remove a
  formatting port that carries no musical meaning?
- The golden dumper does not yet write `vocabulary.json`; phase 1 adds it
  (six `dict[str, int]` from `Vocabulary()`). Agreed that a dumper change
  is in phase 1's scope, given it regenerates 42 files that must stay
  byte-identical?
- Grand staffs (10-line `StaffPoint`s, `isGrandstaff`, non-empty
  `connections`) have no public fixture; the tuple type and the sorted
  `staffs` invariant are unexercised until a two-staff page is typeset.

## Next implementation step

Implement `src/image/numeric.ts` and `src/golden/decode.ts` against
`test/golden.test.ts`, so that every Kesh JSON stage decodes and the
derived-field assertions (corners, median, mean, tolerance) pass before any
geometry is written.

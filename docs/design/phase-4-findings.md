# Phase 4, measured before any code: what the two OpenCV builds actually do

Working notes for the phase-4 port of `bounding_boxes.py`. Everything here was
measured on 2026-09-28 against the checked-in Kesh fixtures, not inferred. The
design record proper is `phase-4-boxes.md`; this file is the evidence it rests
on, and the numbers in it are the acceptance targets.

## The oracle's venv was running the wrong OpenCV, and it was not the cause

`rapidocr` depends on an unpinned `opencv-python`; homr depends on
`opencv-python-headless <5`. Both wheels install the same `cv2` package, so the
last one installed owns the import, and on this machine that was
`opencv-python 5.0.0.93` with `opencv-python-headless 4.14.0.94` shadowed behind
it. `tools/venv.sh` already had the corrective swap but guarded it with
`uname = Linux`, where it had only ever been about libGL, so macOS ran OpenCV 5
unnoticed. The guard is gone and the script now asserts `cv2.__version__`
starts with `4.`.

**The fixtures are not affected.** OpenCV 4.14.0 and 5.0.0 agree on
`minAreaRect` and `boxPoints` for every case tested, and under 4.14.0 the
checked-in `boxes-*.json` reproduce byte for byte, including the `-45` entries.
No re-dump is needed. `meta.json` records the homr version and all three model
hashes but not the OpenCV version, which is why this could hide; phase 4 adds it.

## The real divergence: the two builds use opposite angle conventions

`@techstark/opencv-js` 4.12.0 reports `minAreaRect` angles in `(0, 90]`, the
post-4.5.1 convention. `opencv-python` 4.14.0 reports them in `[-90, 0)`, the
pre-4.5.1 one. Measured on identical integer corner sets across a 0 to 90 degree
sweep:

| true rotation | opencv.js | opencv-python | size |
|---|---|---|---|
| 0 | +90.000 | -90.000 | same |
| 10 | +9.926 | -80.074 | swapped |
| 45 | +45.000 | -45.000 | swapped |
| 80 | +80.074 | -9.926 | swapped |
| 90 | +90.000 | -90.000 | same |

homr's own normalisation absorbs this almost everywhere, because a rectangle at
`-90` and one at `+90` normalise alike. It does **not** absorb it at exactly
`±45`, because homr's comparisons are strict (`angle > 45`, `angle < -45`), so
`+45` stays `+45` while `-45` stays `-45` with the dimensions the other way
round. On the Kesh page that is 23 of 377 `stems_rest` entries and 63 of 340
broken fragments: 86 entries that would have looked like a porting bug.

**The conversion is exact and total.** Subtract 90 from the angle and swap width
with height, repeating while the angle is at or above zero. Applied to the raw
opencv.js rect before homr's normalisation, every angle mismatch disappears:

| golden file | n | worst centre/size delta | angle mismatch > 1e-3 | polygon exact |
|---|---|---|---|---|
| `boxes-bar_lines.json` | 103 | 0 | 0 | 103 / 103 |
| `boxes-clefs_keys.json` | 7 | 1.5e-5 | 0 | 7 / 7 |
| `boxes-staff_fragments.json` | 147 | 1.2e-4 | 0 | 144 / 147 |
| `boxes-stems_rest.json` | 377 | 1.2e-4 | 0 | 376 / 377 |
| `boxes-staff_fragments-broken.json` | 340 | 2.4e-4 | 0 | 326 / 340 |
| `boxes-noteheads.json` | 81 | 2.4e-4 | 0 | 81 / 81 |

## The rect tolerance has to cover the angle, and the plan does not say so

`testing.md` allows boxes "exact count and order, centres and sizes within
1e-3". The angle needs a tolerance too: it is stored as float32 in Python and
recomputed in float64 here, so `0.1752166748046875` against
`0.17521589994430542` is an ordinary outcome, 7.7e-7 apart. 1e-3 on the angle
covers every entry measured with room to spare. The cause is identified, which
is what `testing.md` requires of a widening.

## Polygons cannot be bit-exact, and the reason is not a rounding mode

18 entries of 967 differ by one pixel on one corner. It is not the rounding
mode: `Math.trunc`, `Math.floor` and a `Math.fround` pass all score
identically, and `Math.round` scores worse, which confirms Python's
`.astype(np.int64)` truncation is already reproduced. The cause is the rect
itself, 1e-4 away between the two builds, putting a corner either side of an
integer. Hand-rolling `boxPoints` from OpenCV's C++ formula in float32 matched
0 of 967, so `cv.boxPoints` on the converted rect is the right call and the
ordering it produces is correct.

**This is the one finding with downstream risk.** The polygon feeds
`do_polygons_overlap`, so a one-pixel corner can in principle flip an overlap
for a pair that merely touches, and a flipped overlap changes the merge
grouping and therefore the final list. Whether any of the 18 sits on such a
pair is not knowable until the merge runs; the golden tests are what will say.

## `create_bounding_ellipses` does not store a `fitEllipse` rect

Comparing each notehead's stored rect against `fitEllipse` of its own contour
is wrong and diverges wildly, including in the centre. The stored rect is
`minAreaRect` over the group's concatenated contours, because
`_get_ellipse_for_whole_group` refits with `minAreaRect` and every box reaches
it, singleton groups included. Through that route the 81 noteheads agree to
2.4e-4 with no angle mismatch.

`fitEllipse` still matters, twice: a contour of fewer than 5 points is dropped
before it, and the ellipse's `polygon` comes from `ellipse2Poly` on the raw
fitEllipse rect, which is what the merge's overlap test reads. So fitEllipse
decides *which* boxes merge while `minAreaRect` decides what the merged box is.

**And `fitEllipse` needs no convention conversion, where `minAreaRect` does.**
Measured across all 81 noteheads on 2026-09-28: opencv.js 4.12.0 and
opencv-python 4.14.0 agree **bit for bit**, every centre, size and angle, worst
delta exactly 0. Its angle range on this page is 10.761 to 67.928, consistent
with `[0, 180)` in both builds. So the two raw-rect producers sit in *different*
conventions, and one "raw cv rect" type is the wrong shape: converting a
fitEllipse rect would corrupt it as surely as not converting a minAreaRect one.
The polygon of a returned notehead is `ellipse2Poly` of the **converted
minAreaRect** rect the merge refit produced, and that is exact, 81 of 81.

## opencv.js 4.12.0 has everything this phase needs

Probed directly, all present: `minAreaRect`, `fitEllipse`, `boxPoints`,
`ellipse2Poly`, `HoughLinesP`, `intersectConvexConvex`, `pointPolygonTest`,
`convexHull`, `dilate`, `erode`, `getStructuringElement`,
`rotatedRectPoints`. **`boxPoints` is present and returns a plain JavaScript
array of `{x, y}` at float32 precision**, which contradicts
`docs/design/phase-1-types.md`'s claim that it is absent and must be
reimplemented; that claim should be corrected rather than acted on.

Two call shapes differ from Python and will bite:

- `ellipse2Poly` takes a 7th argument and it must be a `PointVector`, not a
  `Mat`. Passing a `Mat` throws `Expected null or instance of PointVector`.
- `rotatedRectangleIntersection` was not probed; `is_intersecting` needs it.

## What the golden files let phase 4 verify, and what belongs to phase 5

`tools/dump-golden.py` writes `mask-filtered-*.png` after `filter_predictions`
**and** after `make_lines_stronger(staff, (1,2))`, so those five masks are
phase 4's inputs directly, satisfying the Python-input rule with nothing from
phase 5 ported.

Of the seven box files, five come from `predict_symbols` and are phase 4's:
`noteheads`, `staff_fragments`, `clefs_keys`, `stems_rest`, `bar_lines`. The
`bar_lines` one needs `prepare_bar_line_image`, a single `cv2.dilate` with a
5x3 kernel, which lives in `bar_line_detection.py` but has to come along
because it sits inside `predict_symbols`. `boxes-brace_dot.json` is testable
too, since `mask-brace_dot.png` is dumped, but it holds 0 entries on this page
so it proves only that nothing is found. `boxes-staff_fragments-broken.json`
belongs to phase 5: `break_wide_fragments` is `staff_detection.py`.

## Exact arguments each of the five uses

Six call shapes exist in the whole pipeline, so the port's parameters can be
exactly these and no more. The sixth is `brace_dot`'s, below, and it is the only
one anywhere that exercises the "a bound of 0 or less disables that dimension"
rule, so that rule is not dead and must not be trimmed.

- `create_bounding_ellipses(notehead, min_size=(4, 4))`
- `create_rotated_bounding_boxes(staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100))`
- `create_rotated_bounding_boxes(clefs_keys, min_size=(20, 40), max_size=(1000, 1000))`
- `create_rotated_bounding_boxes(stems_rest)`
- `create_rotated_bounding_boxes(dilate(stems_rest), skip_merging=True, min_size=(1, 5))`
- `create_rotated_bounding_boxes(brace_dot, skip_merging=True, max_size=(100, -1))`

`thicken_boxes` is never passed a non-default value anywhere, and neither is
`create_bounding_ellipses`'s `max_size` or `skip_merging`. `create_lines` is
reached only from `staff_position_save_load.detect_staff_simple`, behind
`--read-staff-positions`, and the golden dump never exercises it, so it has no
fixture coverage at all.

## Asymmetries in the Python that a shared helper would erase

- Both `max_size` filters reject if **either** dimension exceeds its bound. The
  asymmetry is elsewhere and sharper: `create_rotated_bounding_boxes` guards each
  dimension with `max_size[k] > 0`, so a bound of `0` or less disables that one
  dimension, and `create_bounding_ellipses` has **no such guard at all**. Passing
  `(100, -1)` to the ellipse factory would therefore reject every ellipse rather
  than disabling the height check. An earlier version of this document said the
  ellipse filter used `and`; it does not, and the `> 0` guard is the real
  difference. Read lines 360 to 440 of the source, not this summary.
- `create_rotated_bounding_box`, singular, skips the `_has_box_valid_size`
  check the plural one applies.
- `BoundingEllipse.make_box_taller` returns a `RotatedBoundingBox`, silently
  rectangularising the shape. Preserve it.
- Merged boxes get `debug_id = 0`, discarding the contour index.
- `_merge_groups_optimized`'s output order is a deterministic consequence of
  the union-by-rank tie-break and Python dict insertion order, and it sets the
  contour concatenation order that the refit `minAreaRect` sees. A JS `Map`
  reproduces it; "any equivalent grouping" does not.
- `do_polygons_overlap` only tests whether a **vertex** of one polygon lies in
  the other. Two quads crossing like a plus sign are reported as not
  overlapping. Reproduce the weaker test, not a correct one.

## Two more things measured while designing, both load-bearing

**`findContours(RETR_TREE, CHAIN_APPROX_SIMPLE)` is identical across the two
builds** on all four filtered masks, same count, same order, same points: 188,
387, 19 and 82 contours. Nothing had checked this, and `debug_id` is the contour
index, so the whole phase rests on it.

**The full `stems_rest` fit, merge included, reproduces 377 of 377**, with
byte-identical concatenated contours and `debug_id`s, in 311 ms. That closes the
one downstream risk this document left open: on this page no polygon corner
difference flips a merge. `clefs_keys` is 7 of 7 fully exact and `bar_lines` 103
of 103 bit-exact on everything. The 18 inexact corners split 3, 1 and 14, so
phase 4 owns 4 boxes and 5 coordinates and the other 14 belong to phase 5's
`staff_fragments-broken`.

**A fixed polygon epsilon would be vacuous.** Over 90 % of the 5072 corner
coordinates are exactly integers, so a 1e-2 bound accepts 95.3 % of them for free.
In all five cases phase 4 owns, the port computes the exact integer and Python's
float32 landed just below it.

## Correction, measured during the port: the polygon difference is a formula
## change in OpenCV, not the rect

Two claims above are wrong, and the port could not reproduce
`boxes-staff_fragments.json` until they were retested.

**"The cause is the rect itself, 1e-4 away between the two builds."** For 3 of
the 18 inexact corners it is not: those entries' rects are bit-identical, centre,
size and angle, and the polygon still differs by one pixel on one corner. No
tolerance conditioned on the stored rect can admit them, which is what the
synthesis's fork 4 asks for.

**"Hand-rolling `boxPoints` from OpenCV's C++ formula in float32 matched 0 of
967."** It matches everything, once the right formula is used. OpenCV's
`RotatedRect::points` derives the first two corners from the centre, the angle
and the size; the older form then *reflects* the other two through the centre
(`pt[2] = 2 * centre - pt[0]`), while opencv-python 4.14.0 derives all four
directly. On a bit-identical rect the reflection can land exactly on an integer
where the direct form lands 3e-5 below it, and `int()` then differs by one.

Measured 2026-09-28 over every entry of the four rotated golden lists whose raw
`minAreaRect` is bit-identical between the builds (131 + 359 + 2 + 103 = 595 of
634), comparing floats and not truncations:

| claim | result |
|---|---|
| `cv.boxPoints` (opencv.js 4.12.0) equals the reflecting formula in float32 | 595 / 595 |
| the direct formula in float32 equals opencv-python 4.14.0's `boxPoints` | 595 / 595 |
| `cv.boxPoints` equals opencv-python 4.14.0's `boxPoints` | 591 / 595 |

So `cv.boxPoints` is not the right call for this port and `src/cv/box-fitting.ts`
reimplements the direct formula. With it, phase 4's own five lists reproduce as
81/81, 147/147, 7/7, 377/377 and 103/103, every polygon bit-exact except one
`stems_rest` corner whose rect is genuinely inexact, and the fork 4 tolerance is
unchanged.

**Phase 5 inherits the rest.** `boxes-staff_fragments-broken.json` goes from 326
to 328 of 340 polygons exact under the direct formula, so 12 remain and they are
`break_wide_fragments`' to explain, not this one's.

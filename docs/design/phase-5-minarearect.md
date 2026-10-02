# Phase 5: the minAreaRect divergence on grand-staff-300dpi noteheads 18 and 48

Measured 2026-10-02 on arm64 macOS. Node 23.5.0, @techstark/opencv-js 4.12.0-release.1, opencv-python-headless 4.14.0.94.
This is the diagnosis as it was written, with the investigation's throwaway scripts left out: they were never committed.
Nothing it proposes is applied. Whether the port should match the arm64 oracle or an x86_64 server is an open decision, so `minAreaRect` and the `KNOWN_DIVERGENCES` pin are as phase 4 left them.
The same cause explains the centre tolerance of `BOX_RECT_TOLERANCE`, see the last section.

## Symptom reproduced

The call that differs is `minAreaRect`, on the merged group's contour, in `refitEllipseFromGroup`. It is not `fitEllipse`.
`fitEllipse` on both contours is bit-identical between opencv.js and Python.
The convex hull is also identical between the two builds on all 2573 golden contours.

Fed the exact golden contour (31 and 33 points), after the port's ladder and homr's normalisation:

| entry | Python 4.14.0 (golden) | opencv.js 4.12.0 |
|---|---|---|
| noteheads[18] | centre (695, 1499), size 22 x 18, angle 0 | centre (695.75006, 1499.25012), size 23.334520 x 16.970562, angle -45 |
| noteheads[48] | centre (1465, 965.5), size 22 x 17, angle 0 | centre (1465.00012, 965.50006), size 24.596748 x 15.205260, angle -26.565048 |

Across every golden box entry of both fixtures (13 lists, 2573 contours):
2111 rects are bit-identical, 460 differ by at most 1e-3, and 2 are wrong. The 2 are these entries.
Across every raw contour of all 24 golden mask PNGs (3657 contours):
3129 identical, 524 within 1e-3, 4 wrong. The 4 are the same two noteheads, once in the raw mask and once in the filtered one.

## Root cause

OpenCV's `rotatingCalipers` breaks exact area ties by float32 rounding, and the two builds round differently.
The native arm64 build uses fused multiply-add. WebAssembly has no fused multiply-add.

The evidence chain:

1. Both rectangles are true minima. In exact rational arithmetic the upright rectangle and the rotated one have the same area.
   Notehead 18: 22 x 18 = 396, and the rectangle on the slope 1:1 hull edge is (33/sqrt 2) x (24/sqrt 2) = 396.
   Notehead 48: 22 x 17 = 374, and the rectangle on the slope 1:2 hull edge is (55/sqrt 5) x (34/sqrt 5) = 374.
   Both products were computed with BigInt fractions.
2. `rotatingCalipers` walks the hull edges and keeps a candidate when `area <= minarea`, in float32. On an exact tie the last candidate wins.
   The upright candidate is visited last on both contours (step k=14 of 15 and k=13 of 14), so on a true tie upright wins.
3. For an axis-aligned candidate the base vector is exactly (1, 0), so its area is exact. For a rotated candidate the base vector
   is a rounded float32, and `width = dx*a + dy*b`, `height = -dx*b + dy*a` are each rounded.
4. Apple clang compiles `dx*a + dy*b` as one `fmul` and one `fmadd`. The disassembly of `cv::minAreaRect` in the venv's
   `cv2.abi3.so` shows 16 fused multiply-add instructions and one `fmls`. The build flags are `-O3` with no `-ffp-contract`, so clang's default applies.
5. With the fused form the rotated candidate's area rounds to exactly 396 (and 374). The tie holds, and upright wins as the last visited.
   With two separate roundings one factor lands one ulp lower and the area becomes 395.99997 (and 373.99997).
   That is strictly less than 396, so the rotated rectangle becomes the minimum and the later upright candidate fails `<=`.
   Trace, notehead 18, edge (-3, -3): fused height 23.334524154663086, area 396. Unfused height 23.334522247314453, area 395.9999694824219.
   Notehead 48, edge (-6, 3): fused width 15.205262184143066, area 374. Unfused width 15.20526123046875, area 373.9999694824219.
6. A float32 port of the algorithm confirms it in both directions.
   With the fusing read off the disassembly and the 4.14 tail, it reproduces Python 4.14.0 bit for bit on 2573 of 2573 raw rects.
   With no fusing it returns the same rotated rectangles opencv.js returns on entries 18 and 48.

The other three candidates are ruled out by measurement:

- Not the OpenCV version. opencv-python-headless 4.12.0.88, installed in a second virtual environment, returns the upright rectangle on both entries.
  Native 4.12 and native 4.14 never differ by more than 1e-3 after normalisation (2164 identical, 409 small, 0 wrong).
  Native 4.12 and wasm 4.12 are the same source and differ on exactly these two entries. So it is the build, not the release.
- Not `findContours` or point order. The contour is the golden one and the hull is identical in both builds.
- Not float32 versus float64. Both builds compute in float32. The difference is one rounding versus two.

One version fact did surface. OpenCV 4.13 rewrote the tail of `minAreaRect` (`rotcalipers.cpp` of 4.12.0 against 4.13.0, 62 diff lines; 4.13 and 4.14 are identical).
It now reports the angle in [-90, 0) natively and derives width and height from the other vector.
That is why opencv-python 4.14 and opencv.js 4.12 sit in different conventions, and it accounts for part of the 460 small differences.
The caliper loop itself did not change.

## Why the others match

The ambiguous class is small, and rounding usually does not flip it.
Of the 2573 golden contours, 53 have an exact tie between two differently oriented rectangles.
By list: grand-staff stems_rest 23, staff_fragments-broken 8, noteheads 2. Kesh stems_rest 15, noteheads 3, staff_fragments-broken 2.
In 51 of the 53 the fused and unfused arithmetic choose the same rectangle. In 2 they do not, and those are entries 18 and 48.
All other contours have a unique minimum with a margin far above one ulp, so both builds agree up to the 1e-3 noise.
The fused and unfused loops stop on a different step in 22 contours overall, but in 20 of them it is the same rectangle reached from another side.

Python's choice inside the tie class has no geometric rule. It picks the upright rectangle in 12 of the 53 ties and a rotated one in 41.
In 38 of those 41 an upright rectangle of equal area was available. So "prefer upright on a tie" would fix these two and break 38 matching boxes.

## Proposed fix

Replace the call to `cv.minAreaRect` with a pure TypeScript port of `rotatingCalipers` and the 4.14 tail of `minAreaRect`, with the fused multiply-adds emulated.
This is what phase 4 did for `boxPoints`. The hull still comes from `cv.convexHull`, which is integer-only and identical in both builds.

Files it would touch:

- New `src/geometry/min-area-rect.ts`, 194 lines, pure, no opencv.js. The prototype passes `tsc --noEmit` under the repository's config and is not committed.
- `src/cv/box-fitting.ts`, the one call site, `minAreaRectOf`. The change is three lines.
- `test/boxes-golden.test.ts`: remove `"grand-staff-300dpi": { noteheads: [18, 48] }` from `KNOWN_DIVERGENCES`. The test's own tripwire demands it.

The fused multiply-add is emulated exactly. The product of two float32 values is exact in a double, the sum is rounded to odd, then `Math.fround` rounds once.

Before and after, measured three ways:

| measure | before | after |
|---|---|---|
| golden box entries, raw rect against Python, 2573 | 2111 identical, 460 within 1e-3, 2 wrong | 2573 bit-identical |
| raw mask contours, 3657, of which 276 have a hull of 1 or 2 points | 3129 identical, 524 within 1e-3, 4 wrong | 3657 bit-identical |
| `test/boxes-golden.test.ts`, 10 lists, 1649 boxes, rect | 1647 inside tolerance, 2 pinned as known | 1649 with centre, size and angle delta exactly 0 |
| same test, polygons | 1641 exact, 6 within 1 px, 2 failing | 1649 exact, 0 within 1 px |
| full vitest suite | 26 files, 406 passed, 1 skipped | 26 files, 406 passed, 1 skipped |

The last three rows were run in a copy of the working tree as it stood during stage 3, with the fix applied.
No currently matching box moves away from Python. Every one of them moves onto it exactly.

What follows from the fix, for whoever applies it:

- The port returns Python's convention directly. `toLegacyAngleConvention`, the `RawMinAreaRect` brand and `rawMinAreaRectOf` lose their only producer and can be deleted. The prototype patch keeps them with a cast to stay minimal.
- The rect tolerance in `src/golden/box-tolerance.ts` and the one-pixel polygon allowance are no longer exercised by these five lists.
- `src/cv/opencv.ts` says the version is pinned at 4.12.0 because of `minAreaRect`'s angle convention. That reason goes away.
- `rotcalipers.cpp` carries OpenCV's BSD-style licence header (Intel, 2000). `NOTICE` should name it.

The honest caveat. Python's result on these two contours is an artefact of one compiler on one architecture, not a property of homr or of OpenCV's algorithm.
The x86_64 manylinux wheel of the same opencv-python-headless 4.14.0.94 has no fused instruction in `minAreaRect`. Its disassembly shows only `mulss`, `addss` and `subss` (near 0xcaad28 in its `cv2.abi3.so`).
The unfused emulation with the 4.14 tail returns the rotated rectangles. So homr on x86_64 Linux very probably returns what opencv.js returns today. I could not run it, since this machine has no Rosetta.
The fix therefore reproduces the arm64 macOS oracle that `meta.json` records (`"machine": "arm64"`), which is the port's stated reference. It does not make the port more correct than it is now.
No rule simpler than copying OpenCV's arithmetic reproduces the oracle. Copying it took 194 lines and one disassembly.

## Downstream effect if unfixed

The consumers of a notehead box in phase 5:

- `combine_noteheads_with_stems` sorts by `box[0][1]`, tests stems against the polygon of `make_box_thicker(15)`, and compares `center[1]` with the stem's for the direction.
- `main.py` takes the median of `size[1]` as the average notehead height for `detect_bar_lines`, and drops bar lines that overlap a notehead polygon.
- `add_notes_to_staffs` reads `center[0]`, `size[0]` and `size[1]` against the unit size, `top_left` and `bottom_right` truncated to int for `split_clumps_of_noteheads`, and the centre for `find_position_in_unit_sizes`.

Measured by running the repository's own dumper into a scratch directory twice, once untouched and once with `cv2.minAreaRect` answering opencv.js's values on the two contours.
The untouched run reproduces the checked-in golden directory byte for byte, so the harness is sound.

- 42 of the 48 stage files are byte-identical. That includes `barlines.json` with its average notehead height, `staffs.json`, `notehead-splits.json`, all four `tokens-*.json`, `voices.json` and `page.musicxml`.
- 6 files differ: `boxes-noteheads.json`, `noteheads-with-stems.json`, `notes.json`, `multistaffs.json`, `canvas-1-staff.json`, `canvas-3-staff.json`.
- The same 102 noteheads get the same stems, the same stem directions and the same staff positions. No clump is split differently.
- The two boxes differ in centre, size, angle and ellipse polygon (81 to 77 points and 77 to 71).
- One ordering change. Notehead 18's centre y moves from 1499.0 to 1499.25, past a neighbour at 1499.00012. Entries 83 and 84 swap in `noteheads-with-stems.json`, in `notes.json` and in that staff's symbol list.

So on this page the music is unchanged, and three phase 5 golden comparisons would each fail on two entries and one swap.
On another page the risk is real but narrow. The truncated bounding box of notehead 48 grows from 22 to 25 wide and shrinks from 17 to 16 tall, and those feed the split and size thresholds.
The patched run changes only these two boxes. It does not model the 1e-3 noise on the others, which the phase 4 tolerance already covers.

## Open questions

- x86_64 Python was not run. The claim that it returns the rotated rectangles rests on the disassembly plus the emulation. One run of the dumper on an x86_64 Linux host would settle it.
- The Linux aarch64 wheel was not analysed. GCC may fuse a different product of each pair than clang does, which would give a third tie-break. I did not locate the function in that stripped binary.
- `Math.atan2` matched the native libm on all 3657 angles after rounding to float32. Nothing guarantees it on every JavaScript engine. A mismatch would be one float32 ulp in the angle and never a different rectangle.
- The unfused emulation matches opencv.js 4.12 bit for bit on 2492 of 2573 rects and within 1e-3 on the other 81. I did not chase those 81. They do not bear on the fix.
- Whether the oracle should be arm64 macOS at all is a project decision. Production homr runs on Linux. If the oracle moved to x86_64, the fix would be the same file with the fusing switched off, and the goldens would need a re-dump.
- Two comments describe the convention split by version and are out of date: `src/geometry/boxes.ts` says opencv-python 4.x reports [-90, 0), which is true from 4.13 only, and `tools/venv.sh` says 4.x reports (0, 90] and 5.0 reports [-90, 0).

## What this says about the rect tolerance

The same arithmetic accounts for the small differences, not only for the two wrong boxes.
Of the 2573 golden rects, 460 differ from Python's by at most 1e-3 under opencv.js and none differs under the emulated fused port.
The broken staff fragments are among them: 142 of 559 on the grand-staff page and 85 of 340 on Kesh are not bit-exact today.
The largest centre delta of those lists is fragment 275 of the grand-staff page, 3.66e-4 in x, which is three float32 ulps at x 1776.
That entry is why `BOX_RECT_TOLERANCE.center` is 4e-4. The bound stays while `cv.minAreaRect` does the fitting, and has nothing left to absorb once the calipers are ported.

# Phase 4: the synthesised design, and which candidate each half came from

Two candidates were drawn independently, `design-A.md` and `design-B.md` in the
session scratchpad. They converged on three forks and split on one, and the split
was settled by measurement rather than by preference. The evidence is
`phase-4-findings.md`; the decisions are `../decisions.tsv`.

## Fork 1, the convention conversion: B's brands, A's conversion

**Two producer-specific brands, not one raw-rect type.** `RawMinAreaRect` and
`RawFitEllipseRect` are distinct branded types, `LegacyConventionRect` is what
either becomes, and `normalizeRotatedRect` narrows to accept only
`LegacyConventionRect | RotatedRect`.

Candidate A proposed one `legacyRectOf(raw, source)` with `source` a string
naming the producer. That is the same information carried where the compiler
cannot check it: passing `"fitEllipse"` for a `minAreaRect` result compiles and
silently corrupts the rect. **The measurement is what makes this fork real**, and
it arrived after both candidates were briefed: `fitEllipse` agrees bit for bit
across the two builds and must *not* be converted, while `minAreaRect` must be.
So the wrong conversion is a live hazard in both directions, and a brand per
producer turns it into a compile error.

The conversion body is A's, measured: subtract 90 from the angle and swap width
with height, while the angle is at or above zero. Two iterations at exactly 90.

`CvRotatedRect` is deleted; it is an unbranded alias nothing uses, and it was the
door phase 1 left open. `axisBoxOf` narrows from `RotatedBox | Ellipse` to
`RotatedBox`, matching Python, where `to_bounding_box` exists only on
`RotatedBoundingBox`. Both were B's findings about already-shipped code.

## Fork 2, the module split: B's placement, A's overlap tester

The merge machinery is **pure and opencv-free**, as B placed it. Union-find and
the insertion-ordered grouping need no `cv` at all, and opencv.js under vitest is
the most fragile part of this repo's test setup, so the one piece of this phase
with a subtle ordering contract should be testable without it.

`predictSymbols` goes in `src/pipeline/`, also B's, because phase 5 already plans
that directory for `detect.ts` and the two belong side by side.

From A, and not in B: **an `OverlapTester` that memoises one polygon Mat per box
for the caller's scope.** The merge is O(n²) in `is_overlapping` over a few
hundred boxes, and building a Mat per comparison is roughly 75 000 allocations
per page. This is a measured necessity, not a refinement, and it is the one place
where a Mat's lifetime crosses more than one call.

**The trap worth repeating from B: `Map<number, S[]>`, never a plain object.**
JavaScript iterates integer-like keys numerically while Python dicts are
insertion-ordered. The substitution compiles, produces the same groups, and
silently reorders the output, which reorders the concatenated contours the refit
`minAreaRect` sees.

## Fork 3, the per-box operations: both, they agreed

`thicker<S extends RotatedBox | Ellipse>(shape: S): S` is kind-preserving through
generic inference, with no `isinstance` chain. `taller(...): RotatedBox` is
kind-collapsing, which is how the deliberate rectangularisation of an ellipse
becomes visible in the return type rather than a comment.

B's catch, which A missed: each of these must recompute the outline with a fresh
`boxPoints` or `ellipse2Poly` call, because Python's constructor recomputes it
every time, **but must not re-normalise**, because the angle is untouched.

## Fork 4, the tolerances: A's contour assertion, both candidates' conditional

Rect: centre and size within 3e-4, angle within 1e-3. Both are tighter than
`testing.md` allows and both causes are named, which is what a widening requires.

Polygon: exact, or one corner differing by at most one pixel **only when that
same entry's rect is not bit-exact**. Both candidates reached this independently,
and it is right because it conditions the allowance on an independently visible
cause instead of on a magnitude. A blanket epsilon would be vacuous anyway: over
90 % of corner coordinates are exactly integers, so 1e-2 accepts 95 % of them for
free. B's additional 0.1 soft ceiling is dropped; B called it its own headroom
choice rather than a derived number, and an undecided constant in a tolerance is
how a real regression gets absorbed later.

From A: **assert the concatenated contour exactly, always.** It is the cheapest
and strongest assertion in the phase, because a wrong merge grouping or a wrong
member order shows up there directly rather than as a drifted rect three steps
later.

## Deferred, with reasons

`rotatedRectangleIntersection`'s JS call shape is unprobed. `is_intersecting` is
its only caller and that is phase 5, with no fixture coverage, so probing it now
would be a fact with nothing to check it against.

`create_lines` stays out. It is reached only behind `--read-staff-positions`,
`dump-golden.py` never exercises it, and porting it would mean writing untested
Hough-transform code whose one filter reads a raw pre-normalisation size and so
depends on exactly the convention difference this phase just pinned down.

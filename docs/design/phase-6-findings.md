# Phase 6 findings: dewarp and staff canvas

What the pinned homr 0.7.0 does between `multistaffs.json` and
`canvas-<n>.png`, read from `staff_parsing.py`, `staff_dewarping.py`,
`staff_regions.py` and `image_utils.py` in the oracle's `.venv`. Line numbers
are omitted; every function named here is short.

## The call chain

`parse_staffs` (`staff_parsing.py`) is the loop, and the dumper reproduces it:

1. `_ensure_same_number_of_staffs(multi_staffs, image)` regroups.
2. `number_of_voices = len(staffs[0].staffs)`.
3. `StaffRegions(staffs)` keeps `(min_y, max_y)` of every staff of every
   multi staff.
4. For each voice, for each multi staff, `prepare_staff_image` on
   `multi_staff.staffs[voice]`. The canvas index counts in that order: voice
   first, then system. Both public pages hold four multi staffs of one staff
   each (on the grand-staff page each is a merged ten-line grand staff), so
   there is one voice and canvases 0 to 3 are the systems top to bottom.

The image is `predictions.preprocessed`, the gray CLAHE page.

## Regrouping

There is no `_find_periodic_core` in 0.7.0; the phase file names a function
of a later homr. The 0.7.0 regrouping is `_ensure_same_number_of_staffs`:

- every multi staff has the same staff count: unchanged;
- more than two multi staffs, and the first is within 50 px of the top or
  bottom and the rest agree: drop the first; the same test on the last;
- otherwise every multi staff is broken into one-staff multi staffs, sorted
  by the first staff's `min_y`.

`_is_close_to_image_top_or_bottom` compares `min(s.min_x, image.shape[0] -
s.max_x)`: **x against the image height**. It is a homr bug (the name says
top or bottom, the code reads x), ported literally. Both public pages take
the first branch, so the other three are proven by vectors only.

## `prepare_staff_image`, step by step

| Step | Python | Port note |
|---|---|---|
| region | `[min_x - 2u, min_y - 4u, max_x + 2u, max_y + 4u]`, y bounded by the neighbouring staffs' extents from `StaffRegions`, then `int()` each | `int()` truncates toward zero; `min_x - 2u` can be negative on a page edge |
| canvas size | `get_tr_omr_canvas_size((h, w))`: if `h / w > 256 / 1280`, `[int(w / h * 256), 256]`, else `[1280, int(h / w * 1280)]`; a numpy `[width, height]` | truncation |
| scale | `image_dimensions[1] / (region[3] - region[1])`, float64 | |
| resize | `cv2.resize(page, (int(W * s), int(H * s)))`, default `INTER_LINEAR` | cv2's fixed-point bilinear; opencv.js runs the same code |
| scaled region | `np.round(region * s)` | half to even |
| first crop | `scaled + [-10, -50, 10, 50]`, `crop_image_and_return_new_top` | `int(round())`, clamp to `[0, size - 1]`: the crop never holds the last row or column |
| staff in crop | `_dewarp_staff(staff, None, top_left / s, s)`: `(x - tl_x) * s` | `StaffPoint.transform_coordinates` takes the **mean** of the five mapped x |
| transform | `dewarp_staff_image(crop, staff_in_crop)` | see below |
| warp | `dewarp.dewarp(crop)`, fill 1, order 1 | see below |
| second crop | `scaled - [tl, tl]`, cut from the warped crop | |
| clean | `remove_black_contours_at_edges_of_image(crop2, staff_in_crop.average_unit_size)` | the unit size is of the staff in crop space |
| centre | `center_image_on_canvas(clean, image_dimensions)` | `cv2.resize` to `[w, h]`, pasted at `x = 0`, `y = (256 - h) // 2` on a 256 by 1280 field of 255 |

**What it returns is not the canvas's staff.** `prepare_staff_image` returns
`staff_in_crop`, the staff in the coordinates of the *first* crop, scaled,
undewarped. The canvas is the second crop, resized again by
`center_image_on_canvas` and moved by `y_offset`. The transformed staff that
matches the canvas is only built under `debug.debug`. `canvas-<n>-staff.json`
is therefore `staff_in_crop`, and the port returns the same thing. Phase 7
reads coordinates off it; whether that matters is phase 7's question.

## `staff_dewarping.py`

### `calculate_span_and_optimal_points(staff, image)`

- `if int(H / 6) == 0: return [], []`.
- Rows `y in range(2, H - 2, int(H / 6))`; columns `x in range(2, W, 80)`.
- At each column, `staff.get_at(x)`; when found, `y_offset = point.y[2]`
  (the middle line). The first `y_offset` ever seen, over all rows, is the
  reference: `if not first_y_offset` is a **truthiness test**, so a reference
  of exactly `0.0` is replaced by the next value. `y_delta = int(y_offset -
  first)`, truncation.
- `is_point_on_image` keeps a point only when `10 <= x <= W - 10` and
  `10 <= y <= H - 10`, so the column at `x = 2` never survives.
- A row is kept when it has **more than two** points; its optimal row puts
  every point at `int(mean(y))`.

On a straight staff every `y_delta` is 0 and source equals destination.

### `calculate_dewarp_transformation(image, source, destination)`

Mutates both lists in place: each row gains `(0, row[0].y)` in front and
`(W, row[-1].y)` behind; then a row `[(0, 0), (W, 0)]` is put first and
`[(0, H), (W, H)]` last. Concatenated, cast to float32. Note `x = W` and
`y = H` lie one past the last pixel.

### `DelaunayTriangulation`

`cv2.Subdiv2D` over `boundingRect(points)` grown by 10 on each side; the
points are inserted in concatenation order; `getTriangleList()` returns
triangles as coordinates, which `_find_point_index` maps back to indices by
the nearest point under 1e-3 (`argmin`, first wins). opencv.js 4.12 does not
export `Subdiv2D`.

The control points form a near-rectangular grid: columns 80 px apart and
rows a sixth of the crop apart. Every cell is a co-circular quad, so either
diagonal is a valid Delaunay choice.

### `find_simplex` and `transform_point`

`find_simplex` returns the first triangle, in `simplices` order, whose
barycentric coordinates are all `>= -1e-10` (a denominator under `1e-10`
skips the triangle). `transform_point` applies that triangle's matrix in
float32 (`np.float32` homogeneous point times the float64 matrix). **Neither
runs without `debug.debug`**: only the debug branch of `prepare_staff_image`
transforms the staff through the dewarp. They are ported for completeness of
the module and tested against Python on the control points, but they sit off
the inference path.

### `PiecewiseAffineTransform.estimate`

Per triangle: `None` when the source or destination triangle is degenerate
(`|cross| / 2 < 1e-6`, computed in float32 under NumPy 2's promotion), else
`cv2.getAffineTransform(src_tri, dst_tri)` on float32, a float64 2 by 3.

### `warp_image(image, fill_color=1, order=1)`

- `output = np.full_like(image, 1)`: **the fill is 1, near black, not 255.**
  Pixels no triangle paints stay 1. `remove_black_contours_at_edges_of_image`
  later clears dark blobs that touch the crop's edge, which is plausibly why
  the fill does not show; the reason is not documented upstream.
- Per triangle in `simplices` order, skipping a `None` matrix or a degenerate
  triangle: `boundingRect` of the float32 source and destination triangles
  (OpenCV floors the minimum and takes `floor(max) - floor(min) + 1`), the
  triangles offset to their rect, `getAffineTransform` again on the offset
  triangles, the source rect sliced from the image (**numpy slicing clamps**,
  so a rect reaching `x = W` is one column short), `warpAffine(src_crop, m,
  (w, h), INTER_LINEAR, BORDER_CONSTANT, 1)`, a mask from
  `fillConvexPoly(dst_tri.astype(int32))` (truncation), clipped to the
  output, and `np.where(mask, warped, output)`.
- **Later triangles overwrite earlier ones** on the pixels both masks cover,
  which is every pixel on a shared edge. The painting order is therefore part
  of the output wherever the two triangles' warps differ.

### `dewarp_staff_image`

Catches any exception, prints it, and returns `StaffDewarping(None)`, whose
`dewarp` returns the image unchanged. The port returns `null` in that case.

## `remove_black_contours_at_edges_of_image(gray, unit_size)`

`threshold(gray, 97, 255, BINARY)`, inverted, `findContours(RETR_TREE,
CHAIN_APPROX_SIMPLE)`. For each contour's `boundingRect` at least
`2 * unit_size` wide and high and touching an edge (`x == 0`, `y == 0`,
`x + w == W` or `y + h == H`): if `np.mean(thresh[rect]) < 127` skip, else
fill the rect with 255 in `gray`. `thresh` is the **inverted** image, so the
variable `is_mostly_dark` is true when fewer than half the pixels are dark:
the rects cleared are the mostly dark ones, which is what the function name
says and the opposite of what the variable says. Writes go to `gray` only, so
the contour order does not matter. Mutates its argument.

## Rounding, truncation and float32 sites

| Site | Python | TypeScript |
|---|---|---|
| region, canvas size, resize size, `int(H / 6)`, `y_delta`, `int(average_y)` | `int()` | `Math.trunc` |
| scaled region, crop corners | `np.round`, `round` | `roundHalfEven` |
| canvas `y_offset` | `//` on ints | `Math.floor` |
| control points | `np.float32` | `Math.fround` |
| degenerate test | float32 arithmetic | `Math.fround` per operation |
| `fillConvexPoly` vertices | `astype(np.int32)` | `Math.trunc` |
| `boundingRect`, `getAffineTransform`, `warpAffine`, `resize`, `threshold`, `findContours`, `fillConvexPoly` | cv2 | opencv.js, same algorithm |

## What the oracle holds

Before this phase the dumper wrote only the endpoints, `canvas-<n>.png` and
`canvas-<n>-staff.json`. Phase 6 unrolls `prepare_staff_image` and writes, per
canvas, `dewarp-<n>.json` (region, sizes, scale, both crops' corners, span
and optimal points, source and destination points, Python's `simplices` and
matrices) and three images: `dewarp-<n>-input.png` (the first crop),
`dewarp-<n>-warped.png` and `dewarp-<n>-cleaned.png`. The script exits
non-zero unless the unrolled canvas equals `prepare_staff_image`'s
byte for byte and the staffs serialise identically.

Six of the eight canvases on the two pages have an identity transform: every
`y_delta` is 0, every matrix is the identity. Kesh canvas 1 (every one of
169 triangles shifted, by at most 1 px) and the grand-staff page's canvas 3
(30 of 122 triangles, by at most 2 px) are the two where the transform does
anything, and so the only two where a different triangulation can change a
pixel.

"""Runs small hand-built inputs through the pinned homr, numpy and OpenCV and
writes what they return under test/golden/vectors/. The TypeScript tests read
these files for the branches no fixture page reaches.

No page and no model. Every file is {"meta": {...}, "cases": [...]}, one case
per line. Inputs come from literals or from a seeded generator, so the script
writes byte-identical files when run twice."""

import importlib.util
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from homr import find_peaks as homr_find_peaks
from homr.bounding_boxes import BoundingEllipse, RotatedBoundingBox
from homr.brace_dot_detection import _merge_multi_staff_if_they_share_a_staff
from homr.model import MultiStaff, Staff, StaffPoint, StemDirection
from homr.note_detection import NoteheadWithStem, add_notes_to_staffs, adjust_bbox, check_bbox_size
from homr.staff_dewarping import calculate_dewarp_transformation, calculate_span_and_optimal_points
from homr.staff_parsing import (
    _calculate_region,
    _ensure_same_number_of_staffs,
    get_tr_omr_canvas_size,
    remove_black_contours_at_edges_of_image,
)
from homr.staff_regions import StaffRegions
from homr.staff_detection import (
    RawStaff,
    StaffAnchor,
    StaffLineSegment,
    are_lines_parallel,
    begins_or_ends_on_one_staff_line,
    connect_staff_lines,
    filter_edge_of_vision,
    filter_line_peaks,
    find_horizontal_lines,
    find_staff_anchors,
    resample_staff,
)

from homr.transformer.staff2score import ConvertToArray  # noqa: E402
from homr.transformer.vocabulary import EncodedSymbol, prior_power_of_two, remove_duplicated_symbols  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
VECTORS = ROOT / "test" / "golden" / "vectors"

# dump-golden.py is a script with a hyphen in its name, so it is loaded by path;
# without this Python would leave a tools/__pycache__ behind.
sys.dont_write_bytecode = True
_spec = importlib.util.spec_from_file_location("dump_golden", ROOT / "tools" / "dump-golden.py")
dump_golden = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump_golden)
jsonable = dump_golden.jsonable

META = {
    "numpy": np.__version__,
    "opencv": cv2.__version__,
    "python": platform.python_version(),
    "machine": platform.machine(),
}


def write_vectors(name: str, cases: list, **notes) -> None:
    lines = [json.dumps(jsonable(case), separators=(",", ":")) for case in cases]
    meta = json.dumps({**META, **notes}, indent=1)
    body = ",\n".join(lines)
    (VECTORS / f"{name}.json").write_text(f'{{"meta": {meta},\n"cases": [\n{body}\n]}}\n')
    print(f"{name}.json: {len(cases)} cases")


def f32(value: float) -> float:
    """opencv.js holds a rect in float32, so a rect input is one float32 can carry."""
    return float(np.float32(value))


def rect(cx: float, cy: float, w: float, h: float, angle: float):
    return ((f32(cx), f32(cy)), (f32(w), f32(h)), f32(angle))


def box(rotated_rect, debug_id: int = 0) -> RotatedBoundingBox:
    return RotatedBoundingBox(rotated_rect, np.array([]), debug_id)


def rows_of(mask: np.ndarray) -> list[str]:
    """A 2D mask as one string of 0 and 1 per row; 1 is any non-zero pixel."""
    return ["".join("1" if v else "0" for v in row) for row in mask]


# pairwise.json


def dump_pairwise() -> None:
    rng = np.random.default_rng(501)
    cases = []
    sizes = [0, 1, 4, 7, 8, 9, 10, 127, 128, 129, 130, 255, 256, 257, 1000, 2718]
    for n in sizes:
        for _ in range(1 if n == 0 else 2 if n >= 1000 else 4):
            values = np.array(rng.standard_normal(n) * 10.0 ** rng.integers(-3, 6, n), dtype=float)
            cases.append(
                {
                    "values": values,
                    "sum": float(np.sum(values)),
                    "mean": None if n == 0 else float(np.mean(values)),
                    "std": None if n == 0 else float(np.std(values)),
                }
            )
    write_vectors("pairwise", cases)


# floor-div.json


def dump_floor_div() -> None:
    rng = np.random.default_rng(502)
    cases = []
    for b in [10.0, 2.0, 3.0, 0.1, 7.5, -10.0]:
        for k in range(-6, 7):
            multiple = k * b
            for a in [float(np.nextafter(multiple, -np.inf)), multiple, float(np.nextafter(multiple, np.inf))]:
                cases.append({"a": a, "b": b, "q": a // b})
    for _ in range(200):
        a = float(rng.standard_normal() * 1000)
        b = float(rng.choice([10.0, 2.0, 3.0, 0.1, 7.5, -10.0]))
        cases.append({"a": a, "b": b, "q": a // b})
    write_vectors("floor-div", cases)


# argsort.json

SMALL_QUICKSORT = 15


def aheapsort(less, order: list[int], lo: int, n: int) -> None:
    """numpy's aheapsort_ over order[lo : lo + n], with its one-based indexing."""

    def get(i: int) -> int:
        return order[lo + i - 1]

    def put(i: int, item: int) -> None:
        order[lo + i - 1] = item

    def sift(held: int, i: int, size: int) -> None:
        j = i * 2
        while j <= size:
            if j < size and less(get(j), get(j + 1)):
                j += 1
            if less(held, get(j)):
                put(i, get(j))
                i = j
                j += j
            else:
                break
        put(i, held)

    for start in range(n >> 1, 0, -1):
        sift(get(start), start, n)
    while n > 1:
        held = get(n)
        put(n, get(1))
        n -= 1
        sift(held, 1, n)


def aquicksort(less, n: int) -> tuple[list[int], int]:
    """numpy's aquicksort_ (npysort/quicksort.cpp) over item indices, with
    `less(i, j)` in place of v[i] < v[j]. Returns the order and how many
    elements the heapsort fallback sorted."""
    order = list(range(n))
    heapsorted = 0
    if n < 2:
        return order, heapsorted
    pl, pr = 0, n - 1
    depth = (n.bit_length() - 1) * 2
    stack: list[tuple[int, int, int]] = []
    while True:
        if depth < 0:
            aheapsort(less, order, pl, pr - pl + 1)
            heapsorted += pr - pl + 1
        else:
            while pr - pl > SMALL_QUICKSORT:
                pm = pl + ((pr - pl) >> 1)
                if less(order[pm], order[pl]):
                    order[pm], order[pl] = order[pl], order[pm]
                if less(order[pr], order[pm]):
                    order[pr], order[pm] = order[pm], order[pr]
                if less(order[pm], order[pl]):
                    order[pm], order[pl] = order[pl], order[pm]
                pivot = order[pm]
                pi, pj = pl, pr - 1
                order[pm], order[pj] = order[pj], order[pm]
                while True:
                    pi += 1
                    while less(order[pi], pivot):
                        pi += 1
                    pj -= 1
                    while less(pivot, order[pj]):
                        pj -= 1
                    if pi >= pj:
                        break
                    order[pi], order[pj] = order[pj], order[pi]
                order[pi], order[pr - 1] = order[pr - 1], order[pi]
                depth -= 1
                if pi - pl < pr - pi:
                    stack.append((pi + 1, pr, depth))
                    pr = pi - 1
                else:
                    stack.append((pl, pi - 1, depth))
                    pl = pi + 1
            for pi in range(pl + 1, pr + 1):
                held = order[pi]
                pj = pi
                while pj > pl and less(held, order[pj - 1]):
                    order[pj] = order[pj - 1]
                    pj -= 1
                order[pj] = held
        if not stack:
            return order, heapsorted
        pl, pr, depth = stack.pop()


def killer_input(n: int) -> list[float]:
    """McIlroy's adversary ("A Killer Adversary for Quicksort", 1999) played
    against aquicksort: every item starts as gas, larger than any solid value,
    and a comparison of two gas items freezes one of them to the next solid
    value. The values it leaves make aquicksort partition as badly as it can."""
    gas = n
    values = [gas] * n
    state = {"solid": 0, "candidate": 0}

    def less(x: int, y: int) -> bool:
        if values[x] == gas and values[y] == gas:
            frozen = x if x == state["candidate"] else y
            values[frozen] = state["solid"]
            state["solid"] += 1
        if values[x] == gas:
            state["candidate"] = x
        elif values[y] == gas:
            state["candidate"] = y
        return values[x] < values[y]

    aquicksort(less, n)
    return [float(v) for v in values]


def dump_argsort() -> None:
    rng = np.random.default_rng(503)
    cases = []
    for n in [5, 16, 17, 18, 20, 30, 40, 80, 200, 1000]:
        for _ in range(6):
            values = rng.integers(0, max(2, n // 3), n).astype(float)
            cases.append({"values": values, "order": np.argsort(values), "heapsorted": 0})
        values = rng.standard_normal(n)
        cases.append({"values": values, "order": np.argsort(values), "heapsorted": 0})
    for case in cases:
        values = case["values"].tolist()
        order, heapsorted = aquicksort(lambda i, j: values[i] < values[j], len(values))
        if order != case["order"].tolist() or heapsorted != 0:
            raise SystemExit("argsort: the aquicksort transcription disagrees with np.argsort")

    for n in [200, 1000]:
        values = killer_input(n)
        order, heapsorted = aquicksort(lambda i, j: values[i] < values[j], n)
        expected = np.argsort(np.array(values)).tolist()
        if heapsorted == 0:
            raise SystemExit(f"argsort: the adversarial input of {n} never reached heapsort")
        if order != expected:
            raise SystemExit(
                f"argsort: on the adversarial input of {n} the transcription disagrees with np.argsort"
            )
        cases.append({"values": values, "order": expected, "heapsorted": heapsorted})
    write_vectors(
        "argsort",
        cases,
        heapsorted="how many elements numpy's heapsort fallback sorted, by a transcription of "
        "aquicksort that equals np.argsort on every case of this file",
    )


# find-peaks.json


def peaks_case(x, height=None, distance=None, prominence=None):
    peaks, _ = homr_find_peaks.find_peaks(
        np.array(x, dtype=float), height=height, distance=distance, prominence=prominence
    )
    return {"x": x, "height": height, "distance": distance, "prominence": prominence, "peaks": peaks}


def dump_find_peaks() -> None:
    rng = np.random.default_rng(504)
    cases = [
        peaks_case([]),
        peaks_case([1.0, 2.0]),
        peaks_case([0.0, 1.0, 0.0]),
        peaks_case([0.0, 2.0, 2.0, 2.0, 0.0]),
        peaks_case([0.0, 2.0, 2.0, 2.0, 2.0, 0.0]),
        peaks_case([0.0, 2.0, 2.0, 3.0, 0.0]),
        peaks_case([3.0, 1.0, 1.0, 0.0, 2.0, 0.0]),
        peaks_case([2.0, 2.0, 0.0, 1.0, 0.0]),
        peaks_case([0.0, 1.0, 2.0, 2.0]),
        peaks_case([0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0], height=2.0),
        peaks_case([0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0], height=3.0),
        peaks_case([0.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0], height=4.0),
        peaks_case([0.0, 3.0, 2.0, 3.5, 0.0, 1.0, 0.5, 4.0, 0.0], prominence=1.0),
        peaks_case([0.0, 3.0, 2.0, 3.5, 0.0, 1.0, 0.5, 4.0, 0.0], prominence=3.0),
        peaks_case([5.0, 3.0, 4.0, 3.0, 5.0], prominence=1.0),
        peaks_case([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0], distance=3.0),
        peaks_case([0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0], distance=2.0),
        peaks_case([0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0], distance=3.0),
        peaks_case([0.0, 2.0, 0.0, 3.0, 0.0, 2.0, 0.0], distance=2.5),
        peaks_case([0.0, 1.0, 0.0, 1.0, 0.0], height=0.0, distance=18.6, prominence=1.0),
    ]
    for n, levels, distance in [(40, 3, 4.0), (120, 3, 5.0), (120, 4, 9.5), (300, 3, 6.0), (300, 5, 12.0)]:
        for _ in range(4):
            x = rng.integers(0, levels, n).astype(float).tolist()
            cases.append(peaks_case(x, height=0.0, distance=distance, prominence=1.0))
            cases.append(peaks_case(x, distance=distance))
    if max(len(case["peaks"]) for case in cases) < 17:
        raise SystemExit("find-peaks: no case sorts 17 or more tied peaks")
    write_vectors("find-peaks", cases)


# line-groups.json


def line_groups_case(name: str, height: int, width: int, row_counts: dict[int, int], unit_size: float):
    image = np.zeros((height, width), dtype=np.uint8)
    for y, count in row_counts.items():
        image[y, :count] = 1
    case = {
        "name": name,
        "height": height,
        "width": width,
        "rowCounts": sorted(row_counts.items()),
        "unitSize": unit_size,
    }
    try:
        case["groups"] = find_horizontal_lines(image, unit_size)
    except Exception as error:  # noqa: BLE001
        case["error"] = type(error).__name__
    return case


def staff_rows(top: int, unit: int, lines: int, width: int, thickness: int = 2) -> dict[int, int]:
    rows = {}
    for line in range(lines):
        for t in range(thickness):
            rows[top + line * unit + t] = width
    return rows


def dump_line_groups() -> None:
    rng = np.random.default_rng(505)
    width = 140
    speckle = {int(y): int(rng.integers(1, 6)) for y in rng.choice(2718, 300, replace=False)}
    three_staffs = {**speckle}
    for top in (400, 1200, 2000):
        three_staffs.update(staff_rows(top, 19, 5, width, 3))
    cases = [
        line_groups_case("one staff", 300, width, staff_rows(100, 18, 5, width), 18.0),
        line_groups_case("one staff, one row per line", 300, width, staff_rows(100, 18, 5, width, 1), 18.0),
        line_groups_case(
            "two staffs",
            600,
            width,
            {**staff_rows(100, 18, 5, width), **staff_rows(400, 18, 5, width)},
            18.0,
        ),
        line_groups_case("four lines", 300, width, staff_rows(100, 18, 4, width), 18.0),
        line_groups_case("six lines", 300, width, staff_rows(100, 18, 6, width), 18.0),
        line_groups_case(
            "five lines and a sixth beyond the gap limit",
            400,
            width,
            {**staff_rows(100, 18, 5, width), **staff_rows(100 + 4 * 18 + 32, 18, 1, width)},
            18.0,
        ),
        line_groups_case(
            "five lines and a sixth within the gap limit",
            400,
            width,
            {**staff_rows(100, 18, 5, width), **staff_rows(100 + 4 * 18 + 30, 18, 1, width)},
            18.0,
        ),
        line_groups_case(
            "a short line between two staff lines",
            300,
            width,
            {**staff_rows(100, 18, 5, width), 109: width // 2, 110: width // 2},
            18.0,
        ),
        line_groups_case("a full page column, three staffs over speckle", 2718, width, three_staffs, 18.625),
        line_groups_case("all zero", 300, width, {}, 18.0),
        line_groups_case("every row equal", 300, width, {y: 7 for y in range(300)}, 18.0),
    ]
    write_vectors(
        "line-groups",
        cases,
        rowCounts="[y, n] pairs: row y of the height by width image has its first n pixels set; "
        "every other row is empty",
    )


# noise.json


def noise_case(name: str, staff: np.ndarray):
    noise, mask = dump_golden.unrolled_noise_grid(staff)
    return {
        "name": name,
        "height": staff.shape[0],
        "width": staff.shape[1],
        "staff": rows_of(staff),
        **noise,
        "mask": None if mask is None else rows_of(mask),
    }


def dump_noise() -> None:
    rng = np.random.default_rng(506)

    def lines(height: int, width: int) -> np.ndarray:
        staff = np.zeros((height, width), dtype=np.uint8)
        staff[height // 2 : height // 2 + 2, :] = 1
        return staff

    def speckled(staff: np.ndarray, x0: int, x1: int, y0: int, y1: int, density: float) -> np.ndarray:
        out = staff.copy()
        region = rng.random((y1 - y0, x1 - x0)) < density
        out[y0:y1, x0:x1] |= region.astype(np.uint8)
        return out

    checker = (np.indices((80, 120)).sum(axis=0) % 2).astype(np.uint8)
    corner = lines(80, 120)
    corner[0:16, 0:24] = checker[0:16, 0:24]
    left_half = np.zeros((80, 120), dtype=np.uint8)
    left_half[:, 0:60] = checker[:, 0:60]
    cases = [
        noise_case("clean lines", lines(80, 120)),
        noise_case("one noisy block of tiles, masked", speckled(lines(80, 120), 0, 30, 0, 20, 0.5)),
        noise_case("one noisy tile with quiet neighbours, kept", speckled(lines(80, 120), 12, 18, 8, 12, 0.6)),
        noise_case("more than half the tiles noisy, skipped", speckled(lines(80, 120), 0, 120, 0, 60, 0.5)),
        noise_case("under half the tiles noisy, masked", speckled(lines(80, 120), 0, 60, 0, 80, 0.5)),
        noise_case("exactly half the tiles noisy, masked", left_half),
        noise_case("checkerboard corner: tile noise above 255", corner),
        noise_case("short last tiles", speckled(lines(90, 130), 100, 130, 60, 90, 0.5)),
        noise_case("checkerboard page", (np.indices((45, 63)).sum(axis=0) % 2).astype(np.uint8)),
    ]
    outcomes = {case["outcome"] for case in cases}
    if outcomes != {"clean", "masked", "skipped"}:
        raise SystemExit(f"noise: the cases reach only {sorted(outcomes)}")
    if not any(case["filtered"] * 2 == case["total"] for case in cases):
        raise SystemExit("noise: no case filters exactly half the tiles")
    write_vectors(
        "noise",
        cases,
        staff="the 0/1 staff mask, one string per row; homr filters 255 * staff",
        mask="the 0/255 mask as 0/1 strings, or null unless the outcome is masked",
    )


# intersections.json


def intersection_case(a, b):
    code, _ = cv2.rotatedRectangleIntersection(a, b)
    return {"a": a, "b": b, "code": code, "intersecting": box(a).is_intersecting(box(b))}


def dump_intersections() -> None:
    rng = np.random.default_rng(507)
    cases = [
        intersection_case(rect(50, 50, 20, 10, 0), rect(70, 50, 20, 10, 0)),
        intersection_case(rect(50, 50, 20, 10, 0), rect(70, 60, 20, 10, 0)),
        intersection_case(rect(50, 50, 20, 10, 0), rect(70.5, 50, 20, 10, 0)),
        intersection_case(rect(50, 50, 20, 10, 0), rect(50, 60, 20, 10, 0)),
        intersection_case(rect(50, 50, 20, 10, 0), rect(50, 50, 20, 10, 0)),
        intersection_case(rect(50, 50, 20, 10, 0), rect(50, 50, 4, 4, 0)),
        intersection_case(rect(50, 50, 4, 4, 30), rect(50, 50, 40, 40, 0)),
        intersection_case(rect(50, 50, 20, 10, 0), rect(50, 50, 10, 20, 0)),
        intersection_case(rect(50, 50, 20, 10, 0), rect(500, 500, 20, 10, 0)),
        intersection_case(rect(50, 50, 100, 2, 0), rect(50, 52, 100, 2, 0)),
        intersection_case(rect(50, 50, 100, 2, 0), rect(50, 52.5, 100, 2, 0)),
        intersection_case(rect(50, 50, 0, 80, 0), rect(50, 50, 100, 2, 0)),
        intersection_case(rect(50, 50, 20, 10, 45), rect(64, 64, 20, 10, -45)),
    ]
    for _ in range(300):
        fragment = rect(
            rng.uniform(50, 400),
            rng.uniform(50, 200),
            rng.uniform(15, 220),
            rng.uniform(2, 5),
            rng.uniform(-4, 4),
        )
        cx, cy = fragment[0]
        symbol = rect(
            cx + rng.uniform(-130, 130),
            cy + rng.uniform(-70, 70),
            rng.uniform(3, 40),
            rng.uniform(60, 140),
            rng.choice([0.0, rng.uniform(-8, 8)]),
        )
        cases.append(intersection_case(fragment, symbol))
    codes = {case["code"] for case in cases}
    if codes != {0, 1, 2}:
        raise SystemExit(f"intersections: the cases reach only codes {sorted(codes)}")
    write_vectors(
        "intersections",
        cases,
        code="cv2.rotatedRectangleIntersection: 0 none, 1 partial, 2 full",
        intersecting="RotatedBoundingBox.is_intersecting, which also applies the centre-distance test",
    )


# bbox-split.json


def blob(mask: np.ndarray, cx: int, cy: int, rx: int, ry: int) -> None:
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)


def bbox_case(name: str, mask: np.ndarray, bbox: list[int], unit_size: float):
    case = {"name": name, "mask": rows_of(mask), "bbox": bbox, "unitSize": unit_size}
    case["adjusted"] = adjust_bbox(bbox, mask)
    try:
        case["boxes"] = check_bbox_size(bbox, mask, unit_size)
    except Exception as error:  # noqa: BLE001
        case["error"] = type(error).__name__
    return case


def dump_bbox_split() -> None:
    def empty() -> np.ndarray:
        return np.zeros((70, 90), dtype=np.uint8)

    unit = 10.0
    one = empty()
    blob(one, 30, 30, 6, 5)
    side_by_side = empty()
    blob(side_by_side, 26, 30, 6, 5)
    blob(side_by_side, 39, 34, 6, 5)
    stacked = empty()
    blob(stacked, 30, 25, 6, 5)
    blob(stacked, 30, 35, 6, 5)
    three_stacked = empty()
    for cy in (20, 30, 40):
        blob(three_stacked, 30, cy, 6, 5)
    four_wide = empty()
    for cx in (20, 33, 46, 59):
        blob(four_wide, cx, 30, 6, 5)
    lopsided = empty()
    blob(lopsided, 26, 30, 6, 5)
    blob(lopsided, 40, 30, 5, 1)
    half_empty = empty()
    blob(half_empty, 24, 30, 6, 5)
    at_top = empty()
    blob(at_top, 26, 4, 6, 5)
    blob(at_top, 39, 4, 6, 5)
    flat = empty()
    blob(flat, 30, 30, 6, 1)
    cases = [
        bbox_case("one notehead", one, [24, 25, 37, 36], unit),
        bbox_case("two side by side, staggered", side_by_side, [20, 25, 46, 40], unit),
        bbox_case("two stacked", stacked, [24, 20, 37, 41], unit),
        bbox_case("three stacked", three_stacked, [24, 15, 37, 46], unit),
        bbox_case("height not a multiple of the count", three_stacked, [24, 15, 37, 44], unit),
        bbox_case("four wide: two levels of width split", four_wide, [14, 25, 66, 36], unit),
        bbox_case("a thin right half, dropped by the second pass", lopsided, [20, 25, 46, 36], unit),
        bbox_case("an empty right half, returned unadjusted", half_empty, [18, 25, 46, 36], unit),
        bbox_case("two side by side on row 0", at_top, [20, 0, 46, 10], unit),
        bbox_case("shorter than half a unit", flat, [24, 29, 37, 32], unit),
        bbox_case("empty box", empty(), [24, 25, 37, 36], unit),
        bbox_case("odd width: the centre rounds half to even", side_by_side, [20, 25, 45, 40], unit),
        bbox_case("unit 18.625", side_by_side, [20, 25, 46, 40], 18.625),
        bbox_case("unit 4.5", stacked, [24, 20, 37, 41], 4.5),
    ]
    write_vectors(
        "bbox-split",
        cases,
        mask="the notehead mask as 0/1 strings, one per row",
        bbox="[x1, y1, x2, y2]",
        adjusted="adjust_bbox(bbox, mask)",
        boxes="check_bbox_size(bbox, mask, unitSize)",
    )


# connect-lines.json


def connect_case(name: str, fragments: list, unit_size: float, probes: list):
    boxes = [box(fragment, i) for i, fragment in enumerate(fragments)]
    lines = connect_staff_lines(boxes, unit_size)
    return {
        "name": name,
        "fragments": fragments,
        "unitSize": unit_size,
        "lines": [dump_golden.positions_in(line.staff_fragments, boxes, name) for line in lines],
        "parallel": are_lines_parallel(lines, unit_size),
        "probes": [
            {"line": probe, "onOneLine": begins_or_ends_on_one_staff_line(box(probe), lines, unit_size)}
            for probe in probes
        ],
    }


def dump_connect_lines() -> None:
    unit = 18.0
    staff = [rect(100 + 110 * i, 100 + unit * line, 100, 3, 0) for line in range(5) for i in range(3)]
    probes = [
        rect(150, 136, 4, 72, 0),
        rect(150, 100, 4, 30, 0),
        rect(150, 109, 4, 10, 0),
        rect(150, 300, 4, 72, 0),
        rect(520, 100, 4, 30, 0),
        rect(40, 100, 4, 30, 0),
        rect(39, 100, 4, 30, 0),
    ]
    cases = [
        connect_case("five lines of three fragments", staff, unit, probes),
        connect_case("given right to left", list(reversed(staff)), unit, probes),
        connect_case(
            "a fragment that joins two chains",
            [rect(100, 100, 100, 3, 0), rect(100, 110, 100, 3, 0), rect(260, 105, 100, 3, 0)],
            unit,
            [rect(260, 110, 4, 30, 0)],
        ),
        connect_case(
            "a gap wider than five units starts a new line",
            [rect(100, 100, 100, 3, 0), rect(300, 100, 100, 3, 0), rect(480, 100, 100, 3, 0)],
            unit,
            [],
        ),
        connect_case(
            "short fragments are skipped",
            [
                rect(100, 100, 100, 3, 0),
                rect(162, 100, 3.5, 3, 0),
                rect(180, 100, 3.75, 3, 0),
                rect(260, 100, 100, 3, 0),
            ],
            unit,
            [],
        ),
        connect_case(
            "equal bottom_left x",
            [
                rect(100, 100, 100, 3, 0),
                rect(100, 118, 100, 3, 0),
                rect(100, 136, 100, 3, 0),
                rect(210, 118, 100, 3, 0),
            ],
            unit,
            [],
        ),
        connect_case(
            "equal bottom_left x in one chain, the wide fragment given first",
            [rect(100, 100, 100, 3, 0), rect(80, 104, 60, 3, 0), rect(260, 95, 100, 3, 0)],
            unit,
            [],
        ),
        connect_case(
            "equal bottom_left x in one chain, the narrow fragment given first",
            [rect(80, 104, 60, 3, 0), rect(100, 100, 100, 3, 0), rect(260, 95, 100, 3, 0)],
            unit,
            [],
        ),
        connect_case(
            "one long fragment off the mean angle",
            [
                rect(100, 100, 100, 3, 0),
                rect(100, 118, 100, 3, 0),
                rect(100, 136, 100, 3, 0),
                rect(100, 154, 100, 3, 20),
            ],
            unit,
            [],
        ),
        connect_case(
            "one short fragment off the mean angle",
            [
                rect(100, 100, 100, 3, 0),
                rect(100, 118, 100, 3, 0),
                rect(100, 136, 100, 3, 0),
                rect(100, 154, 30, 3, 40),
            ],
            unit,
            [],
        ),
        connect_case(
            "sloped fragments connect by extrapolation",
            [rect(100, 100, 100, 3, 2), rect(210, 103.75, 100, 3, 2), rect(320, 114, 100, 3, 2)],
            unit,
            [rect(210, 104, 4, 30, 0), rect(210, 130, 4, 30, 0)],
        ),
        connect_case("no fragments", [], unit, [rect(150, 100, 4, 30, 0)]),
    ]
    if not any(sum(line.count(2) for line in case["lines"]) == 2 for case in cases):
        raise SystemExit("connect-lines: no fragment sits in two lines")
    if all(case["parallel"] for case in cases if case["fragments"]):
        raise SystemExit("connect-lines: no case is non-parallel")
    write_vectors(
        "connect-lines",
        cases,
        fragments="[[cx, cy], [w, h], angle] rects, each a RotatedBoundingBox with no contour",
        lines="connect_staff_lines: per line, indices into fragments in the line's own order",
        parallel="are_lines_parallel(lines, unitSize)",
        onOneLine="begins_or_ends_on_one_staff_line(line, lines, unitSize)",
    )


# staff-merge.json


def staff_of(points: list[tuple[float, list[float], float]]) -> Staff:
    return Staff([StaffPoint(x, y, angle) for x, y, angle in points])


def five(top: float, unit: float = 18.0) -> list[float]:
    return [top + unit * i for i in range(5)]


def merge_case(name: str, a: Staff, b: Staff):
    case = {"name": name, "a": a, "b": b}
    try:
        case["merged"] = a.merge(b)
    except Exception as error:  # noqa: BLE001
        case["error"] = type(error).__name__
    return case


def dump_staff_merge() -> None:
    xs = [100.0, 110.0, 120.0, 130.0]
    upper = staff_of([(x, five(200.0 + 0.3 * i), 0.5) for i, x in enumerate(xs)])
    lower = staff_of([(x, five(330.0 + 0.2 * i, 18.3), -0.25) for i, x in enumerate(xs)])
    shifted = staff_of([(x, five(330.0), 0.0) for x in [120.0, 130.0, 140.0, 150.0]])
    disjoint = staff_of([(x, five(330.0), 0.0) for x in [500.0, 510.0]])
    half_keys = staff_of([(x, five(200.0), 0.0) for x in [100.5, 101.5, 102.5, 110.0]])
    whole_keys = staff_of([(x, five(330.0), 0.0) for x in [100.5, 102.5, 110.0]])
    duplicate_keys = staff_of([(100.0, five(200.0), 0.0), (100.0004, five(204.0), 1.0), (110.0, five(200.0), 0.0)])
    off_key = staff_of([(100.4, five(330.0), 0.0), (110.0, five(330.0), 0.0)])
    uneven = staff_of([(x, [200.0, 217.1, 236.4, 252.2, 270.9], 0.0) for x in xs])
    cases = [
        merge_case("two five-line staffs", upper, lower),
        merge_case("lower into upper: y is sorted", lower, upper),
        merge_case("partly shared x", upper, shifted),
        merge_case("no shared x", upper, disjoint),
        merge_case("x keys round half to even", half_keys, whole_keys),
        merge_case("duplicate x keys: the later point wins", duplicate_keys, lower),
        merge_case("same key, x further apart than 1e-3", upper, off_key),
        merge_case("nine uneven gaps", uneven, lower),
    ]
    write_vectors("staff-merge", cases, staff="homr's Staff as tools/dump-golden.py writes it in staffs.json")


# braces.json


def flat_staff(top: float, min_x: float = 100.0, max_x: float = 500.0, unit: float = 18.0) -> Staff:
    xs = np.arange(min_x, max_x + 1, 10.0)
    return staff_of([(float(x), five(top, unit), 0.0) for x in xs])


def braces_case(name: str, staffs: list[Staff], brace_dot: list):
    boxes = [box(symbol, i) for i, symbol in enumerate(brace_dot)]
    case = {"name": name, "staffs": jsonable(staffs), "braceDot": boxes}
    try:
        unrolled, intermediate = dump_golden.unrolled_braces(staffs, boxes)
        real = dump_golden.find_braces_brackets_and_grand_staff_lines(None, staffs, boxes)
        dump_golden.require_same(f"braces: {name}", unrolled, real)
        case.update(intermediate)
        case["multiStaffs"] = real
    except Exception as error:  # noqa: BLE001
        case["error"] = type(error).__name__
    return case


def dump_braces() -> None:
    def system(top: float) -> list[Staff]:
        return [flat_staff(top), flat_staff(top + 130.0)]

    brace = rect(92, 301, 12, 205, 0)
    far_brace = rect(300, 301, 12, 205, 0)
    bar_line = rect(300, 301, 4, 205, 0)
    cases = [
        braces_case("two staffs and a brace at the left edge", system(200.0), [brace]),
        braces_case("two staffs and no tall element", system(200.0), [rect(92, 301, 12, 30, 0)]),
        braces_case("two staffs and a wide element", system(200.0), [rect(92, 301, 60, 205, 0)]),
        braces_case("a connection in mid staff scores 0", system(200.0), [far_brace]),
        braces_case(
            "a brace and three bar lines",
            system(200.0),
            [bar_line, brace, rect(400, 301, 4, 205, 0), rect(500, 301, 4, 205, 0)],
        ),
        braces_case(
            "three staffs in a chain",
            [flat_staff(200.0), flat_staff(330.0), flat_staff(460.0)],
            [rect(92, 301, 12, 205, 0), rect(92, 431, 12, 205, 0)],
        ),
        braces_case(
            "three staffs, one brace over all",
            [flat_staff(200.0), flat_staff(330.0), flat_staff(460.0)],
            [rect(92, 366, 12, 335, 0)],
        ),
        braces_case(
            "four staffs, the middle pair connected",
            [flat_staff(200.0), flat_staff(330.0), flat_staff(460.0), flat_staff(590.0)],
            [rect(92, 431, 12, 205, 0)],
        ),
        braces_case(
            "two systems",
            system(200.0) + system(520.0),
            [rect(92, 301, 12, 205, 0), rect(92, 621, 12, 205, 0)],
        ),
        braces_case(
            "a brace that reaches the upper staff only where the lower staff has no point",
            [flat_staff(200.0, 100.0, 500.0), flat_staff(330.0, 200.0, 500.0)],
            [rect(92, 301, 12, 205, 0)],
        ),
        braces_case(
            "unit sizes differ: connected from the upper staff and not from the lower",
            [flat_staff(200.0, unit=18.0), flat_staff(330.0, unit=12.0)],
            [rect(80, 289, 12, 190, 0)],
        ),
        braces_case("one staff", [flat_staff(200.0)], [brace]),
        braces_case("no brace_dot boxes", system(200.0), []),
    ]
    if not any(
        {(c["staff"], c["neighbour"]) for c in case.get("connections", [])} == {(0, 1)} for case in cases
    ):
        raise SystemExit("braces: no case connects (0, 1) without (1, 0)")
    write_vectors(
        "braces",
        cases,
        staffs="homr's Staff as tools/dump-golden.py writes it in staffs.json, top to bottom",
        braceDot="RotatedBoundingBox as in boxes-brace_dot.json, with no contour",
        intermediate="notesPerStaff, tall, connections and merged as in a page's braces.json",
    )


# The staff-chain files: the cases a mutation run of the port showed no page
# and no file above reached.


def fragment(cx: float, cy: float, w: float, h: float, angle: float = 0.0) -> RotatedBoundingBox:
    """A line fragment with the two end points of its centre line as its
    contour, which is what RawStaff fits its own box to."""
    slope = np.tan(angle / 180 * np.pi)
    ends = [[int(cx - w / 2), int(cy - w / 2 * slope)], [int(cx + w / 2), int(cy + w / 2 * slope)]]
    return RotatedBoundingBox(rect(cx, cy, w, h, angle), np.array(ends).reshape(-1, 1, 2), 0)


def segments_of(fragments: list, lines: list[list[int]]) -> list[StaffLineSegment]:
    return [StaffLineSegment(i, [fragments[k] for k in line]) for i, line in enumerate(lines)]


def anchor_case(anchor: StaffAnchor, fragments: list, what: str):
    return {
        "symbol": anchor.symbol.box,
        "lines": [
            dump_golden.positions_in(line.staff_fragments, fragments, what)
            for line in anchor.staff_lines
        ],
        "averageUnitSize": anchor.average_unit_size,
    }


LINE_YS = [100, 118, 136, 154, 172]


# connect-lines-cleanup.json


def dump_connect_lines_cleanup() -> None:
    unit = 18.0
    left = rect(100, 100, 100, 3, 0)
    short = rect(201.5, 100, 3, 3, 0)
    right = rect(300, 100, 100, 3, 0)
    cases = [
        connect_case("two fragments 100 px apart", [left, right], unit, []),
        connect_case("a short fragment between them runs the clean-up", [left, short, right], unit, []),
    ]
    if [case["lines"] for case in cases] != [[[0], [1]], [[0, 2]]]:
        raise SystemExit("connect-lines-cleanup: the short fragment no longer joins the two long ones")
    write_vectors(
        "connect-lines-cleanup",
        cases,
        fragments="[[cx, cy], [w, h], angle] rects, each a RotatedBoundingBox with no contour",
        lines="connect_staff_lines: per line, indices into fragments in the line's own order",
    )


# find-anchors.json


def find_anchors_case(name: str, fragments: list, symbol, kind: str):
    anchors = find_staff_anchors(fragments, [symbol], are_clefs=kind == "clef")
    return {
        "name": name,
        "fragments": [f.box for f in fragments],
        "symbol": symbol.box,
        "kind": kind,
        "anchors": [anchor_case(anchor, fragments, name) for anchor in anchors],
    }


def dump_find_anchors() -> None:
    def six(last_width: float) -> list:
        return [fragment(100, y, 20, 1) for y in (95, 97, 99, 101, 103)] + [fragment(100, 105, last_width, 1)]

    uneven = [fragment(100, y, 100, 3) for y in (100, 110, 120, 130, 170)]
    cases = [
        find_anchors_case(
            f"a symbol 10 tall searches with unit round(2.5): a sixth line {width} wide",
            six(width),
            fragment(100, 100, 4, 10),
            "clef",
        )
        for width in (3, 4, 5, 6)
    ] + [
        find_anchors_case("a bar line one unit off the nearest line", uneven, fragment(100, 150, 4, 80), "barLine"),
        find_anchors_case("the same symbol as a clef", uneven, fragment(100, 150, 4, 80), "clef"),
        find_anchors_case("a bar line under one unit off", uneven, fragment(100, 149, 4, 80), "barLine"),
    ]
    counts = [len(case["anchors"]) for case in cases]
    if counts[1] == counts[2] or counts[4] != 0 or counts[6] == 0:
        raise SystemExit(f"find-anchors: the cases no longer sit either side of their limits: {counts}")
    write_vectors(
        "find-anchors",
        cases,
        fragments="[[cx, cy], [w, h], angle] rects of the staff-line fragments",
        kind="clef or barLine: find_staff_anchors' are_clefs",
        anchors="per anchor: the shifted symbol's rect, its five lines as indices into fragments, "
        "and average_unit_size",
    )


# resample.json


def resample_case(name: str, fragments: list, lines: list[list[int]], anchor_lines: list[list[int]], symbol):
    anchor = StaffAnchor(segments_of(fragments, anchor_lines), symbol)
    raw = RawStaff(0, segments_of(fragments, lines), [anchor])
    return {
        "name": name,
        "fragments": [f.box for f in fragments],
        "lines": lines,
        "anchor": anchor_case(anchor, fragments, name),
        "box": raw.box,
        "staff": resample_staff(raw),
    }


def dump_resample() -> None:
    # 0 to 4: the anchor's own lines. 5 and 6 sit 8 px apart where lines 1 and
    # 2 should be 18 apart, left of where line 0 ends.
    shifted = (
        [fragment(350, y, 100, 3) for y in LINE_YS]
        + [fragment(100, 130, 200, 3), fragment(100, 138, 200, 3)]
        + [fragment(310, 118, 180, 3), fragment(310, 136, 180, 3)]
        + [fragment(200, 154, 400, 3), fragment(200, 172, 400, 3)]
    )
    slope = 6.0
    sloped = [fragment(200, y, 400, 3, slope) for y in LINE_YS]
    one_each = [[i] for i in range(5)]
    cases = [
        resample_case(
            "a missing top line beside a too-close pair",
            shifted,
            [[0], [5, 7], [6, 8], [9], [10]],
            one_each,
            fragment(350, 136, 4, 72),
        ),
        resample_case(
            "a sloped staff, anchored right of its centre",
            sloped,
            one_each,
            one_each,
            fragment(380, 136 + 180 * np.tan(slope / 180 * np.pi), 4, 72),
        ),
    ]
    write_vectors(
        "resample",
        cases,
        fragments="[[cx, cy], [w, h], angle] rects; each fragment's contour is the two ends of its centre line",
        lines="the RawStaff's five lines as indices into fragments",
        anchor="the one StaffAnchor: its symbol's rect, its five lines as indices into fragments, "
        "and average_unit_size",
        box="the RawStaff's own rect, fitted by cv2.minAreaRect to the fragments' contours",
        staff="resample_staff, as tools/dump-golden.py writes a Staff in staffs.json",
    )


# edge-of-vision.json


def dump_edge_of_vision() -> None:
    def flat(min_x: float, max_x: float, top: float) -> Staff:
        return staff_of([(x, five(top), 0.0) for x in (min_x, max_x)])

    height, width = 1000, 2000
    named = [
        ("ordinary", flat(100, 1900, 100)),
        ("bottom line on the last row plus one", flat(100, 1900, 928)),
        ("bottom line on the last row", flat(100, 1900, 927)),
        ("top line above the image", flat(100, 1900, -1)),
        ("short, starting in the first 1 %", flat(10, 300, 300)),
        ("short, ending in the last 1 %", flat(1700, 1990, 400)),
        ("short, in the middle", flat(500, 800, 500)),
        ("wide, reaching both edges", flat(10, 1990, 600)),
        ("short, starting exactly at 1 %", flat(20, 300, 700)),
        ("short, ending exactly at 99 %", flat(1700, 1980, 800)),
    ]
    staffs = [staff for _, staff in named]
    kept = filter_edge_of_vision(staffs, (height, width))
    case = {
        "name": "ten staffs in a 2000 by 1000 image",
        "height": height,
        "width": width,
        "labels": [label for label, _ in named],
        "staffs": staffs,
        "kept": dump_golden.positions_in(kept, staffs, "edge-of-vision"),
        "usualWidth": float(np.average([staff.max_x - staff.min_x for staff in staffs])),
    }
    if len(case["kept"]) in (0, len(staffs)):
        raise SystemExit("edge-of-vision: the case drops nothing or everything")
    write_vectors(
        "edge-of-vision",
        [case],
        staffs="homr's Staff as tools/dump-golden.py writes it in staffs.json; labels names each",
        kept="filter_edge_of_vision(staffs, (height, width)) as indices into staffs",
        usualWidth="np.average of max_x - min_x; a staff under half of it is short",
    )


# raw-staff-merge.json


def dump_raw_staff_merge() -> None:
    own = [fragment(100, y, 100, 3) for y in LINE_YS]
    same_x = [fragment(100, y + 1, 100, 5) for y in LINE_YS]
    fragments = own + same_x
    own_lines = [[i] for i in range(5)]
    other_lines = [[i + 5] for i in range(5)]
    anchor = StaffAnchor(segments_of(fragments, own_lines), fragment(100, 136, 4, 72))
    self_staff = RawStaff(7, segments_of(fragments, own_lines), [anchor])
    other_staff = RawStaff(9, segments_of(fragments, other_lines), [anchor])
    merged = self_staff.merge(other_staff)
    name = "two fragments at one centre x"
    case = {
        "name": name,
        "fragments": [f.box for f in fragments],
        "anchor": anchor_case(anchor, fragments, name),
        "self": {"staffId": self_staff.staff_id, "lines": own_lines},
        "other": {"staffId": other_staff.staff_id, "lines": other_lines},
        "merged": {
            "staffId": merged.staff_id,
            "lines": [
                dump_golden.positions_in(line.staff_fragments, fragments, name) for line in merged.lines
            ],
            "anchors": len(merged.anchors),
        },
    }
    write_vectors(
        "raw-staff-merge",
        [case],
        fragments="[[cx, cy], [w, h], angle] rects",
        anchor="the one StaffAnchor both staffs hold: its symbol's rect and its lines as indices into fragments",
        lines="five lines as indices into fragments; in merged, in each line's own order",
        merged="self.merge(other): RawStaff.merge",
    )


# line-peak-groups.json


def dump_line_peak_groups() -> None:
    def case(name: str, peaks: list[int]):
        array = np.array(peaks)
        _valid, groups = filter_line_peaks(array, np.zeros(int(array.max()) + 1))
        return {"name": name, "peaks": peaks, "groups": groups}

    three = [top + 19 * line for top in (100, 206, 312) for line in range(5)]
    cases = [
        case("three staffs 30 apart, lines 19 apart", three),
        case("six even peaks", [10, 20, 30, 40, 50, 60]),
        case("two peaks", [40, 60]),
        case("one peak", [40]),
    ]
    write_vectors(
        "line-peak-groups",
        cases,
        peaks="row ordinates, ascending, as find_horizontal_lines passes them",
        groups="filter_line_peaks(peaks, norm)[1]: the group index of each peak",
    )


# notehead-clumps.json


def clump_case(name: str, staffs: list[Staff], blobs: list, noteheads: list):
    mask = np.zeros((190, 180), dtype=np.uint8)
    for cx, cy, *radii in blobs:
        blob(mask, cx, cy, *(radii or (6, 5)))
    case = {"name": name, "mask": rows_of(mask), "staffs": jsonable(staffs), "noteheads": noteheads}
    unrolled, splits = dump_golden.unrolled_add_notes(staffs, noteheads, mask)
    notes = add_notes_to_staffs(staffs, noteheads, None, mask)
    dump_golden.require_same(f"notehead-clumps: {name}", unrolled, notes)
    case["splits"] = splits
    case["notes"] = notes
    case["notesPerStaff"] = [len(staff.get_notes()) for staff in staffs]
    return case


def dump_notehead_clumps() -> None:
    def lines(top: float) -> list[float]:
        return five(top, 10.0)

    def flat(top: float, unit: float = 10.0) -> Staff:
        return staff_of([(float(x), five(top, unit), 0.0) for x in range(0, 101, 10)])

    def with_point(unit_at_50: float) -> Staff:
        """A flat staff of unit 10 whose one point at x 50 has another unit:
        the staff's unit, a median, stays 10."""
        return staff_of(
            [
                (float(x), five(60.0 - 2 * unit_at_50, unit_at_50) if x == 50 else lines(40.0), 0.0)
                for x in range(0, 101, 10)
            ]
        )

    def stepped() -> Staff:
        """The lines sit 5 px lower from x 80 on, so the point a notehead is
        measured against depends on which x it is looked up at."""
        return staff_of([(float(x), lines(40.0 if x < 80 else 45.0), 0.0) for x in range(0, 101, 10)])

    def head(cx: float, cy: float, w: float, h: float, stem=None, debug_id: int = 0) -> NoteheadWithStem:
        outline = [[int(cx - w / 2), int(cy)], [int(cx), int(cy - h / 2)], [int(cx + w / 2), int(cy)]]
        ellipse = BoundingEllipse(rect(cx, cy, w, h, 0), np.array(outline).reshape(-1, 1, 2), debug_id)
        if stem is None:
            return NoteheadWithStem(ellipse, None, None)
        return NoteheadWithStem(ellipse, fragment(*stem), StemDirection.UP)

    cases = [
        clump_case("one notehead", [flat(40.0)], [(40, 55)], [head(40, 55, 13, 11, (47, 40, 2, 30), 3)]),
        clump_case(
            "two side by side where the staff steps: both are measured at the clump's x",
            [stepped()],
            [(71, 55), (84, 55)],
            [head(77.5, 55, 26, 11, (91, 40, 2, 30), 7)],
        ),
        clump_case("two stacked", [flat(40.0)], [(40, 50), (40, 60)], [head(40, 55, 13, 21)]),
        clump_case(
            "four in a square: a width split, then a height split of each half",
            [flat(40.0)],
            [(34, 50), (47, 50), (34, 60), (47, 60)],
            [head(40.5, 55, 26, 21, (54, 40, 2, 30), 9)],
        ),
        clump_case(
            "a left half of two thin smears: both its halves vanish and it is kept whole",
            [flat(40.0)],
            [(34, 50, 4, 1), (47, 56, 4, 1), (60, 55), (73, 55)],
            [head(53, 55, 52, 13)],
        ),
        clump_case(
            "two and a half units tall: two notes, not three",
            [flat(40.0)],
            [(40, 50), (40, 62)],
            [head(40, 56.5, 13, 25)],
        ),
        clump_case(
            "a clump whose top is above row 0: the corner truncates toward zero",
            [flat(40.0)],
            [(34, 3, 6, 3), (47, 3, 6, 3)],
            [head(40.5, 5, 26, 11)],
        ),
        clump_case(
            "four wide above row 0: the second width split reads from a top of -1",
            [flat(40.0)],
            [(34, 3, 6, 3), (47, 3, 6, 3), (60, 3, 6, 3), (73, 3, 6, 3)],
            [head(53, 5, 52, 11)],
        ),
        clump_case(
            "a unit of two pixels: the second pass drops every box and the notehead stays whole",
            [flat(40.0, 2.0)],
            [(40, 44, 2, 0), (40, 45, 2, 0)],
            [head(40.5, 44.5, 5, 3)],
        ),
        clump_case(
            "a smear four units wide: every half vanishes, the notehead stays whole and is too wide",
            [flat(40.0)],
            [(43, 50, 9, 1), (63, 58, 9, 1)],
            [head(53, 54, 40, 11)],
        ),
        clump_case(
            "a point of unit 5 in a staff of unit 10: split by the staff's unit, refused by the point's",
            [with_point(5.0)],
            [(50, 60)],
            [head(50, 60, 13, 12)],
        ),
        clump_case(
            "a point of unit 28 in a staff of unit 10: the pieces are too narrow for the point",
            [with_point(28.0)],
            [(43, 60), (57, 60)],
            [head(50, 60, 26, 15)],
        ),
        clump_case(
            "two staffs, noteheads given bottom first, one between them: staff by staff, and a note on both",
            [flat(40.0), flat(140.0)],
            [(40, 160), (40, 55), (40, 110)],
            [head(40, 160, 13, 11), head(40, 55, 13, 11), head(40, 110, 13, 11)],
        ),
        clump_case(
            "too small, past the end of the staff, outside the ledger lines, and one kept",
            [flat(40.0)],
            [(20, 55), (160, 55), (60, 125), (60, 45)],
            [head(20, 55, 4, 4), head(160, 55, 13, 11), head(60, 125, 13, 11), head(60, 45, 13, 11)],
        ),
    ]
    counts = [len(case["notes"]) for case in cases]
    if counts != [1, 2, 2, 4, 3, 2, 2, 4, 1, 0, 0, 0, 4, 1]:
        raise SystemExit(f"notehead-clumps: the cases no longer make the notes they are named for: {counts}")
    stepped_case = cases[1]
    if len({note.position for note in stepped_case["notes"]}) != 1:
        raise SystemExit("notehead-clumps: the stepped staff no longer puts both pieces at one position")
    write_vectors(
        "notehead-clumps",
        cases,
        mask="the notehead mask as 0/1 strings, one per row",
        staffs="homr's Staff as tools/dump-golden.py writes it in staffs.json, before any note is added",
        noteheads="NoteheadWithStem as in noteheads-with-stems.json",
        splits="as a page's notehead-splits.json",
        notes="add_notes_to_staffs, as a page's notes.json",
        notesPerStaff="how many of the notes each staff received, in order",
    )


# braces-units.json


def dump_braces_units() -> None:
    def staff(top: float, unit: float) -> Staff:
        return flat_staff(top, unit=unit)

    mixed = [staff(200.0, 18.0), staff(400.0, 10.0)]
    cases = [
        braces_case(
            "rough limits from the first staff, the exact one from the closest, the first on a tie",
            mixed,
            [
                rect(92, 420, 40, 45, 0),
                rect(92, 420, 12, 45, 0),
                rect(92, 236, 12, 60, 0),
                rect(20, 420, 12, 60, 0),
            ],
        ),
        braces_case(
            "the same with the narrow staff first",
            [staff(200.0, 10.0), staff(400.0, 18.0)],
            [rect(92, 436, 12, 60, 0), rect(92, 436, 40, 80, 0), rect(92, 220, 12, 45, 0)],
        ),
        braces_case(
            "a brace that reaches the staffs only with the thickness rounded to 37",
            [staff(200.0, 18.3), staff(330.0, 18.3)],
            [rect(75.5, 301, 12, 205, 0)],
        ),
        braces_case(
            "three staffs, the top pair connected from below only",
            [staff(200.0, 12.0), staff(330.0, 18.0), staff(460.0, 18.0)],
            [rect(80, 289, 12, 190, 0), rect(92, 431, 12, 205, 0)],
        ),
        braces_case(
            "a short symbol at the left edge would score: only tall ones are scored",
            [staff(200.0, 18.0), staff(330.0, 18.0)],
            [rect(300, 301, 12, 205, 0), rect(92, 301, 12, 60, 0)],
        ),
    ]
    seen = [
        (one["tall"], [(c["staff"], c["neighbour"]) for c in one["connections"]], one["merged"])
        for one in cases
    ]
    expected = [
        ([0, 1], [], [[0], [1]]),
        ([2], [], [[0], [1]]),
        ([0], [(0, 1), (1, 0)], [[0, 1]]),
        ([0, 1], [(1, 0), (1, 2), (2, 1)], [[0, 1, 2]]),
        ([0], [(0, 1), (1, 0)], [[0, 1]]),
    ]
    if seen != expected:
        raise SystemExit(f"braces-units: the cases no longer sit where they are named: {seen}")
    if any(s.is_grandstaff for s in cases[4]["multiStaffs"][0].staffs):
        raise SystemExit("braces-units: the short symbol was scored")
    write_vectors(
        "braces-units",
        cases,
        staffs="homr's Staff as tools/dump-golden.py writes it in staffs.json, top to bottom",
        braceDot="RotatedBoundingBox as in boxes-brace_dot.json, with no contour",
        intermediate="notesPerStaff, tall, connections and merged as in a page's braces.json",
    )


# multi-staff-merge.json


def dump_multi_staff_merge() -> None:
    staffs = [flat_staff(200.0 + 130.0 * i) for i in range(5)]
    brace_dot = [box(rect(92, 301 + 130 * i, 12, 205, 0), i) for i in range(3)]
    # The same rect as brace_dot[0] in a second object: equal by value, not by identity.
    brace_dot.append(box(rect(92, 301, 12, 205, 0), 9))

    def case(name: str, groups: list[tuple[list[int], list[int]]]):
        given = [
            MultiStaff([staffs[i] for i in members], [brace_dot[k] for k in links])
            for members, links in groups
        ]
        merged = _merge_multi_staff_if_they_share_a_staff(given)
        return {
            "name": name,
            "given": [{"staffs": members, "connections": links} for members, links in groups],
            "merged": [
                {
                    "staffs": dump_golden.positions_in(multi.staffs, staffs, name),
                    "connections": dump_golden.positions_in(multi.connections, brace_dot, name),
                }
                for multi in merged
            ],
        }

    cases = [
        case("nothing shared", [([0], []), ([1], []), ([2], [])]),
        case(
            "a later entry merges into the first: the merge moves to the end",
            [([0, 1], [0]), ([2], []), ([1, 3], [1])],
        ),
        case(
            "an entry shares a staff with two earlier ones: only the first takes it",
            [([0, 1], [0]), ([2, 3], [2]), ([1, 2], [1])],
        ),
        case("staffs given bottom first come out sorted by min_y", [([3, 1], [1]), ([1, 0], [0])]),
        case(
            "the same connection twice, and an equal rect in another object",
            [([0, 1], [0]), ([1, 0], [0, 3, 1])],
        ),
        case("no entries", []),
    ]
    orders = [[tuple(multi["staffs"]) for multi in one["merged"]] for one in cases]
    if orders[1] != [(2,), (0, 1, 3)] or orders[2] != [(2, 3), (0, 1, 2)]:
        raise SystemExit(f"multi-staff-merge: the cases no longer show the reordering: {orders}")
    write_vectors(
        "multi-staff-merge",
        cases,
        given="MultiStaff(staffs, connections) per entry, as indices: staff i is five lines 18 apart from "
        "y 200 + 130 i over x 100 to 500; connection k is the rect [[92, 301 + 130 k], [12, 205], 0] "
        "for k under 3, and connection 3 is a second object with connection 0's rect",
        merged="_merge_multi_staff_if_they_share_a_staff(given), as indices in each entry's own order; "
        "a connection is named by the first object of the list it is identical to",
    )


# grand-staffs.json


def dump_grand_staffs() -> None:
    def case(
        name: str,
        tops: list[float],
        braces: list,
        min_xs: list[float] | None = None,
        units: list[float] | None = None,
    ):
        staffs = [
            flat_staff(
                top,
                100.0 if min_xs is None else min_xs[i],
                unit=18.0 if units is None else units[i],
            )
            for i, top in enumerate(tops)
        ]
        boxes = [box(brace, i) for i, brace in enumerate(braces)]
        multi = MultiStaff(staffs, [])
        out = {"name": name, "staffs": jsonable(staffs), "braces": boxes}
        out["scores"] = [
            [float(multi._score_brace_with_staff_pair(brace, staffs[i], staffs[i + 1])) for brace in boxes]
            for i in range(len(staffs) - 1)
        ]
        out["multiStaff"] = multi.create_grandstaffs(boxes)
        return out

    cases = [
        case("a brace at the left edge", [200.0, 330.0], [rect(92, 301, 12, 205, 0)]),
        case("the brace further left than five units", [200.0, 330.0], [rect(9, 301, 12, 205, 0)]),
        case("the brace exactly five units left", [200.0, 330.0], [rect(10, 301, 12, 205, 0)]),
        case(
            "a brace overlapping the pair by exactly half its height",
            [200.0, 330.0],
            [rect(92, 200, 12, 200, 0)],
        ),
        case(
            "a brace whose overlap with the pair equals its distance",
            [200.0, 330.0],
            [rect(40, 210, 12, 100, 0)],
        ),
        case(
            "the lower staff starts further left: the distance is to the nearer start",
            [200.0, 330.0],
            [rect(60, 301, 12, 205, 0)],
            [160.0, 100.0],
        ),
        case(
            "units 18 and 12: five units is 75, between the two staffs' own",
            [200.0, 330.0],
            [rect(20, 301, 12, 205, 0), rect(30, 301, 12, 205, 0)],
            units=[18.0, 12.0],
        ),
        case(
            "three staffs: the second pair scores higher and takes the middle staff",
            [200.0, 330.0, 460.0],
            [rect(80, 301, 12, 205, 0), rect(92, 431, 12, 205, 0)],
        ),
        case(
            "three staffs, equal scores: the first pair wins",
            [200.0, 330.0, 460.0],
            [rect(92, 301, 12, 205, 0), rect(92, 431, 12, 205, 0)],
        ),
        case(
            "four staffs, two pairs",
            [200.0, 330.0, 460.0, 590.0],
            [rect(92, 301, 12, 205, 0), rect(92, 561, 12, 205, 0)],
        ),
        case("no pair scores: the multi staff is returned as it is", [200.0, 330.0], [rect(300, 301, 12, 205, 0)]),
        case("one staff", [200.0], [rect(92, 301, 12, 205, 0)]),
    ]
    grand = [[staff.is_grandstaff for staff in one["multiStaff"].staffs] for one in cases]
    if cases[6]["scores"][0][0] != 0 or cases[6]["scores"][0][1] <= 0:
        raise SystemExit(f"grand-staffs: the mixed units no longer straddle the limit: {cases[6]['scores']}")
    if grand[7] != [False, True] or grand[8] != [True, False] or grand[2] != [False, False] or grand[5] != [True]:
        raise SystemExit(f"grand-staffs: the cases no longer sit where they are named: {grand}")
    write_vectors(
        "grand-staffs",
        cases,
        staffs="homr's Staff as tools/dump-golden.py writes it in staffs.json, top to bottom",
        braces="RotatedBoundingBox as in boxes-brace_dot.json, with no contour",
        scores="scores[i][k]: MultiStaff._score_brace_with_staff_pair(braces[k], staffs[i], staffs[i + 1])",
        multiStaff="MultiStaff(staffs, []).create_grandstaffs(braces), as in multistaffs.json",
    )


# staff-regrouping.json, dewarp-points.json and dewarp-warp.json (phase 6)


def regrouping_case(name: str, height: int, systems: list[list[Staff]]):
    staffs = [staff for system in systems for staff in system]
    multi_staffs = [MultiStaff(system, []) for system in systems]
    regions = StaffRegions(multi_staffs)
    probes = sorted({staff.min_y for staff in staffs} | {staff.max_y for staff in staffs} | {0.0, 1e6})
    result = _ensure_same_number_of_staffs(multi_staffs, np.zeros((height, 1), dtype=np.uint8))
    return {
        "name": name,
        "height": height,
        "staffs": staffs,
        "systems": [[staffs.index(s) for s in system] for system in systems],
        "result": [[next(i for i, t in enumerate(staffs) if t is s) for s in ms.staffs] for ms in result],
        "regions": [
            [y, regions.get_start_of_closest_staff_above(y), regions.get_start_of_closest_staff_below(y)]
            for y in probes
        ],
        "calculatedRegions": [_calculate_region(staff, regions) for staff in staffs],
    }


def dump_staff_regrouping() -> None:
    def at(top: float, min_x: float = 100.0) -> Staff:
        return flat_staff(top, min_x=min_x)

    cases = [
        regrouping_case("all systems hold one staff", 2000, [[at(100)], [at(400)], [at(700)]]),
        regrouping_case("all systems hold two", 2000, [[at(100), at(250)], [at(500), at(650)]]),
        regrouping_case("first system odd, its min_x near 0", 2000, [[at(100, 20.0)], [at(400), at(550)], [at(800), at(950)]]),
        regrouping_case("first system odd, far from the edge", 2000, [[at(100)], [at(400), at(550)], [at(800), at(950)]]),
        regrouping_case("last system odd, height minus max_x near 0", 530, [[at(100), at(250)], [at(500), at(650)], [at(900)]]),
        regrouping_case("two systems that differ", 2000, [[at(500), at(650)], [at(100)]]),
        regrouping_case("a middle system odd", 2000, [[at(100), at(250)], [at(500)], [at(800), at(950)]]),
        regrouping_case("two systems, the odd one near the edge", 2000, [[at(100, 20.0)], [at(400), at(550)]]),
        regrouping_case("a staff at the left edge", 2000, [[at(100, 5.0)], [at(400, 30.5)]]),
    ]
    sizes = [[h, w, get_tr_omr_canvas_size((h, w))] for h, w in [(200, 1000), (256, 1280), (257, 1280), (300, 1000), (150, 1500), (1, 3), (333, 1777)]]
    write_vectors(
        "staff-regrouping",
        cases + [{"name": "get_tr_omr_canvas_size", "sizes": sizes}],
        systems="indices into staffs; result is _ensure_same_number_of_staffs as the same indices",
        regions="[y, get_start_of_closest_staff_above(y), get_start_of_closest_staff_below(y)]",
        calculatedRegions="_calculate_region(staff, StaffRegions(systems)) per staff",
    )


def curved_staff(width: int, mid: float, amplitude: float, period: float, unit: float = 12.0, start: float = 30.0) -> Staff:
    xs = np.arange(start, width - start + 1, 10.0)
    return staff_of([
        (float(x), [mid - 2 * unit + unit * i + amplitude * float(np.sin(2 * np.pi * x / period)) for i in range(5)], 0.0)
        for x in xs
    ])


def points_case(name: str, staff: Staff, width: int, height: int):
    image = np.zeros((height, width), dtype=np.uint8)
    span, optimal = calculate_span_and_optimal_points(staff, image)
    case = {"name": name, "width": width, "height": height, "staff": staff, "span": span, "optimal": optimal}
    if span:
        tform = calculate_dewarp_transformation(image, [list(r) for r in span], [list(r) for r in optimal]).tform
        case.update(src=tform.src_points, dst=tform.dst_points, simplices=tform.triangulation.simplices, affine=tform.affine_matrices)
    return case


def dump_dewarp_points() -> None:
    zero_first = staff_of([(x, [-24.0, -12.0, 0.0 if x < 15 else 3.0, 12.0, 24.0], 0.0) for x in np.arange(0.0, 700.0, 10.0)])
    cases = [
        points_case("straight", curved_staff(700, 80.0, 0.0, 600.0), 700, 160),
        points_case("sloped down", staff_of([(float(x), five(40.0 + 0.05 * x, 12.0), 0.0) for x in np.arange(30.0, 671.0, 10.0)]), 700, 160),
        points_case("curved, 4 px", curved_staff(700, 80.0, 4.0, 600.0), 700, 160),
        points_case("curved, 12 px", curved_staff(700, 80.0, 12.0, 450.0), 700, 160),
        points_case("first middle line at y = 0", zero_first, 700, 160),
        points_case("staff covers part of the width", curved_staff(700, 80.0, 6.0, 500.0, start=250.0), 700, 160),
        points_case("too short for six rows", curved_staff(300, 3.0, 0.0, 600.0), 300, 5),
        points_case("too few points per row", curved_staff(170, 80.0, 0.0, 600.0, start=10.0), 170, 160),
        points_case("three points per row, the third on the right margin", curved_staff(252, 80.0, 0.0, 600.0, start=10.0), 252, 160),
        points_case("two points per row", curved_staff(172, 80.0, 0.0, 600.0, start=10.0), 172, 160),
        points_case("rising steeply, rows near the bottom margin", staff_of([(float(x), five(140.0 - 0.03 * x, 4.0), 0.0) for x in np.arange(0.0, 701.0, 10.0)]), 700, 160),
        points_case("a point on the top margin", staff_of([(float(x), five(60.0 + (8.0 if x > 50 else 0.0), 4.0), 0.0) for x in np.arange(0.0, 701.0, 10.0)]), 700, 160),
    ]
    write_vectors(
        "dewarp-points",
        cases,
        transform="src, dst, simplices and affine of calculate_dewarp_transformation, when span is not empty",
    )


def staff_picture(staff: Staff, width: int, height: int) -> np.ndarray:
    image = np.full((height, width), 230, dtype=np.uint8)
    for a, b in zip(staff.grid, staff.grid[1:]):
        for ya, yb in zip(a.y, b.y):
            cv2.line(image, (int(a.x), int(round(ya))), (int(b.x), int(round(yb))), 20, 2)
    for i, point in enumerate(staff.grid[::5]):
        cv2.ellipse(image, (int(point.x), int(round(point.y[i % 5]))), (7, 5), -20, 0, 360, 10, -1)
    return image


def dump_dewarp_warp() -> None:
    cases = []
    for name, amplitude, period in [("curved, 4 px", 4.0, 600.0), ("curved, 12 px", 12.0, 450.0)]:
        width, height = 700, 160
        staff = curved_staff(width, 80.0, amplitude, period)
        image = staff_picture(staff, width, height)
        span, optimal = calculate_span_and_optimal_points(staff, image)
        dewarp = calculate_dewarp_transformation(image, span, optimal)
        warped = dewarp.dewarp(image)
        stem = f"dewarp-warp-{int(amplitude)}px"
        Image.fromarray(image).save(VECTORS / f"{stem}-input.png", optimize=True)
        Image.fromarray(warped).save(VECTORS / f"{stem}-warped.png", optimize=True)
        tform = dewarp.tform
        probes = [(float(x), float(y)) for x in (0.0, 5.5, 82.0, 350.25, 699.0, 700.0) for y in (0.0, 20.0, 80.5, 159.0)]
        cases.append({
            "name": name,
            "input": f"{stem}-input.png",
            "warped": f"{stem}-warped.png",
            "src": tform.src_points,
            "dst": tform.dst_points,
            "simplices": tform.triangulation.simplices,
            "probes": [[p, tform.triangulation.find_simplex(np.array([p]))[0], tform.transform_point(p)] for p in probes],
        })
    write_vectors(
        "dewarp-warp",
        cases,
        probes="[point, find_simplex(point), transform_point(point)]",
    )


def contour_image(height: int, width: int, blobs: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    image = np.full((height, width), 200, dtype=np.uint8)
    for x, y, w, h, value in blobs:
        image[y : y + h, x : x + w] = value
    return image


def dump_black_contours() -> None:
    shapes = [
        ("a dark block on the left edge", [(0, 10, 20, 20, 0)]),
        ("a dark block inside", [(30, 10, 20, 20, 0)]),
        ("a block on the right and the bottom edge", [(40, 20, 20, 20, 10)]),
        ("too narrow", [(0, 10, 7, 30, 0)]),
        ("too short", [(0, 10, 30, 7, 0)]),
        ("exactly the threshold", [(0, 10, 8, 8, 0)]),
        ("gray 98 is light, 97 is dark", [(0, 0, 20, 20, 98), (40, 20, 20, 20, 97)]),
        ("a dark frame, mostly light inside", [(0, 0, 30, 3, 0), (0, 0, 3, 30, 0), (27, 0, 3, 30, 0), (0, 27, 30, 3, 0)]),
        ("an L at the top edge, half dark", [(10, 0, 30, 4, 0), (10, 0, 4, 30, 0)]),
    ]
    cases = []
    for name, blobs in shapes:
        image = contour_image(40, 60, blobs)
        cleaned = remove_black_contours_at_edges_of_image(image.copy(), 4.0)
        cases.append({"name": name, "unitSize": 4.0, "image": image, "cleaned": cleaned})
    write_vectors("black-contours", cases, image="rows of gray values", unitSize="the threshold is 2 * unitSize")


# vocabulary-cleanup.json


def symbol_of(fields: list[str]) -> EncodedSymbol:
    rhythm, pitch, lift, articulation, slur, position = (fields + [".", ".", ".", ".", "."])[:6]
    return EncodedSymbol(rhythm, pitch, lift, articulation, slur, position)


def fields_of(symbol: EncodedSymbol) -> list[str]:
    return [symbol.rhythm, symbol.pitch, symbol.lift, symbol.articulation, symbol.slur, symbol.position]


def note(rhythm: str, pitch: str = "C4", position: str = "upper") -> list[str]:
    return [rhythm, pitch, "_", "_", "_", position]


def dump_vocabulary_cleanup() -> None:
    bar = ["barline"]
    nl = ["newline"]
    chord = ["chord"]
    quarters = [note("note_4"), note("note_4", "D4"), note("note_4", "E4"), note("note_4", "F4"), bar]
    lists = {
        "leading chord token": [chord, note("note_4"), note("note_8", "D4"), note("note_4", "E4")],
        "chord with a longer duplicate pitch": [note("note_8"), chord, note("note_4"), chord, note("note_8", "E4"), bar],
        "chord with a shorter duplicate pitch": [note("note_4"), chord, note("note_8"), bar],
        "duplicate pitch on two positions": [note("note_4"), chord, note("note_4", "C4", "lower"), bar],
        "chord led by a clef": [["clef_G2", "_", "_", "_", "_", "upper"], chord, ["clef_G2", "_", "_", "_", "_", "upper"], note("note_4")],
        "short measure loses its tuplets": quarters + quarters + [note("note_6"), note("note_6", "D4"), note("note_6", "E4"), bar] + quarters,
        "measure at the typical length keeps its tuplets": quarters + [note("note_12"), note("note_12", "D4"), note("note_12", "E4"), note("note_4"), note("note_4"), note("note_4"), bar],
        "fives and sevens": [note("note_20"), note("note_28"), note("note_7"), note("rest_10"), note("note_11"), bar] + quarters + quarters,
        "even measure count takes the upper median": quarters + [note("note_2"), note("note_2"), note("note_2"), bar] + [note("note_6"), bar] + [note("note_2"), note("note_2"), bar],
        "lower clef early keeps lower": [["clef_G2", "_", "_", "_", "_", "upper"], chord, ["clef_F4", "_", "_", "_", "_", "lower"], note("note_4", "C3", "lower"), bar],
        "lower clef too late moves everything up": [note("note_4", "C4", "lower")] * 5 + [["clef_F4", "_", "_", "_", "_", "lower"], note("note_4", "C3", "lower"), bar],
        "redundant clefs keys and times across staffs": [
            ["clef_G2", "_", "_", "_", "_", "upper"], ["keySignature_1"], ["timeSignature/8"], note("note_8"), bar, nl,
            ["clef_G2", "_", "_", "_", "_", "upper"], ["keySignature_1"], ["timeSignature/8"], note("note_8"), bar, nl,
            ["clef_G2", "_", "_", "_", "_", "upper"], ["keySignature_2"], ["timeSignature/4"], ["clef_C3", "_", "_", "_", "_", "upper"], note("note_8"), bar,
        ],
        "clef chord emptied by the redundancy filter": [["clef_G2", "_", "_", "_", "_", "upper"], note("note_4"), ["clef_G2", "_", "_", "_", "_", "upper"], chord, ["clef_G2", "_", "_", "_", "_", "upper"], note("note_4")],
        "lower clef slot": [["clef_F4", "_", "_", "_", "_", "lower"], ["clef_F4", "_", "_", "_", "_", "."], ["clef_F4", "_", "_", "_", "_", "lower"], ["clef_G2", "_", "_", "_", "_", "lower"]],
        "grace notes and multirests": [note("note_8G"), note("note_4"), bar, ["rest_2m", "_", "_", "_", "_", "upper"], bar, note("note_4"), note("note_4"), bar, note("note_16G."), chord, note("note_4", "E4"), bar],
        "repeat ends a measure": [note("note_6")] * 3 + [["repeatEnd"]] + quarters + quarters,
        "empty": [],
    }
    cases = []
    for name, symbols in lists.items():
        built = [symbol_of(s) for s in symbols]
        for cleanup in (True, False):
            out = remove_duplicated_symbols(built, cleanup_tuplets=cleanup)
            cases.append({"kind": "remove", "name": name, "cleanupTuplets": cleanup,
                          "symbols": symbols, "result": [fields_of(s) for s in out]})
    for rhythm in ["note_4", "note_6", "note_12.", "rest_24G", "note_20", "note_10", "note_28", "note_7", "note_14", "note_11", "rest_96", "rest_2m", "clef_G2", "note_15"]:
        cases.append({"kind": "tuplet", "rhythm": rhythm, "result": EncodedSymbol(rhythm).remove_tuplet().rhythm})
    for rhythm in ["note_4", "note_4.", "note_4..", "note_8G", "note_0", "note_0.", "rest_12", "note_6.", "note_3", "rest_2m", "rest_10m", "note_96", "note_7", "note_1", "note_128..", "note_16G..", "clef_G2", "barline"]:
        d = EncodedSymbol(rhythm).get_duration()
        cases.append({"kind": "duration", "rhythm": rhythm, "fraction": [d.fraction.numerator, d.fraction.denominator],
                      "dots": d.dots, "actualNotes": d.actual_notes, "normalNotes": d.normal_notes, "kern": d.kern,
                      "base": [d.base_duration.numerator, d.base_duration.denominator]})
    for n in [-3, 0, 1, 2, 3, 7, 8, 9, 96, 129]:
        cases.append({"kind": "priorPowerOfTwo", "n": n, "result": prior_power_of_two(n)})
    write_vectors("vocabulary-cleanup", cases, symbols="[rhythm, pitch, lift, articulation, slur, position], missing fields '.'")


# normalize.json


def dump_normalize() -> None:
    pixels = np.arange(256, dtype=np.uint8).reshape(1, 256)
    as32 = ConvertToArray()(pixels).reshape(-1)
    as16 = as32.astype(np.float16).view(np.uint16)
    cases = [{"pixel": int(p), "float32": float(a), "float16Bits": int(b)} for p, a, b in zip(range(256), as32, as16, strict=True)]
    write_vectors("normalize", cases, source="staff2score.ConvertToArray, then astype(float16) as encoder_inference.py does")


DUMPERS = {
    "pairwise": dump_pairwise,
    "floor-div": dump_floor_div,
    "argsort": dump_argsort,
    "find-peaks": dump_find_peaks,
    "line-groups": dump_line_groups,
    "noise": dump_noise,
    "intersections": dump_intersections,
    "bbox-split": dump_bbox_split,
    "connect-lines": dump_connect_lines,
    "staff-merge": dump_staff_merge,
    "braces": dump_braces,
    "connect-lines-cleanup": dump_connect_lines_cleanup,
    "find-anchors": dump_find_anchors,
    "resample": dump_resample,
    "edge-of-vision": dump_edge_of_vision,
    "raw-staff-merge": dump_raw_staff_merge,
    "line-peak-groups": dump_line_peak_groups,
    "notehead-clumps": dump_notehead_clumps,
    "braces-units": dump_braces_units,
    "multi-staff-merge": dump_multi_staff_merge,
    "grand-staffs": dump_grand_staffs,
    "staff-regrouping": dump_staff_regrouping,
    "dewarp-points": dump_dewarp_points,
    "dewarp-warp": dump_dewarp_warp,
    "black-contours": dump_black_contours,
    "vocabulary-cleanup": dump_vocabulary_cleanup,
    "normalize": dump_normalize,
}


def main() -> None:
    names = sys.argv[1:] or list(DUMPERS)
    VECTORS.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name not in DUMPERS:
            raise SystemExit(f"unknown vector file {name}; one of {', '.join(DUMPERS)}")
        DUMPERS[name]()


if __name__ == "__main__":
    main()

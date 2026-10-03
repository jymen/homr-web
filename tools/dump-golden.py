"""Runs the pinned homr on every fixture page and writes each pipeline stage's
output under test/golden/<fixture>/. The TypeScript tests read these files.

Nothing in homr is patched: the script calls the same functions main.py
calls, in the same order, on the CPU with the fp32 models, and saves what
they return. Run twice, it writes byte-identical files.

Five homr functions are unrolled into the calls they make, so that their
intermediate values can be saved: create_noise_grid, detect_staff,
add_notes_to_staffs, find_braces_brackets_and_grand_staff_lines and
prepare_staff_image. Each
unrolled block is followed by a call of the real function, and the script
exits non-zero unless the two results serialise identically."""

import copy
import hashlib
import json
import platform
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ort.set_default_logger_severity(3)

from homr import color_adjust, constants  # noqa: E402
from homr.autocrop import autocrop  # noqa: E402
from homr.bar_line_detection import detect_bar_lines  # noqa: E402
from homr.bounding_boxes import RotatedBoundingBox, create_rotated_bounding_boxes  # noqa: E402
from homr.brace_dot_detection import (  # noqa: E402
    _create_grandstaffs,
    _filter_for_tall_elements,
    _get_connections_between_staffs,
    _merge_multi_staff_if_they_share_a_staff,
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug  # noqa: E402
from homr.main import download_weights, get_predictions, predict_symbols  # noqa: E402
from homr.model import MultiStaff, Note  # noqa: E402
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml  # noqa: E402
from homr.noise_filtering import (  # noqa: E402
    apply_noise_filter,
    create_grid,
    filter_predictions,
    handle_filter_results,
)
from homr.note_detection import (  # noqa: E402
    add_notes_to_staffs,
    combine_noteheads_with_stems,
    split_clumps_of_noteheads,
)
from homr.resize import resize_image  # noqa: E402
from homr.segmentation.config import segnet_path_onnx  # noqa: E402
from homr.staff_detection import (  # noqa: E402
    break_wide_fragments,
    detect_staff,
    filter_edge_of_vision,
    filter_unusual_anchors,
    find_horizontal_lines,
    find_raw_staffs_by_connecting_line_fragments,
    find_staff_anchors,
    init_zone,
    make_lines_stronger,
    predict_other_anchors_from_clefs,
    remove_duplicate_staffs,
    resample_staffs,
    sort_staffs_top_to_bottom,
)
from homr.image_utils import crop_image_and_return_new_top  # noqa: E402
from homr.staff_dewarping import (  # noqa: E402
    calculate_dewarp_transformation,
    calculate_span_and_optimal_points,
)
from homr.staff_parsing import (  # noqa: E402
    _calculate_region,
    _dewarp_staff,
    _ensure_same_number_of_staffs,
    _get_number_of_voices,
    center_image_on_canvas,
    get_tr_omr_canvas_size,
    prepare_staff_image,
    remove_black_contours_at_edges_of_image,
)
from homr.staff_parsing_tromr import parse_staff_tromr  # noqa: E402
from homr.staff_position_save_load import save_staff_positions  # noqa: E402
from homr.staff_regions import StaffRegions  # noqa: E402
from homr.transformer.configs import Config  # noqa: E402
from homr.transformer.vocabulary import EncodedSymbol, remove_duplicated_symbols  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "test" / "fixtures"
GOLDEN = ROOT / "test" / "golden"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def jsonable(value, seen=None):
    """Plain data from homr's objects: numpy to lists, enums to names, objects
    to their fields under a __class__ key. Cycles become {"__cycle__": class}."""
    if seen is None:
        seen = set()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(k): jsonable(v, seen) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v, seen) for v in value]
    if hasattr(value, "__dict__"):
        if id(value) in seen:
            return {"__cycle__": type(value).__name__}
        seen = seen | {id(value)}
        out = {"__class__": type(value).__name__}
        for key, field in vars(value).items():
            out[key] = jsonable(field, seen)
        return out
    return repr(value)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(jsonable(value), indent=1, sort_keys=False) + "\n")


def write_gray(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path, optimize=True)


def write_color(path: Path, bgr: np.ndarray) -> None:
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(path, optimize=True)


def write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray((mask > 0).astype(np.uint8) * 255).convert("1").save(path, optimize=True)


def symbols_json(symbols: list[EncodedSymbol]):
    return [
        {
            "rhythm": s.rhythm,
            "pitch": s.pitch,
            "lift": s.lift,
            "articulation": s.articulation,
            "slur": s.slur,
            "position": s.position,
            "coordinates": None if s.coordinates is None else [float(c) for c in s.coordinates],
        }
        for s in symbols
    ]


def positions_in(items, of: list, what: str) -> list[int]:
    """Where each of `items` sits in `of`, by object identity: homr passes the
    same objects from stage to stage, and two of them may compare equal."""
    index = {id(item): i for i, item in enumerate(of)}
    missing = [item for item in items if id(item) not in index]
    if missing:
        raise SystemExit(f"{what}: {len(missing)} object(s) that are not in the list they index")
    return [index[id(item)] for item in items]


def require_same(what: str, unrolled, real) -> None:
    if json.dumps(jsonable(unrolled)) != json.dumps(jsonable(real)):
        raise SystemExit(f"{what}: the unrolled calls and homr's own function disagree")


def unrolled_noise_grid(staff: np.ndarray):
    """create_noise_grid (noise_filtering.py:18) on 255 * staff, as
    filter_predictions calls it. Returns the noise.json value and the mask,
    which is None unless the outcome is masked."""
    gray = 255 * staff
    height, width = gray.shape
    tile_height, tile_width = height // 20, width // 20
    grid = create_grid(gray, tile_height, tile_width)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    debug_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    filtered, total = apply_noise_filter(grid, mask, debug_image, tile_height, tile_width)
    kept_mask = handle_filter_results(filtered, total, mask)
    if kept_mask is not None:
        outcome = "masked"
    elif filtered > 0:
        outcome = "skipped"
    else:
        outcome = "clean"
    noise = {
        "tile": [tile_height, tile_width],
        "grid": grid,
        "filtered": filtered,
        "total": total,
        "outcome": outcome,
    }
    return noise, kept_mask


def anchor_json(anchor, fragments: list):
    return {
        "symbol": anchor.symbol,
        "lines": [
            positions_in(line.staff_fragments, fragments, "anchor line")
            for line in anchor.staff_lines
        ],
        "averageUnitSize": anchor.average_unit_size,
        "minY": anchor.min_y,
        "maxY": anchor.max_y,
        "zone": [anchor.zone.start, anchor.zone.stop],
    }


def unrolled_detect_staff(image: np.ndarray, fragments: list, clefs_keys: list, bar_lines: list):
    """detect_staff (staff_detection.py:694). Returns the staffs and the
    staff-anchors.json, raw-staffs.json and other-clefs.json values."""
    clef_anchors = find_staff_anchors(fragments, clefs_keys, are_clefs=True)
    zones = []
    other_clef_candidates = []
    if len(clef_anchors) > 0:
        # predict_other_anchors_from_clefs (638) computes these and keeps only
        # the boxes that overlap no clef anchor's symbol.
        unit_size = float(np.mean([anchor.average_unit_size for anchor in clef_anchors]))
        for zone in init_zone(clef_anchors, image.shape):
            lines = find_horizontal_lines(image[:, zone], unit_size)
            zones.append({"start": zone.start, "stop": zone.stop, "lines": lines})
            for group in lines:
                min_y = min(group)
                max_y = max(group)
                center_y = (min_y + max_y) / 2
                center_x = zone.start + (zone.stop - zone.start) / 2
                rect = ((int(center_x), int(center_y)), (zone.stop - zone.start, int(max_y - min_y)), 0)
                other_clef_candidates.append(RotatedBoundingBox(rect, np.array([]), 0))
    other_clef_symbols = predict_other_anchors_from_clefs(clef_anchors, image)
    anchor_symbols = [anchor.symbol for anchor in clef_anchors]
    require_same(
        "predict_other_anchors_from_clefs",
        [box for box in other_clef_candidates if not box.is_overlapping_with_any(anchor_symbols)],
        other_clef_symbols,
    )
    other_clef_anchors = find_staff_anchors(fragments, other_clef_symbols, are_clefs=True)
    bar_line_anchors = find_staff_anchors(fragments, bar_lines, are_clefs=False)
    all_anchors = clef_anchors + other_clef_anchors + bar_line_anchors
    kept = filter_unusual_anchors(all_anchors)

    connected = find_raw_staffs_by_connecting_line_fragments(kept, fragments)
    deduplicated = remove_duplicate_staffs(connected)
    resampled = resample_staffs(deduplicated)
    in_view = filter_edge_of_vision(resampled, image.shape)
    staffs = sort_staffs_top_to_bottom(in_view)

    anchors = {
        "clefs": [anchor_json(a, fragments) for a in clef_anchors],
        "zones": zones,
        "otherClefSymbols": other_clef_symbols,
        "otherClefs": [anchor_json(a, fragments) for a in other_clef_anchors],
        "barLines": [anchor_json(a, fragments) for a in bar_line_anchors],
        "kept": positions_in(kept, all_anchors, "kept anchors"),
    }
    kept_in_view = {id(staff) for staff in in_view}
    raw_staffs = {
        "connected": [
            {
                "box": raw.box,
                "polygon": raw.polygon,
                "staffId": raw.staff_id,
                "lines": [
                    positions_in(line.staff_fragments, fragments, "raw staff line")
                    for line in raw.lines
                ],
                "anchors": positions_in(raw.anchors, kept, "raw staff anchors"),
            }
            for raw in connected
        ],
        "deduplicated": positions_in(deduplicated, connected, "deduplicated staffs"),
        "resampledFrom": positions_in(staffs, resampled, "resampled staffs"),
        "droppedAtEdge": [i for i, staff in enumerate(resampled) if id(staff) not in kept_in_view],
    }
    return staffs, anchors, raw_staffs, other_clef_candidates


def unrolled_add_notes(staffs: list, noteheads: list, notehead_pred: np.ndarray):
    """add_notes_to_staffs (note_detection.py:149) without its staff.add_symbol,
    so it leaves the staffs as it found them. Returns the notes and the
    notehead-splits.json value."""
    notes = []
    splits = []
    for i, staff in enumerate(staffs):
        for j, chunk in enumerate(noteheads):
            if not staff.is_on_staff_zone(chunk.notehead):
                continue
            center = chunk.notehead.center
            point = staff.get_at(center[0])
            if point is None:
                continue
            if (
                chunk.notehead.size[0] < 0.5 * point.average_unit_size
                or chunk.notehead.size[1] < 0.5 * point.average_unit_size
            ):
                continue
            pieces = split_clumps_of_noteheads(chunk, notehead_pred, staff)
            if len(pieces) > 1:
                splits.append(
                    {"staff": i, "notehead": j, "pieces": [piece.notehead for piece in pieces]}
                )
            for piece in pieces:
                point = staff.get_at(center[0])
                if point is None:
                    continue
                if (
                    piece.notehead.size[0] < 0.5 * point.average_unit_size
                    or piece.notehead.size[0] > 3 * point.average_unit_size
                    or piece.notehead.size[1] < 0.5 * point.average_unit_size
                    or piece.notehead.size[1] > 2 * point.average_unit_size
                ):
                    continue
                position = point.find_position_in_unit_sizes(piece.notehead)
                notes.append(Note(piece.notehead, position, piece.stem, piece.stem_direction))
    return notes, splits


def unrolled_braces(staffs: list, brace_dot: list):
    """find_braces_brackets_and_grand_staff_lines (brace_dot_detection.py:142).
    Returns the multi staffs and the braces.json value."""
    tall = _filter_for_tall_elements(brace_dot, staffs)
    connections = []
    result = []
    for i, staff in enumerate(staffs):
        neighbours = [k for k in (i - 1, i + 1) if 0 <= k < len(staffs)]
        any_connected_neighbour = False
        for k in neighbours:
            found = _get_connections_between_staffs(staff, staffs[k], tall)
            if len(found) > 0:
                connections.append(
                    {
                        "staff": i,
                        "neighbour": k,
                        "symbols": positions_in(found, brace_dot, "brace connections"),
                    }
                )
            if len(found) >= constants.minimum_connections_to_form_combined_staff:
                result.append(MultiStaff([staff, staffs[k]], found))
                any_connected_neighbour = True
        if not any_connected_neighbour:
            result.append(MultiStaff([staff], []))
    merged = _merge_multi_staff_if_they_share_a_staff(result)
    braces = {
        "notesPerStaff": [len(staff.get_notes()) for staff in staffs],
        "tall": positions_in(tall, brace_dot, "tall brace_dot elements"),
        "connections": connections,
        "merged": [positions_in(multi.staffs, staffs, "merged multi staff") for multi in merged],
    }
    return _create_grandstaffs(merged, tall), braces


def unrolled_prepare_staff_image(staff, image: np.ndarray, regions: StaffRegions):
    """prepare_staff_image's calls without the Debug drawing. Returns the
    canvas, the staff, the intermediates as one dict and the three images
    between the stages: the crop the transform is built on, the warped crop,
    and the second crop after remove_black_contours_at_edges_of_image."""
    region = _calculate_region(staff, regions)
    image_dimensions = get_tr_omr_canvas_size((int(region[3] - region[1]), int(region[2] - region[0])))
    scaling_factor = image_dimensions[1] / (region[3] - region[1])
    resized_size = (int(image.shape[1] * scaling_factor), int(image.shape[0] * scaling_factor))
    resized = cv2.resize(image, resized_size)
    scaled_region = np.round(region * scaling_factor)
    region_step1 = np.array(scaled_region) + np.array([-10, -50, 10, 50])
    cropped, top_left_step1 = crop_image_and_return_new_top(resized, *region_step1)
    region_step2 = np.array(scaled_region) - np.array([*top_left_step1, *top_left_step1])
    top_left = top_left_step1 / scaling_factor
    staff_in_crop = _dewarp_staff(staff, None, top_left, scaling_factor)
    span_points, optimal_points = calculate_span_and_optimal_points(staff_in_crop, cropped)
    dewarp = calculate_dewarp_transformation(
        cropped, copy.deepcopy(span_points), copy.deepcopy(optimal_points)
    )
    tform = dewarp.tform
    warped = dewarp.dewarp(cropped)
    second_crop, top_left_step2 = crop_image_and_return_new_top(warped, *region_step2)
    cleaned = remove_black_contours_at_edges_of_image(second_crop.copy(), staff_in_crop.average_unit_size)
    canvas = center_image_on_canvas(cleaned, image_dimensions)
    intermediates = {
        "region": region,
        "imageDimensions": image_dimensions,
        "scalingFactor": scaling_factor,
        "resizedSize": resized_size,
        "scaledRegion": scaled_region,
        "regionStep1": region_step1,
        "topLeftStep1": top_left_step1,
        "regionStep2": region_step2,
        "topLeftStep2": top_left_step2,
        "spanPoints": span_points,
        "optimalPoints": optimal_points,
        "src": tform.src_points,
        "dst": tform.dst_points,
        "simplices": tform.triangulation.simplices,
        "affine": tform.affine_matrices,
    }
    return canvas, staff_in_crop, intermediates, cropped, warped, cleaned


def golden_dir_for(image_path: Path) -> Path:
    """Private pages live in test/fixtures/local/ and their golden data in
    test/golden/local/; both are git-ignored. Everything else is public."""
    if image_path.parent.name == "local":
        return GOLDEN / "local" / image_path.stem
    return GOLDEN / image_path.stem


def dump(image_path: Path, config: Config) -> None:
    out = golden_dir_for(image_path)
    out.mkdir(parents=True, exist_ok=True)
    print(f"== {image_path.name} -> {out.relative_to(ROOT)}")

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"not an image: {image_path}")
    image = autocrop(image)
    write_color(out / "autocropped.png", image)
    image = resize_image(image)
    write_color(out / "resized.png", image)
    preprocessed = color_adjust.apply_clahe(image)
    write_gray(out / "preprocessed.png", preprocessed)

    predictions = get_predictions(image, preprocessed, str(image_path), False, False)
    for name in ("staff", "symbols", "stems_rest", "notehead", "clefs_keys"):
        write_mask(out / f"mask-{name}.png", getattr(predictions, name))

    debug = Debug(predictions.original, str(image_path), False)
    noise, noise_mask = unrolled_noise_grid(predictions.staff)
    raw_predictions = predictions
    predictions = filter_predictions(predictions, debug)
    if (predictions is raw_predictions) != (noise_mask is None):
        raise SystemExit("noise: the unrolled calls and filter_predictions disagree on masking")
    if noise_mask is not None:
        masked_staff = cv2.bitwise_and(raw_predictions.staff, raw_predictions.staff, mask=noise_mask)
        if not np.array_equal(masked_staff, predictions.staff):
            raise SystemExit("noise: the unrolled mask is not the one filter_predictions applied")
        write_mask(out / "mask-noise.png", noise_mask)
    else:
        (out / "mask-noise.png").unlink(missing_ok=True)
    write_json(out / "noise.json", noise)
    write_mask(out / "mask-denoised-staff.png", predictions.staff)
    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    for name in ("staff", "symbols", "stems_rest", "notehead", "clefs_keys"):
        write_mask(out / f"mask-filtered-{name}.png", getattr(predictions, name))

    symbols = predict_symbols(debug, predictions)
    for name in ("noteheads", "staff_fragments", "clefs_keys", "stems_rest", "bar_lines"):
        write_json(out / f"boxes-{name}.json", getattr(symbols, name))

    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    write_json(out / "boxes-staff_fragments-broken.json", symbols.staff_fragments)

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    write_json(out / "noteheads-with-stems.json", noteheads_with_stems)
    if len(noteheads_with_stems) == 0:
        raise SystemExit("No noteheads found")

    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    all_noteheads = [n.notehead for n in noteheads_with_stems]
    all_stems = [n.stem for n in noteheads_with_stems if n.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    write_json(
        out / "barlines.json",
        {"averageNoteHeadHeight": average_note_head_height, "barLines": bar_line_boxes},
    )

    unrolled_staffs, staff_anchors, raw_staffs, other_clef_candidates = unrolled_detect_staff(
        predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    require_same("detect_staff", unrolled_staffs, staffs)
    write_json(out / "staff-anchors.json", staff_anchors)
    write_json(out / "other-clefs.json", other_clef_candidates)
    write_json(out / "raw-staffs.json", raw_staffs)
    write_json(out / "staffs.json", staffs)
    if len(staffs) == 0:
        raise SystemExit("No staffs found")

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    write_mask(out / "mask-brace_dot.png", brace_dot_img)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))
    write_json(out / "boxes-brace_dot.json", brace_dot)

    unrolled_notes, notehead_splits = unrolled_add_notes(
        staffs, noteheads_with_stems, predictions.notehead
    )
    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )
    require_same("add_notes_to_staffs", unrolled_notes, notes)
    write_json(out / "notehead-splits.json", notehead_splits)
    write_json(out / "notes.json", notes)
    unrolled_multi_staffs, braces = unrolled_braces(staffs, brace_dot)
    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
    require_same(
        "find_braces_brackets_and_grand_staff_lines", unrolled_multi_staffs, multi_staffs
    )
    write_json(out / "braces.json", braces)
    write_json(out / "multistaffs.json", multi_staffs)

    save_staff_positions(multi_staffs, predictions.preprocessed.shape, str(out / "staff-positions.txt"))

    staffs_for_parsing = _ensure_same_number_of_staffs(multi_staffs, predictions.preprocessed)
    number_of_voices = _get_number_of_voices(staffs_for_parsing)
    regions = StaffRegions(staffs_for_parsing)
    voices = []
    index = 0
    for voice in range(number_of_voices):
        result_for_voice: list[EncodedSymbol] = []
        for staff in [s.staffs[voice] for s in staffs_for_parsing]:
            unrolled = unrolled_prepare_staff_image(staff, predictions.preprocessed, regions)
            staff_image, transformed_staff = prepare_staff_image(
                debug, index, staff, predictions.preprocessed, regions=regions
            )
            if not np.array_equal(unrolled[0], staff_image):
                raise SystemExit("prepare_staff_image: the unrolled canvas differs")
            require_same("prepare_staff_image", unrolled[1], transformed_staff)
            write_json(out / f"dewarp-{index}.json", unrolled[2])
            write_gray(out / f"dewarp-{index}-input.png", unrolled[3])
            write_gray(out / f"dewarp-{index}-warped.png", unrolled[4])
            write_gray(out / f"dewarp-{index}-cleaned.png", unrolled[5])
            write_gray(out / f"canvas-{index}.png", staff_image)
            write_json(out / f"canvas-{index}-staff.json", transformed_staff)
            tokens = parse_staff_tromr(staff_image=staff_image, staff=transformed_staff, config=config)
            write_json(out / f"tokens-{index}.json", symbols_json(tokens))
            if len(tokens) > 0:
                tokens.append(EncodedSymbol("newline"))
                result_for_voice.extend(tokens)
            index += 1
        voices.append(remove_duplicated_symbols(result_for_voice))
    write_json(out / "voices.json", [symbols_json(v) for v in voices])

    xml = generate_xml(XmlGeneratorArguments(False, None, None), voices, "")
    xml.write(str(out / "page.musicxml"))

    from importlib.metadata import version

    meta = {
        "fixture": image_path.name,
        "imageSha256": sha256_of(image_path),
        "homrVersion": version("homr"),
        "models": {
            "segnet": f"{Path(segnet_path_onnx).name}:{sha256_of(Path(segnet_path_onnx))}",
            "encoder": f"{Path(config.filepaths.encoder_path).name}:{sha256_of(Path(config.filepaths.encoder_path))}",
            "decoder": f"{Path(config.filepaths.decoder_path).name}:{sha256_of(Path(config.filepaths.decoder_path))}",
        },
        "stages": sorted(p.name for p in out.iterdir() if p.name != "meta.json"),
        "oracle": {
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def dump_vocabulary() -> None:
    """The six decoder vocabularies, token to index, from the installed homr.
    Written once beside the fixtures rather than per fixture: they belong to
    the pinned model, not to a page."""
    from homr.transformer.vocabulary import Vocabulary

    vocabulary = Vocabulary()
    heads = ("rhythm", "pitch", "lift", "articulation", "slur", "position")
    data = {head: getattr(vocabulary, head) for head in heads}
    GOLDEN.mkdir(parents=True, exist_ok=True)
    (GOLDEN / "vocabulary.json").write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")


def main() -> None:
    download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)
    config = Config()
    config.use_gpu_inference = False
    config.use_coreml_encoder = False
    pages = [Path(p).resolve() for p in sys.argv[1:]] or sorted(FIXTURES.glob("*.png")) + sorted(
        (FIXTURES / "local").glob("*.png")
    )
    dump_vocabulary()
    for page in pages:
        dump(page, config)


if __name__ == "__main__":
    main()

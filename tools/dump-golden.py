"""Runs the pinned homr on every fixture page and writes each pipeline stage's
output under test/golden/<fixture>/. The TypeScript tests read these files.

Nothing in homr is patched: the script calls the same functions main.py
calls, in the same order, on the CPU with the fp32 models, and saves what
they return. Run twice, it writes byte-identical files."""

import hashlib
import json
import sys
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

ort.set_default_logger_severity(3)

from homr import color_adjust  # noqa: E402
from homr.autocrop import autocrop  # noqa: E402
from homr.bar_line_detection import detect_bar_lines  # noqa: E402
from homr.bounding_boxes import create_rotated_bounding_boxes  # noqa: E402
from homr.brace_dot_detection import (  # noqa: E402
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr.debug import Debug  # noqa: E402
from homr.main import download_weights, get_predictions, predict_symbols  # noqa: E402
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml  # noqa: E402
from homr.noise_filtering import filter_predictions  # noqa: E402
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems  # noqa: E402
from homr.resize import resize_image  # noqa: E402
from homr.segmentation.config import segnet_path_onnx  # noqa: E402
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger  # noqa: E402
from homr.staff_parsing import (  # noqa: E402
    _ensure_same_number_of_staffs,
    _get_number_of_voices,
    prepare_staff_image,
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


def dump(image_path: Path, config: Config) -> None:
    out = GOLDEN / image_path.stem
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
    predictions = filter_predictions(predictions, debug)
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

    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    write_json(out / "staffs.json", staffs)
    if len(staffs) == 0:
        raise SystemExit("No staffs found")

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    write_mask(out / "mask-brace_dot.png", brace_dot_img)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))
    write_json(out / "boxes-brace_dot.json", brace_dot)

    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )
    write_json(out / "notes.json", notes)
    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
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
            staff_image, transformed_staff = prepare_staff_image(
                debug, index, staff, predictions.preprocessed, regions=regions
            )
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
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")


def main() -> None:
    download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)
    config = Config()
    config.use_gpu_inference = False
    config.use_coreml_encoder = False
    pages = [Path(p) for p in sys.argv[1:]] or sorted(FIXTURES.glob("*.png"))
    for page in pages:
        dump(page, config)


if __name__ == "__main__":
    main()

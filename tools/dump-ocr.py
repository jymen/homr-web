"""Writes test/golden/<page>/ocr.json: every stage of RapidOCR on each chord
strip of the page, the strips cut as the AbcMusicStudio server's
omr_chord_ocr.py cuts them from homr's staff-positions.txt.

RapidOCR.__call__ is unrolled into the calls it makes, in its order, and the
script exits non-zero unless the unrolled result equals the real call's.
test/golden/<page>/texts.json, what the server route answers, is written by
tools/dump-texts.sh with the server's own code; this file is the per-stage
oracle behind it."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime
from rapidocr import RapidOCR
from rapidocr.ch_ppocr_cls import TextClsOutput
from rapidocr.ch_ppocr_rec import TextRecOutput
from rapidocr.utils.process_img import apply_vertical_padding

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "test" / "golden"
FIXTURES = ROOT / "test" / "fixtures"


def strips(positions: Path, height: int, width: int):
    for i, line in enumerate(positions.read_text().splitlines(keepends=True)):
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        top = int((cy - h / 2 - 1.9 * h) * height)
        bot = int((cy - h / 2 - 0.05 * h) * height)
        x0 = max(int((cx - w / 2 - 0.03) * width), 0)
        x1 = int((cx + w / 2 + 0.02) * width)
        yield i, top, bot, x0, x1


def boxes_json(boxes) -> list:
    return [[[float(v) for v in p] for p in box] for box in boxes]


def unrolled(engine: RapidOCR, strip: np.ndarray) -> dict:
    ori = engine.load_img(strip)
    img, op_record = engine.preprocess_img(ori)
    stage = {
        "preprocess": {
            "height": img.shape[0],
            "width": img.shape[1],
            "ratioH": op_record["preprocess"]["ratio_h"],
            "ratioW": op_record["preprocess"]["ratio_w"],
        }
    }
    padded, op_record = apply_vertical_padding(
        img, op_record, engine.width_height_ratio, engine.min_height
    )
    stage["paddingTop"] = op_record["padding_1"]["top"]
    det = engine.text_det(padded)
    if det.boxes is None:
        stage["det"] = {"boxes": [], "scores": []}
        stage["final"] = {"boxes": [], "txts": [], "scores": []}
        return stage
    stage["det"] = {"boxes": boxes_json(det.boxes), "scores": [float(s) for s in det.scores]}
    crops = engine.crop_text_regions(padded, det.boxes)
    stage["crops"] = [[c.shape[0], c.shape[1]] for c in crops]
    cls_images, cls = engine.cls_and_rotate(crops)
    stage["cls"] = [[label, float(score)] for label, score in cls.cls_res]
    rec = engine.recognize_txt(cls_images)
    stage["rec"] = [[txt, float(score)] for txt, score in zip(rec.txts, rec.scores, strict=True)]
    final = engine.build_final_output(ori, det, cls, rec, crops, op_record)
    if final.txts is None:
        stage["final"] = {"boxes": [], "txts": [], "scores": []}
    else:
        stage["final"] = {
            "boxes": boxes_json(final.boxes),
            "txts": list(final.txts),
            "scores": [float(s) for s in final.scores],
        }
    return stage


def same(a: dict, b) -> bool:
    if b.txts is None:
        return a["txts"] == []
    return a["txts"] == list(b.txts) and a["scores"] == [float(s) for s in b.scores] and a[
        "boxes"
    ] == boxes_json(b.boxes)


def dump(page: Path, engine: RapidOCR) -> None:
    out = GOLDEN / page.stem if page.parent == FIXTURES else GOLDEN / "local" / page.stem
    image = cv2.imread(str(page))
    height, width = image.shape[:2]
    result = []
    for line, top, bot, x0, x1 in strips(out / "staff-positions.txt", height, width):
        strip = image[max(top, 0) : bot, x0:x1]
        stage = unrolled(engine, strip)
        if not same(stage["final"], engine(strip)):
            raise SystemExit(f"{page.name} line {line}: the unrolled RapidOCR differs from the real call")
        result.append({"line": line, "top": top, "bottom": bot, "x0": x0, "x1": x1, **stage})
    meta = {"opencv": cv2.__version__, "numpy": np.__version__, "onnxruntime": onnxruntime.__version__}
    (out / "ocr.json").write_text(json.dumps({"oracle": meta, "strips": result}, indent=1, ensure_ascii=False) + "\n")
    print(f"== {page.name}: {sum(len(s['final']['txts']) for s in result)} texts")


def main() -> None:
    pages = [Path(p).resolve() for p in sys.argv[1:]] or sorted(FIXTURES.glob("*.png")) + sorted(
        (FIXTURES / "local").glob("*.png")
    )
    engine = RapidOCR(params={"Global.log_level": "error"})
    for page in pages:
        dump(page, engine)


if __name__ == "__main__":
    main()

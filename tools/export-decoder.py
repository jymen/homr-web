"""Re-exports homr's transformer decoder for the WebGPU execution provider.

The shipped decoder is fused (com.microsoft SkipLayerNormalization, which the
WebGPU EP has no kernel for) and int8-quantised. This runs homr's own
training/onnx/convert.py on the public pytorch_model_396 checkpoint, at the
commit tagged v0.7.0, and stops before fuse.py and quantization.py.

    tools/export-venv.sh
    .venv-export/bin/python tools/export-decoder.py

Writes into models/:
    decoder_pytorch_model_396-<hash>_web.onnx        fp32, unfused
    decoder_pytorch_model_396-<hash>_web_fp16.onnx   fp16 weights, fp32 inputs and outputs

Idempotent: the clone and the checkpoint under .export/ are reused, and an
output already in models/ is left alone unless --overwrite is given.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

HOMR_URL = "https://github.com/liebharc/homr.git"
HOMR_COMMIT = "8b5dcf7d7bdd1a47911dc0c661c573b957271eab"  # tag v0.7.0
MODEL_NAME = "pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903"
CHECKPOINT_URL = f"https://github.com/liebharc/homr/releases/download/checkpoints/{MODEL_NAME}.zip"

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / ".export"
CLONE = EXPORT / "homr"
MODELS = ROOT / "models"
OUT_FP32 = MODELS / f"decoder_{MODEL_NAME}_web.onnx"
OUT_FP16 = MODELS / f"decoder_{MODEL_NAME}_web_fp16.onnx"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(CLONE), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def ensure_clone() -> None:
    if not CLONE.exists():
        subprocess.run(["git", "clone", "--quiet", HOMR_URL, str(CLONE)], check=True)
    if git("rev-parse", "HEAD") != HOMR_COMMIT:
        git("fetch", "--quiet", "--tags", "origin")
        git("checkout", "--quiet", HOMR_COMMIT)
    if git("rev-parse", "HEAD") != HOMR_COMMIT:
        sys.exit(f"{CLONE} is not at {HOMR_COMMIT}")


def ensure_checkpoint() -> Path:
    """The .pth where homr's FilePaths().checkpoint looks for it, relative to the clone."""
    target = CLONE / "training" / "architecture" / "transformer" / f"{MODEL_NAME}.pth"
    if target.exists():
        return target
    archive = EXPORT / f"{MODEL_NAME}.zip"
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"fetching {CHECKPOINT_URL}", file=sys.stderr)
        urllib.request.urlretrieve(CHECKPOINT_URL, partial)
        partial.rename(archive)
    with zipfile.ZipFile(archive) as z:
        members = [m for m in z.namelist() if m.endswith(".pth")]
        if len(members) != 1:
            sys.exit(f"{archive} holds {members}, expected one .pth")
        with z.open(members[0]) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target


def export_fp32() -> None:
    os.chdir(CLONE)
    sys.path.insert(0, str(CLONE))
    from homr.transformer.configs import Config
    from training.onnx.convert import convert_decoder
    from training.onnx.split_weights import split_weights

    config = Config()
    split_weights(config.filepaths.checkpoint)
    try:
        path = convert_decoder(overwrite=True)
    finally:
        for weights in ("decoder_weights.pt", "encoder_weights.pt"):
            Path(weights).unlink(missing_ok=True)
    if path is None:
        sys.exit("convert_decoder wrote nothing")
    shutil.move(path, OUT_FP32)


def export_fp16() -> None:
    """keep_io_types: the graph casts at its edges, so the feeds and the caches stay float32 on every backend."""
    import onnx
    from onnxconverter_common import float16

    model = float16.convert_float_to_float16(onnx.load(str(OUT_FP32)), keep_io_types=True)
    # Two things onnxconverter-common 1.16 leaves float32 inside the graph, each
    # of which stops the model loading with an Einsum fed one half and one float:
    # 1. It casts each cache_out<i> back to float32 at the graph's edge but
    #    leaves the layer's own attention reading the float32 name. Those
    #    readers are pointed at the half the edge Cast consumes.
    # 2. x-transformers casts the attention softmax to float, and the converter
    #    does not rewrite a Cast's `to`. Internal casts to float become half.
    graph = model.graph
    outputs = {o.name for o in graph.output}
    edge_casts = [n for n in graph.node if n.op_type == "Cast" and n.output[0] in outputs]
    half_of = {n.output[0]: n.input[0] for n in edge_casts}
    for node in graph.node:
        if node in edge_casts:
            continue
        for i, name in enumerate(node.input):
            if name in half_of:
                node.input[i] = half_of[name]
        if node.op_type == "Cast":
            for attribute in node.attribute:
                if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                    attribute.i = onnx.TensorProto.FLOAT16
    del graph.value_info[:]
    onnx.checker.check_model(model)
    onnx.save(model, str(OUT_FP16))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-fp16", action="store_true")
    args = parser.parse_args()

    EXPORT.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    ensure_clone()
    ensure_checkpoint()
    if args.overwrite or not OUT_FP32.exists():
        export_fp32()
    if not args.no_fp16 and (args.overwrite or not OUT_FP16.exists()):
        export_fp16()
    for out in (OUT_FP32, OUT_FP16):
        if out.exists():
            print(f"{sha256(out)}  {out.stat().st_size:>10}  {out.name}")


if __name__ == "__main__":
    main()

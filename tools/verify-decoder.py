"""Decodes every golden canvas with the shipped decoder and with each
re-exported one, through homr's own ScoreDecoder, and diffs the tokens.

    .venv-export/bin/python tools/verify-decoder.py

The encoder is the shipped fp32 file on the CPU EP, as the oracle runs it, so
the decoders are the only variable. Exits non-zero on any difference in any of
the six heads. The attention coordinates are reported, not compared: phase 7
found they move by whole pixels under changes that leave every token alone.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLONE = ROOT / ".export" / "homr"
MODELS = ROOT / "models"
GOLDEN = ROOT / "test" / "golden"
FIXTURES = ("grand-staff-300dpi", "the-kesh-300dpi")
MODEL_NAME = "pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903"
SHIPPED = f"decoder_{MODEL_NAME}.onnx"
CANDIDATES = (f"decoder_{MODEL_NAME}_web.onnx", f"decoder_{MODEL_NAME}_web_fp16.onnx")
HEADS = ("rhythm", "pitch", "lift", "articulation", "slur", "position")

sys.path.insert(0, str(CLONE))
from homr.transformer.configs import Config  # noqa: E402
from homr.transformer.decoder_inference import ScoreDecoder  # noqa: E402
from homr.transformer.staff2score import ConvertToArray  # noqa: E402


def session(name: str) -> ort.InferenceSession:
    return ort.InferenceSession(str(MODELS / name), providers=["CPUExecutionProvider"])


def decode(decoder: ScoreDecoder, context: np.ndarray, config: Config) -> list:
    return decoder.generate(
        np.array([[1]], dtype=np.int64),
        np.array([[0]], dtype=np.int64),
        seq_len=config.max_seq_len,
        eos_token=config.eos_token,
        context=context,
    )


def main() -> None:
    config = Config()
    encoder = session(f"encoder_{MODEL_NAME}.onnx")
    to_array = ConvertToArray()
    decoders = {
        name: ScoreDecoder(session(name), fp16=False, use_gpu=False, config=config)
        for name in (SHIPPED, *CANDIDATES)
        if (MODELS / name).exists()
    }
    failed = False
    for fixture in FIXTURES:
        for canvas in sorted((GOLDEN / fixture).glob("canvas-[0-9].png")):
            image = np.array(Image.open(canvas).convert("L"))
            context = encoder.run(["output"], {"input": to_array(image)})[0]
            reference = decode(decoders[SHIPPED], context, config)
            for name, decoder in decoders.items():
                if name == SHIPPED:
                    continue
                symbols = decode(decoder, context, config)
                same = len(symbols) == len(reference) and all(
                    getattr(a, h) == getattr(b, h)
                    for a, b in zip(symbols, reference, strict=True)
                    for h in HEADS
                )
                drift = max(
                    (
                        float(np.abs(np.asarray(a.coordinates) - np.asarray(b.coordinates)).max())
                        for a, b in zip(symbols, reference, strict=False)
                    ),
                    default=0.0,
                )
                failed |= not same
                print(
                    f"{fixture} {canvas.stem} {name.removeprefix(f'decoder_{MODEL_NAME}')}: "
                    f"{len(symbols)} tokens against {len(reference)}, "
                    f"{'equal' if same else 'DIFFERENT'}, attention drift {drift:.3f} px"
                )
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

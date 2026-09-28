#!/usr/bin/env bash
# Builds the pinned Python oracle: homr 0.7.0 in .venv, with its models.
# Mirrors the production install recipe in AbcGoDb's CLAUDE.md, and forces the
# headless OpenCV 4.x wheel on every platform: on Linux it is what avoids
# rapidocr's opencv-python needing libGL, and everywhere it is what keeps the
# oracle off OpenCV 5's older minAreaRect angle convention.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3.12}"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r tools/requirements.txt
# homr asks for opencv-python-headless <5, but rapidocr asks for an unpinned
# opencv-python, and both wheels install the same cv2 package, so whichever
# landed last owns the import. On 2026-09-28 that was opencv-python 5.0.0.93,
# whose minAreaRect reports the pre-4.5.1 [-90, 0) angle convention, and homr's
# geometry is written against 4.x's (0, 90]. The two agree after homr's own
# normalisation everywhere except an angle of exactly +-45, where they disagree
# in sign, and boxPoints starts from a different corner. That silently poisons
# every dumped box list. This swap used to be guarded by `uname = Linux`, where
# it was only about libGL, so on macOS the oracle ran OpenCV 5 unnoticed.
.venv/bin/python -m pip uninstall --quiet -y opencv-python opencv-python-headless || true
.venv/bin/python -m pip install --quiet --no-deps "opencv-python-headless>=4.13,<5"
.venv/bin/python - <<'CHECK'
import cv2, sys
if not cv2.__version__.startswith("4."):
    sys.exit(f"cv2 is {cv2.__version__}; the golden dump needs OpenCV 4.x")
print("cv2", cv2.__version__)
CHECK
# The oracle runs on the CPU with the fp32 models; download exactly those.
.venv/bin/python - <<'PY'
from homr.main import download_weights
download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)
PY
.venv/bin/python -c "import homr, importlib.metadata as m; print('homr', m.version('homr'))"

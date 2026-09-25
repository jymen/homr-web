#!/usr/bin/env bash
# Builds the pinned Python oracle: homr 0.7.0 in .venv, with its models.
# Mirrors the production install recipe in AbcGoDb's CLAUDE.md (headless
# OpenCV swap on Linux, where rapidocr's opencv-python needs libGL).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3.12}"
if [ ! -x .venv/bin/python ]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r tools/requirements.txt
if [ "$(uname)" = "Linux" ]; then
  .venv/bin/python -m pip uninstall --quiet -y opencv-python opencv-python-headless || true
  .venv/bin/python -m pip install --quiet --no-deps "opencv-python-headless<5"
fi
# The oracle runs on the CPU with the fp32 models; download exactly those.
.venv/bin/python - <<'PY'
from homr.main import download_weights
download_weights(segnet_use_gpu=False, transformer_use_gpu=False, coreml_encoder=False)
PY
.venv/bin/python -c "import homr, importlib.metadata as m; print('homr', m.version('homr'))"

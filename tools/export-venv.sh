#!/usr/bin/env bash
# Builds .venv-export, the torch environment tools/export-decoder.py runs in.
# Kept apart from tools/venv.sh's .venv so the oracle's packages never move.
# CPU torch is enough: the export traces the model once.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON="${PYTHON:-python3.12}"
if [ ! -x .venv-export/bin/python ]; then
  "$PYTHON" -m venv .venv-export
fi
.venv-export/bin/python -m pip install --quiet --upgrade pip
.venv-export/bin/python -m pip install --quiet -r tools/export-requirements.txt
.venv-export/bin/python -c "import torch, x_transformers, importlib.metadata as m; print('torch', torch.__version__, 'x-transformers', m.version('x-transformers'))"

#!/usr/bin/env bash
# Rebuilds the tab digit classifier (src/tab/digits.ts) from scratch: a venv
# with torch in .venv-digits, MNIST in .digits/ (torchvision's download, its
# MD5s checked), the training crops from tools/tab-digits-data.ts, then
# tools/train-tab-digits.py. Every step is seeded; the result is written to
# models/tab-digits-<first 12 hex of its SHA-256>.onnx and its name printed.
# Rerun gen-manifest.mjs after it (FILES names the file) when the model changes.
#
# MNIST (Yann LeCun, Corinna Cortes, Christopher J.C. Burges) is published
# under CC BY-SA 3.0; the weights are released under the same licence (NOTICE).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$root/.venv-digits"
data="$root/.digits"
python="${PYTHON:-python3.12}"
steps="${STEPS:-8000}"

if [ ! -x "$venv/bin/python" ]; then
  "$python" -m venv "$venv"
  "$venv/bin/pip" install --quiet -r "$root/tools/digits-requirements.txt"
fi
mkdir -p "$data"
"$venv/bin/python" -c "import torchvision, sys; [torchvision.datasets.MNIST(sys.argv[1], train=t, download=True) for t in (True, False)]" "$data"

gen() { (cd "$root" && npx vite-node tools/tab-digits-data.ts "$data" "$@"); }
# Four synthetic training sets on four seeds (run side by side), one held-out set, the fixtures.
for k in 1 2 3 4; do
  [ -f "$data/synthetic-train-$k-x.f32" ] || gen "$data/synthetic-train-$k" train 10000 "$k" &
done
[ -f "$data/synthetic-test-x.f32" ] || gen "$data/synthetic-test" test 2000 99 &
[ -f "$data/fixtures-x.f32" ] || gen "$data/fixtures" train 0 1 --fixtures &
wait

scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
"$venv/bin/python" "$root/tools/train-tab-digits.py" "$data" "$scratch/tab-digits.onnx" "$steps"
if command -v sha256sum >/dev/null; then hash="$(sha256sum "$scratch/tab-digits.onnx" | cut -c1-12)"
else hash="$(shasum -a 256 "$scratch/tab-digits.onnx" | cut -c1-12)"; fi
mkdir -p "$root/models"
mv "$scratch/tab-digits.onnx" "$root/models/tab-digits-$hash.onnx"
echo "models/tab-digits-$hash.onnx"

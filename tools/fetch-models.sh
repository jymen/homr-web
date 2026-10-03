#!/usr/bin/env bash
# Assembles eight of the nine ONNX artifacts phase 2 pins into models/, which is
# git-ignored. Six come from the pinned venv (tools/venv.sh installs homr
# 0.7.0 and rapidocr); the two fp16 ones are homr's own release assets, which
# the venv only downloads when it runs on a GPU. Idempotent: anything already
# in models/ is left alone, so a second run does nothing and costs nothing.
#
# A browser cannot fetch these release assets (the redirect carries no
# access-control-allow-origin), which is why the models are self-hosted and
# tools/gen-manifest.mjs reads them from here.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models="$root/models"
release="https://github.com/liebharc/homr/releases/download/onnx_checkpoints"

site="$(echo "$root"/.venv/lib/python3.*/site-packages)"
if [ ! -d "$site" ]; then
  echo "no venv at $root/.venv; run tools/venv.sh first" >&2
  exit 1
fi

mkdir -p "$models"

copied=0
skipped=0
fetched=0

# The six the venv already carries, by the directory homr and rapidocr keep them in.
for source in \
  "$site/homr/segmentation/segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f.onnx" \
  "$site/homr/transformer/encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx" \
  "$site/homr/transformer/decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903.onnx" \
  "$site/rapidocr/models/PP-OCRv6_det_small.onnx" \
  "$site/rapidocr/models/PP-OCRv6_rec_small.onnx" \
  "$site/rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx"; do
  name="$(basename "$source")"
  if [ -f "$models/$name" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  if [ ! -f "$source" ]; then
    echo "missing from the venv: $source" >&2
    exit 1
  fi
  cp "$source" "$models/$name"
  copied=$((copied + 1))
done

# The two fp16 ones, from homr's release. Each asset is "<stem>.zip" holding
# "<stem>.onnx" (homr/main.py download_weights).
for stem in \
  "segnet_308-3296ccd40960f90ca6ab9c035cca945675d30a0f_fp16" \
  "encoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903_fp16"; do
  if [ -f "$models/$stem.onnx" ]; then
    skipped=$((skipped + 1))
    continue
  fi
  # Unzip into a scratch directory and move the result into place only once it
  # is complete, so an interrupted download never leaves a half file in models/
  # for gen-manifest.mjs to hash.
  scratch="$(mktemp -d)"
  trap 'rm -rf "$scratch"' EXIT
  echo "fetching $stem.zip"
  curl --fail --location --progress-bar --output "$scratch/$stem.zip" "$release/$stem.zip"
  unzip -q -j "$scratch/$stem.zip" -d "$scratch"
  if [ ! -f "$scratch/$stem.onnx" ]; then
    echo "$stem.zip did not contain $stem.onnx" >&2
    exit 1
  fi
  mv "$scratch/$stem.onnx" "$models/$stem.onnx"
  rm -rf "$scratch"
  trap - EXIT
  fetched=$((fetched + 1))
done

echo "models/: $copied copied, $fetched fetched, $skipped already present"
ls -1 "$models"
if [ ! -f "$models/decoder_pytorch_model_396-f6feedb42ff90087d898b0941a55d040fa6b2903_web_fp16.onnx" ]; then
  echo "the WebGPU decoder is not released anywhere: run tools/export-venv.sh, then .venv-export/bin/python tools/export-decoder.py" >&2
fi

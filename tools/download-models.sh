#!/usr/bin/env bash
# Downloads the ONNX artifacts src/models/manifest.ts pins into models/, from
# where each is published, with no Python: what CI runs, since
# tools/fetch-models.sh copies six of them out of a local homr venv.
#
# Every file is checked against the SHA-256 the manifest records before it is
# moved into models/, so a changed upstream asset fails the run instead of
# testing the wrong model. Idempotent: a file already in models/ with the right
# hash is left alone.
#
# The WebGPU decoder re-export (phase 8) is published nowhere upstream; it is
# taken from this repository's own release when HOMR_WEB_MODELS_RELEASE names
# one, and reported missing otherwise.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models="$root/models"
manifest="$root/src/models/manifest.ts"
homr="https://github.com/liebharc/homr/releases/download/onnx_checkpoints"
rapidocr="https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx"
own_release="${HOMR_WEB_MODELS_RELEASE:-}"

sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# Where each artifact is published, and whether it arrives zipped (homr's
# release holds "<stem>.zip" around "<stem>.onnx").
source_of() {
  case "$1" in
    PP-OCRv6_det_small.onnx) echo "plain $rapidocr/PP-OCRv6/det/$1" ;;
    PP-OCRv6_rec_small.onnx) echo "plain $rapidocr/PP-OCRv6/rec/$1" ;;
    ch_ppocr_mobile_v2.0_cls_mobile.onnx) echo "plain $rapidocr/PP-OCRv4/cls/$1" ;;
    *_web_fp16.onnx)
      if [ -n "$own_release" ]; then
        echo "plain https://github.com/jymen/homr-web/releases/download/$own_release/$1"
      else
        echo "none"
      fi
      ;;
    *) echo "zip $homr/${1%.onnx}.zip" ;;
  esac
}

mkdir -p "$models"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT

ok=0
fetched=0
missing=0
while IFS=/ read -r hash file; do
  target="$models/$file"
  if [ -f "$target" ] && [ "$(sha256 "$target")" = "$hash" ]; then
    ok=$((ok + 1))
    continue
  fi
  read -r kind url <<<"$(source_of "$file")"
  if [ "$kind" = "none" ]; then
    echo "missing   $file: published nowhere upstream; set HOMR_WEB_MODELS_RELEASE" >&2
    missing=$((missing + 1))
    continue
  fi
  echo "fetching  $file"
  if [ "$kind" = "zip" ]; then
    curl --fail --silent --show-error --location --output "$scratch/a.zip" "$url"
    unzip -q -o -j "$scratch/a.zip" "$file" -d "$scratch"
    rm -f "$scratch/a.zip"
  else
    curl --fail --silent --show-error --location --output "$scratch/$file" "$url"
  fi
  actual="$(sha256 "$scratch/$file")"
  if [ "$actual" != "$hash" ]; then
    echo "mismatch  $file is $actual, the manifest pins $hash" >&2
    exit 1
  fi
  mv "$scratch/$file" "$target"
  fetched=$((fetched + 1))
done < <(grep -oE '[0-9a-f]{64}/[A-Za-z0-9._-]+\.onnx' "$manifest" | sort -u)

echo "models/: $fetched fetched, $ok already present, $missing missing"
[ "$missing" -eq 0 ]

#!/bin/sh
# Writes test/golden/<page>/texts.json with the app server's own chord OCR:
# parseOmrStaves and runOmrChordOcr in AbcGoDb's abcsql/omr.go, which run the
# embedded omr_chord_ocr.py with the homr virtualenv's python on the page and
# its staff-positions.txt, then remap and sort the texts as the route does.
# Usage: tools/dump-texts.sh ~/development/AbcGoDb_V03 ~/development/abcmusicstudio-current-home/tools/homr-venv
set -eu
GO_REPO=${1:?path to the AbcGoDb checkout}
HOMR_VENV=${2:?path to the server homr virtualenv}
HOMR_WEB=$(cd "$(dirname "$0")/.." && pwd)
PROBE="$GO_REPO/abcsql/zz_homrweb_texts_test.go"
trap 'rm -f "$PROBE"' EXIT
cat > "$PROBE" <<'GO'
package abcsql

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestZZHomrWebTexts(t *testing.T) {
	files, _ := filepath.Glob(os.Getenv("HOMR_WEB") + "/test/golden/*/staff-positions.txt")
	cfg := omrRunConfig{binaryPath: filepath.Join(os.Getenv("HOMR_VENV"), "bin", "homr"), timeout: 10 * time.Minute}
	for _, file := range files {
		dir := filepath.Dir(file)
		image := filepath.Join(os.Getenv("HOMR_WEB"), "test", "fixtures", filepath.Base(dir)+".png")
		if strings.Contains(dir, string(filepath.Separator)+"local"+string(filepath.Separator)) {
			image = filepath.Join(os.Getenv("HOMR_WEB"), "test", "fixtures", "local", filepath.Base(dir)+".png")
		}
		_, lineToIndex, err := parseOmrStaves(file)
		if err != nil {
			t.Fatal(err)
		}
		texts, notes := runOmrChordOcr(context.Background(), cfg, t.TempDir(), image, file, lineToIndex)
		if texts == nil {
			t.Fatalf("%s: %v", image, notes)
		}
		out, _ := json.MarshalIndent(texts, "", " ")
		if err := os.WriteFile(filepath.Join(dir, "texts.json"), append(out, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
GO
(cd "$GO_REPO" && HOMR_WEB="$HOMR_WEB" HOMR_VENV="$HOMR_VENV" go test ./abcsql/ -run TestZZHomrWebTexts -count=1 -v)

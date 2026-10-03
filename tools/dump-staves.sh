#!/bin/sh
# Writes test/golden/<page>/staves.json with the app server's own parser:
# parseOmrStaves in AbcGoDb's abcsql/omr.go, run on the page's
# staff-positions.txt through a throwaway test file that is removed again.
# Usage: tools/dump-staves.sh ~/development/AbcGoDb_V03
set -eu
GO_REPO=${1:?path to the AbcGoDb checkout}
HOMR_WEB=$(cd "$(dirname "$0")/.." && pwd)
PROBE="$GO_REPO/abcsql/zz_homrweb_staves_test.go"
trap 'rm -f "$PROBE"' EXIT
cat > "$PROBE" <<'GO'
package abcsql

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestZZHomrWebStaves(t *testing.T) {
	dirs, _ := filepath.Glob(os.Getenv("HOMR_WEB") + "/test/golden/*/staff-positions.txt")
	for _, file := range dirs {
		staves, _, err := parseOmrStaves(file)
		if err != nil {
			t.Fatal(err)
		}
		out, _ := json.MarshalIndent(staves, "", " ")
		if err := os.WriteFile(filepath.Join(filepath.Dir(file), "staves.json"), append(out, '\n'), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}
GO
(cd "$GO_REPO" && HOMR_WEB="$HOMR_WEB" go test ./abcsql/ -run TestZZHomrWebStaves -count=1)

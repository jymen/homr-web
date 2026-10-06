#!/usr/bin/env bash
# Publish the version in package.json to npm, and only that version.
#
# Usage: tools/npm-publish.sh [--check-only] [--skip-tests] [--yes] [--otp CODE | --web]
#
#   --check-only  run every check and the pack dry run, publish nothing
#   --skip-tests  skip the test suite (about ten minutes with the models);
#                 the type check, lint and build still run
#   --yes         do not ask for confirmation before publishing
#   --otp CODE    one-time code for an npm account with two-factor auth
#                 (HOMR_NPM_OTP in the environment works too)
#   --web         approve the publish in the browser instead (npm
#                 --auth-type=web): the way for a passkey or security key.
#                 npm prints a link and waits, so run it from a terminal
#
# What it refuses, each before anything is published:
#   - a working tree with changes, a branch other than main, or a main that
#     is not exactly origin/main: what is published is what is on GitHub;
#   - a missing release tag: v<version>+homr<HOMR_VERSION> must exist and be
#     pushed, as v0.1.0 and v0.2.0 were, and HEAD must build the same package
#     as the tagged commit (commits after the tag may only touch files that
#     are not shipped or compiled, such as tools/ or docs/);
#   - a version already on npm (exits 0, there is nothing to do);
#   - no npm login: run `npm login` first (it is interactive, so this script
#     only starts it when run from a terminal).
#
# dist/ is not tracked: it is rebuilt from scratch here, so the tarball holds
# exactly what HEAD compiles to.
set -euo pipefail

cd "$(dirname "$0")/.."

check_only=false
skip_tests=false
assume_yes=false
otp="${HOMR_NPM_OTP:-}"
web=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) check_only=true ;;
    --skip-tests) skip_tests=true ;;
    --yes) assume_yes=true ;;
    --otp) shift; otp="${1:-}" ;;
    --web) web=true ;;
    -h|--help) sed -n '2,29p' "$0"; exit 0 ;;
    *) echo "npm-publish: unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

step() { printf '\n== %s\n' "$*"; }
fail() { echo "npm-publish: $*" >&2; exit 1; }

name=$(node -p "require('./package.json').name")
version=$(node -p "require('./package.json').version")
homr=$(sed -n 's/.*HOMR_VERSION = "\([^"]*\)".*/\1/p' src/version.ts)
[[ -n "$homr" ]] || fail "could not read HOMR_VERSION from src/version.ts"
tag="v${version}+homr${homr}"
echo "Package ${name}@${version}, tag ${tag}"

step "Repository state"
[[ -z "$(git status --porcelain)" ]] || { git status --short; fail "the working tree has changes; commit or remove them first"; }
branch=$(git rev-parse --abbrev-ref HEAD)
[[ "$branch" == "main" ]] || fail "on branch $branch, not main"
git fetch -q origin main --tags
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] \
  || fail "main is not origin/main; push or pull first"
git rev-parse -q --verify "refs/tags/${tag}" >/dev/null \
  || fail "tag ${tag} does not exist; create it on the release commit: git tag ${tag} && git push origin ${tag}"
# Everything that is compiled into dist/ or shipped beside it.
shipped=(src package.json package-lock.json tsconfig.json tsconfig.build.json README.md LICENSE NOTICE)
git diff --quiet "${tag}" HEAD -- "${shipped[@]}" \
  || { git diff --stat "${tag}" HEAD -- "${shipped[@]}"; fail "HEAD changes what ${tag} ships; release a new version instead"; }
git ls-remote --exit-code --tags origin "refs/tags/${tag}" >/dev/null \
  || fail "tag ${tag} is not pushed; git push origin ${tag}"
echo "clean, main = origin/main, ${tag} pushed and HEAD ships the same files"

step "npm"
published=$(npm view "${name}@${version}" version 2>/dev/null || true)
if [[ "$published" == "$version" ]]; then
  echo "${name}@${version} is already on npm; nothing to do"
  exit 0
fi
echo "latest on npm: $(npm view "${name}" version 2>/dev/null || echo none)"
if ! npm_user=$(npm whoami 2>/dev/null); then
  if $check_only; then
    echo "not logged in to npm (publishing will need: npm login)"
  elif [[ -t 0 && -t 1 ]]; then
    echo "not logged in to npm; starting npm login"
    npm login
    npm_user=$(npm whoami) || fail "npm login did not complete"
  else
    fail "not logged in to npm; run npm login in a terminal, then run this again"
  fi
fi
[[ -n "${npm_user:-}" ]] && echo "logged in as ${npm_user}"

step "Type check and lint"
npm run --silent check
npm run --silent lint

if $skip_tests; then
  step "Tests skipped (--skip-tests)"
else
  step "Tests"
  npm test
fi

step "Build"
rm -rf dist
npm run --silent build
[[ -f dist/index.js ]] || fail "the build produced no dist/index.js"

step "Package contents"
pack=$(npm pack --dry-run --json 2>/dev/null)
packed_version=$(node -e 'const p=JSON.parse(require("fs").readFileSync(0,"utf8"))[0];console.log(p.version)' <<<"$pack")
[[ "$packed_version" == "$version" ]] || fail "the tarball says ${packed_version}, package.json says ${version}"
node -e '
  const p = JSON.parse(require("fs").readFileSync(0, "utf8"))[0];
  const files = p.files.map((f) => f.path);
  for (const want of ["dist/index.js", "LICENSE", "NOTICE", "README.md"]) {
    if (!files.includes(want)) { console.error("missing from the tarball: " + want); process.exit(1); }
  }
  console.log(`${p.filename}: ${files.length} files, ${(p.size / 1000).toFixed(1)} kB packed, ${(p.unpackedSize / 1000).toFixed(1)} kB unpacked`);
' <<<"$pack"

if $check_only; then
  step "Check only: everything is ready, nothing was published"
  exit 0
fi

[[ -n "${npm_user:-}" ]] || fail "not logged in to npm; run npm login, then run this again"

if ! $assume_yes; then
  [[ -t 0 ]] || fail "not a terminal; pass --yes to publish without the prompt"
  read -r -p "Publish ${name}@${version} to npm as ${npm_user}? [y/N] " answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || fail "not published"
fi

step "Publish"
if [[ -n "$otp" ]]; then
  npm publish --otp "$otp"
elif $web; then
  [[ -t 0 && -t 1 ]] || fail "--web needs a terminal: npm prints a link to approve in the browser, then waits"
  npm publish --auth-type=web
else
  npm publish || fail "npm publish failed; an EOTP error means two-factor: pass --otp <code>, or --web from a terminal for a passkey or security key"
fi

step "Verify"
for _ in 1 2 3 4 5 6; do
  now=$(npm view "${name}@${version}" version 2>/dev/null || true)
  [[ "$now" == "$version" ]] && { echo "${name}@${version} is on npm"; exit 0; }
  sleep 5
done
fail "npm publish returned, but npm view does not list ${version} yet; check https://www.npmjs.com/package/${name}"

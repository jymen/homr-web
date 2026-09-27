/**
 * Test-side access to test/golden/<fixture>/: file reading and PNG
 * decoding, the two things the library itself must not depend on. Public
 * fixtures and the git-ignored local ones are listed alike, as
 * test/fixtures.test.ts already does.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { PNG } from "pngjs";
import {
  createGoldenPage,
  type GoldenPage,
  type GoldenReader,
} from "../../src/golden/page.js";

const goldenRoot = join(import.meta.dirname, "..", "golden");

export interface GoldenFixture {
  /** test/golden/<name> or test/golden/local/<name>. */
  readonly dir: string;
  readonly name: string;
}

const fixturesIn = (dir: string): GoldenFixture[] =>
  existsSync(dir)
    ? readdirSync(dir, { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && entry.name !== "local")
        .map((entry) => ({ dir: join(dir, entry.name), name: entry.name }))
    : [];

/** Every fixture that has a golden directory, public first. */
export function listGoldenFixtures(): GoldenFixture[] {
  return [...fixturesIn(goldenRoot), ...fixturesIn(join(goldenRoot, "local"))];
}

/** pngjs hands back RGBA whatever the file's depth; the golden PNGs are gray, so channel 0 is the pixel. */
function decodePng(bytes: Uint8Array): {
  width: number;
  height: number;
  gray: Uint8Array;
} {
  const png = PNG.sync.read(Buffer.from(bytes));
  const gray = new Uint8Array(png.width * png.height);
  for (let i = 0; i < gray.length; i += 1) {
    gray[i] = png.data[i * 4] ?? 0;
  }
  return { gray, height: png.height, width: png.width };
}

export function readerFor(fixture: GoldenFixture): GoldenReader {
  return {
    png: (name) => decodePng(readFileSync(join(fixture.dir, name))),
    text: (name) => readFileSync(join(fixture.dir, name), "utf8"),
  };
}

export function goldenPageOf(fixture: GoldenFixture): GoldenPage {
  return createGoldenPage(readerFor(fixture));
}

export function readGoldenJson(name: string): unknown {
  return JSON.parse(readFileSync(join(goldenRoot, name), "utf8"));
}

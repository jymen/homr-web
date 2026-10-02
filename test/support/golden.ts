/**
 * Test-side access to test/golden/<fixture>/: file reading and PNG
 * decoding, the two things the library itself must not depend on. Public
 * fixtures and the git-ignored local ones are listed alike, as
 * test/fixtures.test.ts already does.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { PNG } from "pngjs";
import {
  createGoldenPage,
  type GoldenPage,
  type GoldenPng,
  type GoldenReader,
} from "../../src/golden/page.js";
import { type ColorImage, colorImageFromRgba } from "../../src/image/plane.js";

const goldenRoot = join(import.meta.dirname, "..", "golden");
const fixtureRoot = join(import.meta.dirname, "..", "fixtures");

export interface GoldenFixture {
  /** test/golden/<name> or test/golden/local/<name>. */
  readonly dir: string;
  readonly name: string;
}

/** Directories under test/golden/ that hold something other than one page's dump. */
const NOT_A_FIXTURE: ReadonlySet<string> = new Set(["local", "vectors"]);

const fixturesIn = (dir: string): GoldenFixture[] =>
  existsSync(dir)
    ? readdirSync(dir, { withFileTypes: true })
        .filter(
          (entry) => entry.isDirectory() && !NOT_A_FIXTURE.has(entry.name)
        )
        .map((entry) => ({ dir: join(dir, entry.name), name: entry.name }))
    : [];

/** Every fixture that has a golden directory, public first. */
export function listGoldenFixtures(): GoldenFixture[] {
  return [...fixturesIn(goldenRoot), ...fixturesIn(join(goldenRoot, "local"))];
}

/** pngjs hands back RGBA whatever the file's depth, which is exactly what GoldenPng wants. */
function decodePng(bytes: Uint8Array): GoldenPng {
  const png = PNG.sync.read(Buffer.from(bytes));
  return { height: png.height, rgba: png.data, width: png.width };
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

/**
 * The fixture page itself as BGR: what `cv2.imread` hands autocrop, and the one
 * stage input that is not a golden file. Same pngjs decode and
 * colorImageFromRgba as the golden reader, so the two can only differ by file.
 */
export function fixtureImageOf(fixture: GoldenFixture): ColorImage {
  const inLocal = basename(dirname(fixture.dir)) === "local";
  const file = `${fixture.name}.png`;
  const png = decodePng(
    readFileSync(
      inLocal ? join(fixtureRoot, "local", file) : join(fixtureRoot, file)
    )
  );
  return colorImageFromRgba(png.width, png.height, png.rgba);
}

export function readGoldenJson(name: string): unknown {
  return JSON.parse(readFileSync(join(goldenRoot, name), "utf8"));
}

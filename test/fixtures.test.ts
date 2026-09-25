import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const fixturesDir = join(import.meta.dirname, "fixtures");
const localDir = join(fixturesDir, "local");
const goldenDir = join(import.meta.dirname, "golden");
const PNG_SUFFIX = /\.png$/;

const pngsIn = (dir: string) =>
  existsSync(dir)
    ? readdirSync(dir).filter((name) => name.endsWith(".png"))
    : [];

/** Public pages, then the private ones a developer keeps in fixtures/local (git-ignored). */
const fixtures = [
  ...pngsIn(fixturesDir).map((name) => ({
    dir: fixturesDir,
    golden: goldenDir,
    name,
  })),
  ...pngsIn(localDir).map((name) => ({
    dir: localDir,
    golden: join(goldenDir, "local"),
    name,
  })),
];

const sha256 = (path: string) =>
  createHash("sha256").update(readFileSync(path)).digest("hex");

interface GoldenMeta {
  fixture: string;
  homrVersion: string;
  imageSha256: string;
  models: Record<string, string>;
}

describe("fixtures", () => {
  it("has at least one public page", () => {
    expect(pngsIn(fixturesDir).length).toBeGreaterThan(0);
  });

  const sources = readFileSync(join(fixturesDir, "SOURCE.md"), "utf8");
  for (const fixture of fixtures) {
    const isPublic = fixture.dir === fixturesDir;
    it(`${fixture.name} ${isPublic ? "names its origin in SOURCE.md" : "is private"}`, () => {
      if (isPublic) {
        expect(sources).toContain(`\`${fixture.name}\``);
      } else {
        expect(sources).not.toContain(fixture.name);
      }
    });

    const golden = join(fixture.golden, fixture.name.replace(PNG_SUFFIX, ""));
    it(`${fixture.name} has golden data from this exact image`, () => {
      expect(existsSync(join(golden, "meta.json"))).toBe(true);
      const meta = JSON.parse(
        readFileSync(join(golden, "meta.json"), "utf8")
      ) as GoldenMeta;
      expect(meta.fixture).toBe(fixture.name);
      expect(meta.imageSha256).toBe(sha256(join(fixture.dir, fixture.name)));
      expect(meta.homrVersion).toBe("0.7.0");
      expect(Object.keys(meta.models).sort()).toEqual([
        "decoder",
        "encoder",
        "segnet",
      ]);
    });
  }
});

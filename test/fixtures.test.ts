import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

const fixturesDir = join(import.meta.dirname, "fixtures");
const goldenDir = join(import.meta.dirname, "golden");

const fixtures = readdirSync(fixturesDir).filter((name) =>
  name.endsWith(".png")
);

const PNG_SUFFIX = /\.png$/;

const sha256 = (path: string) =>
  createHash("sha256").update(readFileSync(path)).digest("hex");

interface GoldenMeta {
  fixture: string;
  homrVersion: string;
  imageSha256: string;
  models: Record<string, string>;
}

describe("fixtures", () => {
  it("has at least one page", () => {
    expect(fixtures.length).toBeGreaterThan(0);
  });

  const sources = readFileSync(join(fixturesDir, "SOURCE.md"), "utf8");
  for (const fixture of fixtures) {
    it(`${fixture} names its origin in SOURCE.md`, () => {
      expect(sources).toContain(`\`${fixture}\``);
    });

    const golden = join(goldenDir, fixture.replace(PNG_SUFFIX, ""));
    it(`${fixture} has golden data from this exact image`, () => {
      expect(existsSync(join(golden, "meta.json"))).toBe(true);
      const meta = JSON.parse(
        readFileSync(join(golden, "meta.json"), "utf8")
      ) as GoldenMeta;
      expect(meta.fixture).toBe(fixture);
      expect(meta.imageSha256).toBe(sha256(join(fixturesDir, fixture)));
      expect(meta.homrVersion).toBe("0.7.0");
      expect(Object.keys(meta.models).sort()).toEqual([
        "decoder",
        "encoder",
        "segnet",
      ]);
    });
  }
});

/**
 * The MusicXML writer against homr's: generate_xml on hand-built voices for
 * every branch (musicxml.json), and on each page's voices.json against its
 * page.musicxml. Compared canonically, so attribute order and indentation do
 * not count and a moved, missing or changed element does.
 */

import { describe, expect, it } from "vitest";
import { generateMusicXml, MusicXmlError } from "../src/musicxml/generate.js";
import {
  createEncodedSymbol,
  type EncodedSymbol,
} from "../src/transformer/symbol.js";
import { canonicalXml } from "./support/canonical-xml.js";
import {
  goldenPageOf,
  listGoldenFixtures,
  readerFor,
} from "./support/golden.js";
import { vectorSet } from "./support/vectors.js";

const symbolOf = ([
  rhythm = ".",
  pitch = ".",
  lift = ".",
  articulation = ".",
  slur = ".",
  position = ".",
]: readonly string[]): EncodedSymbol =>
  createEncodedSymbol(rhythm, {
    articulation,
    lift: lift as EncodedSymbol["lift"],
    pitch: pitch as EncodedSymbol["pitch"],
    position: position as EncodedSymbol["position"],
    slur,
  });

interface MusicXmlCase {
  readonly error?: string;
  readonly log: readonly string[];
  readonly name: string;
  readonly voices: readonly (readonly (readonly string[])[])[];
  readonly xml?: string;
}

const cases = vectorSet("musicxml").cases as unknown as readonly MusicXmlCase[];

describe("generateMusicXml against homr's generate_xml", () => {
  it.each(cases.map((one) => [one.name, one] as const))("%s", (_, one) => {
    const log: string[] = [];
    const voices = one.voices.map((voice) => voice.map(symbolOf));
    if (one.error !== undefined) {
      expect(() =>
        generateMusicXml(voices, "", (line) => log.push(line))
      ).toThrow(MusicXmlError);
      return;
    }
    const xml = generateMusicXml(voices, "", (line) => log.push(line));
    expect(canonicalXml(xml)).toEqual(canonicalXml(one.xml ?? ""));
    expect(log).toEqual(one.log);
  });
});

describe("the canonical comparison", () => {
  const xml = cases.find((one) => one.name === "articulations")?.xml ?? "";

  it("ignores attribute order and indentation", () => {
    const reformatted = xml
      .replaceAll(/\n\s*/g, "")
      .replace(
        '<slur type="start" number="1" />',
        '<slur number="1" type="start" />'
      );
    expect(canonicalXml(reformatted)).toEqual(canonicalXml(xml));
  });

  it("fails on a missing tie", () => {
    expect(xml).toContain('<tied type="start" />');
    expect(canonicalXml(xml.replace('<tied type="start" />', ""))).not.toEqual(
      canonicalXml(xml)
    );
  });

  it("fails on a reordered child", () => {
    const swapped = xml.replace(
      "<type>eighth</type>\n        <staff>1</staff>",
      "<staff>1</staff>\n        <type>eighth</type>"
    );
    expect(swapped).not.toEqual(xml);
    expect(canonicalXml(swapped)).not.toEqual(canonicalXml(xml));
  });
});

describe.each(listGoldenFixtures().map((f) => [f.name, f] as const))(
  "%s: voices.json to page.musicxml",
  (_, fixture) => {
    it("is homr's file, byte for byte", () => {
      const { title } = JSON.parse(readerFor(fixture).text("title.json")) as {
        title: string;
      };
      const xml = generateMusicXml(goldenPageOf(fixture).voices(), title);
      const expected = readerFor(fixture).text("page.musicxml");
      expect(canonicalXml(xml)).toEqual(canonicalXml(expected));
      expect(xml).toBe(expected);
    });
  }
);

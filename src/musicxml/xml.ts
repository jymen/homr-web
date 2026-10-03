/**
 * The XML tree homr's `musicxml` package builds, cut down to what
 * music_xml_generator.py uses, and its `write` format.
 *
 * That package places a child where the MusicXML schema's sequence puts it,
 * not where it was added: homr adds a note's `<voice>` before its `<type>`
 * and a pitch's `<octave>` before its `<alter>`, and the file has them the
 * other way round. `SCHEMA_ORDER` holds the four sequences where homr's
 * insertion order and the schema's differ; every other element keeps the
 * order its children were added in, which for the schema's choice groups
 * (`<measure>`, `<notations>`, `<articulations>`) is what the package does too.
 */

const SCHEMA_ORDER: Readonly<Record<string, readonly string[]>> = {
  attributes: [
    "divisions",
    "key",
    "time",
    "staves",
    "part-symbol",
    "clef",
    "measure-style",
  ],
  barline: ["bar-style", "ending", "repeat"],
  note: [
    "grace",
    "chord",
    "pitch",
    "rest",
    "duration",
    "voice",
    "type",
    "dot",
    "time-modification",
    "staff",
    "notations",
  ],
  pitch: ["step", "alter", "octave"],
};

export class XmlElement {
  readonly attributes: [string, string][] = [];
  readonly children: XmlElement[] = [];
  readonly name: string;
  text: string | undefined;

  constructor(
    name: string,
    attributes: Readonly<Record<string, string | number>> = {},
    text?: string | number
  ) {
    this.name = name;
    for (const [key, value] of Object.entries(attributes)) {
      this.attributes.push([key, String(value)]);
    }
    this.text = text === undefined ? undefined : String(text);
  }

  /** Adds `child` in schema position where this element has a sequence, else last. Returns the child. */
  add(child: XmlElement): XmlElement {
    const order = SCHEMA_ORDER[this.name];
    const rank = order?.indexOf(child.name) ?? -1;
    if (order === undefined || rank < 0) {
      this.children.push(child);
      return child;
    }
    let at = this.children.length;
    while (at > 0) {
      const before = this.children[at - 1];
      if (before === undefined || order.indexOf(before.name) <= rank) {
        break;
      }
      at -= 1;
    }
    this.children.splice(at, 0, child);
    return child;
  }

  attribute(name: string): string | undefined {
    return this.attributes.find(([key]) => key === name)?.[1];
  }

  childrenNamed(name: string): XmlElement[] {
    return this.children.filter((child) => child.name === name);
  }

  setAttribute(name: string, value: string): void {
    const existing = this.attributes.find(([key]) => key === name);
    if (existing === undefined) {
      this.attributes.push([name, value]);
    } else {
      existing[1] = value;
    }
  }
}

/** A leaf with text, the shape of nearly every MusicXML value element. */
export const leaf = (name: string, text: string | number): XmlElement =>
  new XmlElement(name, {}, text);

const escapeText = (text: string): string =>
  text.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");

const escapeAttribute = (text: string): string =>
  escapeText(text).replaceAll('"', "&quot;");

function writeElement(element: XmlElement, depth: number, out: string[]): void {
  const indent = "  ".repeat(depth);
  const attributes = element.attributes
    .map(([key, value]) => ` ${key}="${escapeAttribute(value)}"`)
    .join("");
  const open = `${indent}<${element.name}${attributes}`;
  const text = element.text === undefined ? "" : escapeText(element.text);
  if (element.children.length === 0) {
    out.push(text === "" ? `${open} />` : `${open}>${text}</${element.name}>`);
    return;
  }
  out.push(`${open}>${text}`);
  for (const child of element.children) {
    writeElement(child, depth + 1, out);
  }
  out.push(`${indent}</${element.name}>`);
}

/** The document `XMLElement.write` produces: declaration, two-space indent, `<empty />`, a final newline. */
export function writeXmlDocument(root: XmlElement): string {
  const lines = ['<?xml version="1.0" encoding="UTF-8" standalone="no"?>'];
  writeElement(root, 0, lines);
  return `${lines.join("\n")}\n`;
}

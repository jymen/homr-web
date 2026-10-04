/**
 * An ONNX file's metadata_props, read from its bytes. onnxruntime-web has no
 * custom-metadata accessor, and PP-OCR's recognition model carries its
 * character list there, which RapidOCR reads through onnxruntime's
 * `get_modelmeta().custom_metadata_map`. Reading the verified bytes keeps the
 * list tied to the model it belongs to.
 *
 * Only ModelProto's top level is walked; the graph and everything else is
 * skipped by its length.
 */

import { ModelError } from "./errors.js";

const METADATA_PROPS = 14;
const ENTRY_KEY = 1;
const ENTRY_VALUE = 2;
const WIRE = { fixed32: 5, fixed64: 1, lengthDelimited: 2, varint: 0 } as const;

class ProtoReader {
  #at = 0;
  readonly #bytes: Uint8Array;
  readonly #end: number;

  constructor(bytes: Uint8Array, start = 0, end = bytes.length) {
    this.#bytes = bytes;
    this.#at = start;
    this.#end = end;
  }

  get done(): boolean {
    return this.#at >= this.#end;
  }

  varint(): number {
    let value = 0;
    let scale = 1;
    for (;;) {
      const byte = this.#bytes[this.#at];
      if (byte === undefined || this.#at >= this.#end) {
        throw new ModelError("manifest", "ONNX file ends inside a varint");
      }
      this.#at += 1;
      value += (byte % 128) * scale;
      if (byte < 0x80) {
        return value;
      }
      scale *= 128;
    }
  }

  /** The field number and a reader over its bytes, or null for a field that is not length-delimited, which is skipped. */
  next(): { readonly field: number; readonly body: ProtoReader | null } {
    const tag = this.varint();
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (wire === WIRE.varint) {
      this.varint();
      return { body: null, field };
    }
    if (wire === WIRE.fixed64 || wire === WIRE.fixed32) {
      this.#at += wire === WIRE.fixed64 ? 8 : 4;
      return { body: null, field };
    }
    if (wire !== WIRE.lengthDelimited) {
      throw new ModelError("manifest", `ONNX file has wire type ${wire}`);
    }
    const length = this.varint();
    const body = new ProtoReader(this.#bytes, this.#at, this.#at + length);
    this.#at += length;
    return { body, field };
  }

  text(): string {
    return new TextDecoder().decode(this.#bytes.subarray(this.#at, this.#end));
  }
}

export function readOnnxMetadata(
  bytes: Uint8Array
): ReadonlyMap<string, string> {
  const metadata = new Map<string, string>();
  const model = new ProtoReader(bytes);
  while (!model.done) {
    const { body, field } = model.next();
    if (field !== METADATA_PROPS || body === null) {
      continue;
    }
    let key = "";
    let value = "";
    while (!body.done) {
      const entry = body.next();
      if (entry.field === ENTRY_KEY && entry.body !== null) {
        key = entry.body.text();
      } else if (entry.field === ENTRY_VALUE && entry.body !== null) {
        value = entry.body.text();
      }
    }
    metadata.set(key, value);
  }
  return metadata;
}

/**
 * MusicXML as a comparable value: element names, attributes sorted by name,
 * trimmed text, children in document order. Whitespace between elements and
 * attribute order are formatting; everything else (a missing tie, a child
 * moved, a changed value) is a difference.
 *
 * The parser covers what homr and the port write: one declaration, elements,
 * attributes in double quotes, text with the five predefined entities. No
 * comments, CDATA or DOCTYPE, and it throws on anything else rather than
 * guessing.
 */

export interface CanonicalElement {
  readonly attributes: readonly (readonly [string, string])[];
  readonly children: readonly CanonicalElement[];
  readonly name: string;
  readonly text: string;
}

const ENTITIES: Readonly<Record<string, string>> = {
  amp: "&",
  apos: "'",
  gt: ">",
  lt: "<",
  quot: '"',
};

const decode = (text: string): string =>
  text.replace(
    /&(amp|apos|gt|lt|quot);/g,
    (_, name: string) => ENTITIES[name] ?? ""
  );

const DECLARATION = /^<\?xml[^?]*\?>/;
const TOKEN = /<[^>]*>|[^<]+/g;
const TAG = /^<(\/?)([A-Za-z][\w.-]*)((?:\s+[\w:.-]+="[^"]*")*)\s*(\/?)>$/;
const ATTRIBUTE = /([\w:.-]+)="([^"]*)"/g;

interface Building {
  attributes: [string, string][];
  children: CanonicalElement[];
  name: string;
  text: string;
}

/** Closes `done` into its parent; answers it when it is the root. */
function finish(
  done: Building,
  stack: readonly Building[]
): CanonicalElement | undefined {
  const element: CanonicalElement = {
    attributes: [...done.attributes].sort(([a], [b]) => (a < b ? -1 : 1)),
    children: done.children,
    name: done.name,
    text: done.text.trim(),
  };
  const parent = stack.at(-1);
  if (parent === undefined) {
    return element;
  }
  parent.children.push(element);
  return undefined;
}

function opened(tag: RegExpMatchArray): Building {
  const attributes: [string, string][] = [];
  for (const [, key = "", value = ""] of (tag[3] ?? "").matchAll(ATTRIBUTE)) {
    attributes.push([key, decode(value)]);
  }
  return { attributes, children: [], name: tag[2] ?? "", text: "" };
}

function addText(stack: readonly Building[], token: string): void {
  if (token.startsWith("<")) {
    throw new Error(`not a tag: ${token}`);
  }
  const top = stack.at(-1);
  if (top !== undefined) {
    top.text += decode(token);
  } else if (token.trim() !== "") {
    throw new Error(`text outside the root: ${token}`);
  }
}

function close(stack: Building[], name: string): CanonicalElement | undefined {
  const done = stack.pop();
  if (done === undefined || done.name !== name) {
    throw new Error(`</${name}> closes <${done?.name}>`);
  }
  return finish(done, stack);
}

export function canonicalXml(xml: string): CanonicalElement {
  const stack: Building[] = [];
  let root: CanonicalElement | undefined;
  for (const [token] of xml.replace(DECLARATION, "").matchAll(TOKEN)) {
    const tag = token.match(TAG);
    if (!tag) {
      addText(stack, token);
    } else if (tag[1] === "/") {
      root = close(stack, tag[2] ?? "") ?? root;
    } else if (tag[4] === "/") {
      root = finish(opened(tag), stack) ?? root;
    } else {
      stack.push(opened(tag));
    }
  }
  if (root === undefined || stack.length > 0) {
    throw new Error("not one closed root element");
  }
  return root;
}

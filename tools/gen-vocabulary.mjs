// Regenerates src/transformer/vocabulary.ts's token tables from
// test/golden/vocabulary.json, which tools/dump-golden.py writes from the
// installed homr. Run after a re-pin; the diff is the vocabulary change.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = JSON.parse(
  readFileSync(join(root, "test/golden/vocabulary.json"), "utf8")
);
const target = join(root, "src/transformer/vocabulary.ts");
const heads = ["rhythm", "pitch", "lift", "articulation", "slur", "position"];

const orderedTokens = (table) =>
  Object.entries(table)
    .sort((a, b) => a[1] - b[1])
    .map(([token, index], position) => {
      if (index !== position) {
        throw new Error(
          `gap in vocabulary at ${token}: index ${index}, position ${position}`
        );
      }
      return token;
    });

const literal = (tokens) => {
  const lines = [];
  for (let i = 0; i < tokens.length; i += 8) {
    lines.push(
      `  ${tokens
        .slice(i, i + 8)
        .map((t) => JSON.stringify(t))
        .join(", ")},`
    );
  }
  return lines.join("\n");
};

const tables = heads
  .map((head) => {
    const tokens = orderedTokens(source[head]);
    return `/** ${tokens.length} tokens, index = decoder output column. */\nexport const ${head.toUpperCase()}_TOKENS = [\n${literal(tokens)}\n] as const;\n`;
  })
  .join("\n");

const file = readFileSync(target, "utf8");
const begin = "// BEGIN GENERATED (tools/gen-vocabulary.mjs)\n";
const end = "// END GENERATED\n";
const start = file.indexOf(begin);
const stop = file.indexOf(end);
if (start < 0 || stop < 0) {
  throw new Error(`${target} lacks the generated-block markers`);
}
writeFileSync(
  target,
  `${file.slice(0, start + begin.length)}${tables}${file.slice(stop)}`
);
console.log(
  `wrote ${heads.map((h) => `${h}=${Object.keys(source[h]).length}`).join(" ")}`
);

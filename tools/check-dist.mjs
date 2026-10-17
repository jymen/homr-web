// Refuses to publish a dist/ a file-sync tool has touched: on 2026-10-17
// 0.5.0 went out with dist/index.js replaced by "index [conflicted 36].js".
import { existsSync, readdirSync } from "node:fs";
import { join } from "node:path";

const CONFLICT = /conflict/i;
const REQUIRED = ["dist/index.js", "dist/index.d.ts", "dist/models/dtype.js"];

const walk = (dir) =>
  readdirSync(dir, { withFileTypes: true }).flatMap((e) =>
    e.isDirectory() ? walk(join(dir, e.name)) : [join(dir, e.name)]
  );
const problems = [
  ...walk("dist")
    .filter((f) => CONFLICT.test(f))
    .map((f) => `stray: ${f}`),
  ...REQUIRED.filter((f) => !existsSync(f)).map((f) => `missing: ${f}`),
];
if (problems.length > 0) {
  console.error(problems.join("\n"));
  process.exit(1);
}

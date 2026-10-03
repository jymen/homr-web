/**
 * The static server the bench page needs, and nothing else.
 *
 * Three of its rules are not defaults. It sends COOP same-origin with COEP
 * require-corp on every response, because without cross-origin isolation there
 * is no SharedArrayBuffer, the wasm backend is pinned to one thread and the
 * page measures something other than the gate. It sends Content-Length on every
 * response, because ModelStore's httpFetch reads it to report download progress
 * and reports a total of 0 without it. And it sends no-store, so a rebuilt
 * dist/ is never served stale and the cache-hit proof measures the model
 * store's own cache rather than the browser's HTTP cache.
 *
 * It serves the repository root rather than bench/, because the page reaches
 * dist/, models/, node_modules/ and test/fixtures/ by absolute path.
 *
 * Node's own http, fs and path only: the bench is a measurement, and a
 * measurement harness that adds a dependency to the thing it measures is one
 * more variable.
 */

import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const PORT = 8099;
const HOST = "127.0.0.1";
const PAGE = "/bench/index.html";

const ROOT = resolve(fileURLToPath(new URL("..", import.meta.url)));

const MIME_TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".onnx": "application/octet-stream",
  ".png": "image/png",
  ".wasm": "application/wasm",
};

const BASE_HEADERS = {
  "Cache-Control": "no-store",
  "Cross-Origin-Embedder-Policy": "require-corp",
  "Cross-Origin-Opener-Policy": "same-origin",
};

const MODEL_URL = /^\/models\/[0-9a-f]{64}\/([^/]+)$/;

/**
 * manifest.ts gives every artifact a content-addressed urlPath,
 * `{sha256}/{filename}`, and the checkout under models/ is flat. The hash
 * directory exists in the URL space and nowhere on disk.
 */
function flattenModelUrl(pathname) {
  return pathname.replace(MODEL_URL, "/models/$1");
}

/** The file this URL names, or undefined when it names something outside the repository. */
function fileFor(pathname) {
  let decoded;
  try {
    decoded = decodeURIComponent(pathname);
  } catch {
    return;
  }
  const target = resolve(ROOT, `.${flattenModelUrl(decoded)}`);
  if (target !== ROOT && !target.startsWith(ROOT + sep)) {
    return;
  }
  return target;
}

function headersFor(pathname, size) {
  return {
    ...BASE_HEADERS,
    "Content-Length": String(size),
    "Content-Type": MIME_TYPES[extname(pathname)] ?? "application/octet-stream",
    ...(pathname.startsWith("/models/")
      ? { "Cross-Origin-Resource-Policy": "same-origin" }
      : {}),
  };
}

/**
 * The bench page's import map, applied by the server to dist/. A module
 * Worker gets no import map in Chrome, and dist/worker.js reaches the same
 * three packages the page does, so the bare specifiers are rewritten here
 * instead. A bundler does this for a real consumer.
 */
const BARE_SPECIFIERS = {
  "@techstark/opencv-js": "/bench/opencv-esm.js",
  delaunator: "/node_modules/delaunator/index.js",
  "onnxruntime-web": "/node_modules/onnxruntime-web/dist/ort.bundle.min.mjs",
  "robust-predicates": "/node_modules/robust-predicates/index.js",
};

const rewriteBareSpecifiers = (source) =>
  source.replace(
    /(from\s*|import\(\s*)"([^"./][^"]*)"/g,
    (whole, lead, name) =>
      name in BARE_SPECIFIERS ? `${lead}"${BARE_SPECIFIERS[name]}"` : whole
  );

function refuse(response, status, message) {
  const body = Buffer.from(`${message}\n`, "utf8");
  response.writeHead(status, {
    ...BASE_HEADERS,
    "Content-Length": String(body.length),
    "Content-Type": "text/plain; charset=utf-8",
  });
  response.end(body);
}

async function serve(request, response) {
  if (request.method !== "GET" && request.method !== "HEAD") {
    refuse(response, 405, `${request.method} is not served here`);
    return;
  }
  const url = new URL(request.url, `http://${HOST}:${PORT}`);
  const pathname = url.pathname === "/" ? PAGE : url.pathname;
  const file = fileFor(pathname);
  if (file === undefined) {
    refuse(response, 403, `${pathname} is outside the repository`);
    return;
  }
  let stats;
  try {
    stats = await stat(file);
  } catch {
    refuse(response, 404, `${pathname} is not here`);
    return;
  }
  if (!stats.isFile()) {
    refuse(response, 404, `${pathname} is not a file`);
    return;
  }
  if (pathname.startsWith("/dist/") && pathname.endsWith(".js")) {
    const body = Buffer.from(
      rewriteBareSpecifiers(await readFile(file, "utf8")),
      "utf8"
    );
    response.writeHead(200, headersFor(pathname, body.length));
    response.end(request.method === "HEAD" ? undefined : body);
    return;
  }
  const headers = headersFor(pathname, stats.size);
  if (request.method === "HEAD") {
    response.writeHead(200, headers);
    response.end();
    return;
  }
  response.writeHead(200, headers);
  const body = createReadStream(file);
  body.on("error", () => response.destroy());
  body.pipe(response);
}

createServer((request, response) => {
  serve(request, response).catch(() => {
    if (!response.headersSent) {
      refuse(response, 500, "the server failed to answer");
      return;
    }
    response.destroy();
  });
}).listen(PORT, HOST, () => {
  console.log(`serving ${ROOT}`);
  console.log(`bench page: http://${HOST}:${PORT}${PAGE}`);
});

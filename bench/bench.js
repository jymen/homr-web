/**
 * Phase 3's go/no-go numbers, taken in the browser they have to be decided in.
 *
 * The design doc's measurement section says this machine cannot be trusted with
 * a stopwatch under Node: the same page's tiles came out at 525, 1380, 1582 and
 * 1705 ms across four runs, with Backblaze and pCloud on the cores. So the three
 * things the decision log is owed are measured here instead. The WebGPU page
 * time against the wasm path, the batch-size A/B at 8, 16 and 32, and a proof
 * that a second run downloads no model bytes at all.
 *
 * Everything the page shows comes out of `runs`, one record per measured run.
 * A number that is not on a record is a number nobody can trace back to the run
 * that produced it.
 *
 * The chain stops at segmentPage. predictSymbols wants noise-filtered masks,
 * which is phase 5 and not ported, and feeding it golden PNGs would measure a
 * pipeline this library does not have.
 *
 * This is also the first exercise of the default opencv.js path: the Node tests
 * inject a createRequire source because the dynamic import hangs under vitest,
 * so `preprocessPage(image)` is called with no `cv` argument on purpose, and the
 * import map routes `@techstark/opencv-js` through bench/opencv-esm.js.
 *
 * Plain JavaScript against dist/, on the main thread. The bench needs no build
 * step of its own, and SegmentationWorker's transport is already proven over a
 * MessageChannel in test/segment.test.ts; what is missing is a timing, not a
 * transport.
 */

import {
  BACKENDS,
  browserCache,
  colorImageFromRgba,
  loadOpenCv,
  MASK_CLASS_NAMES,
  ModelError,
  ModelStore,
  planeAgreement,
  preprocessPage,
  segmentPage,
  startRuntime,
} from "../dist/index.js";

const BATCHES = [8, 16, 32];
const DEFAULT_BACKEND = "webgpu";
const FIXTURE_URL = "/test/fixtures/the-kesh-300dpi.png";
const MODELS_BASE_URL = "/models/";

/**
 * sha256 over the decoded ColorImage, measured on the Node golden path. Getting
 * BGR the wrong way round, or letting the canvas convert colour space, produces
 * a segmentation that is wrong and still looks like a segmentation, and no
 * other signal in a run would report it.
 */
const FIXTURE_DIGEST =
  "0c14d207bd8ea5c49b76bc1dd8f3da69618ad6c767c8e9b5733df136f084d5d0";

const ui = {
  backend: document.querySelectorAll('input[name="backend"]'),
  empty: document.querySelector("#empty"),
  head: document.querySelector("thead tr"),
  isolation: document.querySelector("#isolation"),
  log: document.querySelector("#log"),
  progress: document.querySelector("#progress"),
  run: document.querySelector("#run"),
  runs: document.querySelector("#runs"),
};

/** One record per measured run, and the only source the table reads. */
const runs = [];
let image;
let runtime;
/** The first run's class map and the batch that produced it. Batch size must not change what segnet decides, and an agreement below 1 is the only thing that would say otherwise. */
let reference;

function log(line) {
  ui.log.textContent += `${line}\n`;
  ui.log.scrollTop = ui.log.scrollHeight;
}

const ms = (value) => `${Math.round(value).toLocaleString()} ms`;
const count = (value) => value.toLocaleString();

function requestedBackend() {
  const asked = new URLSearchParams(window.location.search).get("backend");
  return BACKENDS.includes(asked) ? asked : DEFAULT_BACKEND;
}

function showIsolation() {
  const isolated = globalThis.crossOriginIsolated === true;
  ui.isolation.className = isolated ? "ok" : "bad";
  ui.isolation.textContent = isolated
    ? "crossOriginIsolated is true. SharedArrayBuffer is available, so the wasm backends can have threads."
    : "crossOriginIsolated is false. The COOP and COEP headers are not arriving, every wasm run below is pinned to one thread, and none of these numbers answer the gate.";
}

function wireBackendChooser() {
  const selected = requestedBackend();
  for (const input of ui.backend) {
    input.checked = input.value === selected;
    input.addEventListener("change", () => {
      window.location.search = `?backend=${input.value}`;
    });
  }
  return selected;
}

/**
 * A download event carries a cumulative `received` for one artifact, so the
 * bytes that crossed the network are the largest of those per artifact, added
 * up across artifacts. Adding every event instead would report a 57 MB model as
 * gigabytes. A cached event carries the artifact's whole size once.
 */
function byteCounts(events) {
  const peak = new Map();
  let cached = 0;
  for (const event of events) {
    if (event.kind === "download") {
      peak.set(
        event.artifact,
        Math.max(peak.get(event.artifact) ?? 0, event.received)
      );
    }
    if (event.kind === "cached") {
      cached += event.bytes;
    }
  }
  let downloaded = 0;
  for (const received of peak.values()) {
    downloaded += received;
  }
  return { cached, downloaded };
}

function kindsOf(events) {
  const counts = new Map();
  for (const event of events) {
    counts.set(event.kind, (counts.get(event.kind) ?? 0) + 1);
  }
  return [...counts].map(([kind, seen]) => `${kind}×${seen}`).join(" ");
}

/**
 * ModelStore verifies an artifact's digest on every load, cache hits included,
 * and phase 2 chose that on an 82 ms figure for 157 MB taken under Node. It
 * names the browser figure as one this page owes, so the verified events are
 * added up rather than only counted.
 */
function verifyMs(events) {
  let total = 0;
  for (const event of events) {
    if (event.kind === "verified") {
      total += event.ms;
    }
  }
  return total;
}

async function sha256Of(data) {
  const digest = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

async function decodeFixture() {
  const response = await fetch(FIXTURE_URL);
  if (!response.ok) {
    throw new Error(`${FIXTURE_URL} answered ${response.status}`);
  }
  const bitmap = await createImageBitmap(await response.blob(), {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const { height, width } = bitmap;
  const context = new OffscreenCanvas(width, height).getContext("2d", {
    colorSpace: "srgb",
  });
  if (context === null) {
    throw new Error("this browser gave no 2d context to decode the fixture in");
  }
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  // colorImageFromRgba does the RGBA to BGR swap itself, and it is the same
  // function the Node golden reader calls, which is what makes the two decodes
  // one code path.
  return colorImageFromRgba(
    width,
    height,
    context.getImageData(0, 0, width, height).data
  );
}

async function ensureImage() {
  if (image !== undefined) {
    return image;
  }
  const startedAt = performance.now();
  const decoded = await decodeFixture();
  const digest = await sha256Of(decoded.data);
  log(
    `fixture ${decoded.width}x${decoded.height} decoded in ${ms(performance.now() - startedAt)}`
  );
  if (digest === FIXTURE_DIGEST) {
    log(`decode confirmed against the golden path: ${digest}`);
  } else {
    log(
      "DECODE MISMATCH. These pixels are not the ones the golden path sees, so every number below is untrustworthy."
    );
    log(`  expected ${FIXTURE_DIGEST}`);
    log(`  actual   ${digest}`);
  }
  image = decoded;
  return image;
}

async function ensureRuntime() {
  if (runtime !== undefined) {
    return runtime;
  }
  runtime = await startRuntime({ maxBackend: requestedBackend() });
  log(
    `runtime: ${runtime.backend} on ${runtime.numThreads} thread(s), adapter ${runtime.probe.adapterInfo ?? "none"}`
  );
  return runtime;
}

async function measure(batch) {
  const events = [];
  const store = new ModelStore({
    baseUrl: MODELS_BASE_URL,
    cache: await browserCache(),
    onEvent: (event) => events.push(event),
    runtime,
  });
  const startedAt = performance.now();
  try {
    // The same batch goes to open and to segmentPage, because
    // freeDimensionOverrides pins the dimension on the session.
    const session = await store.open("segnet", { batch });
    const openedAt = performance.now();
    const { preprocessed } = await preprocessPage(image);
    const preprocessedAt = performance.now();
    const result = await segmentPage(session, preprocessed, {
      batch,
      onProgress: (done, total) => {
        ui.progress.textContent = `batch ${batch}: tile ${done} of ${total}`;
      },
    });
    const finishedAt = performance.now();
    const plan = store.plan("segnet");
    const bytes = byteCounts(events);
    return {
      result,
      run: {
        backend: runtime.backend,
        batch,
        cachedBytes: bytes.cached,
        cacheState: bytes.downloaded === 0 ? "warm" : "cold",
        crossOriginIsolated: runtime.probe.crossOriginIsolated,
        downloadedBytes: bytes.downloaded,
        events,
        modelOpenMs: openedAt - startedAt,
        modelReason: plan.reason,
        numThreads: runtime.numThreads,
        preprocessMs: preprocessedAt - openedAt,
        provider: plan.provider,
        runtimeReason: runtime.reason,
        segmentMs: finishedAt - preprocessedAt,
        totalMs: finishedAt - startedAt,
      },
    };
  } finally {
    await store.close();
  }
}

function cell(row, text, className) {
  const td = row.insertCell();
  td.textContent = text;
  if (className !== undefined) {
    td.className = className;
  }
}

function renderRuns() {
  ui.empty.hidden = runs.length > 0;
  ui.runs.replaceChildren();
  for (const run of runs) {
    const row = ui.runs.insertRow();
    row.className = "run";
    cell(row, String(run.batch));
    cell(row, run.backend);
    cell(row, run.provider);
    cell(row, String(run.numThreads));
    cell(row, String(run.crossOriginIsolated));
    cell(row, run.cacheState, run.cacheState);
    cell(row, ms(run.modelOpenMs));
    cell(row, ms(run.preprocessMs));
    cell(row, ms(run.segmentMs));
    cell(row, ms(run.totalMs));
    cell(row, count(run.downloadedBytes));
    cell(row, count(run.cachedBytes));
    cell(row, kindsOf(run.events));

    const why = ui.runs.insertRow().insertCell();
    why.className = "why";
    why.colSpan = ui.head.cells.length;
    why.textContent = `${run.runtimeReason} · ${run.modelReason}`;
  }
}

function logMasks(result) {
  const parts = [];
  for (const name of MASK_CLASS_NAMES) {
    let set = 0;
    for (const byte of result.masks[name].data) {
      if (byte !== 0) {
        set += 1;
      }
    }
    parts.push(`${name} ${count(set)}`);
  }
  log(`  masks: ${parts.join(", ")}`);
}

function logAgreement(batch, classes) {
  if (reference === undefined) {
    reference = { batch, classes };
    log(`  class map: the reference, from batch ${batch}`);
    return;
  }
  const agreement = planeAgreement(reference.classes, classes);
  const verdict = agreement === 1 ? "identical" : "DIVERGED";
  log(
    `  class map against batch ${reference.batch}: ${agreement.toFixed(9)} ${verdict}`
  );
}

async function runAll() {
  ui.run.disabled = true;
  try {
    await ensureImage();
    await ensureRuntime();
    // loadOpenCv memoises, so without this the first run's preprocess would
    // carry the 10.9 MB bundle and its wasm init while the other two do not,
    // and the batch A/B would read as a preprocess difference. preprocessPage
    // still gets no cv argument: this is the same default path, warmed.
    const openCvAt = performance.now();
    await loadOpenCv();
    log(`opencv.js ready in ${ms(performance.now() - openCvAt)}`);

    for (const batch of BATCHES) {
      log(`batch ${batch}: running`);
      const { result, run } = await measure(batch);
      runs.push(run);
      renderRuns();
      log(
        `  ${run.cacheState}, open ${ms(run.modelOpenMs)}, preprocess ${ms(run.preprocessMs)}, segment ${ms(run.segmentMs)}, total ${ms(run.totalMs)}`
      );
      log(
        `  downloaded ${count(run.downloadedBytes)} bytes, cached ${count(run.cachedBytes)} bytes, verified in ${ms(verifyMs(run.events))}, events ${kindsOf(run.events)}`
      );
      logMasks(result);
      logAgreement(batch, result.classes);
    }
    log("done");
  } catch (error) {
    if (error instanceof ModelError) {
      log(`ModelError ${error.code}: ${error.message}`);
    } else {
      log(`failed: ${error instanceof Error ? error.message : String(error)}`);
    }
  } finally {
    ui.progress.textContent = "";
    ui.run.disabled = false;
  }
}

showIsolation();
wireBackendChooser();
renderRuns();
ui.run.addEventListener("click", () => {
  runAll().catch((error) => {
    log(`failed: ${String(error)}`);
  });
});

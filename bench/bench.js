/**
 * Phase 3's go/no-go numbers, taken in the browser they have to be decided in,
 * and the detection stage's runtime check: the staffs it finds, drawn over the
 * page they were found on.
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
 * The chain runs from the fixture PNG through preprocessPage and segmentPage to
 * detectStaffsInImage, on this library's own masks and never on golden PNGs,
 * so what the overlay shows is what a musician's page would get. Every staff
 * line drawn comes from `staff.grid`: if the lines sit on the printed ones,
 * the whole chain agrees with the page.
 *
 * After the last run, staffCanvases turns that run's multi staffs into the
 * encoder canvases, and each is drawn above homr's canvas-<n>.png with their
 * mean absolute difference. Detection here ran on the browser's own masks, so
 * a difference includes anything detection did differently.
 *
 * Under each pair, the transformer runs twice: on homr's canvas, which is the
 * golden test's input and isolates the encoder's backend, and on this port's
 * canvas, which is what a musician gets. Both token lists are compared with
 * tokens-<n>.json on the six heads, with the per-staff and per-step time.
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
  createInputPredictions,
  createStaffCanvas,
  DetectionError,
  detectStaffsInImage,
  ensureSameNumberOfStaffs,
  loadOpenCv,
  MASK_CLASS_NAMES,
  ModelError,
  ModelStore,
  parseStaffCanvas,
  planeAgreement,
  planeFromBytes,
  preprocessPage,
  rgbaFromPlane,
  segmentPage,
  staffCanvases,
  staffRegion,
  staffRegions,
  startRuntime,
} from "../dist/index.js";

const BATCHES = [8, 16, 32];
const DEFAULT_BACKEND = "webgpu";
const DEFAULT_FIXTURE = "the-kesh-300dpi";
const MODELS_BASE_URL = "/models/";

/**
 * Each public fixture with the sha256 of its decoded ColorImage, measured on
 * the Node golden path. Getting BGR the wrong way round, or letting the canvas
 * convert colour space, produces a segmentation that is wrong and still looks
 * like a segmentation, and no other signal in a run would report it.
 */
const FIXTURE_DIGESTS = {
  "grand-staff-300dpi":
    "1cbd050f2d38e700954680c9a37d8156d58e399e5a983e4266c3e50bb51bf8e3",
  "the-kesh-300dpi":
    "0c14d207bd8ea5c49b76bc1dd8f3da69618ad6c767c8e9b5733df136f084d5d0",
};

/** What the overlay draws in, chosen to stay apart from black print on a gray page. */
const OVERLAY = {
  connection: "#ffb000",
  grandStaff: "#e0218a",
  line: "#00a2ff",
  multiStaff: "#19c37d",
};

const ui = {
  backend: document.querySelectorAll('input[name="backend"]'),
  canvases: document.querySelector("#canvases"),
  detection: document.querySelector("#detection"),
  empty: document.querySelector("#empty"),
  fixture: document.querySelectorAll('input[name="fixture"]'),
  head: document.querySelector("thead tr"),
  isolation: document.querySelector("#isolation"),
  log: document.querySelector("#log"),
  overlay: document.querySelector("#overlay"),
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

function requestedFixture() {
  const asked = new URLSearchParams(window.location.search).get("fixture");
  return Object.hasOwn(FIXTURE_DIGESTS, asked) ? asked : DEFAULT_FIXTURE;
}

/** Both choosers reload the page with the other's choice kept. */
function reloadWith(name, value) {
  const query = new URLSearchParams(window.location.search);
  query.set(name, value);
  window.location.search = `?${query}`;
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
      reloadWith("backend", input.value);
    });
  }
  return selected;
}

function wireFixtureChooser() {
  const selected = requestedFixture();
  for (const input of ui.fixture) {
    input.checked = input.value === selected;
    input.addEventListener("change", () => {
      reloadWith("fixture", input.value);
    });
  }
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
  const url = `/test/fixtures/${requestedFixture()}.png`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} answered ${response.status}`);
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
  const expected = FIXTURE_DIGESTS[requestedFixture()];
  log(`fixture ${requestedFixture()}`);
  if (digest === expected) {
    log(`decode confirmed against the golden path: ${digest}`);
  } else {
    log(
      "DECODE MISMATCH. These pixels are not the ones the golden path sees, so every number below is untrustworthy."
    );
    log(`  expected ${expected}`);
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

/**
 * The detection stage on this run's own masks. A page homr cannot read is an
 * outcome and is returned as one; anything else thrown is a defect and
 * propagates.
 */
function detect(cv, page, result) {
  const startedAt = performance.now();
  try {
    const detection = detectStaffsInImage(
      cv,
      createInputPredictions(page.resized, page.preprocessed, result.masks)
    );
    return { detection, detectMs: performance.now() - startedAt };
  } catch (error) {
    if (error instanceof DetectionError) {
      return { detectMs: performance.now() - startedAt, failure: error };
    }
    throw error;
  }
}

async function measure(batch, cv) {
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
    const page = await preprocessPage(image);
    const preprocessedAt = performance.now();
    const result = await segmentPage(session, page.preprocessed, {
      batch,
      onProgress: (done, total) => {
        ui.progress.textContent = `batch ${batch}: tile ${done} of ${total}`;
      },
    });
    const finishedAt = performance.now();
    const detected = detect(cv, page, result);
    const plan = store.plan("segnet");
    const bytes = byteCounts(events);
    return {
      detected,
      page,
      result,
      run: {
        backend: runtime.backend,
        batch,
        cachedBytes: bytes.cached,
        cacheState: bytes.downloaded === 0 ? "warm" : "cold",
        crossOriginIsolated: runtime.probe.crossOriginIsolated,
        detectMs: detected.detectMs,
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
    cell(row, ms(run.detectMs));
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

function strokePath(context, color, width, points) {
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  for (const [i, [x, y]] of points.entries()) {
    if (i === 0) {
      context.moveTo(x, y);
    } else {
      context.lineTo(x, y);
    }
  }
  context.stroke();
}

/** A square bracket left of the staffs, `inset` px from their left edge, from the top line of the first to the bottom line of the last. */
function drawBracket(context, color, staffs, inset) {
  const x = Math.min(...staffs.map((staff) => staff.minX)) - inset;
  const top = Math.min(...staffs.map((staff) => staff.minY));
  const bottom = Math.max(...staffs.map((staff) => staff.maxY));
  strokePath(context, color, 5, [
    [x + 14, top],
    [x, top],
    [x, bottom],
    [x + 14, bottom],
  ]);
}

/**
 * One polyline per staff line through the grid points, a bracket per multi
 * staff, a second one per grand staff, and the outline of every connection.
 * The grid is drawn in x order, which it is not stored in.
 */
function drawDetection(detection) {
  const { height, width } = detection.preprocessed;
  ui.overlay.width = width;
  ui.overlay.height = height;
  const context = ui.overlay.getContext("2d");
  context.putImageData(
    new ImageData(rgbaFromPlane(detection.preprocessed), width, height),
    0,
    0
  );
  for (const multiStaff of detection.multiStaffs) {
    for (const staff of multiStaff.staffs) {
      const grid = [...staff.grid].sort((a, b) => a.x - b.x);
      for (let line = 0; line < grid[0].y.length; line += 1) {
        strokePath(
          context,
          OVERLAY.line,
          2,
          grid.map((point) => [point.x, point.y[line]])
        );
      }
      if (staff.isGrandstaff) {
        drawBracket(context, OVERLAY.grandStaff, [staff], 70);
      }
    }
    drawBracket(context, OVERLAY.multiStaff, multiStaff.staffs, 90);
    for (const connection of multiStaff.connections) {
      const corners = [];
      for (let i = 0; i < connection.polygon.length; i += 2) {
        corners.push([connection.polygon[i], connection.polygon[i + 1]]);
      }
      strokePath(context, OVERLAY.connection, 3, [...corners, corners[0]]);
    }
  }
}

function showDetection(detected) {
  if (detected.failure !== undefined) {
    ui.detection.textContent = `no detection: ${detected.failure.code}, "${detected.failure.message}"`;
    return;
  }
  const { detection } = detected;
  const staffs = detection.multiStaffs.flatMap((multi) => multi.staffs);
  const summary = `${detection.multiStaffs.length} multi staffs holding ${staffs.length} staffs, ${staffs.filter((staff) => staff.isGrandstaff).length} of them grand staffs, ${detection.multiStaffs.reduce((n, multi) => n + multi.connections.length, 0)} connections, ${detection.notes.length} notes, noise ${detection.noise.kind}`;
  ui.detection.textContent = summary;
  log(`  detection: ${summary}`);
  drawDetection(detection);
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

/** A grayscale PNG's samples, decoded without colour conversion; channel 0 is the gray. */
async function fetchGray(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${url} answered ${response.status}`);
  }
  const bitmap = await createImageBitmap(await response.blob(), {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const { height, width } = bitmap;
  const context = new OffscreenCanvas(width, height).getContext("2d");
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = context.getImageData(0, 0, width, height).data;
  const data = new Uint8Array(width * height);
  for (let i = 0; i < data.length; i += 1) {
    data[i] = rgba[i * 4];
  }
  return { data, height, width };
}

function drawGray(gray) {
  const element = document.createElement("canvas");
  element.width = gray.width;
  element.height = gray.height;
  const rgba = new Uint8ClampedArray(gray.width * gray.height * 4);
  gray.data.forEach((value, i) => {
    rgba.set([value, value, value, 255], i * 4);
  });
  element.getContext("2d").putImageData(new ImageData(rgba, gray.width), 0, 0);
  return element;
}

function meanAbsoluteDifference(a, b) {
  if (a.width !== b.width || a.height !== b.height) {
    return Number.POSITIVE_INFINITY;
  }
  let total = 0;
  for (let i = 0; i < a.data.length; i += 1) {
    total += Math.abs(a.data[i] - b.data[i]);
  }
  return total / a.data.length;
}

const HEADS = ["rhythm", "pitch", "lift", "articulation", "slur", "position"];
const symbolText = (s) => HEADS.map((h) => s[h]).join(" ");

/** parseStaffCanvas with its timing, and where its heads first leave homr's. */
async function transcribe(sessions, canvas, expected) {
  let steps = 0;
  let firstStepAt = 0;
  const startedAt = performance.now();
  const tokens = await parseStaffCanvas(sessions, canvas, {
    onStep: (step) => {
      steps = step;
      if (step === 1) {
        firstStepAt = performance.now();
      }
    },
  });
  const endedAt = performance.now();
  const length = Math.max(tokens.length, expected.length);
  let firstDifference = -1;
  for (let i = 0; i < length && firstDifference < 0; i += 1) {
    const got = tokens[i];
    const want = expected[i];
    if (
      got === undefined ||
      want === undefined ||
      symbolText(got) !== symbolText(want)
    ) {
      firstDifference = i;
    }
  }
  return {
    encodeMs: firstStepAt - startedAt,
    firstDifference,
    perStepMs: (endedAt - firstStepAt) / Math.max(steps - 1, 1),
    staffMs: endedAt - startedAt,
    steps,
    tokens,
  };
}

function tokenLine(label, run, expected) {
  const verdict =
    run.firstDifference < 0
      ? `equal to tokens-<n>.json (${expected.length})`
      : `DIFFERS from token ${run.firstDifference}: ${run.tokens[run.firstDifference] ? symbolText(run.tokens[run.firstDifference]) : "end"} where homr has ${expected[run.firstDifference] ? symbolText(expected[run.firstDifference]) : "end"}`;
  return `${label}: ${run.tokens.length} tokens, ${run.steps} steps, staff ${ms(run.staffMs)} (encoder and step 0 ${ms(run.encodeMs)}, then ${run.perStepMs.toFixed(1)} ms/step), ${verdict}`;
}

async function showCanvases(cv, detected, page) {
  ui.canvases.replaceChildren();
  if (detected.failure !== undefined) {
    return;
  }
  const store = new ModelStore({
    baseUrl: MODELS_BASE_URL,
    cache: await browserCache(),
    runtime,
  });
  try {
    const sessions = {
      decoder: await store.open("decoder"),
      encoder: await store.open("encoder"),
    };
    log(`  encoder: ${store.plan("encoder").reason}`);
    log(`  decoder: ${store.plan("decoder").reason}`);
    await drawCanvases(cv, detected, page, sessions);
  } finally {
    await store.close();
  }
}

async function drawCanvases(cv, detected, page, sessions) {
  const startedAt = performance.now();
  const canvases = staffCanvases(
    cv,
    detected.detection.multiStaffs,
    page.preprocessed
  );
  log(
    `  staff canvases: ${canvases.length} in ${ms(performance.now() - startedAt)}`
  );
  const systems = ensureSameNumberOfStaffs(
    detected.detection.multiStaffs,
    page.preprocessed.height
  );
  const regions = staffRegions(systems);
  // staffCanvases' order: every system's first staff, then every second staff.
  const staffs = (systems[0]?.staffs ?? []).flatMap((_, voice) =>
    systems.map((system) => system.staffs[voice])
  );
  for (const [n, canvas] of canvases.entries()) {
    const base = `/test/golden/${requestedFixture()}`;
    const golden = await fetchGray(`${base}/canvas-${n}.png`);
    const dewarp = await (await fetch(`${base}/dewarp-${n}.json`)).json();
    const difference = meanAbsoluteDifference(canvas.image, golden);
    log(
      `  canvas ${n}: mean absolute difference from homr's ${difference.toFixed(4)}, region ${staffRegion(staffs[n], regions).join(",")} where homr's is ${dewarp.region.join(",")}`
    );
    const expected = await (await fetch(`${base}/tokens-${n}.json`)).json();
    const onHomr = await transcribe(
      sessions,
      createStaffCanvas(
        planeFromBytes("gray", golden.width, golden.height, golden.data),
        canvas.staff
      ),
      expected
    );
    const onPort = await transcribe(sessions, canvas, expected);
    const lines = [
      tokenLine("homr's canvas", onHomr, expected),
      tokenLine("this port's canvas", onPort, expected),
    ];
    for (const line of lines) {
      log(`  ${line}`);
    }
    const figure = document.createElement("figure");
    const caption = document.createElement("figcaption");
    caption.textContent = `canvas ${n}: this port above, homr below, mean absolute difference ${difference.toFixed(4)}`;
    const tokens = document.createElement("pre");
    tokens.style.whiteSpace = "pre-wrap";
    tokens.textContent = [
      ...lines,
      onPort.tokens.map(symbolText).join(" | "),
    ].join("\n");
    figure.append(caption, drawGray(canvas.image), drawGray(golden), tokens);
    ui.canvases.append(figure);
  }
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
    const cv = await loadOpenCv();
    log(`opencv.js ready in ${ms(performance.now() - openCvAt)}`);

    let last;
    for (const batch of BATCHES) {
      log(`batch ${batch}: running`);
      const { detected, page, result, run } = await measure(batch, cv);
      last = { detected, page };
      runs.push(run);
      renderRuns();
      log(
        `  ${run.cacheState}, open ${ms(run.modelOpenMs)}, preprocess ${ms(run.preprocessMs)}, segment ${ms(run.segmentMs)}, detect ${ms(run.detectMs)}, total ${ms(run.totalMs)}`
      );
      log(
        `  downloaded ${count(run.downloadedBytes)} bytes, cached ${count(run.cachedBytes)} bytes, verified in ${ms(verifyMs(run.events))}, events ${kindsOf(run.events)}`
      );
      logMasks(result);
      logAgreement(batch, result.classes);
      showDetection(detected);
    }
    await showCanvases(cv, last.detected, last.page);
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
wireFixtureChooser();
renderRuns();
ui.run.addEventListener("click", () => {
  runAll().catch((error) => {
    log(`failed: ${String(error)}`);
  });
});

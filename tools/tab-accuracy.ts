// Re-measures the tab reader (src/tab/read.ts) on the owner's tablature
// samples in test/fixtures/local/tablature/, which may not be redistributed:
// local only, never part of `npm test`. Prints per page the systems, events,
// techniques and unread marks, the reader's time, and three scores:
//   - against the hand transcription of one system per page (exact events, LCS),
//   - Cripple Creek tab-only against Cripple Creek with score (the same tab
//     printed twice: events identical between the two readings, LCS),
//   - against homr's reading of the staff above the tab (pitch sets aligned
//     by Needleman-Wunsch, best octave shift), on the pages with staves.
// The prototype's figures (.scratch/tab-sketch/REPORT.md) are 15/15, 35/35,
// 33/33, 161/169 and 115/115.
//
// Needs models/ (tools/fetch-models.sh), the rasterised pages in
// test/fixtures/local/tablature/.pages/ (test/tab-local.test.ts makes them)
// and test/fixtures/local/tablature/tab-truth.json (git-ignored, beside the pages):
//   [{ "png": "<file in .pages>", "key": "<name>", "tuning": ["D4", ...],
//      "capo": 0, "staves": 0, "truth": { "system": 0, "events": "4/2 3/0 1/0+2/0" } }]
//
// With --text it scores the tuning and capo read off the page text instead
// (src/tab/text.ts), on the pages of test/fixtures/local/tablature/tab-text-truth.json
// (git-ignored too), each found, correct, wrong or missing against the text the
// page prints; an entry with a "pdf" is rasterised into .pages/ first:
//   [{ "png": "<file in .pages>", "name": "<title>", "tuning": ["D4", ...] | null,
//      "capo": 2 | null, "pdf": "<optional source PDF>" }]
//
// usage: npx vite-node tools/tab-accuracy.ts [--no-homr | --text]
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { PNG } from "pngjs";
import { type AcquiredOpenCv, loadOpenCv } from "../src/cv/opencv.js";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { startRuntime } from "../src/models/backend.js";
import { memoryCache } from "../src/models/cache.js";
import { ModelStore } from "../src/models/store.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { RapidOcr, recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { recognizePage } from "../src/pipeline/recognize.js";
import { detectTablature, type ReadCrops } from "../src/tab/detect.js";
import { pitchTab } from "../src/tab/pitch.js";
import { readTablature, type TabReading } from "../src/tab/read.js";
import { readTabText } from "../src/tab/text.js";
import { midiOfPitch } from "../src/tab/tuning.js";

const ROOT = join(
  import.meta.dirname,
  "..",
  "test",
  "fixtures",
  "local",
  "tablature"
);

interface Entry {
  readonly capo: number;
  readonly key: string;
  readonly png: string;
  readonly staves: number;
  readonly truth?: { readonly events: string; readonly system: number };
  readonly tuning: readonly string[];
}

const entries: readonly Entry[] = process.argv.includes("--text")
  ? []
  : JSON.parse(readFileSync(join(ROOT, "tab-truth.json"), "utf8"));
const withHomr = !process.argv.includes("--no-homr");

function pageOf(png: string): ColorImage {
  const decoded = PNG.sync.read(readFileSync(join(ROOT, ".pages", png)));
  return colorImageFromRgba(decoded.width, decoded.height, decoded.data);
}

const eventKey = (notes: readonly { string: number; fret: number }[]) =>
  notes
    .map((n) => `${n.string}/${n.fret}`)
    .sort()
    .join("+");

function lcs(a: readonly string[], b: readonly string[]): number {
  const table = Array.from(
    { length: a.length + 1 },
    () => new Int32Array(b.length + 1)
  );
  for (let i = 1; i <= a.length; i += 1) {
    for (let j = 1; j <= b.length; j += 1) {
      table[i][j] =
        a[i - 1] === b[j - 1]
          ? table[i - 1][j - 1] + 1
          : Math.max(table[i - 1][j], table[i][j - 1]);
    }
  }
  return table[a.length][b.length];
}

const STEP: Record<string, number> = {
  A: 9,
  B: 11,
  C: 0,
  D: 2,
  E: 4,
  F: 5,
  G: 7,
};

const STEP_TAG = /<step>([A-G])<\/step>/;
const ALTER_TAG = /<alter>(-?\d+)<\/alter>/;
const OCTAVE_TAG = /<octave>(-?\d+)<\/octave>/;

/** homr's MusicXML as pitch-set events per system, split at each new system; grace notes and a second staff skipped. */
function staffEvents(xml: string): number[][][] {
  const systems: number[][][] = [[]];
  for (const token of xml.match(
    /<print new-system="yes"\s*\/?>|<note>[\s\S]*?<\/note>/g
  ) ?? []) {
    if (token.startsWith("<print")) {
      systems.push([]);
      continue;
    }
    const step = token.match(STEP_TAG);
    if (
      !step ||
      token.includes("<grace") ||
      token.includes("<staff>2</staff>")
    ) {
      continue;
    }
    const alter = Number(token.match(ALTER_TAG)?.[1] ?? 0);
    const octave = Number(token.match(OCTAVE_TAG)?.[1]);
    const midi = 12 * (octave + 1) + STEP[step[1]] + alter;
    const current = systems.at(-1) as number[][];
    if (token.includes("<chord/>") && current.length > 0) {
      (current.at(-1) as number[]).push(midi);
    } else {
      current.push([midi]);
    }
  }
  return systems;
}

const sameSet = (a: readonly number[], b: readonly number[]) => {
  const left = [...new Set(a)].sort((m, n) => m - n);
  const right = [...new Set(b)].sort((m, n) => m - n);
  return left.length === right.length && left.every((v, k) => v === right[k]);
};

/** Needleman-Wunsch as the prototype's compare.ts: 2 for equal sets, 1 for a shared note, -1 otherwise, gaps -0.5. */
function align(
  tab: number[][],
  staff: number[][],
  shift: number
): [number, number][] {
  const score = (p: number, q: number) => {
    const a = tab[p].map((v) => v + shift);
    if (sameSet(a, staff[q])) {
      return 2;
    }
    return a.some((v) => staff[q].includes(v)) ? 1 : -1;
  };
  const d = Array.from(
    { length: tab.length + 1 },
    () => new Float64Array(staff.length + 1)
  );
  for (let i = 1; i <= tab.length; i += 1) {
    d[i][0] = -0.5 * i;
  }
  for (let j = 1; j <= staff.length; j += 1) {
    d[0][j] = -0.5 * j;
  }
  for (let i = 1; i <= tab.length; i += 1) {
    for (let j = 1; j <= staff.length; j += 1) {
      d[i][j] = Math.max(
        d[i - 1][j - 1] + score(i - 1, j - 1),
        d[i - 1][j] - 0.5,
        d[i][j - 1] - 0.5
      );
    }
  }
  const pairs: [number, number][] = [];
  let i = tab.length;
  let j = staff.length;
  while (i > 0 && j > 0) {
    if (d[i][j] === d[i - 1][j - 1] + score(i - 1, j - 1)) {
      pairs.push([i - 1, j - 1]);
      i -= 1;
      j -= 1;
    } else if (d[i][j] === d[i - 1][j] - 0.5) {
      i -= 1;
    } else {
      j -= 1;
    }
  }
  return pairs.reverse();
}

function staffScore(
  readings: readonly TabReading[],
  entry: Entry,
  xml: string
) {
  const tuning = { capo: entry.capo, strings: entry.tuning.map(midiOfPitch) };
  const tabs = readings.map((r) =>
    pitchTab(r, tuning).map((e) => e.notes.map((n) => n.midi))
  );
  const staves = staffEvents(xml);
  let best = { exact: 0, n: 0, overlap: 0, shift: 0 };
  for (const shift of [-24, -12, 0, 12, 24]) {
    let exact = 0;
    let overlap = 0;
    let n = 0;
    for (const [k, tab] of tabs.entries()) {
      const staff = staves[k] ?? [];
      n += tab.length;
      for (const [i, j] of align(tab, staff, shift)) {
        const a = tab[i].map((v) => v + shift);
        exact += sameSet(a, staff[j]) ? 1 : 0;
        overlap += a.some((v) => staff[j].includes(v)) ? 1 : 0;
      }
    }
    if (
      exact > best.exact ||
      (exact === best.exact && overlap > best.overlap)
    ) {
      best = { exact, n, overlap, shift };
    }
  }
  return best;
}

const MODELS = join(import.meta.dirname, "..", "models");
const requireCjs = createRequire(import.meta.url);
const cv = await loadOpenCv(() =>
  Promise.resolve({
    module: requireCjs("@techstark/opencv-js"),
  } as AcquiredOpenCv)
);
const store = new ModelStore({
  baseUrl: "file:///models/",
  cache: memoryCache(),
  fetchBytes: ({ url }) =>
    Promise.resolve(
      new Uint8Array(
        readFileSync(join(MODELS, url.slice(url.lastIndexOf("/") + 1)))
      )
    ),
  placement: { artifactsFor: "wasm", provider: "wasm" },
  runtime: await startRuntime({ maxBackend: "wasm" }),
});
const opened = performance.now();
const session = await store.open("ocrRecognize");
const characters = ctcCharacters(session.metadata.get("character") ?? "");
console.log(
  `recogniser opened in ${Math.round(performance.now() - opened)} ms`
);
const read: ReadCrops = async (crops) =>
  (await recognizeCrops(cv, session, characters, crops)).map((r) => r.text);

if (process.argv.includes("--text")) {
  await scoreText();
  await store.close();
  process.exit(0);
}

const readingsByKey = new Map<string, TabReading[]>();
for (const entry of entries) {
  const page = pageOf(entry.png);
  const started = performance.now();
  const tabs = await detectTablature(cv, page, read);
  const detected = performance.now();
  const readings = await readTablature(cv, page, tabs, read);
  const done = performance.now();
  readingsByKey.set(entry.key, readings);
  const notes = readings.reduce(
    (sum, r) => sum + r.events.reduce((s, e) => s + e.notes.length, 0),
    0
  );
  const techniques = readings.flatMap((r) =>
    r.annotations.map((a) => a.technique)
  );
  console.log(
    [
      entry.key,
      `tabs ${readings.map((r) => r.lines).join(",")}`,
      `events ${readings.reduce((sum, r) => sum + r.events.length, 0)} (${readings.map((r) => r.events.length).join(" ")})`,
      `notes ${notes}`,
      `techniques ${techniques.length === 0 ? "-" : [...new Set(techniques)].map((t) => `${t}x${techniques.filter((u) => u === t).length}`).join(" ")}`,
      `unread ${readings.reduce((sum, r) => sum + r.unread, 0)}`,
      `detect ${Math.round(detected - started)} ms, read ${Math.round(done - detected)} ms`,
    ].join(" | ")
  );
  if (entry.truth !== undefined) {
    const want = entry.truth.events.split(" ").map((e) =>
      eventKey(
        e.split("+").map((n) => ({
          fret: Number(n.split("/")[1]),
          string: Number(n.split("/")[0]),
        }))
      )
    );
    const got = (readings[entry.truth.system]?.events ?? []).map((e) =>
      eventKey(e.notes)
    );
    console.log(
      `  hand truth, system ${entry.truth.system + 1}: ${lcs(got, want)}/${want.length} exact (read ${got.length})`
    );
  }
}

const tabOnly = readingsByKey.get(
  "banjo__Cripple_Creek_Beginner_Banjo_Tab_tef-p1"
);
const withScore = readingsByKey.get(
  "banjo__Cripple_Creek_Beginner_Banjo_Tab_with_Score_tef-p1"
);
if (tabOnly && withScore) {
  const flat = (rs: readonly TabReading[]) =>
    rs.flatMap((r) => r.events.map((e) => eventKey(e.notes)));
  console.log(
    `Cripple Creek consistency: ${lcs(flat(tabOnly), flat(withScore))}/${flat(withScore).length} events identical`
  );
}

if (withHomr) {
  for (const entry of entries.filter((e) => e.staves > 0)) {
    const result = await recognizePage(
      pageOf(entry.png),
      {
        backend: "wasm",
        cv,
        open: (role, batch) =>
          store.open(role, batch === undefined ? {} : { batch }),
      },
      { ocr: false }
    );
    if (!result.ok) {
      console.log(`${entry.key}: homr ${result.error}`);
      continue;
    }
    const best = staffScore(result.tablature, entry, result.musicXml);
    console.log(
      `${entry.key} vs homr's staff: exact ${best.exact}/${best.n}, sharing a note ${best.overlap}/${best.n}, octave shift ${best.shift}`
    );
  }
}
await store.close();

interface TextEntry {
  readonly capo: number | null;
  readonly name: string;
  readonly pdf?: string;
  readonly png: string;
  readonly tuning: readonly string[] | null;
}

type Verdict = "correct" | "missing" | "none" | "wrong";

/** found and right, nothing printed and nothing found, printed and not found, or found and not what is printed. */
function verdict<T>(
  want: T | null,
  got: T | undefined,
  same: (a: T, b: T) => boolean
): Verdict {
  if (want === null) {
    return got === undefined ? "none" : "wrong";
  }
  if (got === undefined) {
    return "missing";
  }
  return same(want, got) ? "correct" : "wrong";
}

type TabTexts = Awaited<ReturnType<typeof readTabText>>;

/** The systems' most common reading: the guard can take a staff for a tab, a false system with no text beside it. */
function commonText(texts: TabTexts): TabTexts[number] {
  const keyOf = (t: TabTexts[number]) =>
    JSON.stringify([
      t.tuning?.status === "read" ? t.tuning.strings : null,
      t.capo?.fret ?? null,
    ]);
  const counts = new Map<string, number>();
  for (const t of texts) {
    counts.set(keyOf(t), (counts.get(keyOf(t)) ?? 0) + 1);
  }
  return (
    [...texts].sort(
      (a, b) => (counts.get(keyOf(b)) ?? 0) - (counts.get(keyOf(a)) ?? 0)
    )[0] ?? {}
  );
}

function describeTuning(tuning: TabTexts[number]["tuning"]): string {
  if (tuning === undefined) {
    return "-";
  }
  const strings =
    tuning.status === "unknown_name" ? "" : ` ${tuning.strings.join(" ")}`;
  return `${tuning.status}${strings} from ${JSON.stringify(tuning.text)} @${tuning.confidence.toFixed(2)}`;
}

function rasteriseSamples(textEntries: readonly TextEntry[]): void {
  const tool = join(ROOT, ".pages", "rasterise-pdf");
  for (const entry of textEntries) {
    const png = join(ROOT, ".pages", entry.png);
    if (entry.pdf !== undefined && !existsSync(png)) {
      execFileSync(tool, [entry.pdf, png, "300", "1"]);
    }
  }
}

async function scoreText(): Promise<void> {
  const textEntries: readonly TextEntry[] = JSON.parse(
    readFileSync(join(ROOT, "tab-text-truth.json"), "utf8")
  );
  rasteriseSamples(textEntries);
  const ocr = new RapidOcr(cv, {
    classify: await store.open("ocrClassify"),
    detect: await store.open("ocrDetect"),
    recognize: session,
  });
  const totals = {
    capo: new Map<Verdict, number>(),
    tuning: new Map<Verdict, number>(),
  };
  for (const entry of textEntries) {
    const page = pageOf(entry.png);
    const started = performance.now();
    const tabs = await detectTablature(cv, page, read);
    const detected = performance.now();
    const texts = await readTabText(
      (image, minScore, limit) => ocr.read(image, minScore, limit),
      page,
      tabs
    );
    const done = performance.now();
    const common = commonText(texts);
    const strings =
      common.tuning?.status === "read" ? common.tuning.strings : undefined;
    const tuning = verdict(
      entry.tuning,
      strings,
      (a, b) => a.join(" ") === b.join(" ")
    );
    const capo = verdict(entry.capo, common.capo?.fret, (a, b) => a === b);
    totals.tuning.set(tuning, (totals.tuning.get(tuning) ?? 0) + 1);
    totals.capo.set(capo, (totals.capo.get(capo) ?? 0) + 1);
    const differ = texts.some((t) => t.tuning?.text !== common.tuning?.text);
    console.log(
      [
        entry.name,
        `tabs ${tabs.length}`,
        `tuning ${tuning} (${describeTuning(common.tuning)})${differ ? " [systems differ]" : ""}`,
        `capo ${capo} (${common.capo === undefined ? "-" : `${common.capo.fret} from ${JSON.stringify(common.capo.text)}`})`,
        `detect ${Math.round(detected - started)} ms, text ${Math.round(done - detected)} ms`,
      ].join(" | ")
    );
  }
  for (const field of ["tuning", "capo"] as const) {
    console.log(
      `${field}: ${[...totals[field]].map(([v, n]) => `${v} ${n}`).join(", ")}`
    );
  }
}

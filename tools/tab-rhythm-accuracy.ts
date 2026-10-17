// Measures the tab rhythm reader (src/tab/rhythm.ts, src/tab/read.ts) and the
// prerequisites (src/tab/tune.ts) on the owner's tab pages in
// test/fixtures/local/tablature/, which may not be redistributed: local only,
// never part of `npm test`. Per page it prints the events, how many carry a
// stem, the bars and how many sum to the meter, the time signature and slurs
// read, each prerequisite's status and whether resolveTab makes a tune; and,
// where a truth exists, the events matched and their durations against it:
//   - "hand": a hand transcription, bars split by "|", events "s/f+s/f:v"
//     with v one of w h q e s (dotted: "q."), and "slurs": "s/f>s/f:kind";
//   - "xml": the MusicXML the page was exported with (TablEdit, Guitar Pro),
//     whose fretted notes give each event's time to the next.
// Durations are compared as the time from an event to the next, in quarters.
//
// Needs models/ (tools/fetch-models.sh) and
// test/fixtures/local/tablature/tab-rhythm-truth.json (git-ignored):
//   [{ "name": "...", "png": "<file in .pages>", "pdf": "<optional, under
//      the tablature folder>", "page": 1, "hand": "...", "slurs": "...",
//      "xml": "<optional absolute path>" }]
// An entry with a "pdf" is rasterised into .pages/ first.
//
// usage: npx vite-node tools/tab-rhythm-accuracy.ts [name filter]
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { PNG } from "pngjs";
import { type AcquiredOpenCv, loadOpenCv } from "../src/cv/opencv.js";
import { type ColorImage, colorImageFromRgba } from "../src/image/plane.js";
import { startRuntime } from "../src/models/backend.js";
import { memoryCache } from "../src/models/cache.js";
import { ModelStore } from "../src/models/store.js";
import { ctcCharacters } from "../src/ocr/ctc.js";
import { RapidOcr, recognizeCrops } from "../src/ocr/rapid-ocr.js";
import { detectTablature, type ReadCrops } from "../src/tab/detect.js";
import { glyphsOf, linesIn, type PageTextItem } from "../src/tab/layer.js";
import { readTablature, type TabReading } from "../src/tab/read.js";
import { readTabText } from "../src/tab/text.js";
import {
  durationOfStem,
  resolveTab,
  type TimeSignature,
  tabPrerequisites,
  ticksOf,
} from "../src/tab/tune.js";
import { textLayerOf } from "../test/support/pdf-text.js";

const ROOT = join(
  import.meta.dirname,
  "..",
  "test",
  "fixtures",
  "local",
  "tablature"
);

interface Entry {
  readonly hand?: string;
  readonly name: string;
  readonly page?: number;
  readonly pdf?: string;
  readonly png: string;
  readonly slurs?: string;
  readonly xml?: string;
}

const withLayer = process.argv.includes("--layer");
const filter = process.argv.slice(2).find((a) => !a.startsWith("--"));
const entries: readonly Entry[] = (
  JSON.parse(
    readFileSync(join(ROOT, "tab-rhythm-truth.json"), "utf8")
  ) as Entry[]
).filter((e) => filter === undefined || e.name.includes(filter));

for (const entry of entries) {
  const png = join(ROOT, ".pages", entry.png);
  if (entry.pdf !== undefined && !existsSync(png)) {
    execFileSync(join(ROOT, ".pages", "rasterise-pdf"), [
      resolve(ROOT, entry.pdf),
      png,
      "300",
      String(entry.page ?? 1),
    ]);
  }
}

function pageOf(png: string): ColorImage {
  const decoded = PNG.sync.read(readFileSync(join(ROOT, ".pages", png)));
  return colorImageFromRgba(decoded.width, decoded.height, decoded.data);
}

/** An event as compared: its frets, and its time to the next event, in quarters. */
interface Timed {
  readonly key: string;
  readonly quarters: number;
}

const keyOf = (notes: readonly { string: number; fret: number }[]) =>
  notes
    .map((n) => `${n.string}/${n.fret}`)
    .sort()
    .join("+");

const HAND_VALUES: Readonly<Record<string, number>> = {
  e: 0.5,
  h: 2,
  q: 1,
  s: 0.25,
  w: 4,
};

const SPACES = /\s+/;
const DOTS = /\./g;

function handEvents(hand: string): Timed[] {
  return hand
    .split(SPACES)
    .filter((t) => t !== "|" && t !== "")
    .map((token) => {
      const [frets = "", value = "q"] = token.split(":");
      const base = HAND_VALUES[value.replace(DOTS, "")] ?? 1;
      const dots = value.length - value.replace(DOTS, "").length;
      return {
        key: keyOf(
          frets.split("+").map((n) => ({
            fret: Number(n.split("/")[1]),
            string: Number(n.split("/")[0]),
          }))
        ),
        quarters: base * (2 - 2 ** -dots),
      };
    });
}

const tag = (xml: string, name: string) =>
  xml.match(new RegExp(`<${name}>([^<]*)</${name}>`))?.[1];

type Onsets = Map<number, { string: number; fret: number }[]>;

/** One measure's fretted notes by onset, and where the measure ends, in divisions. */
function measureOnsets(measure: string): { end: number; onsets: Onsets } {
  const onsets: Onsets = new Map();
  let time = 0;
  let last = 0;
  let end = 0;
  for (const token of measure.match(
    /<note>[\s\S]*?<\/note>|<backup>[\s\S]*?<\/backup>|<forward>[\s\S]*?<\/forward>/g
  ) ?? []) {
    const duration = Number(tag(token, "duration") ?? 0);
    if (token.startsWith("<backup")) {
      time -= duration;
    } else if (token.startsWith("<forward")) {
      time += duration;
      end = Math.max(end, time);
    } else if (!token.includes("<grace")) {
      const chord = token.includes("<chord");
      addNote(onsets, chord ? last : time, token);
      if (!chord) {
        last = time;
        time += duration;
        end = Math.max(end, time);
      }
    }
  }
  return { end, onsets };
}

function addNote(onsets: Onsets, at: number, token: string): void {
  const fret = tag(token, "fret");
  const string = tag(token, "string");
  if (fret === undefined || string === undefined) {
    return;
  }
  const notes = onsets.get(at) ?? [];
  if (!notes.some((n) => n.string === Number(string))) {
    notes.push({ fret: Number(fret), string: Number(string) });
  }
  onsets.set(at, notes);
}

/** The fretted notes of a MusicXML file as events: grouped by onset, each timed to the next onset in its measure. */
function xmlEvents(path: string): Timed[] {
  const xml = readFileSync(path, "utf8");
  const events: Timed[] = [];
  let divisions = 1;
  for (const measure of xml.match(/<measure[\s\S]*?<\/measure>/g) ?? []) {
    divisions = Number(tag(measure, "divisions") ?? divisions);
    const { end, onsets } = measureOnsets(measure);
    const times = [...onsets.keys()].sort((a, b) => a - b);
    for (const [k, at] of times.entries()) {
      events.push({
        key: keyOf(onsets.get(at) ?? []),
        quarters: ((times[k + 1] ?? end) - at) / divisions,
      });
    }
  }
  return events;
}

const truthOf = (entry: Entry): Timed[] | undefined => {
  if (entry.hand !== undefined) {
    return handEvents(entry.hand);
  }
  return entry.xml === undefined ? undefined : xmlEvents(entry.xml);
};

/** Pairs of indices of a longest common subsequence of keys. */
function lcsPairs(
  a: readonly Timed[],
  b: readonly Timed[]
): [number, number][] {
  const table = Array.from(
    { length: a.length + 1 },
    () => new Int32Array(b.length + 1)
  );
  for (let i = a.length - 1; i >= 0; i -= 1) {
    for (let j = b.length - 1; j >= 0; j -= 1) {
      table[i][j] =
        a[i]?.key === b[j]?.key
          ? table[i + 1][j + 1] + 1
          : Math.max(table[i + 1][j], table[i][j + 1]);
    }
  }
  const pairs: [number, number][] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i]?.key === b[j]?.key) {
      pairs.push([i, j]);
      i += 1;
      j += 1;
    } else if (table[i + 1][j] >= table[i][j + 1]) {
      i += 1;
    } else {
      j += 1;
    }
  }
  return pairs;
}

/** The read events timed by their durations, the bars' unstemmed notes given what each bar has left; grace notes left out, as the truths leave them. */
function readEvents(
  readings: readonly TabReading[],
  meter: TimeSignature
): Timed[] {
  const { durations } = tabPrerequisites(readings, { meter });
  const settled =
    durations.status === "missing" ? durations.candidates[0] : durations.value;
  return readings.flatMap((r, s) =>
    r.events.flatMap((e, k) => {
      const d = settled?.[s]?.[k] ?? durationOfStem(e.stem);
      return d === "grace"
        ? []
        : [
            {
              key: keyOf(e.notes),
              quarters: d === undefined ? 0 : ticksOf(d) / 960,
            },
          ];
    })
  );
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
const session = await store.open("ocrRecognize");
const characters = ctcCharacters(session.metadata.get("character") ?? "");
const read: ReadCrops = async (crops) =>
  (await recognizeCrops(cv, session, characters, crops)).map((r) => r.text);
const ocr = new RapidOcr(cv, {
  classify: await store.open("ocrClassify"),
  detect: await store.open("ocrDetect"),
  recognize: session,
});

const totals = { bars: 0, fit: 0, matched: 0, same: 0, truth: 0 };
for (const entry of entries) {
  const page = pageOf(entry.png);
  const started = performance.now();
  const layer: PageTextItem[] =
    withLayer && entry.pdf !== undefined
      ? await textLayerOf(resolve(ROOT, entry.pdf), entry.page ?? 1)
      : [];
  const glyphs = glyphsOf(layer, page.width, page.height);
  const tabs = await detectTablature(cv, page, read);
  const readings = await readTablature(cv, page, tabs, read, undefined, layer);
  const texts = await readTabText(
    layer.length > 0
      ? (_image, _score, _limit, region) =>
          Promise.resolve(linesIn(glyphs, region))
      : (image, minScore, limit) => ocr.read(image, minScore, limit),
    page,
    tabs
  );
  const systems = readings.map((r, k) => ({ ...r, ...texts[k] }));
  const ms = Math.round(performance.now() - started);
  const events = systems.reduce((n, s) => n + s.events.length, 0);
  const stemmed = systems.reduce(
    (n, s) => n + s.events.filter((e) => e.stem.kind === "stem").length,
    0
  );
  const printed = systems.find((s) => s.timeSignature)?.timeSignature;
  const asRead = tabPrerequisites(systems).meter;
  const meter = printed ??
    (asRead.status === "missing" ? asRead.candidates[0] : undefined) ?? {
      den: 4,
      num: 4,
    };
  const checked = tabPrerequisites(systems, { meter });
  const fit = checked.bars.filter((b) => b.fits).length;
  totals.bars += checked.bars.length;
  totals.fit += fit;
  const prerequisites = tabPrerequisites(systems);
  const resolution = resolveTab(systems);
  const slurs = systems.flatMap((s) => s.slurs.map((x) => x.kind));
  console.log(
    [
      `${entry.name}${layer.length > 0 ? ` [text layer, ${readings.filter((r) => r.readFrom === "text_layer").length}/${readings.length} systems]` : ""}`,
      `systems ${systems.map((s) => s.lines).join(",")}`,
      `events ${events}, stemmed ${stemmed}`,
      `bars ${fit}/${checked.bars.length} sum to ${meter.num}/${meter.den}${printed ? "" : " (not printed: the best-fitting candidate)"}`,
      `slurs ${slurs.length === 0 ? "-" : [...new Set(slurs)].map((k) => `${k}x${slurs.filter((u) => u === k).length}`).join(" ")}`,
      `${ms} ms`,
    ].join(" | ")
  );
  console.log(
    `  prerequisites: ${(
      ["instrument", "tuning", "capo", "meter", "durations"] as const
    )
      .map((f) => {
        const p = prerequisites[f];
        return `${f} ${p.status}${p.status === "missing" ? "" : ` ${JSON.stringify(p.value).slice(0, 60)}`}`;
      })
      .join("; ")}`
  );
  console.log(
    `  resolveTab: ${resolution.ok ? `tune, ${resolution.tune.bars.length} bars` : `needs ${resolution.needs.map((n) => n.field).join(", ")}`}`
  );
  const unfit = checked.bars.filter((b) => !b.fits);
  if (unfit.length > 0) {
    console.log(
      `  bars off: ${unfit.map((b) => `${b.bar + 1}:${b.read}/${b.expected}${b.open > 0 ? ` (${b.open} unstemmed)` : ""}`).join(" ")}`
    );
  }
  const truth = truthOf(entry);
  if (truth !== undefined) {
    const got = readEvents(systems, meter);
    const pairs = lcsPairs(got, truth);
    const same = pairs.filter(
      ([i, j]) =>
        Math.abs((got[i]?.quarters ?? 0) - (truth[j]?.quarters ?? 0)) < 1e-6
    ).length;
    totals.truth += entry.hand === undefined ? 0 : truth.length;
    totals.matched += pairs.length;
    totals.same += same;
    console.log(
      `  vs ${entry.hand === undefined ? "MusicXML" : "hand truth"}: events matched ${pairs.length} (read ${got.length}, truth ${truth.length}); durations agree on ${same}/${pairs.length}`
    );
    if (process.argv.includes("--diff")) {
      for (const [i, j] of pairs) {
        if (
          Math.abs((got[i]?.quarters ?? 0) - (truth[j]?.quarters ?? 0)) > 1e-6
        ) {
          console.log(
            `    ${got[i]?.key} read ${got[i]?.quarters} truth ${truth[j]?.quarters}`
          );
        }
      }
    }
  }
  if (entry.slurs !== undefined) {
    const want = entry.slurs.split(" ");
    const gotSlurs = systems.flatMap((s) =>
      s.slurs.map((x) => {
        const from = s.events[x.from]?.notes.find((n) => n.string === x.string);
        const to = s.events[x.to]?.notes.find((n) => n.string === x.string);
        return `${x.string}/${from?.fret}>${x.string}/${to?.fret}:${x.kind}`;
      })
    );
    const left = [...gotSlurs];
    const found = want.filter((w) => {
      const k = left.indexOf(w);
      if (k >= 0) {
        left.splice(k, 1);
      }
      return k >= 0;
    }).length;
    console.log(
      `  slurs vs hand truth: ${found}/${want.length}, ${left.length} extra`
    );
  }
}
console.log(
  `all pages: ${totals.fit}/${totals.bars} bars sum to the meter; durations agree on ${totals.same}/${totals.matched} matched events`
);
await store.close();

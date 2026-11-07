// Typesets the line-tablature fixtures in test/fixtures/tab/ and writes their
// truth: A4 pages at 300 dpi (2480 x 3508), each built as an SVG, rasterised
// by rsvg-convert and re-encoded as 8-bit grayscale PNG. Three fret pages,
// one per instrument, from the column specs in PAGES: truth.json holds every
// event's page-pixel x and its (string, fret) notes, every technique letter
// and the count of slur arcs, all read from the layout that drew them. Eleven
// header pages from HEADER_PAGES, one short system each with the tuning and
// capo text a page prints: truth.json's `headers` holds what that text says.
//
// Glyph placement is measured, not assumed: each label is first rasterised
// alone and its ink box read back, so a number is placed with its ink centred
// on its x and its line, and the two digits of "10" are set to share one ink
// column. Anchors are whole pixels, so the page's ink is the calibration's ink
// translated, and truth x is exact to the half pixel.
//
// Needs rsvg-convert and fontconfig (Homebrew: librsvg), Roboto installed, and
// TeX Gyre Bonum either installed or in a directory named by TAB_FONT_DIRS
// (colon-separated). Rerun after editing PAGES:
//   TAB_FONT_DIRS=/path/to/texgyrebonum node tools/typeset-tab.mjs
// The output is byte-identical from run to run.
//
// The same run typesets the melodeon button tab (CADB) pages in
// test/fixtures/button-tab/ from BUTTON_TUNE: the staff is abcjs's, rendered
// by headless Chrome (CHROME, default the macOS app) from ABCJS_FILE (default
// the app checkout's abcjs 6.6.4 beside this repository), and the P/T bands,
// button numbers and left-hand letters are drawn here under each note head's
// column, measured the same way. Its truth.json holds one ButtonTabReading
// (docs/homr-web-plan/button-tab.md, H2) per staff.
import { execFileSync } from "node:child_process";
import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const outDir = join(root, "test/fixtures/tab");

const WIDTH = 2480;
const HEIGHT = 3508;
const SYSTEM_X0 = 200;
const SYSTEM_X1 = 2280;
const FIRST_TOP = 520;
const LINE_THICKNESS = 2;
const BAR_THICKNESS = 3;
const ARC_STROKE = 3;
/** Fret digits are this share of the line spacing tall. */
const DIGIT_HEIGHT = 0.8;
/** The ink of the two digits of a two-digit fret shares this many columns. */
const DIGIT_OVERLAP = 1;
const SANS = "Roboto";
const SERIF = "TeX Gyre Bonum";
const TECHNIQUES = new Set(["Sl", "Po", "H", "R", "p", "h", "x", "Harm."]);
const WHITESPACE = /\s+/;
const ANNOTATION_TOKEN = /^(.+)@(\d)$/;
const NOTE_TOKEN = /^(\d):(\d{1,2})$/;
const PNG_SUFFIX = /\.png$/;
/** truth.json's key order, at every depth. */
const TRUTH_KEYS = [
  "generator",
  "pages",
  "headers",
  "file",
  "width",
  "height",
  "tuning",
  "capo",
  "systems",
  "lines",
  "spacing",
  "staff",
  "top",
  "events",
  "x",
  "notes",
  "string",
  "fret",
  "annotations",
  "technique",
  "arcs",
  "texts",
  "at",
  "text",
  "expect",
  "status",
  "strings",
  "rotation",
  "keyboard",
  "staves",
  "buttonTabs",
  "x0",
  "x1",
  "bottom",
  "bandTop",
  "rule",
  "bandBottom",
  "marks",
  "bellows",
  "button",
  "row",
  "held",
  "leftHand",
  "unread",
];
const LINE_WIDTH = 80;

/**
 * A system's columns, left to right: "s:f" is one note (string 1 is the top
 * line), "s:f+s:f" a chord at one x, "Tech@s" a technique letter standing on
 * string s midway between its two neighbouring events, and "arc" a slur from
 * the event before it to the event after it, both single notes on one string.
 */
const PAGES = [
  {
    clef: true,
    file: "banjo-5-lines.png",
    font: SANS,
    lines: 5,
    spacing: 42,
    systems: [
      {
        columns:
          "1:0 2:1 3:2 4:3 5:0 1:4 2:5 3:6 4:7 5:0 1:8 2:9 3:10 1:12 1:0+2:1+3:2 4:0 Sl@4 4:2",
        knockout: true,
      },
      {
        columns:
          "3:2 Po@3 3:0 2:0+3:0+4:2 5:0 1:3 arc 1:2 2:3 H@2 2:5 4:4 5:0 1:0+3:2+4:2 1:10 2:12",
        knockout: true,
      },
      {
        columns:
          "4:5 R@4 4:4 3:2 2:1 1:0+2:0+3:0 5:0 4:0 3:7 2:8 1:9 2:10 Sl@2 2:12 1:2+2:3+4:4",
        knockout: true,
      },
    ],
    title: "homr-web tab fixture: banjo, 5 lines",
    tuning: ["D4", "B3", "G3", "D3", "G4"],
  },
  {
    clef: true,
    file: "guitar-6-lines.png",
    font: SERIF,
    lines: 6,
    spacing: 31,
    systems: [
      {
        columns:
          "6:0 5:2 4:2 3:1 2:0 1:0 5:3+4:2 4:0 h@4 4:2 3:0 p@3 3:2 1:0+2:1+3:0+4:2+5:3 6:3 x@6 6:5 2:3+1:3 1:5 1:7 2:8 3:9 4:10 5:11 6:12 1:0+2:0+3:1+4:2+5:2+6:0",
        knockout: true,
      },
      {
        columns:
          "1:12 Harm.@1 1:12 2:10 3:9 3:7+4:9+5:9 6:7 4:5 h@4 4:7 2:5+3:5+4:5+5:7 3:4 p@3 3:2 5:0 x@5 5:3 1:3+2:3+3:4+4:5+5:5+6:3 6:12 1:10 2:11",
        knockout: true,
      },
    ],
    title: "homr-web tab fixture: guitar, 6 lines",
    tuning: ["E4", "B3", "G3", "D3", "A2", "E2"],
  },
  {
    clef: false,
    file: "mandolin-4-lines.png",
    font: SANS,
    lines: 4,
    spacing: 36,
    systems: [
      {
        columns:
          "4:0 4:2 4:4 3:0 3:2 3:4 2:0 2:2 2:4 1:0 1:2 1:3 1:5 1:7 1:10 1:12 2:0+3:2 1:0+2:2+3:2+4:0 2:10 3:11",
        knockout: true,
      },
      {
        columns:
          "1:0 2:2 3:4 4:5 1:7 2:9 3:12 4:10 1:0+2:2 3:3+4:5 1:5 1:3 1:2 2:5 2:3 2:2 3:7 4:0 1:12+2:10",
        knockout: false,
      },
    ],
    title: "homr-web tab fixture: mandolin, 4 lines",
    tuning: ["E5", "A4", "D4", "G3"],
  },
];

const HEADER_HEIGHT = HEIGHT;
const HEADER_TOP = 440;
const BANJO_COLUMNS = "1:0 2:1 3:2 4:3 5:0 1:2 2:3 3:0 4:2 5:0 1:0+2:1";
const GUITAR_COLUMNS = "6:0 5:2 4:2 3:1 2:0 1:0 5:3+4:2 6:3 1:3 2:1";
const MANDOLIN_COLUMNS = "4:0 4:2 3:0 3:2 2:0 2:2 1:0 1:2 2:0+3:2";

/**
 * Pages for the tuning and capo text (src/tab/text.ts): one short system
 * each, with the text where tab software prints it, a centred subtitle under
 * the title, a line at the right or left above the system, a label turned a
 * quarter in the left margin, one string name per line in the margin, or a
 * line under the system. `expect` is what the page says, top line first: a
 * tuning that fits, one for another string count, an unknown name, or none.
 */
const HEADER_PAGES = [
  {
    columns: BANJO_COLUMNS,
    expect: { capo: 2, strings: ["E4", "D4", "A3", "D3", "A4"] },
    file: "header-banjo-en.png",
    lines: 5,
    texts: [
      {
        at: "subtitle",
        family: SERIF,
        text: "aDADE tuning, Capo 2",
        weight: "bold",
      },
    ],
  },
  {
    columns: BANJO_COLUMNS,
    expect: { capo: 3, strings: ["D4", "C4", "G3", "C3", "G4"] },
    file: "header-banjo-fr.png",
    lines: 5,
    texts: [
      { at: "left", family: SANS, text: "Accordage : Double C" },
      { at: "right", family: SANS, text: "Capodastre en 3e case" },
    ],
  },
  {
    columns: BANJO_COLUMNS,
    expect: { capo: 2, strings: ["D4", "B3", "G3", "D3", "G4"] },
    file: "header-banjo-margin.png",
    lines: 5,
    texts: [
      { at: "margin", family: SERIF, text: "gDGBD" },
      { at: "right", family: SERIF, text: "Capo II" },
    ],
  },
  {
    columns: GUITAR_COLUMNS,
    expect: { capo: 2, strings: ["D4", "A3", "G3", "D3", "A2", "D2"] },
    file: "header-guitar-en.png",
    lines: 6,
    texts: [
      { at: "left", family: SERIF, text: "Tuning: D A D G A D" },
      { at: "right", family: SERIF, text: "capo on 2nd fret" },
    ],
  },
  {
    columns: GUITAR_COLUMNS,
    expect: { capo: 5, strings: ["D4", "B3", "G3", "D3", "G2", "D2"] },
    file: "header-guitar-fr.png",
    lines: 6,
    texts: [
      { at: "subtitle", family: SERIF, text: "Accord : Open G" },
      { at: "right", family: SERIF, text: "Capo : 5" },
    ],
  },
  {
    columns: GUITAR_COLUMNS,
    expect: { capo: 3, strings: ["E4", "B3", "G3", "D3", "G2", "D2"] },
    file: "header-guitar-strings.png",
    lines: 6,
    texts: [
      { at: "left", family: SANS, text: "(6) = D   (5) = G" },
      { at: "right", family: SANS, text: "Capo 3rd" },
    ],
  },
  {
    columns: GUITAR_COLUMNS,
    expect: { capo: null, strings: ["E4", "B3", "G3", "D3", "A2", "D2"] },
    file: "header-guitar-below.png",
    lines: 6,
    texts: [{ at: "below", family: SANS, text: "Drop D" }],
  },
  {
    columns: GUITAR_COLUMNS,
    expect: { capo: null, strings: ["E4", "B3", "G3", "D3", "A2", "D2"] },
    file: "header-guitar-labels.png",
    lines: 6,
    texts: [{ at: "labels", family: SANS, text: "e B G D A D" }],
  },
  {
    columns: MANDOLIN_COLUMNS,
    expect: { capo: 2, strings: ["E5", "A4", "D4", "G3"] },
    file: "header-mandolin.png",
    lines: 4,
    texts: [
      { at: "subtitle", family: SANS, text: "Standard tuning (GDAE)" },
      { at: "right", family: SANS, text: "capo 2 (sounds in A)" },
    ],
  },
  {
    columns: BANJO_COLUMNS,
    expect: {
      capo: null,
      status: "string_count",
      strings: ["D4", "A3", "G3", "D3", "A2", "D2"],
    },
    file: "header-banjo-mismatch.png",
    lines: 5,
    texts: [{ at: "subtitle", family: SERIF, text: "DADGAD tuning" }],
  },
  {
    columns: GUITAR_COLUMNS,
    expect: { capo: 2, status: "unknown_name", strings: null },
    file: "header-guitar-unknown.png",
    lines: 6,
    texts: [
      { at: "left", family: SERIF, text: "Open Zeta tuning" },
      { at: "right", family: SERIF, text: "Capodastre 2" },
    ],
  },
];

/** Where each header text goes, in page pixels, and how it is anchored. */
function headerText(page, label) {
  const s = page.spacing;
  const top = HEADER_TOP;
  const bottom = top + (page.lines - 1) * s;
  const place = {
    below: { anchor: "start", size: 40, x: SYSTEM_X0, y: bottom + 2.4 * s },
    left: { anchor: "start", size: 38, x: SYSTEM_X0, y: top - 2.2 * s },
    margin: {
      anchor: "middle",
      rotate: true,
      size: 40,
      x: SYSTEM_X0 - 0.9 * s,
      y: (top + bottom) / 2,
    },
    right: { anchor: "end", size: 38, x: SYSTEM_X1, y: top - 2.2 * s },
    subtitle: { anchor: "middle", size: 48, x: WIDTH / 2, y: 280 },
  }[label.at];
  if (label.at === "labels") {
    return label.text
      .split(" ")
      .map(
        (name, k) =>
          `<text x="${SYSTEM_X0 - 0.5 * s}" y="${top + k * s + 0.35 * s}" font-family="${label.family}" font-size="${Math.round(0.9 * s)}" text-anchor="end">${escapeXml(name)}</text>`
      )
      .join("");
  }
  const rotate = place.rotate
    ? ` transform="rotate(-90 ${place.x} ${place.y})"`
    : "";
  return `<text x="${place.x}" y="${place.y}" font-family="${label.family}" font-size="${place.size}" font-weight="${label.weight ?? "normal"}" text-anchor="${place.anchor}"${rotate}>${escapeXml(label.text)}</text>`;
}

const HEADER_STYLE = {
  4: { font: SANS, spacing: 36 },
  5: { font: SANS, spacing: 42 },
  6: { font: SERIF, spacing: 31 },
};

function typesetHeaderPage(spec, work, env) {
  const page = {
    clef: spec.lines !== 4,
    file: spec.file,
    lines: spec.lines,
    ...HEADER_STYLE[spec.lines],
  };
  const glyphs = glyphsFor(page, work, env);
  const system = layoutSystem(
    page,
    { columns: spec.columns, knockout: true },
    HEADER_TOP,
    glyphs
  );
  const title = `<text x="${WIDTH / 2}" y="190" font-family="${SERIF}" font-size="80" text-anchor="middle">${escapeXml(`Fixture tune, ${spec.lines} lines`)}</text>`;
  const texts = spec.texts.map((label) => headerText(page, label)).join("\n");
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEADER_HEIGHT}"><rect width="100%" height="100%" fill="#fff"/>\n${title}\n${texts}\n${system.svg}\n</svg>`;
  const png = toGrayPng(
    rasterise(svg, work, env, spec.file.replace(PNG_SUFFIX, ""))
  );
  writeFileSync(join(outDir, spec.file), png);
  console.log(`${spec.file}: ${png.length} bytes`);
  return {
    expect: spec.expect,
    file: spec.file,
    height: HEADER_HEIGHT,
    lines: spec.lines,
    texts: spec.texts.map(({ at, text }) => ({ at, text })),
    width: WIDTH,
  };
}

function parseColumns(text) {
  return text
    .trim()
    .split(WHITESPACE)
    .map((token) => {
      if (token === "arc") {
        return { kind: "arc" };
      }
      if (ANNOTATION_TOKEN.test(token)) {
        const [, technique, string] = token.match(ANNOTATION_TOKEN) ?? [];
        if (!TECHNIQUES.has(technique)) {
          throw new Error(`unknown technique ${token}`);
        }
        return { kind: "annotation", string: Number(string), technique };
      }
      const notes = token.split("+").map((note) => {
        if (!NOTE_TOKEN.test(note)) {
          throw new Error(`bad column ${token}`);
        }
        const [, string, fret] = note.match(NOTE_TOKEN) ?? [];
        return { fret: Number(fret), string: Number(string) };
      });
      return {
        kind: "event",
        notes: notes.sort((a, b) => a.string - b.string),
      };
    });
}

function fontConfig(work) {
  const dirs = (process.env.TAB_FONT_DIRS ?? "")
    .split(":")
    .filter((dir) => dir !== "")
    .map((dir) => `<dir>${dir}</dir>`)
    .join("");
  const file = join(work, "fonts.conf");
  writeFileSync(
    file,
    `<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><include ignore_missing="yes">/opt/homebrew/etc/fonts/fonts.conf</include><include ignore_missing="yes">/etc/fonts/fonts.conf</include>${dirs}<cachedir>${join(work, "cache")}</cachedir></fontconfig>`
  );
  // pango's CoreText backend on macOS ignores fontconfig, and with it TAB_FONT_DIRS.
  const env = {
    ...process.env,
    FONTCONFIG_FILE: file,
    PANGOCAIRO_BACKEND: "fc",
  };
  for (const family of [SANS, SERIF]) {
    const found = execFileSync("fc-match", ["-f", "%{family}", family], {
      encoding: "utf8",
      env,
    });
    if (!found.split(",").includes(family)) {
      throw new Error(
        `fontconfig resolves "${family}" to "${found}"; install it or name its directory in TAB_FONT_DIRS`
      );
    }
  }
  return env;
}

function rasterise(svg, work, env, name) {
  const svgPath = join(work, `${name}.svg`);
  const pngPath = join(work, `${name}.png`);
  writeFileSync(svgPath, svg);
  execFileSync(
    "rsvg-convert",
    ["--background-color=white", svgPath, "-o", pngPath],
    {
      env,
    }
  );
  return PNG.sync.read(readFileSync(pngPath));
}

const escapeXml = (text) =>
  text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const textElement = (label, x, y) =>
  `<text x="${x}" y="${y}" font-family="${label.family}" font-size="${label.size}" font-weight="${label.weight}">${escapeXml(label.text)}</text>`;

/**
 * Ink boxes of each label drawn with its anchor at (0, 0): columns l..r and
 * rows t..b, end-exclusive, relative to the start of the baseline.
 */
function measure(labels, work, env) {
  const cell = 400;
  const columns = 6;
  const rows = Math.ceil(labels.length / columns);
  const anchor = (index) => ({
    x: (index % columns) * cell + 100,
    y: Math.floor(index / columns) * cell + 250,
  });
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${columns * cell}" height="${rows * cell}"><rect width="100%" height="100%" fill="#fff"/>${labels
    .map((label, index) => {
      const { x, y } = anchor(index);
      return textElement(label, x, y);
    })
    .join("")}</svg>`;
  const png = rasterise(svg, work, env, "measure");
  return labels.map((label, index) => {
    const { x: ax, y: ay } = anchor(index);
    const cx0 = (index % columns) * cell;
    const cy0 = Math.floor(index / columns) * cell;
    let l = Number.POSITIVE_INFINITY;
    let r = -1;
    let t = Number.POSITIVE_INFINITY;
    let b = -1;
    for (let y = cy0; y < cy0 + cell; y += 1) {
      for (let x = cx0; x < cx0 + cell; x += 1) {
        if (png.data[(y * png.width + x) * 4] < 128) {
          l = Math.min(l, x);
          r = Math.max(r, x);
          t = Math.min(t, y);
          b = Math.max(b, y);
        }
      }
    }
    if (r < 0) {
      throw new Error(`no ink for "${label.text}" in ${label.family}`);
    }
    return { b: b + 1 - ay, l: l - ax, r: r + 1 - ax, t: t - ay };
  });
}

/** A label's glyph runs with whole-pixel anchors relative to its ink centre, and its ink box. */
function composeFret(fret, digits) {
  const parts = String(fret)
    .split("")
    .map((digit) => digits[Number(digit)]);
  const runs = [];
  let offset = 0;
  for (const [index, part] of parts.entries()) {
    if (index > 0) {
      const previous = parts[index - 1];
      offset += previous.box.r - DIGIT_OVERLAP - part.box.l;
    }
    runs.push({ dx: offset, label: part.label });
  }
  const l = runs[0].dx + parts[0].box.l;
  const r = runs.at(-1).dx + parts.at(-1).box.r;
  const t = Math.min(...parts.map((part) => part.box.t));
  const b = Math.max(...parts.map((part) => part.box.b));
  return { box: { b, l, r, t }, runs };
}

function placed(glyph, cx, cy) {
  const ax = Math.round(cx - (glyph.box.l + glyph.box.r) / 2);
  const ay = Math.round(cy - (glyph.box.t + glyph.box.b) / 2);
  return {
    box: {
      b: ay + glyph.box.b,
      l: ax + glyph.box.l,
      r: ax + glyph.box.r,
      t: ay + glyph.box.t,
    },
    svg: glyph.runs
      .map((run) => textElement(run.label, ax + run.dx, ay))
      .join(""),
    x: ax + (glyph.box.l + glyph.box.r) / 2,
  };
}

/** `glyph` with its ink centred on cx, its text set on the baseline y. */
function onBaseline(glyph, cx, y) {
  const ax = Math.round(cx - (glyph.box.l + glyph.box.r) / 2);
  return {
    box: {
      b: y + glyph.box.b,
      l: ax + glyph.box.l,
      r: ax + glyph.box.r,
      t: y + glyph.box.t,
    },
    svg: glyph.runs
      .map((run) => textElement(run.label, ax + run.dx, y))
      .join(""),
    x: ax + (glyph.box.l + glyph.box.r) / 2,
  };
}

function glyphsFor(page, work, env) {
  const s = page.spacing;
  const [probe] = measure(
    [{ family: page.font, size: 100, text: "0", weight: "normal" }],
    work,
    env
  );
  const size =
    Math.round(((DIGIT_HEIGHT * s) / (probe.b - probe.t)) * 1000) / 10;
  const [clefProbe] = measure(
    [{ family: page.font, size: 100, text: "T", weight: "bold" }],
    work,
    env
  );
  const clefHeight = Math.min(1.1 * s, ((page.lines - 1) * s) / 3.6);
  const clefSize =
    Math.round((clefHeight / (clefProbe.b - clefProbe.t)) * 1000) / 10;
  const labels = [
    ..."0123456789".split("").map((text) => ({ size, text, weight: "normal" })),
    ...[...TECHNIQUES].map((text) => ({ size, text, weight: "normal" })),
    ..."TAB"
      .split("")
      .map((text) => ({ size: clefSize, text, weight: "bold" })),
  ].map((label) => ({ ...label, family: page.font }));
  const boxes = measure(labels, work, env);
  const single = labels.map((label, index) => ({
    box: boxes[index],
    label,
    runs: [{ dx: 0, label }],
  }));
  const digits = single.slice(0, 10);
  return {
    clef: Object.fromEntries(
      single.slice(-3).map((glyph) => [glyph.label.text, glyph])
    ),
    fret: (fret) => composeFret(fret, digits),
    technique: Object.fromEntries(
      single.slice(10, -3).map((glyph) => [glyph.label.text, glyph])
    ),
  };
}

const halfWidth = (glyph) => (glyph.box.r - glyph.box.l) / 2;

function columnsWithGlyphs(system, glyphs) {
  return parseColumns(system.columns).map((column) => {
    if (column.kind === "event") {
      const notes = column.notes.map((note) => ({
        ...note,
        glyph: glyphs.fret(note.fret),
      }));
      return {
        ...column,
        half: Math.max(...notes.map((n) => halfWidth(n.glyph))),
        notes,
      };
    }
    if (column.kind === "annotation") {
      const glyph = glyphs.technique[column.technique];
      return { ...column, glyph, half: halfWidth(glyph) };
    }
    return column;
  });
}

/** Sets each visible column's x: events 2 spacings apart or more, a technique midway between its two events. */
function positionColumns(page, visible) {
  const s = page.spacing;
  let x = SYSTEM_X0 + (page.clef ? 2.2 : 1.2) * s + visible[0].half;
  for (const [index, item] of visible.entries()) {
    const previous = visible[index - 1];
    if (previous === undefined) {
      item.x = x;
      continue;
    }
    if (item.kind === "annotation") {
      const next = visible[index + 1];
      if (previous.kind !== "event" || next?.kind !== "event") {
        throw new Error(
          `${page.file}: a technique needs an event on each side`
        );
      }
      item.step = Math.max(
        1.3 * s,
        previous.half + item.half + 0.45 * s,
        item.half + next.half + 0.45 * s
      );
    }
    x +=
      previous.kind === "annotation"
        ? previous.step
        : (item.step ?? Math.max(2 * s, previous.half + item.half + 0.9 * s));
    item.x = x;
  }
  if (x + visible.at(-1).half > SYSTEM_X1 - 0.8 * s) {
    throw new Error(
      `${page.file}: system overflows by ${Math.ceil(x - SYSTEM_X1)} px`
    );
  }
}

function drawColumns(page, system, visible, lineY) {
  const s = page.spacing;
  const padX = Math.round(0.12 * s);
  const padY = Math.round(0.08 * s);
  const knockouts = [];
  const ink = [];
  const events = [];
  const annotations = [];
  const draw = (glyph, x, string) => {
    const d = placed(glyph, x, lineY(string));
    if (system.knockout) {
      knockouts.push(
        `<rect x="${d.box.l - padX}" y="${d.box.t - padY}" width="${d.box.r - d.box.l + 2 * padX}" height="${d.box.b - d.box.t + 2 * padY}" fill="#fff"/>`
      );
    }
    ink.push(d.svg);
    return d.x;
  };
  for (const item of visible) {
    if (item.kind === "event") {
      const xs = item.notes.map((note) =>
        draw(note.glyph, item.x, note.string)
      );
      const cx = xs.reduce((sum, x) => sum + x, 0) / xs.length;
      events.push({
        notes: item.notes.map(({ fret, string }) => ({ fret, string })),
        x: Math.round(cx * 10) / 10,
      });
    } else {
      annotations.push({
        string: item.string,
        technique: item.technique,
        x: draw(item.glyph, item.x, item.string),
      });
    }
  }
  return { annotations, events, ink, knockouts };
}

/** A slur from just above one number to just above the next, its ends dipping to 0.2 spacing over the line. */
function drawArcs(page, items, lineY) {
  const s = page.spacing;
  return items.flatMap((item, index) => {
    if (item.kind !== "arc") {
      return [];
    }
    const from = items[index - 1];
    const to = items[index + 1];
    if (
      from?.kind !== "event" ||
      to?.kind !== "event" ||
      from.notes.length !== 1 ||
      to.notes.length !== 1 ||
      from.notes[0].string !== to.notes[0].string
    ) {
      throw new Error(
        `${page.file}: an arc joins two single notes on one string`
      );
    }
    const y = lineY(from.notes[0].string);
    const x0 = from.x + 0.45 * s;
    const x1 = to.x - 0.45 * s;
    return [
      `<path d="M ${x0} ${y - 0.2 * s} Q ${(x0 + x1) / 2} ${y - 1.4 * s} ${x1} ${y - 0.2 * s}" fill="none" stroke="#000" stroke-width="${ARC_STROKE}" stroke-linecap="round"/>`,
    ];
  });
}

function drawStaff(page, top, glyphs, lineY) {
  const bottom = lineY(page.lines);
  const staff = [];
  for (let string = 1; string <= page.lines; string += 1) {
    staff.push(
      `<rect x="${SYSTEM_X0}" y="${lineY(string) - LINE_THICKNESS / 2}" width="${SYSTEM_X1 - SYSTEM_X0}" height="${LINE_THICKNESS}" fill="#000"/>`
    );
  }
  for (const barX of [SYSTEM_X0, SYSTEM_X1 - BAR_THICKNESS]) {
    staff.push(
      `<rect x="${barX}" y="${top - LINE_THICKNESS / 2}" width="${BAR_THICKNESS}" height="${bottom - top + LINE_THICKNESS}" fill="#000"/>`
    );
  }
  const clef = page.clef
    ? ["T", "A", "B"].map(
        (letter, index) =>
          placed(
            glyphs.clef[letter],
            SYSTEM_X0 + page.spacing,
            top + ((bottom - top) * (index + 0.5)) / 3
          ).svg
      )
    : [];
  return { clef, staff };
}

function layoutSystem(page, system, top, glyphs) {
  const lineY = (string) => top + (string - 1) * page.spacing;
  const items = columnsWithGlyphs(system, glyphs);
  const visible = items.filter((item) => item.kind !== "arc");
  positionColumns(page, visible);
  const { annotations, events, ink, knockouts } = drawColumns(
    page,
    system,
    visible,
    lineY
  );
  const arcs = drawArcs(page, items, lineY);
  const { clef, staff } = drawStaff(page, top, glyphs, lineY);
  return {
    svg: [...staff, ...knockouts, ...ink, ...clef, ...arcs].join("\n"),
    truth: {
      annotations,
      arcs: arcs.length,
      events,
      lines: page.lines,
      spacing: page.spacing,
      top,
    },
  };
}

/** JSON laid out the way the repository's formatter lays it: a node goes on one line when it fits. */
function toJson(value, indent = "", prefix = "") {
  if (typeof value !== "object" || value === null) {
    return JSON.stringify(value);
  }
  const isArray = Array.isArray(value);
  const keys = isArray
    ? [...value.keys()]
    : TRUTH_KEYS.filter((key) => key in value);
  const label = (key) => (isArray ? "" : `${JSON.stringify(key)}: `);
  const [open, close] = isArray ? ["[", "]"] : ["{ ", " }"];
  const flat = keys.map((key) => `${label(key)}${toJson(value[key])}`);
  const oneLine = isArray
    ? `[${flat.join(", ")}]`
    : `${open}${flat.join(", ")}${close}`;
  if (
    !oneLine.includes("\n") &&
    indent.length + prefix.length + oneLine.length + 1 <= LINE_WIDTH
  ) {
    return oneLine;
  }
  const inner = `${indent}  `;
  const lines = keys.map(
    (key) => `${inner}${label(key)}${toJson(value[key], inner, label(key))}`
  );
  return `${open.trim()}\n${lines.join(",\n")}\n${indent}${close.trim()}`;
}

function toGrayPng(rgba) {
  const gray = Buffer.alloc(rgba.width * rgba.height);
  for (let i = 0; i < gray.length; i += 1) {
    gray[i] = Math.round(
      0.299 * rgba.data[i * 4] +
        0.587 * rgba.data[i * 4 + 1] +
        0.114 * rgba.data[i * 4 + 2]
    );
  }
  return PNG.sync.write(
    { data: gray, height: rgba.height, width: rgba.width },
    { colorType: 0, deflateLevel: 9, inputColorType: 0, inputHasAlpha: false }
  );
}

// Melodeon button tab (CADB): a treble staff with two bands under it, P (push)
// above T (pull), split by one rule, and a line of left-hand letters under T.

const buttonOutDir = join(root, "test/fixtures/button-tab");
const ABCJS_FILE =
  process.env.ABCJS_FILE ??
  join(
    root,
    "../AbcMusicStudioSvelte/node_modules/abcjs/dist/abcjs-basic-min.js"
  );
const CHROME =
  process.env.CHROME ??
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const BUTTON_STAFFWIDTH = 700;
/** Band and letter-line geometry, in staff spacings. */
const BAND = {
  /** From the staff's bottom line to the top of the P band. */
  gap: 3.4,
  /** One band's height. */
  height: 2,
  /** From the bottom of the T band to the centre of the letter line. */
  letters: 1.3,
  /** From the letter line's centre to the next staff's top line. */
  next: 6.8,
};
/** A button number's digits are this share of a band tall. */
const BUTTON_DIGIT_HEIGHT = 0.62;
const BAND_RULE = 2;
const BAND_BAR = 2;
const BUTTON_FIRST_TOP = 560;
const MARK_TOKEN = /^([PT])(\d{1,2})('{0,2})(-?)$/;
const LETTER_TOKEN = /^[A-Ga-g]$/;
const DUMP_OUT = /<pre id="out">([^<]+)<\/pre>/;
const BUTTON_PAGES = [
  { file: "gc-cadb-straight.png", rotation: 0 },
  { file: "gc-cadb-rotated.png", rotation: 0.3 },
];

/**
 * An original waltz for a three-row G/C melodeon (the app's melodeonGC3:
 * G row outer, C row ', accidentals row ''), every button sounding the staff
 * note on its bellows. One ABC line per system. `tab` holds one token per note
 * head, bar by bar: P or T, the button as printed and its suffix, or "." for a
 * note with no number. `bass` holds the left hand's letter or "." per note:
 * a capital for a bass, lower case for a chord (owner, 2026-11-07).
 */
const BUTTON_TUNE = {
  abc: "X:1\nM:3/4\nL:1/8\n%%stretchlast true\nK:C\n",
  keyboard: "melodeonGC3",
  systems: [
    {
      abc: "c2 e2 g2 | d2 B2 G2 | Acf a3 | g6 |",
      bass: "C c c | G g g | F . f . | C",
      tab: "P6' P7' P8' | T7' T6' T3'' | T5' T8 T8' T9' | P8'-",
    },
    {
      abc: "edc def | g2 b2 d'2 | c'2 ba g2 | ^f2 d4 |",
      bass: "C . c . G . | G g g | C g . c | D d",
      tab: "P7' T7' P6' T7' P7' T8' | P9 P10 P11 | P9' T10' T9' P8' | T10 T7'-",
    },
    {
      abc: "a2 ^g2 a2 | fed cBA | G2 B2 d2 | c6 |]",
      bass: "A e a | F . c . G . | G b g | C",
      tab: "T9' P7'' T9' | T8' P7' T7' P6' T6' T7 | . P7 P8 | P6'-",
    },
  ],
  title: "homr-web button tab fixture: G/C melodeon, CADB",
};

const bars = (text) =>
  text.split("|").map((bar) => bar.trim().split(WHITESPACE));

function parseButtonSystem(system) {
  const tab = bars(system.tab).map((bar) =>
    bar.map((token) => {
      if (token === ".") {
        return null;
      }
      const match = token.match(MARK_TOKEN);
      if (match === null) {
        throw new Error(`bad button mark ${token}`);
      }
      const [, bellows, button, row, held] = match;
      return {
        bellows: bellows === "P" ? "push" : "pull",
        button: Number(button),
        digits: button,
        held: held === "-",
        row: row.length,
        text: token.slice(1),
      };
    })
  );
  const bass = bars(system.bass).map((bar) =>
    bar.map((token) => {
      if (token !== "." && !LETTER_TOKEN.test(token)) {
        throw new Error(`bad left-hand letter ${token}`);
      }
      return token === "." ? null : token;
    })
  );
  return { bass, tab };
}

/** Renders the tune with abcjs in headless Chrome and reads back each line's staff, bars, note heads and markup. */
function renderWithAbcjs(abc, work) {
  const html = join(work, "button-tab.html");
  writeFileSync(join(work, "abcjs.js"), readFileSync(ABCJS_FILE));
  writeFileSync(
    html,
    `<!doctype html><html><body><div id="score"></div><pre id="out"></pre>
<script src="abcjs.js"></script>
<script>
"use strict";
ABCJS.renderAbc("score", ${JSON.stringify(abc)}, { add_classes: true, staffwidth: ${BUTTON_STAFFWIDTH} });
const svg = document.querySelector("#score svg");
const box = (el) => { const b = el.getBBox(); return { h: b.height, w: b.width, x: b.x, y: b.y }; };
const numberOf = (el, prefix) => Number([...el.classList].find((c) => c.startsWith(prefix) && /^\\d+$/.test(c.slice(prefix.length))).slice(prefix.length));
const lines = [...svg.querySelectorAll("g.abcjs-staff-wrapper")].map((wrapper) => ({
  bars: [...wrapper.querySelectorAll("g.abcjs-bar")].map(box),
  line: numberOf(wrapper, "abcjs-l"),
  notes: [...wrapper.querySelectorAll("g.abcjs-note")].map((g) => ({ heads: [...g.querySelectorAll(".abcjs-notehead")].map(box), m: numberOf(g, "abcjs-m") })),
  staffLines: [...wrapper.querySelectorAll("g.abcjs-staff path")].map(box),
  svg: new XMLSerializer().serializeToString(wrapper),
}));
document.getElementById("out").textContent = btoa(unescape(encodeURIComponent(JSON.stringify(lines))));
</script></body></html>`
  );
  const dom = execFileSync(
    CHROME,
    [
      "--headless=new",
      "--disable-gpu",
      "--no-first-run",
      "--no-default-browser-check",
      `--user-data-dir=${join(work, "chrome")}`,
      "--virtual-time-budget=10000",
      "--dump-dom",
      `file://${html}`,
    ],
    {
      encoding: "utf8",
      maxBuffer: 64 * 1024 * 1024,
      stdio: ["ignore", "pipe", "ignore"],
      timeout: 120_000,
    }
  );
  const encoded = dom.match(DUMP_OUT)?.[1];
  if (encoded === undefined) {
    throw new Error("abcjs rendered nothing: check ABCJS_FILE and CHROME");
  }
  return JSON.parse(Buffer.from(encoded, "base64").toString("utf8"));
}

/** Pairs each note head of a line with its tab token and letter, bar by bar, refusing any count that disagrees. */
function notesWithTab(line, parsed, index) {
  const measures = [...new Set(line.notes.map((note) => note.m))].sort(
    (a, b) => a - b
  );
  if (
    measures.length !== parsed.tab.length ||
    measures.length !== parsed.bass.length
  ) {
    throw new Error(
      `system ${index + 1}: ${measures.length} bars drawn, ${parsed.tab.length} in tab, ${parsed.bass.length} in bass`
    );
  }
  return measures.flatMap((m, bar) => {
    const notes = line.notes.filter((note) => note.m === m);
    if (
      notes.length !== parsed.tab[bar].length ||
      notes.length !== parsed.bass[bar].length
    ) {
      throw new Error(
        `system ${index + 1} bar ${bar + 1}: ${notes.length} notes, ${parsed.tab[bar].length} tab tokens, ${parsed.bass[bar].length} letters`
      );
    }
    return notes.map((note, k) => {
      if (note.heads.length !== 1) {
        throw new Error(
          `system ${index + 1} bar ${bar + 1}: one head per note`
        );
      }
      const [head] = note.heads;
      return {
        letter: parsed.bass[bar][k],
        mark: parsed.tab[bar][k],
        x: head.x + head.w / 2,
      };
    });
  });
}

function buttonGlyphs(systems, spacing, work, env) {
  const bandHeight = BAND.height * spacing;
  const [probe] = measure(
    [{ family: SANS, size: 100, text: "0", weight: "normal" }],
    work,
    env
  );
  const size =
    Math.round(
      ((BUTTON_DIGIT_HEIGHT * bandHeight) / (probe.b - probe.t)) * 1000
    ) / 10;
  const texts = new Set();
  const letters = new Set();
  for (const system of systems) {
    for (const mark of system.parsed.tab.flat()) {
      if (mark !== null) {
        texts.add(mark.text);
        texts.add(mark.digits);
      }
    }
    for (const letter of system.parsed.bass.flat()) {
      if (letter !== null) {
        letters.add(letter);
      }
    }
  }
  const labels = [
    ...[...texts].map((text) => ({ size, text, weight: "normal" })),
    ...[...letters].map((text) => ({ size, text, weight: "normal" })),
    ...["P", "T"].map((text) => ({ size, text, weight: "bold" })),
  ].map((label) => ({ ...label, family: SANS }));
  const boxes = measure(labels, work, env);
  const byText = new Map();
  for (const [index, label] of labels.entries()) {
    byText.set(`${label.weight}:${label.text}`, {
      box: boxes[index],
      label,
      runs: [{ dx: 0, label }],
    });
  }
  return (text, weight = "normal") => byText.get(`${weight}:${text}`);
}

/** Draws a button number with its digits' ink centred on (cx, cy), the suffix trailing; x is the digits' ink centre. */
function placeMark(glyph, mark, cx, cy) {
  const digits = glyph(mark.digits).box;
  const full = glyph(mark.text);
  const ax = Math.round(cx - (digits.l + digits.r) / 2);
  const ay = Math.round(cy - (digits.t + digits.b) / 2);
  return {
    l: ax + full.box.l,
    r: ax + full.box.r,
    svg: textElement(full.label, ax, ay),
    x: ax + (digits.l + digits.r) / 2,
  };
}

function rotatePoint(x, y, degrees) {
  const angle = (degrees * Math.PI) / 180;
  const dx = x - WIDTH / 2;
  const dy = y - HEIGHT / 2;
  return {
    x: WIDTH / 2 + dx * Math.cos(angle) - dy * Math.sin(angle),
    y: HEIGHT / 2 + dx * Math.sin(angle) + dy * Math.cos(angle),
  };
}

const norm = (value, size) => Math.round((value / size) * 1e6) / 1e6;

function layoutButtonSystem(system, index, scale, top, glyph) {
  const { line, parsed } = system;
  const staffLeft = Math.min(...line.staffLines.map((b) => b.x));
  const lineY = line.staffLines.map((b) => b.y + b.h / 2).sort((a, b) => a - b);
  const spacing = ((lineY[4] - lineY[0]) / 4) * scale;
  const tx = SYSTEM_X0 - scale * staffLeft;
  const ty = top - scale * lineY[0];
  const pageX = (x) => tx + scale * x;
  const bottom = top + 4 * spacing;
  const bandTop = Math.round(bottom + BAND.gap * spacing);
  const rule = Math.round(bandTop + BAND.height * spacing);
  const bandBottom = Math.round(rule + BAND.height * spacing);
  const lettersY = bandBottom + BAND.letters * spacing;
  // The left hand's letters share one baseline, as print sets them: a capital
  // (a bass) has its ink centred on lettersY, a lower-case chord sits lower.
  const capital = glyph("C").box;
  const lettersBaseline = Math.round(lettersY - (capital.t + capital.b) / 2);
  const svg = [
    `<g transform="translate(${tx} ${ty}) scale(${scale})">${line.svg}</g>`,
    `<rect x="${SYSTEM_X0}" y="${rule - BAND_RULE / 2}" width="${SYSTEM_X1 - SYSTEM_X0}" height="${BAND_RULE}" fill="#000"/>`,
  ];
  // The last bar closes the band at the staff's right end, also under a final |], whose box is thin and thick bars together.
  const barXs = [
    SYSTEM_X0 + BAND_BAR / 2,
    ...line.bars.slice(0, -1).map((b) => pageX(b.x + b.w / 2)),
    SYSTEM_X1 - BAND_BAR / 2,
  ];
  for (const x of barXs) {
    svg.push(
      `<rect x="${Math.round(x - BAND_BAR / 2)}" y="${bandTop}" width="${BAND_BAR}" height="${bandBottom - bandTop}" fill="#000"/>`
    );
  }
  for (const [letter, cy] of [
    ["P", (bandTop + rule) / 2],
    ["T", (rule + bandBottom) / 2],
  ]) {
    svg.push(
      placed(glyph(letter, "bold"), SYSTEM_X0 - 0.9 * BAND.height * spacing, cy)
        .svg
    );
  }
  const marks = [];
  const leftHand = [];
  const lastRight = { pull: -1, push: -1 };
  for (const note of notesWithTab(line, parsed, index)) {
    const cx = pageX(note.x);
    if (note.mark !== null) {
      const { bellows } = note.mark;
      const cy =
        bellows === "push" ? (bandTop + rule) / 2 : (rule + bandBottom) / 2;
      const drawn = placeMark(glyph, note.mark, cx, cy);
      if (drawn.l - lastRight[bellows] < 6) {
        throw new Error(
          `system ${index + 1}: ${note.mark.text} touches the mark before it`
        );
      }
      lastRight[bellows] = drawn.r;
      svg.push(drawn.svg);
      marks.push({
        bellows,
        button: note.mark.button,
        held: note.mark.held,
        row: note.mark.row,
        x: drawn.x,
        y: cy,
      });
    }
    if (note.letter !== null) {
      const drawn = onBaseline(glyph(note.letter), cx, lettersBaseline);
      svg.push(drawn.svg);
      leftHand.push({ text: note.letter, x: drawn.x, y: lettersY });
    }
  }
  return {
    geometry: { bandBottom, bandTop, bottom, rule, top },
    leftHand,
    marks,
    next: lettersY + BAND.next * spacing,
    svg: svg.join("\n"),
  };
}

/** One system's truth on a page turned by `rotation` degrees about its centre. */
function buttonTruth(laid, index, rotation) {
  const at = (x, y) => rotatePoint(x, y, rotation);
  const { bandBottom, bandTop, bottom, rule, top } = laid.geometry;
  const midY = (bandTop + bandBottom) / 2;
  const midX = (SYSTEM_X0 + SYSTEM_X1) / 2;
  const pageY = (value) => norm(at(midX, value).y, HEIGHT);
  return {
    reading: {
      leftHand: laid.leftHand.map((item) => ({
        text: item.text,
        x: norm(at(item.x, item.y).x, WIDTH),
      })),
      marks: laid.marks.map((mark) => ({
        bellows: mark.bellows,
        button: mark.button,
        held: mark.held,
        row: mark.row,
        x: norm(at(mark.x, mark.y).x, WIDTH),
      })),
      staff: index,
      unread: 0,
    },
    staff: {
      bandBottom: pageY(bandBottom),
      bandTop: pageY(bandTop),
      bottom: pageY(bottom),
      rule: pageY(rule),
      staff: index,
      top: pageY(top),
      x0: norm(at(SYSTEM_X0, midY).x, WIDTH),
      x1: norm(at(SYSTEM_X1, midY).x, WIDTH),
    },
  };
}

function typesetButtonPages(work, env) {
  const abc = `${BUTTON_TUNE.abc}${BUTTON_TUNE.systems.map((s) => s.abc).join("\n")}\n`;
  const lines = renderWithAbcjs(abc, work).sort((a, b) => a.line - b.line);
  if (lines.length !== BUTTON_TUNE.systems.length) {
    throw new Error(
      `abcjs drew ${lines.length} lines for ${BUTTON_TUNE.systems.length} systems`
    );
  }
  const widths = lines.map((line) => {
    const xs = line.staffLines.flatMap((b) => [b.x, b.x + b.w]);
    return Math.max(...xs) - Math.min(...xs);
  });
  if (Math.max(...widths) - Math.min(...widths) > 0.5) {
    throw new Error(`abcjs lines differ in width: ${widths.join(", ")}`);
  }
  const scale = (SYSTEM_X1 - SYSTEM_X0) / widths[0];
  const systems = BUTTON_TUNE.systems.map((spec, index) => ({
    line: lines[index],
    parsed: parseButtonSystem(spec),
  }));
  const firstLines = lines[0].staffLines.map((b) => b.y).sort((a, b) => a - b);
  const spacing = ((firstLines[4] - firstLines[0]) / 4) * scale;
  const glyph = buttonGlyphs(systems, spacing, work, env);
  let top = BUTTON_FIRST_TOP;
  const laid = systems.map((system, index) => {
    const result = layoutButtonSystem(system, index, scale, top, glyph);
    top = Math.round(result.next);
    return result;
  });
  if (top > HEIGHT - 200) {
    throw new Error(`button tab page overflows by ${top - HEIGHT + 200} px`);
  }
  mkdirSync(buttonOutDir, { recursive: true });
  const title = `<text x="${SYSTEM_X0}" y="220" font-family="${SANS}" font-size="44">${escapeXml(BUTTON_TUNE.title)}</text>`;
  const body = `${title}\n${laid.map((system) => system.svg).join("\n")}`;
  const pages = BUTTON_PAGES.map((page) => {
    const content =
      page.rotation === 0
        ? body
        : `<g transform="rotate(${page.rotation} ${WIDTH / 2} ${HEIGHT / 2})">${body}</g>`;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}"><rect width="100%" height="100%" fill="#fff"/>\n${content}\n</svg>`;
    const png = toGrayPng(
      rasterise(svg, work, env, page.file.replace(PNG_SUFFIX, ""))
    );
    writeFileSync(join(buttonOutDir, page.file), png);
    console.log(`${page.file}: ${png.length} bytes, ${laid.length} systems`);
    const truths = laid.map((system, index) =>
      buttonTruth(system, index, page.rotation)
    );
    return {
      buttonTabs: truths.map((truth) => truth.reading),
      file: page.file,
      height: HEIGHT,
      rotation: page.rotation,
      staves: truths.map((truth) => truth.staff),
      width: WIDTH,
    };
  });
  writeFileSync(
    join(buttonOutDir, "truth.json"),
    `${toJson({ generator: "tools/typeset-tab.mjs", keyboard: BUTTON_TUNE.keyboard, pages })}\n`
  );
}

const scratch = mkdtempSync(join(tmpdir(), "typeset-tab-"));
try {
  const env = fontConfig(scratch);
  mkdirSync(outDir, { recursive: true });
  const truthPages = [];
  for (const page of PAGES) {
    const glyphs = glyphsFor(page, scratch, env);
    const gap = Math.round(7 * page.spacing);
    let top = FIRST_TOP;
    const systems = page.systems.map((system) => {
      const laid = layoutSystem(page, system, top, glyphs);
      top += (page.lines - 1) * page.spacing + gap;
      return laid;
    });
    const title = `<text x="${SYSTEM_X0}" y="220" font-family="${SANS}" font-size="44">${escapeXml(page.title)}</text>`;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${WIDTH}" height="${HEIGHT}"><rect width="100%" height="100%" fill="#fff"/>\n${title}\n${systems.map((system) => system.svg).join("\n")}\n</svg>`;
    const png = toGrayPng(
      rasterise(svg, scratch, env, page.file.replace(PNG_SUFFIX, ""))
    );
    writeFileSync(join(outDir, page.file), png);
    truthPages.push({
      capo: 0,
      file: page.file,
      height: HEIGHT,
      systems: systems.map((system) => system.truth),
      tuning: page.tuning,
      width: WIDTH,
    });
    console.log(`${page.file}: ${png.length} bytes, ${systems.length} systems`);
  }
  const headers = HEADER_PAGES.map((spec) =>
    typesetHeaderPage(spec, scratch, env)
  );
  writeFileSync(
    join(outDir, "truth.json"),
    `${toJson({ generator: "tools/typeset-tab.mjs", headers, pages: truthPages })}\n`
  );
  typesetButtonPages(scratch, env);
} finally {
  rmSync(scratch, { force: true, recursive: true });
}

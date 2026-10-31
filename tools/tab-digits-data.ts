// Writes the training crops of the tab digit classifier (src/tab/digits.ts),
// cut the way the tab reader cuts them: a page's ink, its tab lines erased
// (bandWithoutLines), its line-sitting marks found (marksOf at READ_MARK) and
// each mark cropped (cropOf) and put in the classifier's form (digitInput).
// Two sources, both reproducible from this repository alone:
//
//   synthetic  MNIST digits drawn on synthetic tab systems at random line
//              spacings, sizes and slants, with pen-like stems hanging from
//              them, brush arrows and slur arcs drawn among them. A mark is
//              labelled with the digit it came from, or 10 (not a digit) when
//              it came from no digit or only a piece of one: a stem stub, an
//              arrow head, a piece of arc.
//   fixtures   the typeset tab pages committed in test/fixtures/tab/, read
//              through the same reader: a mark on a truth note with a fret
//              under 10 is that digit (print, which the reader must keep
//              reading as it did), any other mark is 10 (the TAB clef's
//              letters, Sl, Po, H, R, h, p, x, a slur's pieces). A mark on a
//              two-digit fret is left out.
//
// usage: npx vite-node tools/tab-digits-data.ts <mnist dir> <out prefix> \
//          <synthetic train|test> <pages> <seed> [--fixtures]
// writes <out prefix>-x.f32 (n x 784 float32) and <out prefix>-y.u8 (n labels)
// and prints the count per class. tools/train-tab-digits.sh runs it.
import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { PNG } from "pngjs";
import {
  type AcquiredOpenCv,
  loadOpenCv,
  withMatScope,
} from "../src/cv/opencv.js";
import {
  type ColorImage,
  colorImageFromRgba,
  createColor,
} from "../src/image/plane.js";
import {
  cropOf,
  detectTablature,
  inkOf,
  inkWithoutLines,
  type LineGroup,
  type Mark,
  marksOf,
  READ_MARK,
} from "../src/tab/detect.js";
import { DIGIT_INPUT, digitInput, NOT_A_DIGIT } from "../src/tab/digits.js";

const [
  mnistDir = "",
  out = "",
  split = "train",
  pagesArg = "0",
  seedArg = "1",
] = process.argv.slice(2);
const withFixtures = process.argv.includes("--fixtures");

// mulberry32: 32-bit integer arithmetic throughout (an LCG in doubles loses
// its low bits past 2^53 and falls into a short cycle).
let seed = Number(seedArg) >>> 0;
const rnd = () => {
  seed = (seed + 0x6d_2b_79_f5) >>> 0;
  let t = seed;
  t = Math.imul(t ^ (t >>> 15), t | 1);
  t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
  return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
};
const uni = (a: number, b: number) => a + (b - a) * rnd();
const pick = (n: number) => Math.floor(rnd() * n);

const requireCjs = createRequire(import.meta.url);
const cv = await loadOpenCv(() =>
  Promise.resolve({
    module: requireCjs("@techstark/opencv-js"),
  } as AcquiredOpenCv)
);

const inputs: Float32Array[] = [];
const targets: number[] = [];
const keep = (crop: ColorImage, label: number) => {
  const input = new Float32Array(DIGIT_INPUT);
  digitInput(crop, input, 0);
  inputs.push(input);
  targets.push(label);
};

/** Whether the mark's ink is crossed by a tab line, which the reader then crops from the line-erased ink (read.ts, struck). */
const crops = (
  page: ColorImage,
  ink: Uint8Array,
  lineless: Uint8Array,
  mark: Mark,
  spacing: number
) => [cropOf(page, ink, mark, spacing), cropOf(page, lineless, mark, spacing)];

// MNIST's idx files, as torchvision leaves them under <dir>/MNIST/raw.
const idx = (name: string, header: number) =>
  readFileSync(join(mnistDir, "MNIST", "raw", name)).subarray(header);
const prefix = split === "test" ? "t10k" : "train";
const images = idx(`${prefix}-images-idx3-ubyte`, 16);
const labels = idx(`${prefix}-labels-idx1-ubyte`, 8);

const W = 260;
const H = 220;

interface DigitBox {
  readonly label: number;
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

/** A synthetic page's ink, 1 for ink, and the pen that draws on it. */
class Canvas {
  readonly ink = new Uint8Array(W * H);

  set(x: number, y: number): void {
    if (x >= 0 && x < W && y >= 0 && y < H) {
      this.ink[y * W + x] = 1;
    }
  }

  /** A pen dot, `fat` pixels more to the right and below. */
  dot(x: number, y: number, fat: number): void {
    for (let dy = 0; dy <= fat; dy += 1) {
      for (let dx = 0; dx <= fat; dx += 1) {
        this.set(x + dx, y + dy);
      }
    }
  }

  stroke(x0: number, y0: number, x1: number, y1: number, thick: number): void {
    const n = Math.max(Math.abs(x1 - x0), Math.abs(y1 - y0), 1);
    for (let k = 0; k <= n; k += 1) {
      const x = Math.round(x0 + ((x1 - x0) * k) / n);
      const y = Math.round(y0 + ((y1 - y0) * k) / n);
      for (let a = 0; a < thick; a += 1) {
        for (let b = 0; b < thick; b += 1) {
          this.set(x + a, y + b);
        }
      }
    }
  }
}

/** The MNIST image's ink box (pixels over 100), or undefined when blank. */
function sourceBox(src: Uint8Array) {
  const box = { x0: 28, x1: -1, y0: 28, y1: -1 };
  for (let y = 0; y < 28; y += 1) {
    for (let x = 0; x < 28; x += 1) {
      if ((src[y * 28 + x] ?? 0) > 100) {
        box.x0 = Math.min(box.x0, x);
        box.x1 = Math.max(box.x1, x);
        box.y0 = Math.min(box.y0, y);
        box.y1 = Math.max(box.y1, y);
      }
    }
  }
  return box.x1 < 0 ? undefined : box;
}

/** Whether pixel (x, y) of the digit drawn w x h, slanted, falls on the MNIST image's ink. */
function inkAt(
  src: Uint8Array,
  box: { x0: number; x1: number; y0: number; y1: number },
  { h, w }: { h: number; w: number },
  slant: number,
  x: number,
  y: number
): boolean {
  const sx =
    box.x0 +
    Math.floor(((x - slant * (y - h / 2)) / w) * (box.x1 - box.x0 + 1));
  const sy = box.y0 + Math.floor((y / h) * (box.y1 - box.y0 + 1));
  return sx >= 0 && sx <= 27 && (src[sy * 28 + sx] ?? 0) >= 128;
}

/** MNIST digit `i` drawn on a tab line, slanted, with perhaps a pen's stem; its box, or undefined for a blank image. */
function drawDigit(
  canvas: Canvas,
  lineY: (k: number) => number,
  s: number,
  slot: number,
  count: number
): DigitBox | undefined {
  const i = pick(labels.length);
  const src = images.subarray(i * 784, (i + 1) * 784);
  const box = sourceBox(src);
  if (box === undefined) {
    return;
  }
  const h = Math.round(uni(0.9, 1.8) * s);
  const w = Math.max(
    3,
    Math.round(
      ((box.x1 - box.x0 + 1) / (box.y1 - box.y0 + 1)) * h * uni(0.8, 1.25)
    )
  );
  const cx = Math.round(30 + (slot + uni(0.1, 0.9)) * ((W - 70) / count));
  const cy = Math.round(lineY(pick(5)) + uni(-0.2, 0.2) * s);
  const slant = uni(-0.25, 0.25);
  const fat = s > 15 && rnd() < 0.5 ? 1 : 0;
  const left = cx - Math.floor(w / 2);
  const upper = cy - Math.floor(h / 2);
  const at = { h, left, upper, w };
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (inkAt(src, box, at, slant, x, y)) {
        canvas.dot(left + x, upper + y, fat);
      }
    }
  }
  // A pen's stem, hanging from the digit to below the tab, or rising above it.
  if (rnd() < 0.45) {
    const down = rnd() < 0.8;
    const sx0 = cx + Math.round(uni(-0.2, 0.2) * w);
    const from = down ? upper + h - 1 : upper;
    const to = down
      ? Math.round(lineY(4) + uni(1, 2.6) * s)
      : Math.round(lineY(0) - uni(1, 2) * s);
    canvas.stroke(
      sx0,
      from,
      sx0 + Math.round(uni(-0.12, 0.12) * (to - from)),
      to,
      1 + pick(2)
    );
  }
  return {
    label: labels[i] ?? 0,
    x0: left,
    x1: left + w,
    y0: upper,
    y1: upper + h,
  };
}

/** Not digits, but drawn where frets sit: a stem standing alone through the tab, a clawhammer brush arrow (a shaft, its head at a line) and a slur arc. */
function drawOthers(
  canvas: Canvas,
  lineY: (k: number) => number,
  s: number
): void {
  if (rnd() < 0.3) {
    const x = Math.round(uni(20, W - 20));
    canvas.stroke(
      x,
      Math.round(lineY(pick(3)) + uni(-0.3, 0.3) * s),
      x + Math.round(uni(-2, 2)),
      Math.round(lineY(4) + uni(1, 2.6) * s),
      1 + pick(2)
    );
  }
  if (rnd() < 0.5) {
    const x = Math.round(uni(20, W - 20));
    const tip = Math.round(lineY(pick(2)) - uni(0, 0.6) * s);
    const thick = 1 + pick(2);
    canvas.stroke(
      x,
      tip,
      x + Math.round(uni(-2, 2)),
      Math.round(lineY(4) + uni(0.5, 2) * s),
      thick
    );
    const arm = uni(0.3, 0.7) * s;
    canvas.stroke(
      x,
      tip,
      Math.round(x - arm),
      Math.round(tip + arm * uni(0.8, 1.4)),
      thick
    );
    canvas.stroke(
      x,
      tip,
      Math.round(x + arm),
      Math.round(tip + arm * uni(0.8, 1.4)),
      thick
    );
  }
  if (rnd() < 0.4) {
    const cx = uni(30, W - 30);
    const half = uni(0.6, 1.5) * s;
    const base = lineY(pick(5)) - uni(0.4, 1.0) * s;
    const rise = uni(0.2, 0.6) * s;
    const thick = 1 + pick(2);
    let px = cx - half;
    let py = base;
    for (let k = 1; k <= 20; k += 1) {
      const u = -1 + (2 * k) / 20;
      const x = cx + u * half;
      const y = base - rise * (1 - u * u);
      canvas.stroke(
        Math.round(px),
        Math.round(py),
        Math.round(x),
        Math.round(y),
        thick
      );
      px = x;
      py = y;
    }
  }
}

/** The digit a mark came from when it covers more than 0.6 of that digit's box, else not a digit. */
function labelOf(mark: Mark, boxes: readonly DigitBox[]): number {
  const mx = mark.x + mark.w / 2;
  const my = mark.y + mark.h / 2;
  const owner = boxes.find(
    (b) => mx >= b.x0 && mx <= b.x1 && my >= b.y0 && my <= b.y1
  );
  if (owner === undefined) {
    return NOT_A_DIGIT;
  }
  const overlap =
    ((Math.min(mark.x + mark.w, owner.x1) - Math.max(mark.x, owner.x0)) *
      (Math.min(mark.y + mark.h, owner.y1) - Math.max(mark.y, owner.y0))) /
    ((owner.x1 - owner.x0) * (owner.y1 - owner.y0));
  return overlap > 0.6 ? owner.label : NOT_A_DIGIT;
}

function syntheticPage(): void {
  const s = uni(10, 22);
  const t = Math.round(uni(1, 3.4));
  const top = 50;
  const canvas = new Canvas();
  const lines = Array.from({ length: 5 }, (_, k) => {
    const y = Math.round(top + k * s);
    const y0 = y - Math.floor((t - 1) / 2);
    for (let yy = y0; yy < y0 + t; yy += 1) {
      canvas.ink.fill(1, yy * W + 8, yy * W + W - 8);
    }
    return { breakLength: 0, x0: 8, x1: W - 9, y, y0, y1: y0 + t - 1 };
  });
  const group: LineGroup = { lines, spacing: s, x0: 8, x1: W - 9 };
  const lineY = (k: number) => lines[k]?.y ?? top;
  const boxes: DigitBox[] = [];
  const count = 1 + pick(3);
  for (let d = 0; d < count; d += 1) {
    const box = drawDigit(canvas, lineY, s, d, count);
    if (box !== undefined) {
      boxes.push(box);
    }
  }
  drawOthers(canvas, lineY, s);
  const page = createColor(W, H);
  page.data.fill(255);
  for (let i = 0; i < W * H; i += 1) {
    if (canvas.ink[i] === 1 && rnd() < 0.97) {
      page.data.fill(0, i * 3, i * 3 + 3);
    }
  }
  withMatScope((scope) => {
    const pageInk = new Uint8Array(inkOf(cv, scope, page).data);
    const lineless = inkWithoutLines(pageInk, W, H, group);
    for (const mark of marksOf(cv, pageInk, W, H, group, READ_MARK).marks) {
      const both = crops(page, pageInk, lineless, mark, s);
      keep(both[pick(2)] as ColorImage, labelOf(mark, boxes));
    }
  });
}

interface TruthSystem {
  readonly events: readonly {
    readonly x: number;
    readonly notes: readonly {
      readonly string: number;
      readonly fret: number;
    }[];
  }[];
}

interface TruthPage {
  readonly file: string;
  readonly systems: readonly TruthSystem[];
}

/** One typeset system's marks, labelled from the truth: a note's single-digit fret, or not a digit; a two-digit fret's marks left out. */
function fixtureSystem(
  page: ColorImage,
  ink: Uint8Array,
  group: LineGroup,
  system: TruthSystem
): void {
  const s = group.spacing;
  const lineless = inkWithoutLines(ink, page.width, page.height, group);
  const { marks, short } = marksOf(
    cv,
    ink,
    page.width,
    page.height,
    group,
    READ_MARK
  );
  for (const mark of [...marks, ...short]) {
    const mx = mark.x + mark.w / 2;
    const note = system.events
      .filter((e) => Math.abs(e.x - mx) < 0.6 * s)
      .flatMap((e) => e.notes)
      .find((n) => n.string === mark.line + 1);
    if (note !== undefined && note.fret >= 10) {
      continue;
    }
    const label = note === undefined ? NOT_A_DIGIT : note.fret;
    for (const crop of crops(page, ink, lineless, mark, s)) {
      keep(crop, label);
    }
  }
}

async function fixturePages(): Promise<void> {
  const dir = join(import.meta.dirname, "..", "test", "fixtures", "tab");
  const truth = JSON.parse(readFileSync(join(dir, "truth.json"), "utf8")) as {
    pages: readonly TruthPage[];
  };
  for (const entry of truth.pages) {
    const png = PNG.sync.read(readFileSync(join(dir, entry.file)));
    const page = colorImageFromRgba(png.width, png.height, png.data);
    // Every group on these pages is a tab: a reader that answers a digit for
    // every crop lets the guard keep them all.
    const tabs = await detectTablature(cv, page, (c) =>
      Promise.resolve(c.map(() => "1"))
    );
    withMatScope((scope) => {
      const ink = new Uint8Array(inkOf(cv, scope, page).data);
      for (const [k, tab] of tabs.entries()) {
        const system = entry.systems[k];
        if (system !== undefined) {
          fixtureSystem(page, ink, tab.group, system);
        }
      }
    });
  }
}

for (let p = 0; p < Number(pagesArg); p += 1) {
  syntheticPage();
}
if (withFixtures) {
  await fixturePages();
}
const all = new Float32Array(inputs.length * DIGIT_INPUT);
for (const [i, x] of inputs.entries()) {
  all.set(x, i * DIGIT_INPUT);
}
writeFileSync(`${out}-x.f32`, Buffer.from(all.buffer));
writeFileSync(`${out}-y.u8`, Buffer.from(Uint8Array.from(targets)));
const hist = new Array<number>(11).fill(0);
for (const y of targets) {
  hist[y] = (hist[y] ?? 0) + 1;
}
console.log(`${out}: ${targets.length} crops; per class ${hist.join(" ")}`);
process.exit(0);

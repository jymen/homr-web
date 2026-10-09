/**
 * The tuning and capo a tab page prints, read with RapidOCR around the tab
 * systems the guard found. Three places hold them on the pages measured
 * (docs/design/tab-3-findings.md): the header above the first tab ("aDADE
 * tuning", "Key Of A (Capo 2)", "Capo 2" over the first bar), a label in the
 * left margin of a system, often turned a quarter ("gDGBD" beside the TAB
 * clef), and a line under the first system ("DADGAD"). The header and the
 * line under the first system speak for the page; a margin label speaks for
 * its own system.
 */

import { type ColorImage, slicePlane } from "../image/plane.js";
import type { DetectionLimit, OcrLine } from "../ocr/rapid-ocr.js";
import type { DetectedTab } from "./detect.js";
import {
  type ParsedTuning,
  parseLetterLabels,
  parseTabText,
  resolveTuning,
  type TabCapo,
  type TabTuning,
} from "./tuning.js";

/** What a system's text says; either may be absent, and nothing is ever filled in. */
export interface TabText {
  readonly capo?: TabCapo;
  readonly tuning?: TabTuning;
}

/** Reads text lines off an image, boxes in the image's pixels: RapidOcr.read. */
export type ReadText = (
  image: ColorImage,
  minTextScore: number,
  limit: DetectionLimit
) => Promise<OcrLine[]>;

/**
 * Detection with the long side brought down to 1280. RapidOCR's default
 * brings the short side up to 736, which turns a 2000 by 100 strip into
 * 15 000 by 736 (2.3 s in Node for a strip with nothing in it). Header text
 * at 300 dpi stays 20 px tall or more at 1280; recognition still crops from
 * the region at RapidOCR's own scale (long side 2000). Measured on 23 pages:
 * the same readings at 2000, 1280 and 960; text 2.4 to 7.2 s a page at 2000,
 * 1.3 to 5.2 s at 1280, 1.0 to 3.5 s at 960.
 */
const LIMIT: DetectionLimit = { side: 1280, type: "max" };
const MIN_SCORE = 0.5;

/** Region bounds in line spacings of the tab they belong to. */
const AROUND = {
  /** The tab's own band, where the fret numbers sit: the header stops above it. */
  band: 0.75,
  /** The line under the first system reaches this far below its bottom line. */
  below: 3.5,
  /**
   * A margin reaches this far above the top line: a label centred on a staff
   * and its tab starts about three spacings above the tab (Blackberry
   * Blossom's "gDGBD"), and stops at the system above.
   */
  marginAbove: 5,
  marginBelow: 1,
  /** A margin stops this short of the lines, clear of a bracket or a bar. */
  marginGap: 0.25,
  /** A per-line string name sits within this of its line. */
  onLine: 0.4,
} as const;

type Scope = "page" | number;

interface Region {
  readonly scope: Scope;
  readonly x0: number;
  readonly x1: number;
  readonly y0: number;
  readonly y1: number;
}

interface Candidate {
  readonly confidence: number;
  readonly scope: Scope;
  readonly text: string;
}

interface TuningCandidate extends Candidate {
  readonly parsed: ParsedTuning;
}

interface CapoCandidate extends Candidate {
  readonly fret: number;
}

function regionsOf(page: ColorImage, tabs: readonly DetectedTab[]): Region[] {
  const [first] = tabs;
  if (first === undefined) {
    return [];
  }
  const s = first.group.spacing;
  const top = (first.group.lines[0]?.y ?? 0) - AROUND.band * s;
  const bottom = (first.group.lines.at(-1)?.y ?? 0) + AROUND.band * s;
  const regions: Region[] = [
    { scope: "page", x0: 0, x1: page.width, y0: 0, y1: top },
    {
      scope: "page",
      x0: 0,
      x1: page.width,
      y0: bottom,
      y1: bottom + (AROUND.below - AROUND.band) * s,
    },
  ];
  for (const [k, { group }] of tabs.entries()) {
    const previous = tabs[k - 1]?.group;
    const above =
      previous === undefined
        ? 0
        : (previous.lines.at(-1)?.y ?? 0) + AROUND.band * previous.spacing;
    regions.push({
      scope: k,
      x0: 0,
      x1: group.x0 - AROUND.marginGap * group.spacing,
      y0: Math.max(
        above,
        (group.lines[0]?.y ?? 0) - AROUND.marginAbove * group.spacing
      ),
      y1: (group.lines.at(-1)?.y ?? 0) + AROUND.marginBelow * group.spacing,
    });
  }
  return regions.map((r) => ({
    scope: r.scope,
    x0: Math.max(0, Math.round(r.x0)),
    x1: Math.min(page.width, Math.round(r.x1)),
    y0: Math.max(0, Math.round(r.y0)),
    y1: Math.min(page.height, Math.round(r.y1)),
  }));
}

/** Big enough for the detector to find a line of text in. */
const MIN_REGION = 16;

const centreY = (line: OcrLine, region: Region) =>
  region.y0 + line.box.reduce((sum, [, y]) => sum + y, 0) / line.box.length;

/**
 * String names printed one per line in a system's margin, top line first
 * ("e B G D A E"), when every line has exactly one.
 */
function letterLabels(
  lines: readonly OcrLine[],
  region: Region,
  tab: DetectedTab
): TuningCandidate | undefined {
  const { group } = tab;
  const perLine: (OcrLine | undefined)[] = group.lines.map(() => undefined);
  for (const line of lines) {
    const y = centreY(line, region);
    const k = group.lines.findIndex(
      (l) => Math.abs(l.y - y) < AROUND.onLine * group.spacing
    );
    if (k < 0 || perLine[k] !== undefined) {
      return;
    }
    perLine[k] = line;
  }
  if (perLine.some((line) => line === undefined)) {
    return;
  }
  const found = perLine as OcrLine[];
  const parsed = parseLetterLabels(found.map((line) => line.text));
  return parsed === undefined
    ? undefined
    : {
        confidence: Math.min(...found.map((line) => line.score)),
        parsed,
        scope: region.scope,
        text: found.map((line) => line.text).join(" "),
      };
}

const STATUS_RANK = { read: 2, string_count: 1, unknown_name: 0 } as const;

/** Best first: fits the system, from its own margin, printed letters over a name, then the recogniser's score. */
function rankOf(tuning: TabTuning, scope: Scope): number[] {
  return [
    STATUS_RANK[tuning.status],
    scope === "page" ? 0 : 1,
    tuning.status !== "unknown_name" && tuning.source === "text" ? 1 : 0,
    tuning.confidence,
  ];
}

const compareRanks = (a: readonly number[], b: readonly number[]): number => {
  for (const [k, value] of a.entries()) {
    const other = b[k] ?? 0;
    if (value !== other) {
      return other - value;
    }
  }
  return 0;
};

/**
 * The candidates that speak for system `index`: the page's, and those of the
 * nearest margin at or above it that says anything. A margin label holds
 * until the next one, as an instrument label does: Guitar Pro prints the
 * tuning beside the first system only.
 */
function speakingFor<T extends Candidate>(
  candidates: readonly T[],
  index: number
): T[] {
  const margin = Math.max(
    -1,
    ...candidates.flatMap((c) =>
      c.scope !== "page" && c.scope <= index ? [c.scope] : []
    )
  );
  return candidates.filter((c) => c.scope === "page" || c.scope === margin);
}

function tuningFor(
  candidates: readonly TuningCandidate[],
  index: number,
  tab: DetectedTab
): TabTuning | undefined {
  return speakingFor(candidates, index)
    .map((c) => ({
      scope: c.scope,
      tuning: resolveTuning(c.parsed, tab.system.lines, c.text, c.confidence),
    }))
    .sort((a, b) =>
      compareRanks(rankOf(a.tuning, a.scope), rankOf(b.tuning, b.scope))
    )[0]?.tuning;
}

/** A margin's capo over the page's, then the recogniser's score. */
function capoFor(
  candidates: readonly CapoCandidate[],
  index: number
): TabCapo | undefined {
  const [best] = speakingFor(candidates, index).sort(
    (a, b) =>
      (a.scope === "page" ? 1 : 0) - (b.scope === "page" ? 1 : 0) ||
      b.confidence - a.confidence
  );
  return best === undefined
    ? undefined
    : { confidence: best.confidence, fret: best.fret, text: best.text };
}

/**
 * Each tab's tuning and capo as the page prints them, in the tabs' order.
 * One recogniser pass per region: the header, the line under the first
 * system and every system's margin.
 */
export async function readTabText(
  read: ReadText,
  page: ColorImage,
  tabs: readonly DetectedTab[]
): Promise<TabText[]> {
  const tunings: TuningCandidate[] = [];
  const capos: CapoCandidate[] = [];
  for (const region of regionsOf(page, tabs)) {
    if (
      region.x1 - region.x0 < MIN_REGION ||
      region.y1 - region.y0 < MIN_REGION
    ) {
      continue;
    }
    // biome-ignore lint/performance/noAwaitInLoops: one OCR session, one region at a time
    const lines = await read(
      slicePlane(page, region.x0, region.y0, region.x1, region.y1),
      MIN_SCORE,
      LIMIT
    );
    for (const line of lines) {
      const said = parseTabText(line.text);
      const base = {
        confidence: line.score,
        scope: region.scope,
        text: line.text,
      };
      if (said.tuning !== undefined) {
        tunings.push({ ...base, parsed: said.tuning });
      }
      if (said.capo !== undefined) {
        capos.push({ ...base, fret: said.capo });
      }
    }
    const tab = region.scope === "page" ? undefined : tabs[region.scope];
    const labels =
      tab === undefined ? undefined : letterLabels(lines, region, tab);
    if (labels !== undefined) {
      tunings.push(labels);
    }
  }
  return tabs.map((tab, index) => {
    const tuning = tuningFor(tunings, index, tab);
    const capo = capoFor(capos, index);
    return {
      ...(capo === undefined ? {} : { capo }),
      ...(tuning === undefined ? {} : { tuning }),
    };
  });
}

/**
 * homr's process_image without file IO or debug images, then the app server's
 * chord OCR: one BGR page in, the app server's result shape out. The Worker
 * and the Node golden test both call `recognizePage`; neither has to know
 * what a stage is. Before homr, the tab guard reads line tablature's frets
 * and paints it white, so homr never reads a tab as a staff.
 */

import type { OpenCv } from "../cv/opencv.js";
import type { ColorImage } from "../image/plane.js";
import { createInputPredictions, DetectionError } from "../model/pipeline.js";
import type { Staff } from "../model/staff.js";
import { ModelError } from "../models/errors.js";
import type { ModelRole } from "../models/manifest.js";
import type { ModelSession } from "../models/session.js";
import { generateMusicXml } from "../musicxml/generate.js";
import { ctcCharacters } from "../ocr/ctc.js";
import { detectTitle, readStripTexts } from "../ocr/page.js";
import { RapidOcr, recognizeCrops } from "../ocr/rapid-ocr.js";
import {
  type Backend,
  failedResult,
  isTimeoutAbort,
  type PageText,
  type Progress,
  type RecognizeError,
  type RecognizeResult,
  type StaffBox,
  tablatureOnlyResult,
} from "../result.js";
import { preprocessPage } from "../segmentation/preprocess.js";
import { DEFAULT_SEGNET_BATCH, segmentPage } from "../segmentation/segment.js";
import {
  type DetectedTab,
  detectTablature,
  type ReadCrops,
  whitenTablature,
} from "../tab/detect.js";
import { readTablature, type TabReading } from "../tab/read.js";
import { detectStaffsInImage } from "./detect.js";
import { parseStaffs } from "./parse-staffs.js";
import { staffBoxes, staffPositions } from "./staff-positions.js";

/** What a page is read with. `open` is lazy, so a page that is not music never opens the transformer. */
export interface RecognizeEngine {
  readonly backend: Backend;
  readonly cv: OpenCv;
  /** `signal` is the page's: a cancel during the first download stops it. */
  readonly open: (
    role: ModelRole,
    batch?: number,
    signal?: AbortSignal
  ) => Promise<ModelSession>;
}

export interface PipelineOptions {
  /** RapidOCR's text_score for the chord strips: lines scoring below it are dropped. Default 0.5, the server's. The title keeps homr's 0.5. */
  readonly minTextScore?: number;
  /** Read the chord strips and the title. Default true. */
  readonly ocr?: boolean;
  readonly onProgress?: (progress: Progress) => void;
  /** Checked between stages and before every decoder step. */
  readonly signal?: AbortSignal;
}

const NOT_MUSIC: ReadonlySet<string> = new Set(["no-noteheads", "no-staffs"]);
const MODELS_MISSING: ReadonlySet<string> = new Set([
  "digest-mismatch",
  "fetch-failed",
]);

/**
 * The code for an error out of the pipeline. A detection failure homr itself
 * crashes on is `engine_failed`, as the server reports it; only a page with
 * no staff or no notehead is `not_music`.
 */
export function classifyFailure(
  cause: unknown,
  signal: AbortSignal | undefined
): Exclude<RecognizeError, "tablature_only"> {
  if (signal?.aborted) {
    return isTimeoutAbort(signal) ? "timeout" : "cancelled";
  }
  if (cause instanceof DetectionError) {
    return NOT_MUSIC.has(cause.code) ? "not_music" : "engine_failed";
  }
  if (cause instanceof ModelError) {
    return MODELS_MISSING.has(cause.code) ? "engine_missing" : "engine_failed";
  }
  return "engine_failed";
}

export const describeError = (cause: unknown): string =>
  cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause);

async function openOcr(
  engine: RecognizeEngine,
  signal: AbortSignal | undefined
): Promise<RapidOcr> {
  const detect = await engine.open("ocrDetect", undefined, signal);
  const classify = await engine.open("ocrClassify", undefined, signal);
  const recognize = await engine.open("ocrRecognize", undefined, signal);
  return new RapidOcr(engine.cv, { classify, detect, recognize });
}

interface PageForText {
  readonly page: ColorImage;
  readonly staves: readonly StaffBox[];
  /** homr's resized page and first staff, when the title is read too. */
  readonly title?: { readonly resized: ColorImage; readonly topStaff: Staff };
}

/**
 * The chord strips and, given the first staff, the title. Each fails alone,
 * into an empty result and a log line, as the server's OCR never fails a job;
 * a cancel still cancels.
 */
async function readPageText(
  engine: RecognizeEngine,
  { page, staves, title: titleBand }: PageForText,
  log: string[],
  { minTextScore, onProgress, signal }: PipelineOptions
): Promise<{ readonly texts: PageText[]; readonly title: string }> {
  const total = staves.length + (titleBand === undefined ? 0 : 1);
  const report = (done: number) => onProgress?.({ done, stage: "ocr", total });
  signal?.throwIfAborted();
  const ocr = await openOcr(engine, signal);
  report(0);
  const attempt = async <T>(what: string, read: () => Promise<T>, empty: T) => {
    try {
      return await read();
    } catch (cause) {
      signal?.throwIfAborted();
      log.push(`${what} failed: ${describeError(cause)}`);
      return empty;
    }
  };
  const texts = await attempt(
    "chord OCR",
    () =>
      readStripTexts(ocr, page, staves, {
        ...(minTextScore === undefined ? {} : { minTextScore }),
        onStrip: (done) => {
          signal?.throwIfAborted();
          report(done);
        },
      }),
    [] as PageText[]
  );
  if (titleBand === undefined) {
    return { texts, title: "" };
  }
  const title = await attempt(
    "title detection",
    () => detectTitle(ocr, titleBand.resized, titleBand.topStaff),
    ""
  );
  report(total);
  return { texts, title };
}

/**
 * Rows the guard paints white beyond a tab's outer lines, in line spacings:
 * the fret numbers on them reach half a spacing out. Measured on the
 * TablEdit Cripple Creek page, where it turns homr's four grand staves (the
 * tab read as a bass staff) into the four treble staves.
 */
const TAB_MARGIN = 0.75;

/** RapidOCR's recogniser, opened on the first five-line group that needs it, read text only. */
function digitReader(
  engine: RecognizeEngine,
  signal: AbortSignal | undefined
): ReadCrops {
  let opened:
    | Promise<{ characters: readonly string[]; session: ModelSession }>
    | undefined;
  return async (crops) => {
    opened ??= engine
      .open("ocrRecognize", undefined, signal)
      .then((recognizer) => ({
        characters: ctcCharacters(recognizer.metadata.get("character") ?? ""),
        session: recognizer,
      }));
    const { characters, session } = await opened;
    signal?.throwIfAborted();
    const read = await recognizeCrops(engine.cv, session, characters, crops);
    return read.map((text) => text.text);
  };
}

async function readPage(
  page: ColorImage,
  engine: RecognizeEngine,
  log: string[],
  options: PipelineOptions
): Promise<
  Omit<
    RecognizeResult & { ok: true },
    "backend" | "durationMs" | "log" | "tablature"
  >
> {
  const { onProgress, signal } = options;
  const { cv } = engine;
  const report = (stage: Progress["stage"], done: number, total: number) =>
    onProgress?.({ done, stage, total });

  signal?.throwIfAborted();
  const { preprocessed, resized } = await preprocessPage(page, cv);
  const segnet = await engine.open("segnet", DEFAULT_SEGNET_BATCH, signal);
  signal?.throwIfAborted();
  const { masks } = await segmentPage(segnet, preprocessed, {
    batch: DEFAULT_SEGNET_BATCH,
    onProgress: (done, total) => report("segment", done, total),
  });

  signal?.throwIfAborted();
  report("detect", 0, 1);
  const detection = detectStaffsInImage(
    cv,
    createInputPredictions(resized, preprocessed, masks)
  );
  report("detect", 1, 1);
  const staves = staffBoxes(
    staffPositions(detection.multiStaffs, detection.preprocessed)
  );

  const encoder = await engine.open("encoder", undefined, signal);
  const decoder = await engine.open("decoder", undefined, signal);
  const voices = await parseStaffs(
    cv,
    { decoder, encoder },
    detection.multiStaffs,
    detection.preprocessed,
    { onStaff: report, ...(signal === undefined ? {} : { signal }) }
  );

  const { texts, title } =
    options.ocr === false
      ? { texts: [], title: "" }
      : await readPageText(
          engine,
          {
            page,
            staves,
            title: { resized, topStaff: detection.topStaff },
          },
          log,
          options
        );

  report("xml", 0, 1);
  const musicXml = generateMusicXml(voices, title, (line) => log.push(line));
  report("xml", 1, 1);
  log.push(`Finished parsing ${voices.length} staves`);
  return {
    engine: "browser",
    error: "",
    musicXml,
    ok: true,
    staves,
    texts,
  };
}

const hasTabs = (
  tablature: readonly TabReading[]
): tablature is readonly [TabReading, ...TabReading[]] => tablature.length > 0;

/**
 * Never throws: every failure is `ok: false` with its reason as the last log
 * line. A page homr finds no staff on once its tabs are painted out is
 * `tablature_only`, not `not_music`.
 */
export async function recognizePage(
  page: ColorImage,
  engine: RecognizeEngine,
  options: PipelineOptions = {}
): Promise<RecognizeResult> {
  const started = performance.now();
  const log: string[] = [];
  const elapsed = () => Math.round(performance.now() - started);
  let tablature: TabReading[] = [];
  try {
    options.signal?.throwIfAborted();
    const read = digitReader(engine, options.signal);
    const tabs: DetectedTab[] = await detectTablature(engine.cv, page, read);
    tablature = await readTablature(
      engine.cv,
      page,
      tabs,
      read,
      (done, total) => options.onProgress?.({ done, stage: "tab", total })
    );
    if (hasTabs(tablature)) {
      log.push(
        `Tab guard: ${tabs.length} tablature systems (${tablature.map((t) => t.lines).join(", ")} lines) painted out`,
        `Tab reader: ${tablature.map((t) => t.events.length).join(", ")} events`
      );
    }
    const reading = await readPage(
      tabs.length === 0 ? page : whitenTablature(page, tabs, TAB_MARGIN),
      engine,
      log,
      options
    );
    return {
      ...reading,
      backend: engine.backend,
      durationMs: elapsed(),
      log: log.join("\n"),
      tablature,
    };
  } catch (cause) {
    log.push(describeError(cause));
    const error = classifyFailure(cause, options.signal);
    if (error === "not_music" && hasTabs(tablature)) {
      log.push("Every system on the page is tablature");
      return tablatureOnlyResult(
        engine.backend,
        tablature,
        log.join("\n"),
        elapsed()
      );
    }
    return failedResult(
      engine.backend,
      classifyFailure(cause, options.signal),
      log.join("\n"),
      Math.round(performance.now() - started)
    );
  }
}

/**
 * The chord strips alone, above staves the caller already has (the server's,
 * or an earlier page result's): `musicXml` is empty and `staves` are the ones
 * given. Never throws.
 */
export async function recognizeTexts(
  page: ColorImage,
  staves: readonly StaffBox[],
  engine: RecognizeEngine,
  options: PipelineOptions = {}
): Promise<RecognizeResult> {
  const started = performance.now();
  const log: string[] = [];
  try {
    const { texts } = await readPageText(
      engine,
      { page, staves },
      log,
      options
    );
    log.push(`Read ${texts.length} texts above ${staves.length} staves`);
    return {
      backend: engine.backend,
      durationMs: Math.round(performance.now() - started),
      engine: "browser",
      error: "",
      log: log.join("\n"),
      musicXml: "",
      ok: true,
      staves: [...staves],
      tablature: [],
      texts,
    };
  } catch (cause) {
    log.push(describeError(cause));
    return failedResult(
      engine.backend,
      classifyFailure(cause, options.signal),
      log.join("\n"),
      Math.round(performance.now() - started)
    );
  }
}

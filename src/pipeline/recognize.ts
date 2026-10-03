/**
 * homr's process_image without file IO, title detection or debug images: one
 * BGR page in, the app server's result shape out. The Worker and the Node
 * golden test both call `recognizePage`; neither has to know what a stage is.
 */

import type { OpenCv } from "../cv/opencv.js";
import type { ColorImage } from "../image/plane.js";
import { createInputPredictions, DetectionError } from "../model/pipeline.js";
import { ModelError } from "../models/errors.js";
import type { ModelRole } from "../models/manifest.js";
import type { ModelSession } from "../models/session.js";
import { generateMusicXml } from "../musicxml/generate.js";
import {
  type Backend,
  failedResult,
  type Progress,
  type RecognizeError,
  type RecognizeResult,
} from "../result.js";
import { preprocessPage } from "../segmentation/preprocess.js";
import { DEFAULT_SEGNET_BATCH, segmentPage } from "../segmentation/segment.js";
import { detectStaffsInImage } from "./detect.js";
import { parseStaffs } from "./parse-staffs.js";
import { staffBoxes, staffPositions } from "./staff-positions.js";

/** What a page is read with. `open` is lazy, so a page that is not music never opens the transformer. */
export interface RecognizeEngine {
  readonly backend: Backend;
  readonly cv: OpenCv;
  readonly open: (role: ModelRole, batch?: number) => Promise<ModelSession>;
}

export interface PipelineOptions {
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
): RecognizeError {
  if (signal?.aborted) {
    const reason: unknown = signal.reason;
    return reason instanceof Error && reason.name === "TimeoutError"
      ? "timeout"
      : "cancelled";
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

async function readPage(
  page: ColorImage,
  engine: RecognizeEngine,
  log: string[],
  { onProgress, signal }: PipelineOptions
): Promise<
  Omit<RecognizeResult & { ok: true }, "backend" | "durationMs" | "log">
> {
  const { cv } = engine;
  const report = (stage: Progress["stage"], done: number, total: number) =>
    onProgress?.({ done, stage, total });

  signal?.throwIfAborted();
  const { preprocessed, resized } = await preprocessPage(page, cv);
  const segnet = await engine.open("segnet", DEFAULT_SEGNET_BATCH);
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

  const encoder = await engine.open("encoder");
  const decoder = await engine.open("decoder");
  const voices = await parseStaffs(
    cv,
    { decoder, encoder },
    detection.multiStaffs,
    detection.preprocessed,
    { onStaff: report, ...(signal === undefined ? {} : { signal }) }
  );

  report("xml", 0, 1);
  const musicXml = generateMusicXml(voices, "", (line) => log.push(line));
  report("xml", 1, 1);
  log.push(`Finished parsing ${voices.length} staves`);
  return {
    engine: "browser",
    error: "",
    musicXml,
    ok: true,
    staves,
    texts: [],
  };
}

/** Never throws: every failure is `ok: false` with its reason as the last log line. */
export async function recognizePage(
  page: ColorImage,
  engine: RecognizeEngine,
  options: PipelineOptions = {}
): Promise<RecognizeResult> {
  const started = performance.now();
  const log: string[] = [];
  try {
    const read = await readPage(page, engine, log, options);
    return {
      ...read,
      backend: engine.backend,
      durationMs: Math.round(performance.now() - started),
      log: log.join("\n"),
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

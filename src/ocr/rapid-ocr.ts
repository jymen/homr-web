/**
 * RapidOCR 3.9.2's RapidOCR.__call__ with its default configuration
 * (rapidocr/config.yaml), on a BGR image: PP-OCRv6 small detection,
 * the PP-OCRv4 mobile direction classifier, PP-OCRv6 small recognition.
 * Each step is exported for tests/ocr-golden.test.ts, which checks it against
 * the stage tools/dump-ocr.py wrote.
 */

import type { Mat } from "@techstark/opencv-js";
import { Tensor } from "onnxruntime-web";
import { type MatScope, type OpenCv, withMatScope } from "../cv/opencv.js";
import { npArgsort } from "../image/argsort.js";
import { roundHalfEven, toFloat32 } from "../image/numeric.js";
import { type ColorImage, planeFromBytes } from "../image/plane.js";
import { ModelError } from "../models/errors.js";
import { type ModelSession, readFloat32 } from "../models/session.js";
import { ctcCharacters, ctcDecode, type RecognizedText } from "./ctc.js";
import { detectionBoxes, type Quad } from "./db-postprocess.js";

export const RAPID_OCR = {
  clsBatch: 6,
  clsShape: { height: 48, width: 192 },
  clsThresh: 0.9,
  detLimitSideLen: 736,
  maxSideLen: 2000,
  minHeight: 30,
  minSideLen: 30,
  recBatch: 6,
  recShape: { height: 48, width: 320 },
  textScore: 0.5,
  widthHeightRatio: 8,
} as const;

/** One text line: its corners in the input image, as RapidOCR returns them. */
export interface OcrLine {
  readonly box: Quad;
  readonly score: number;
  readonly text: string;
}

export interface OcrSessions {
  readonly classify: ModelSession;
  readonly detect: ModelSession;
  readonly recognize: ModelSession;
}

function toMat(cv: OpenCv, scope: MatScope, image: ColorImage): Mat {
  const mat = scope.keep(new cv.Mat(image.height, image.width, cv.CV_8UC3));
  mat.data.set(image.data);
  return mat;
}

const fromMat = (mat: Mat): ColorImage =>
  planeFromBytes("bgr", mat.cols, mat.rows, Uint8Array.from(mat.data));

function resized(
  cv: OpenCv,
  image: ColorImage,
  width: number,
  height: number
): ColorImage {
  return withMatScope((scope) => {
    const dst = scope.keep(new cv.Mat());
    cv.resize(
      toMat(cv, scope, image),
      dst,
      new cv.Size(width, height),
      0,
      0,
      cv.INTER_LINEAR
    );
    return fromMat(dst);
  });
}

/** `int(round(int(side * ratio) / 32) * 32)`, the multiple of 32 every RapidOCR resize lands on. */
const multipleOf32 = (side: number, ratio: number): number =>
  roundHalfEven(Math.trunc(side * ratio) / 32) * 32;

export interface Preprocessed {
  readonly image: ColorImage;
  readonly ratioH: number;
  readonly ratioW: number;
}

/** resize_image_within_bounds: the long side down to 2000, then the short side up to 30. */
export function preprocessImage(cv: OpenCv, image: ColorImage): Preprocessed {
  let current = image;
  let ratioH = 1;
  let ratioW = 1;
  const rescale = (ratio: number) => {
    const { height, width } = current;
    const h = multipleOf32(height, ratio);
    const w = multipleOf32(width, ratio);
    if (h <= 0 || w <= 0) {
      throw new ModelError(
        "unknown-tensor",
        `OCR input ${width}x${height} resizes to nothing`
      );
    }
    current = resized(cv, current, w, h);
    ratioH = height / h;
    ratioW = width / w;
  };
  if (Math.max(current.height, current.width) > RAPID_OCR.maxSideLen) {
    const { height, width } = current;
    rescale(RAPID_OCR.maxSideLen / (height > width ? height : width));
  }
  if (Math.min(current.height, current.width) < RAPID_OCR.minSideLen) {
    const { height, width } = current;
    rescale(RAPID_OCR.minSideLen / (height < width ? height : width));
  }
  return { image: current, ratioH, ratioW };
}

/** apply_vertical_padding: a black band above and below a wide or short image. */
export function verticalPadding(
  cv: OpenCv,
  image: ColorImage
): { readonly image: ColorImage; readonly top: number } {
  const { height, width } = image;
  if (
    !(
      height <= RAPID_OCR.minHeight ||
      width / height > RAPID_OCR.widthHeightRatio
    )
  ) {
    return { image, top: 0 };
  }
  const target =
    Math.max(
      Math.trunc(width / RAPID_OCR.widthHeightRatio),
      RAPID_OCR.minHeight
    ) * 2;
  const top = Math.trunc(Math.abs(target - height) / 2);
  return withMatScope((scope) => {
    const dst = scope.keep(new cv.Mat());
    cv.copyMakeBorder(
      toMat(cv, scope, image),
      dst,
      top,
      top,
      0,
      0,
      cv.BORDER_CONSTANT,
      new cv.Scalar(0, 0, 0, 0)
    );
    return { image: fromMat(dst), top };
  });
}

/**
 * RapidOCR's Det.limit_type and Det.limit_side_len. The default, "min" 736,
 * brings the short side up to 736, which on a wide strip of a page (2000 by
 * 100) makes an image of 15 000 by 736 to detect on; "max" only brings the
 * long side down.
 */
export interface DetectionLimit {
  readonly side: number;
  readonly type: "max" | "min";
}

export const DEFAULT_DETECTION_LIMIT: DetectionLimit = {
  side: RAPID_OCR.detLimitSideLen,
  type: "min",
};

/** DetPreProcess: the short side up to (or the long side down to) the limit, both sides to multiples of 32, then (x/255 - 0.5)/0.5 in CHW. */
export function detectionInput(
  cv: OpenCv,
  image: ColorImage,
  limit: DetectionLimit = DEFAULT_DETECTION_LIMIT
): Tensor {
  const { height, width } = image;
  const side =
    limit.type === "min" ? Math.min(height, width) : Math.max(height, width);
  const beyond = limit.type === "min" ? side < limit.side : side > limit.side;
  const ratio = beyond ? limit.side / side : 1;
  const h = multipleOf32(height, ratio);
  const w = multipleOf32(width, ratio);
  const input = resized(cv, image, w, h);
  const plane = h * w;
  const data = new Float32Array(3 * plane);
  const scale = toFloat32(1 / 255);
  for (let i = 0; i < plane; i += 1) {
    for (let c = 0; c < 3; c += 1) {
      const scaled = toFloat32((input.data[i * 3 + c] ?? 0) * scale);
      data[c * plane + i] = (scaled - 0.5) / 0.5;
    }
  }
  return new Tensor("float32", data, [1, 3, h, w]);
}

const pointDistance = (
  a: readonly [number, number],
  b: readonly [number, number]
) =>
  toFloat32(
    Math.sqrt(
      toFloat32(toFloat32((a[0] - b[0]) ** 2) + toFloat32((a[1] - b[1]) ** 2))
    )
  );

/** get_rotate_crop_image: the quadrilateral warped upright, bicubic, edges replicated; a tall result turned a quarter left. */
export function cropText(cv: OpenCv, image: ColorImage, box: Quad): ColorImage {
  const width = Math.trunc(
    Math.max(pointDistance(box[0], box[1]), pointDistance(box[2], box[3]))
  );
  const height = Math.trunc(
    Math.max(pointDistance(box[0], box[3]), pointDistance(box[1], box[2]))
  );
  return withMatScope((scope) => {
    const source = scope.keep(cv.matFromArray(4, 1, cv.CV_32FC2, box.flat()));
    const target = scope.keep(
      cv.matFromArray(4, 1, cv.CV_32FC2, [
        0,
        0,
        width,
        0,
        width,
        height,
        0,
        height,
      ])
    );
    const transform = scope.keep(cv.getPerspectiveTransform(source, target));
    const warped = scope.keep(new cv.Mat());
    cv.warpPerspective(
      toMat(cv, scope, image),
      warped,
      transform,
      new cv.Size(width, height),
      cv.INTER_CUBIC,
      cv.BORDER_REPLICATE,
      new cv.Scalar()
    );
    if (warped.rows / warped.cols < 1.5) {
      return fromMat(warped);
    }
    const turned = scope.keep(new cv.Mat());
    cv.rotate(warped, turned, cv.ROTATE_90_COUNTERCLOCKWISE);
    return fromMat(turned);
  });
}

function rotated180(cv: OpenCv, image: ColorImage): ColorImage {
  return withMatScope((scope) => {
    const dst = scope.keep(new cv.Mat());
    cv.rotate(toMat(cv, scope, image), dst, cv.ROTATE_180);
    return fromMat(dst);
  });
}

/**
 * The classifier's and the recogniser's resize_norm_img: height 48, the width
 * the aspect ratio gives up to `canvasWidth`, (x/255 - 0.5)/0.5 in float32,
 * zeros to the right.
 */
function normalizedLine(
  cv: OpenCv,
  image: ColorImage,
  canvasWidth: number,
  into: Float32Array,
  offset: number
): void {
  const canvasHeight = RAPID_OCR.recShape.height;
  const ratio = image.width / image.height;
  const fitted = Math.ceil(canvasHeight * ratio);
  const width = fitted > canvasWidth ? canvasWidth : fitted;
  const line = resized(cv, image, width, canvasHeight);
  const plane = canvasHeight * canvasWidth;
  for (let y = 0; y < canvasHeight; y += 1) {
    for (let x = 0; x < width; x += 1) {
      for (let c = 0; c < 3; c += 1) {
        const value = line.data[(y * width + x) * 3 + c] ?? 0;
        into[offset + c * plane + y * canvasWidth + x] = toFloat32(
          toFloat32(toFloat32(value / 255) - 0.5) / 0.5
        );
      }
    }
  }
}

/** `np.argsort` of the width-to-height ratios: numpy's own quicksort, so equal ratios batch as they do in Python. */
const byAspect = (images: readonly ColorImage[]): Int32Array =>
  npArgsort(images.map((image) => image.width / image.height));

/**
 * A turn of the event loop. onnxruntime-web's wasm runs settle as microtasks,
 * so a page's OCR, a hundred runs or so, would otherwise hold off every
 * message to its realm: a Worker's cancel, and vitest's own RPC, which times
 * out after 60 s (measured: test/ocr-golden.test.ts alone tripped it).
 */
const nextTask = (): Promise<void> =>
  new Promise((resolve) => {
    setTimeout(resolve, 0);
  });

async function runSingle(
  session: ModelSession,
  input: Tensor
): Promise<{
  readonly data: Float32Array;
  readonly dims: readonly number[];
}> {
  const [output] = session.plan.artifact.outputs;
  const [inputSpec] = session.plan.artifact.inputs;
  if (output === undefined || inputSpec === undefined) {
    throw new ModelError(
      "manifest",
      `${session.role} declares no input or output`
    );
  }
  const outputs = await session.session.run({ [inputSpec.name]: input });
  await nextTask();
  const tensor = outputs[output.name];
  if (tensor === undefined) {
    throw new ModelError(
      "unknown-tensor",
      `the ${session.role} run produced no ${output.name}`,
      { expected: output.name, id: session.plan.artifactId }
    );
  }
  try {
    return {
      data: Float32Array.from(await readFloat32(tensor)),
      dims: tensor.dims,
    };
  } finally {
    tensor.dispose();
    input.dispose();
  }
}

/** TextClassifier: label "0" or "180" and its probability per crop; a "180" above 0.9 is turned. */
export async function classifyCrops(
  cv: OpenCv,
  session: ModelSession,
  crops: readonly ColorImage[]
): Promise<{
  readonly crops: ColorImage[];
  readonly labels: [string, number][];
}> {
  const order = byAspect(crops);
  const out = [...crops];
  const labels: [string, number][] = crops.map(() => ["", 0]);
  const { height, width } = RAPID_OCR.clsShape;
  const threshold = toFloat32(RAPID_OCR.clsThresh);
  for (let start = 0; start < crops.length; start += RAPID_OCR.clsBatch) {
    const batch = Array.from(order.subarray(start, start + RAPID_OCR.clsBatch));
    const data = new Float32Array(batch.length * 3 * height * width);
    for (const [slot, index] of batch.entries()) {
      const crop = crops[index];
      if (crop !== undefined) {
        normalizedLine(cv, crop, width, data, slot * 3 * height * width);
      }
    }
    const { data: probabilities } = await runSingle(
      session,
      new Tensor("float32", data, [batch.length, 3, height, width])
    );
    for (const [slot, index] of batch.entries()) {
      const zero = probabilities[slot * 2] ?? 0;
      const half = probabilities[slot * 2 + 1] ?? 0;
      const turned = half > zero;
      const score = turned ? half : zero;
      labels[index] = [turned ? "180" : "0", score];
      const crop = out[index];
      if (turned && score > threshold && crop !== undefined) {
        out[index] = rotated180(cv, crop);
      }
    }
  }
  return { crops: out, labels };
}

/** TextRecognizer: batches of six by aspect, each padded to its widest member's ratio, CTC-decoded. */
export async function recognizeCrops(
  cv: OpenCv,
  session: ModelSession,
  characters: readonly string[],
  crops: readonly ColorImage[]
): Promise<RecognizedText[]> {
  const order = byAspect(crops);
  const results: RecognizedText[] = crops.map(() => ({ score: 0, text: "" }));
  const { height, width } = RAPID_OCR.recShape;
  for (let start = 0; start < crops.length; start += RAPID_OCR.recBatch) {
    const batch = Array.from(order.subarray(start, start + RAPID_OCR.recBatch));
    let maxRatio = width / height;
    for (const index of batch) {
      const crop = crops[index];
      if (crop !== undefined) {
        maxRatio = Math.max(maxRatio, crop.width / crop.height);
      }
    }
    const canvasWidth = Math.trunc(height * maxRatio);
    const size = 3 * height * canvasWidth;
    const data = new Float32Array(batch.length * size);
    for (const [slot, index] of batch.entries()) {
      const crop = crops[index];
      if (crop !== undefined) {
        normalizedLine(cv, crop, canvasWidth, data, slot * size);
      }
    }
    const { data: probabilities, dims } = await runSingle(
      session,
      new Tensor("float32", data, [batch.length, 3, height, canvasWidth])
    );
    const steps = dims[1] ?? 0;
    const classes = dims[2] ?? 0;
    for (const [slot, index] of batch.entries()) {
      results[index] = ctcDecode(
        probabilities,
        slot * steps * classes,
        steps,
        classes,
        characters
      );
    }
  }
  return results;
}

/** Python's str.isspace set, which `strip()` removes; JavaScript's \s differs on \x1c-\x1f and \ufeff. */
const PYTHON_BLANK =
  // biome-ignore lint/suspicious/noControlCharactersInRegex: \x1c to \x1f are whitespace to Python's str.strip.
  /^[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]*$/;

/** map_boxes_to_original: the padding taken off, the preprocess ratio undone, clipped to the image. numpy casts the Python-float ratio to float32 before multiplying. */
function toOriginal(
  box: Quad,
  top: number,
  pre: Preprocessed,
  height: number,
  width: number
): Quad {
  return box.map(([x, y]) => {
    const ox = Math.max(toFloat32(x * toFloat32(pre.ratioW)), 0);
    const oy = Math.max(
      toFloat32(toFloat32(y - top) * toFloat32(pre.ratioH)),
      0
    );
    return [Math.min(ox, width), Math.min(oy, height)] as const;
  }) as unknown as Quad;
}

/** What one image passed through, for the per-stage golden test. */
export interface OcrTrace {
  readonly boxes: readonly Quad[];
  readonly crops: readonly ColorImage[];
  readonly labels: readonly [string, number][];
  readonly padded: ColorImage;
  readonly paddingTop: number;
  readonly preprocessed: Preprocessed;
  readonly recognized: readonly RecognizedText[];
}

/** RapidOCR's engine with its three models open: one image in, its text lines out, as `engine(img)` answers. */
export class RapidOcr {
  readonly #characters: readonly string[];
  readonly #cv: OpenCv;
  readonly #sessions: OcrSessions;

  constructor(cv: OpenCv, sessions: OcrSessions) {
    const alphabet = sessions.recognize.metadata.get("character");
    if (alphabet === undefined) {
      throw new ModelError(
        "manifest",
        "the recognition model carries no character metadata",
        { id: sessions.recognize.plan.artifactId }
      );
    }
    this.#characters = ctcCharacters(alphabet);
    this.#cv = cv;
    this.#sessions = sessions;
  }

  /** `minTextScore` is RapidOCR's Global.text_score: lines scoring below it are dropped. */
  async read(
    image: ColorImage,
    minTextScore: number = RAPID_OCR.textScore,
    limit: DetectionLimit = DEFAULT_DETECTION_LIMIT
  ): Promise<OcrLine[]> {
    return (await this.trace(image, minTextScore, limit)).lines;
  }

  async trace(
    image: ColorImage,
    minTextScore: number = RAPID_OCR.textScore,
    limit: DetectionLimit = DEFAULT_DETECTION_LIMIT
  ): Promise<{ readonly lines: OcrLine[]; readonly trace: OcrTrace }> {
    const cv = this.#cv;
    const preprocessed = preprocessImage(cv, image);
    const { image: padded, top } = verticalPadding(cv, preprocessed.image);
    const map = await runSingle(
      this.#sessions.detect,
      detectionInput(cv, padded, limit)
    );
    const boxes = withMatScope((scope) =>
      detectionBoxes(
        cv,
        scope,
        { data: map.data, height: map.dims[2] ?? 0, width: map.dims[3] ?? 0 },
        padded.height,
        padded.width
      )
    );
    const empty: OcrTrace = {
      boxes,
      crops: [],
      labels: [],
      padded,
      paddingTop: top,
      preprocessed,
      recognized: [],
    };
    if (boxes.length === 0) {
      return { lines: [], trace: empty };
    }
    const crops = boxes.map((box) => cropText(cv, padded, box));
    const classified = await classifyCrops(cv, this.#sessions.classify, crops);
    const recognized = await recognizeCrops(
      cv,
      this.#sessions.recognize,
      this.#characters,
      classified.crops
    );
    const lines: OcrLine[] = [];
    for (const [i, { score, text }] of recognized.entries()) {
      const box = boxes[i];
      if (
        box === undefined ||
        PYTHON_BLANK.test(text) ||
        score < minTextScore
      ) {
        continue;
      }
      lines.push({
        box: toOriginal(box, top, preprocessed, image.height, image.width),
        score,
        text,
      });
    }
    return {
      lines,
      trace: { ...empty, crops, labels: classified.labels, recognized },
    };
  }
}

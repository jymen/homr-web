/**
 * homr-web: homr's optical music recognition in the browser. One page image
 * in, MusicXML and the staff rectangles out, in the shape the AbcMusicStudio
 * server's homr route answers. Everything runs in a module Worker.
 */

export type {
  PageInput,
  RecognizeOptions,
  Recognizer,
  RecognizerOptions,
} from "./client.js";
export { createRecognizer } from "./client.js";
export type {
  Backend,
  PageText,
  Progress,
  ProgressStage,
  RecognizeError,
  RecognizeFailure,
  RecognizeResult,
  RecognizeSuccess,
  StaffBox,
} from "./result.js";
export { BACKENDS, PROGRESS_STAGES, RECOGNIZE_ERRORS } from "./result.js";
export { HOMR_COMMIT, HOMR_VERSION } from "./version.js";

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
  TextOptions,
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
  TabAnnotation,
  TabCapo,
  TabEvent,
  TabNote,
  TabReading,
  TabSystem,
  TabTechnique,
  TabTuning,
  TuningSource,
} from "./result.js";
export { BACKENDS, PROGRESS_STAGES, RECOGNIZE_ERRORS } from "./result.js";
export type { PitchedEvent, PitchedNote, Tuning } from "./tab/pitch.js";
export { pitchTab, standardTuning, tuningOf } from "./tab/pitch.js";
export { TAB_TECHNIQUES } from "./tab/read.js";
export { midiOfPitch } from "./tab/tuning.js";
export { HOMR_COMMIT, HOMR_VERSION } from "./version.js";

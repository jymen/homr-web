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
  TabBarline,
  TabCapo,
  TabEvent,
  TabInstrumentKind,
  TabInstrumentName,
  TabNote,
  TabPlacedEvent,
  TabReading,
  TabSlur,
  TabSlurKind,
  TabStem,
  TabSystem,
  TabTechnique,
  TabTimeSignature,
  TabTuning,
  TuningSource,
} from "./result.js";
export { BACKENDS, PROGRESS_STAGES, RECOGNIZE_ERRORS } from "./result.js";
export type { PageTextItem } from "./tab/layer.js";
export type { PitchedEvent, PitchedNote, Tuning } from "./tab/pitch.js";
export { pitchTab, standardTuning, tuningOf } from "./tab/pitch.js";
export { TAB_SLUR_KINDS, TAB_TECHNIQUES } from "./tab/read.js";
export type {
  Duration,
  EventDuration,
  NoteValue,
  Prerequisite,
  PrerequisiteSource,
  TabAnswers,
  TabBarCheck,
  TabDurations,
  TabInstrument,
  TabNeed,
  TabPrerequisiteField,
  TabPrerequisites,
  TabResolution,
  TabTune,
  TabTuneBar,
  TabTuneEvent,
  TabTuneNote,
  TimeSignature,
} from "./tab/tune.js";
export {
  NOTE_VALUES,
  needsOf,
  resolveTab,
  TAB_INSTRUMENTS,
  tabPrerequisites,
  ticksOf,
} from "./tab/tune.js";
export {
  midiOfPitch,
  TAB_INSTRUMENT_KINDS,
  tuningCandidates,
} from "./tab/tuning.js";
export { HOMR_COMMIT, HOMR_VERSION } from "./version.js";

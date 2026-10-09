/**
 * Tuning and capo from the text a tab page prints: "aDADE tuning",
 * "Tuning: D A D G A D", "(6)=D", "Capo II", "Accord : Open G", "Capodastre
 * en 3e case". Pure: one line of recognised text in, what it says out, and a
 * resolver that ties it to a tab system's line count.
 *
 * Pages list open strings low to high, which is bottom tab line first: a
 * guitar's "EADGBE", a mandolin's "GDAE", and a banjo's "gDGBD", whose first
 * letter is the short fifth string, the bottom line in banjo tab, lowercase
 * by convention. The resolver turns them around to the library's top line
 * first. A letter carries no octave; each string takes the octave nearest
 * the same string of the standard tuning for that string count, the octave
 * any player assumes ("aEAC#E" is A4 E3 A3 C#4 E4 like "gDGBD" is G4 D3 G3
 * B3 D4).
 *
 * Names ("Open G", "Sawmill", "Drop D") go through one registry per string
 * count, since "Open G" is DGDGBD on a guitar and gDGBD on a banjo. A name the
 * registry lacks is kept as text, never guessed.
 */

import type { TabLineCount } from "./detect.js";

/** Where a tuning came from: letters printed on the page, or a name looked up in the registry. */
export type TuningSource = "text" | "named";

/**
 * A tuning read from the page. `read` fits the system: `strings` are as many
 * scientific pitch names as it has lines, top line first, ready for
 * `pitchTab`. `string_count` is a tuning for another number of strings
 * (DADGAD printed over a five-line tab); its `strings` are that tuning's,
 * reported, not forced. `unknown_name` is a tuning word with no letters the
 * registry knows ("Open Zeta tuning"). `text` is the line it was read from
 * and `confidence` the recogniser's score for that line.
 */
export type TabTuning =
  | {
      readonly confidence: number;
      readonly source: TuningSource;
      readonly status: "read" | "string_count";
      readonly strings: readonly string[];
      readonly text: string;
    }
  | {
      readonly confidence: number;
      readonly status: "unknown_name";
      readonly text: string;
    };

/** A capo read from the page: `fret` 0 is "no capo" printed as such. */
export interface TabCapo {
  readonly confidence: number;
  readonly fret: number;
  readonly text: string;
}

const STEP_SEMITONES = {
  A: 9,
  B: 11,
  C: 0,
  D: 2,
  E: 4,
  F: 5,
  G: 7,
} as const;
type Step = keyof typeof STEP_SEMITONES;

/** A note name without octave. */
interface Note {
  readonly alter: -1 | 0 | 1;
  readonly step: Step;
}

/** What one line of text says about the tuning, before a line count is known. */
export type ParsedTuning =
  /** Open strings as printed, bottom line first. */
  | { readonly form: "letters"; readonly notes: readonly Note[] }
  /** "(6)=D 5=G": strings by number, 1 the top line, the rest standard. */
  | {
      readonly assignments: readonly (readonly [number, Note])[];
      readonly form: "strings";
    }
  | { readonly form: "named"; readonly name: string }
  | { readonly form: "unknown_name" };

export interface ParsedText {
  readonly capo?: number;
  readonly tuning?: ParsedTuning;
}

/** Standard tuning per string count, bottom line first, the octave reference for letters. */
const STANDARD: Readonly<Record<TabLineCount, readonly string[]>> = {
  4: ["G3", "D4", "A4", "E5"],
  5: ["G4", "D3", "G3", "B3", "D4"],
  6: ["E2", "A2", "D3", "G3", "B3", "E4"],
};

interface NamedTuning {
  /** As printed, bottom line first. */
  readonly letters: string;
  /** Lowercase, accents stripped, single spaces. */
  readonly names: readonly string[];
}

/**
 * Names per string count: mandolin, five-string banjo, guitar. French names
 * use the solfège the French books print ("sol ouvert" is open G).
 */
const NAMED: Readonly<Record<TabLineCount, readonly NamedTuning[]>> = {
  4: [
    { letters: "GDAE", names: ["standard", "standard tuning", "normal"] },
    { letters: "AEAE", names: ["cross", "cross tuning", "cross a"] },
  ],
  5: [
    {
      letters: "gDGBD",
      names: [
        "standard",
        "standard tuning",
        "open g",
        "g tuning",
        "sol ouvert",
        "open sol",
      ],
    },
    { letters: "gCGCD", names: ["double c", "double do"] },
    {
      letters: "gDGCD",
      names: ["sawmill", "mountain minor", "g modal", "modal"],
    },
    { letters: "aDADE", names: ["double d", "double re"] },
    { letters: "f#DF#AD", names: ["open d", "re ouvert", "open re"] },
    { letters: "gCGBD", names: ["c tuning", "drop c"] },
    { letters: "aEAC#E", names: ["open a", "la ouvert", "open la"] },
  ],
  6: [
    {
      letters: "EADGBE",
      names: ["standard", "standard tuning", "normal", "e standard"],
    },
    { letters: "DADGBE", names: ["drop d", "drop re"] },
    { letters: "DADGBD", names: ["double drop d"] },
    { letters: "DADGAD", names: ["dadgad"] },
    { letters: "DADF#AD", names: ["open d", "re ouvert", "open re"] },
    { letters: "DGDGBD", names: ["open g", "sol ouvert", "open sol"] },
    { letters: "EBEG#BE", names: ["open e", "mi ouvert", "open mi"] },
    { letters: "EAEAC#E", names: ["open a", "la ouvert", "open la"] },
    { letters: "CGCGCE", names: ["open c", "do ouvert", "open do"] },
    { letters: "CGCFAD", names: ["drop c", "drop do"] },
  ],
};

const LINE_COUNTS: readonly TabLineCount[] = [4, 5, 6];
const isLineCount = (n: number): n is TabLineCount =>
  (LINE_COUNTS as readonly number[]).includes(n);

/** Every registered name, longest first, so "double drop d" wins over "drop d". */
const ALL_NAMES = [
  ...new Set(
    Object.values(NAMED).flatMap((family) => family.flatMap((t) => t.names))
  ),
].sort((a, b) => b.length - a.length);

const SOLFEGE: Readonly<Record<string, Step>> = {
  do: "C",
  fa: "F",
  la: "A",
  mi: "E",
  re: "D",
  si: "B",
  sol: "G",
};

/** Words that make a line about tuning: "tuning", "accord", "accordage", "accorde". */
const TUNING_WORD = /\b(?:tunings?|accord(?:age|e|ee)?)\b/;
const COMPACT =
  /^([A-Ga-g](?:#|b)?)((?:[A-G](?:#|b)?){3,5})$|^([A-G](?:#|b)?){4,6}$/;
const NOTE = /([A-Ga-g])(#|b)?/g;
const SINGLE_NOTE = /^([A-Ga-g])(#|b)?$/;
const SOLFEGE_NOTE = /^(do|re|mi|fa|sol|la|si)(#|b)?$/;
const STRING_ASSIGNMENT =
  /[(]?([1-6])[)]?\s*(?:=|:|->|→)\s*([A-Ga-g](?:#|b)?|do|re|mi|fa|sol|la|si)(?![a-z])/gi;
const ROMAN: Readonly<Record<string, number>> = {
  i: 1,
  ii: 2,
  iii: 3,
  iv: 4,
  ix: 9,
  v: 5,
  vi: 6,
  vii: 7,
  viii: 8,
  x: 10,
  xi: 11,
  xii: 12,
};
const CAPO =
  /\b(?:capo(?:dastre)?)\b\s*[:=.]?\s*(?:(?:on|at|sur|en|a)\s+)?(?:(?:the|la|le)\s+)?(?:(?:fret|case|frette)\s+)?(\d{1,2}|[ivx]{1,4})(?:st|nd|rd|th|eme|e|re|ere)?(?![a-z0-9])/;
const NO_CAPO = /\b(?:no|sans|without)\s+capo(?:dastre)?\b/;
const MAX_CAPO = 12;
const DIGITS = /^\d+$/;
/** Where a line splits into tokens: a compact tuning keeps its #, a spaced one splits at dashes too. */
const TOKEN_GAP = /[\s,;:()[\]/]+/;
const WORD_GAP = /[\s,;:()[\]/-]+/;
const NOT_WORD = /[^a-z0-9#]+/g;

/**
 * The line as compared: NFC, accents and OCR's stray diacritics dropped
 * ("Štandard"), sharp and flat signs as # and b, dashes and separators as
 * spaces. Case is kept: it is what tells a banjo's fifth string.
 */
function normalised(text: string): string {
  return text
    .normalize("NFD")
    .replace(/\p{M}/gu, "")
    .replace(/[♯＃]/g, "#")
    .replace(/♭/g, "b")
    .replace(/[‐-―−]/g, "-")
    .replace(/[①②③④⑤⑥]/g, (c) => `(${c.charCodeAt(0) - 0x24_5f})`)
    .replace(/\s+/g, " ")
    .trim();
}

const ALTER: Readonly<Record<string, -1 | 1>> = { "#": 1, b: -1 };
const ACCIDENTAL = { "-1": "b", "0": "", "1": "#" } as const;

const noteOf = (letter: string, accidental: string | undefined): Note => ({
  alter: ALTER[accidental ?? ""] ?? 0,
  step: letter.toUpperCase() as Step,
});

function compactNotes(token: string): Note[] | undefined {
  if (!COMPACT.test(token)) {
    return;
  }
  return [...token.matchAll(NOTE)].map(([, letter, accidental]) =>
    noteOf(letter as string, accidental)
  );
}

/** One note name standing alone: a letter, or a solfège syllable when `solfege` (French books write "Ré La Ré Sol La Ré"). */
function noteOfToken(token: string, solfege: boolean): Note | undefined {
  const letter = token.match(SINGLE_NOTE);
  if (letter) {
    return noteOf(letter[1] as string, letter[2]);
  }
  const word = solfege ? token.toLowerCase().match(SOLFEGE_NOTE) : null;
  return word
    ? noteOf(SOLFEGE[word[1] as string] as string, word[2])
    : undefined;
}

/** A run of four to six single-note tokens ("D A D G A D", "Re La Re Sol La Re" after a tuning word). */
function spacedNotes(tokens: readonly string[], solfege: boolean): Note[] {
  let best: Note[] = [];
  let run: Note[] = [];
  for (const token of tokens) {
    const note = noteOfToken(token, solfege);
    if (note === undefined) {
      run = [];
      continue;
    }
    run.push(note);
    if (run.length > best.length) {
      best = [...run];
    }
  }
  return best.length >= 4 && best.length <= 6 ? best : [];
}

/** The capo fret the line names, if any. */
function capoOf(lower: string): number | undefined {
  if (NO_CAPO.test(lower)) {
    return 0;
  }
  const match = lower.match(CAPO);
  if (!match) {
    return;
  }
  const value = match[1] as string;
  const fret = DIGITS.test(value) ? Number(value) : ROMAN[value];
  return fret !== undefined && fret >= 0 && fret <= MAX_CAPO ? fret : undefined;
}

/** The longest registered name standing as whole words in the line. */
const nameIn = (lower: string): string | undefined =>
  ALL_NAMES.find((name) =>
    new RegExp(`(?:^|[^a-z0-9#])${name}(?:$|[^a-z0-9#])`).test(lower)
  );

/** The line with every capo phrase, tuning word and punctuation gone: empty when a name is all it says. */
const leftover = (lower: string, name: string): string =>
  lower
    .replace(CAPO, " ")
    .replace(TUNING_WORD, " ")
    .replace(name, " ")
    .replace(NOT_WORD, "");

/**
 * What one recognised line says about tuning and capo. Letters printed bare
 * ("gDGBD") count; a run of spaced single letters ("D A D G B E") and a
 * name count beside a tuning word or alone on their line, so "Jazz Standard"
 * in a title is not a tuning and neither is "A B C D" inside a sentence.
 */
export function parseTabText(text: string): ParsedText {
  const line = normalised(text);
  const lower = line.toLowerCase();
  const capo = capoOf(lower);
  const tuning = tuningOf(line, lower);
  return {
    ...(capo === undefined ? {} : { capo }),
    ...(tuning === undefined ? {} : { tuning }),
  };
}

function tuningOf(line: string, lower: string): ParsedTuning | undefined {
  const keyword = TUNING_WORD.test(lower);
  const tokens = line.split(TOKEN_GAP).filter((t) => t !== "");
  for (const token of tokens) {
    const notes = compactNotes(token);
    if (notes !== undefined) {
      return { form: "letters", notes };
    }
  }
  const assignments = [...line.matchAll(STRING_ASSIGNMENT)].map(
    ([, position, pitch]): readonly [number, Note] => [
      Number(position),
      noteOfToken(pitch as string, true) as Note,
    ]
  );
  if (assignments.length > 0) {
    return { assignments, form: "strings" };
  }
  const words = line.split(WORD_GAP).filter((t) => t !== "");
  const spaced = spacedNotes(words, keyword);
  if (spaced.length > 0 && (keyword || spaced.length === words.length)) {
    return { form: "letters", notes: spaced };
  }
  const name = nameIn(lower);
  if (name !== undefined && (keyword || leftover(lower, name) === "")) {
    return { form: "named", name };
  }
  return keyword ? { form: "unknown_name" } : undefined;
}

/**
 * String names printed one per tab line, top line first ("e B G D A E"):
 * letters, bottom line first, when every text is one note name.
 */
export function parseLetterLabels(
  topFirst: readonly string[]
): ParsedTuning | undefined {
  const notes: Note[] = [];
  for (const text of [...topFirst].reverse()) {
    const match = normalised(text).match(SINGLE_NOTE);
    if (!match) {
      return;
    }
    notes.push(noteOf(match[1] as string, match[2]));
  }
  return notes.length >= 4 ? { form: "letters", notes } : undefined;
}

const PITCH = /^([A-G])([#b]?)(-?\d)$/;

/** MIDI number of a scientific pitch name ("G4", "C#4", "Bb3"; C4 is 60). Throws a RangeError on anything else. */
export function midiOfPitch(name: string): number {
  const match = name.trim().match(PITCH);
  if (!match) {
    throw new RangeError(`not a pitch name: ${JSON.stringify(name)}`);
  }
  const note = noteOf(match[1] as string, match[2]);
  return 12 * (Number(match[3]) + 1) + STEP_SEMITONES[note.step] + note.alter;
}

/** The note in the octave nearest `reference` (a pitch name), the lower one a tritone away. */
function nearest(note: Note, reference: string): string {
  const target = midiOfPitch(reference);
  const semitones = STEP_SEMITONES[note.step] + note.alter;
  const up = (((semitones - target) % 12) + 12) % 12;
  const midi = up < 6 ? target + up : target + up - 12;
  return `${note.step}${ACCIDENTAL[note.alter]}${(midi - semitones) / 12 - 1}`;
}

/** Bottom-first notes to top-first pitch names, octaves from the standard tuning of that string count. */
function pitched(notes: readonly Note[]): string[] | undefined {
  if (!isLineCount(notes.length)) {
    return;
  }
  const reference = STANDARD[notes.length];
  return notes
    .map((note, k) => nearest(note, reference[k] as string))
    .reverse();
}

const lettersNotes = (letters: string): Note[] =>
  [...letters.matchAll(NOTE)].map(([, letter, accidental]) =>
    noteOf(letter as string, accidental)
  );

/**
 * The tuning a parsed line gives a system of `lines` lines, with the line's
 * text and score. A name is looked up for this line count first, then for
 * the others, which gives `string_count` ("Sawmill" over a guitar tab).
 */
export function resolveTuning(
  parsed: ParsedTuning,
  lines: TabLineCount,
  text: string,
  confidence: number
): TabTuning {
  const unknown: TabTuning = { confidence, status: "unknown_name", text };
  const fitting = (strings: string[] | undefined, source: TuningSource) =>
    strings === undefined
      ? unknown
      : {
          confidence,
          source,
          status:
            strings.length === lines
              ? ("read" as const)
              : ("string_count" as const),
          strings,
          text,
        };
  switch (parsed.form) {
    case "letters":
      return fitting(pitched(parsed.notes), "text");
    case "named": {
      for (const count of [lines, ...LINE_COUNTS.filter((n) => n !== lines)]) {
        const found = NAMED[count].find((t) => t.names.includes(parsed.name));
        if (found !== undefined) {
          return fitting(pitched(lettersNotes(found.letters)), "named");
        }
      }
      return unknown;
    }
    case "strings": {
      const highest = Math.max(...parsed.assignments.map(([s]) => s));
      const count = highest <= lines ? lines : 6;
      const base = [...STANDARD[count as TabLineCount]].reverse();
      for (const [position, note] of parsed.assignments) {
        base[position - 1] = nearest(note, base[position - 1] as string);
      }
      return fitting(base, "text");
    }
    case "unknown_name":
      return unknown;
    default:
      return unknown;
  }
}

/** The standard tuning's open strings for a line count, top line first: mandolin GDAE, banjo gDGBD, guitar EADGBE. */
export const standardStrings = (lines: TabLineCount): readonly string[] =>
  [...STANDARD[lines]].reverse();

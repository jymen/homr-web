/**
 * The note-to-notehead alignment (src/pipeline/note-columns.ts) on synthetic
 * staffs: every note found, heads missed, notes missed, chords, and the
 * column carried through the MusicXML writer into document order. The two
 * real pages it was measured on are in docs/design/button-tab-findings.md.
 */

import { describe, expect, it } from "vitest";
import {
  ellipseFromParts,
  legacyConventionRectOf,
  normalizeRotatedRect,
  pointListFromPairs,
} from "../src/geometry/boxes.js";
import { createStaff, createStaffPoint } from "../src/model/staff.js";
import { createNote } from "../src/model/symbols.js";
import {
  generateMusicXml,
  generateMusicXmlWithColumns,
} from "../src/musicxml/generate.js";
import {
  alignNoteEvents,
  clefOffset,
  diatonicStep,
  groupNoteheadColumns,
  type NoteEvent,
  type NoteheadColumn,
  noteEventsOfStaff,
  withNoteColumns,
} from "../src/pipeline/note-columns.js";
import { joinVoice } from "../src/pipeline/parse-staffs.js";
import {
  createEncodedSymbol,
  type EncodedSymbol,
} from "../src/transformer/symbol.js";

const IDENTITY = { offset: 0, scale: 1 };

/** Columns 40 px apart; a lagging attention point 30 px to the left of its own column. */
const columnsAt = (positions: readonly number[]): NoteheadColumn[] =>
  positions.map((position, k) => ({ positions: [position], x: 100 + 40 * k }));
const eventsFor = (columns: readonly NoteheadColumn[], lag = 30): NoteEvent[] =>
  columns.map((column) => ({
    positions: [...column.positions],
    x: column.x - lag,
  }));

/** A run with no two neighbours on the same position, as a tune mostly is. */
const TUNE = [1, 3, 5, 4, 2, 3, 6, 5, 7, 4, 2, 1];

describe("alignNoteEvents", () => {
  it("pairs every event with its own column when nothing is missing, despite a lag of most of a spacing", () => {
    const columns = columnsAt(TUNE);
    expect(alignNoteEvents(eventsFor(columns), columns)).toEqual(
      TUNE.map((_, k) => k)
    );
  });

  it("leaves an event unpaired, and the rest right, when its head was missed", () => {
    const all = columnsAt(TUNE);
    const events = eventsFor(all);
    const columns = all.filter((_, k) => k !== 4 && k !== 8);
    const paired = alignNoteEvents(events, columns);
    expect(paired[4]).toBeNull();
    expect(paired[8]).toBeNull();
    for (const [k, j] of paired.entries()) {
      if (j !== null) {
        expect(columns[j]).toBe(all[k]);
      }
    }
  });

  it("skips a column, and pairs the rest right, when the decoder missed its note", () => {
    const columns = columnsAt(TUNE);
    const events = eventsFor(columns).filter((_, k) => k !== 6);
    const paired = alignNoteEvents(events, columns);
    expect(paired).toEqual(TUNE.map((_, k) => k).filter((k) => k !== 6));
  });

  it("lets the attention decide between two columns of the same pitch", () => {
    const columns = columnsAt([3, 3]);
    expect(
      alignNoteEvents([{ positions: [3], x: columns[1]?.x ?? 0 }], columns)
    ).toEqual([1]);
    expect(
      alignNoteEvents([{ positions: [3], x: columns[0]?.x ?? 0 }], columns)
    ).toEqual([0]);
  });

  it("pairs a chord with the column holding its heads", () => {
    const columns: NoteheadColumn[] = [
      { positions: [1], x: 100 },
      { positions: [3, 7], x: 140 },
      { positions: [5], x: 180 },
    ];
    const events: NoteEvent[] = [
      { positions: [1], x: 90 },
      { positions: [7, 3], x: 120 },
      { positions: [5], x: 150 },
    ];
    expect(alignNoteEvents(events, columns)).toEqual([0, 1, 2]);
  });

  it("answers nothing for nothing, and all null with no column", () => {
    expect(alignNoteEvents([], [])).toEqual([]);
    expect(alignNoteEvents([{ positions: [1], x: null }], [])).toEqual([null]);
  });
});

describe("groupNoteheadColumns", () => {
  it("joins heads closer than 0.8 unit size and averages their x", () => {
    const columns = groupNoteheadColumns(
      [
        { position: 5, x: 207 },
        { position: 1, x: 100 },
        { position: 3, x: 200 },
        { position: 4, x: 300 },
      ],
      10
    );
    expect(columns).toEqual([
      { positions: [1], x: 100 },
      { positions: [3, 5], x: 203.5 },
      { positions: [4], x: 300 },
    ]);
  });
});

describe("pitches as staff positions", () => {
  it("puts each clef's note on its line, the bottom line being 1", () => {
    expect(clefOffset("clef_G2")).toBe(29);
    expect(clefOffset("clef_F4")).toBe(17);
    expect(clefOffset("clef_C3")).toBe(23);
    expect(clefOffset("note_4")).toBeUndefined();
    const e4 = diatonicStep("E4");
    expect(e4 === null ? null : e4 - 29).toBe(1);
    const g2 = diatonicStep("G2");
    expect(g2 === null ? null : g2 - 17).toBe(1);
    expect(diatonicStep(".")).toBeNull();
  });
});

const note = (
  pitch: string,
  x: number | null,
  rhythm = "note_8"
): EncodedSymbol =>
  createEncodedSymbol(rhythm, {
    coordinates: x === null ? null : { x, y: 128 },
    lift: "_",
    pitch: pitch as EncodedSymbol["pitch"],
    position: "upper",
  });
const CHORD = createEncodedSymbol("chord");
const STEP = /<step>(\w)<\/step>/;

describe("noteEventsOfStaff", () => {
  it("joins chord tones into one event, skips rests, follows the clef and maps x to the page", () => {
    const symbols = [
      createEncodedSymbol("clef_F4", { position: "upper" }),
      note("G2", 10),
      CHORD,
      note("B2", 14),
      createEncodedSymbol("rest_8", { pitch: "." as EncodedSymbol["pitch"] }),
      note("D3", null),
      createEncodedSymbol("barline"),
      createEncodedSymbol("clef_G2", { position: "upper" }),
      note("E4", 40),
    ];
    const { events, members } = noteEventsOfStaff(symbols, {
      offset: 5,
      scale: 2,
    });
    expect(events).toEqual([
      { positions: [1, 3], x: 25 },
      { positions: [5], x: null },
      { positions: [1], x: 85 },
    ]);
    expect(members).toEqual([[1, 3], [5], [8]]);
  });
});

const noContour = pointListFromPairs([]);
const headAt = (x: number, position: number) =>
  createNote(
    ellipseFromParts(
      normalizeRotatedRect(
        legacyConventionRectOf({ angle: 0, cx: x, cy: 100, h: 8, w: 10 })
      ),
      noContour,
      noContour,
      0
    ),
    position,
    null
  );
const staffWith = (heads: ReturnType<typeof headAt>[]) =>
  createStaff(
    [
      createStaffPoint(50, [80, 90, 100, 110, 120], 0),
      createStaffPoint(450, [80, 90, 100, 110, 120], 0),
    ],
    { symbols: heads }
  );

describe("withNoteColumns", () => {
  it("writes each paired note's column, normalised by the page width, on every tone of its event", () => {
    const staff = staffWith([
      headAt(100, 1),
      headAt(200, 3),
      headAt(202, 7),
      headAt(300, 5),
    ]);
    const symbols = [
      createEncodedSymbol("clef_G2", { position: "upper" }),
      note("E4", 90),
      note("G4", 180),
      CHORD,
      note("D5", 182),
      note("B4", null),
      note("A5", 380),
    ];
    const out = withNoteColumns(symbols, staff, IDENTITY, 1000);
    expect(out.map((s) => s.column)).toEqual([
      undefined,
      0.1,
      0.201,
      undefined,
      0.201,
      0.3,
      undefined,
    ]);
    expect(symbols.every((s) => s.column === undefined)).toBe(true);
  });

  it("leaves a grand staff alone", () => {
    const staff = { ...staffWith([headAt(100, 1)]), isGrandstaff: true };
    const out = withNoteColumns([note("E4", 90)], staff, IDENTITY, 1000);
    expect(out[0]?.column).toBeUndefined();
  });
});

describe("generateMusicXmlWithColumns", () => {
  const voice = joinVoice([
    [
      createEncodedSymbol("clef_G2", { position: "upper" }),
      createEncodedSymbol("timeSignature/4", { position: "upper" }),
      { ...note("C5", 1, "note_4"), column: 0.2 },
      CHORD,
      { ...note("E4", 1, "note_4"), column: 0.21 },
      CHORD,
      { ...note("G4", 1, "note_2"), column: 0.22 },
      createEncodedSymbol("rest_4", {
        pitch: "_" as EncodedSymbol["pitch"],
        position: "upper",
      }),
      { ...note("A4", 1, "note_4") },
      createEncodedSymbol("barline"),
    ],
  ]);

  it("gives one entry per <note>, in document order, and the same XML as generateMusicXml", () => {
    const { musicXml, noteColumns } = generateMusicXmlWithColumns([voice], "");
    expect(musicXml).toBe(generateMusicXml([voice], ""));
    const notes = musicXml.match(/<note[ >]/g) ?? [];
    expect(noteColumns).toHaveLength(notes.length);
    // The chord is written sorted by token text (G4, E4, C5 here) and split
    // by duration, so the columns follow the notes, not the tokens.
    const steps = [...musicXml.matchAll(/<note[ >][\s\S]*?<\/note>/g)].map(
      ([element]) => element.match(STEP)?.[1] ?? "rest"
    );
    const byStep: Record<string, number | null> = {
      A: null,
      C: 0.2,
      E: 0.21,
      G: 0.22,
      rest: null,
    };
    expect(noteColumns).toEqual(steps.map((step) => byStep[step]));
  });
});

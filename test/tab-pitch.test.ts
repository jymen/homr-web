/**
 * The tab reader's pure parts: what a recogniser read means on a tab line,
 * how frets become events, and pitch from tuning and capo.
 */

import { describe, expect, it } from "vitest";
import { pitchTab, standardTuning, tuningOf } from "../src/tab/pitch.js";
import { eventsOf, readMark } from "../src/tab/read.js";
import { midiOfPitch } from "../src/tab/tuning.js";

describe("readMark", () => {
  it("reads frets, an O as a zero, and the letters stuck to a fret as its technique", () => {
    expect(readMark("7")).toEqual({
      fret: 7,
      kind: "fret",
      technique: undefined,
    });
    expect(readMark("12")).toEqual({
      fret: 12,
      kind: "fret",
      technique: undefined,
    });
    expect(readMark("O")).toEqual({
      fret: 0,
      kind: "fret",
      technique: undefined,
    });
    expect(readMark("S0")).toEqual({ fret: 0, kind: "fret", technique: "Sl" });
    expect(readMark("7h")).toEqual({ fret: 7, kind: "fret", technique: "h" });
  });

  it("takes a technique before a fret, so Po is no P stuck to a zero", () => {
    for (const [text, technique] of [
      ["Po", "Po"],
      ["P", "Po"],
      ["Sl", "Sl"],
      ["S", "Sl"],
      ["SI", "Sl"],
      ["H", "H"],
      ["R", "R"],
      ["p", "p"],
      ["h", "h"],
      ["x", "x"],
      ["X", "x"],
      ["Harm.", "Harm."],
      ["Harm", "Harm."],
    ] as const) {
      expect(readMark(text), text).toEqual({ kind: "technique", technique });
    }
  });

  it("leaves clef letters, frets past 24 and noise unread", () => {
    for (const text of ["", "T", "A", "B", "25", "123", "#", "日"]) {
      expect(readMark(text), text).toEqual({ kind: "unread" });
    }
  });
});

describe("eventsOf", () => {
  const at = (centre: number, string: number, fret: number) => ({
    centre,
    note: { fret, string },
  });

  it("stacks frets within reach into one chord ordered by string, x page-normalised", () => {
    const events = eventsOf(
      [at(100, 3, 0), at(102, 1, 2), at(101, 2, 3), at(200, 1, 5)],
      40,
      1000
    );
    expect(events).toEqual([
      {
        notes: [
          { fret: 2, string: 1 },
          { fret: 3, string: 2 },
          { fret: 0, string: 3 },
        ],
        x: 0.1,
      },
      { notes: [{ fret: 5, string: 1 }], x: 0.2 },
    ]);
  });

  it("never puts two frets of one string in one event", () => {
    const events = eventsOf([at(100, 1, 2), at(110, 1, 3)], 40, 1000);
    expect(events.map((e) => e.notes)).toEqual([
      [{ fret: 2, string: 1 }],
      [{ fret: 3, string: 1 }],
    ]);
  });

  it("opens a new event once a fret is out of reach of the event's first", () => {
    expect(eventsOf([at(100, 1, 0), at(118, 2, 0)], 40, 1000)).toHaveLength(2);
    expect(eventsOf([at(100, 1, 0), at(117, 2, 0)], 40, 1000)).toHaveLength(1);
  });
});

describe("midiOfPitch", () => {
  it("reads scientific pitch names, C4 being 60", () => {
    expect(midiOfPitch("C4")).toBe(60);
    expect(midiOfPitch("G4")).toBe(67);
    expect(midiOfPitch("C#4")).toBe(61);
    expect(midiOfPitch("Bb3")).toBe(58);
    expect(midiOfPitch("E2")).toBe(40);
  });

  it("refuses anything else", () => {
    for (const name of ["", "H4", "c4", "C", "C##4", "G 4"]) {
      expect(() => midiOfPitch(name), name).toThrow(RangeError);
    }
  });
});

describe("pitchTab", () => {
  const banjoOpenG = ["D4", "B3", "G3", "D3", "G4"].map(midiOfPitch);

  it("is open string plus capo plus fret, the banjo's fifth string (bottom line) taking the capo", () => {
    const reading = {
      events: [
        { notes: [{ fret: 0, string: 5 }], x: 0.1 },
        {
          notes: [
            { fret: 0, string: 1 },
            { fret: 2, string: 3 },
          ],
          x: 0.2,
        },
      ] as const,
      lines: 5,
    } as const;
    expect(pitchTab(reading, { capo: 0, strings: banjoOpenG })).toEqual([
      { notes: [{ fret: 0, midi: midiOfPitch("G4"), string: 5 }], x: 0.1 },
      {
        notes: [
          { fret: 0, midi: midiOfPitch("D4"), string: 1 },
          { fret: 2, midi: midiOfPitch("A3"), string: 3 },
        ],
        x: 0.2,
      },
    ]);
    const capo2 = pitchTab(reading, { capo: 2, strings: banjoOpenG });
    expect(capo2.flatMap((e) => e.notes.map((n) => n.midi))).toEqual(
      ["A4", "E4", "B3"].map(midiOfPitch)
    );
  });

  it("refuses a tuning whose string count is not the system's, and a capo off the neck", () => {
    const reading = { events: [], lines: 4 } as const;
    expect(() => pitchTab(reading, { capo: 0, strings: banjoOpenG })).toThrow(
      RangeError
    );
    const mandolin = ["E5", "A4", "D4", "G3"].map(midiOfPitch);
    expect(() => pitchTab(reading, { capo: -1, strings: mandolin })).toThrow(
      RangeError
    );
    expect(() => pitchTab(reading, { capo: 1.5, strings: mandolin })).toThrow(
      RangeError
    );
    expect(pitchTab(reading, { capo: 0, strings: mandolin })).toEqual([]);
  });
});

describe("pitchTab with the tuning read off the page", () => {
  const events = [{ notes: [{ fret: 0, string: 5 }], x: 0.1 }] as const;
  const sawmill = {
    confidence: 1,
    source: "text",
    status: "read",
    strings: ["D4", "C4", "G3", "D3", "G4"],
    text: "gDGCD",
  } as const;

  it("takes the read tuning and capo when no tuning is passed", () => {
    const reading = {
      capo: { confidence: 1, fret: 2, text: "Capo 2" },
      events,
      lines: 5,
      tuning: sawmill,
    } as const;
    expect(tuningOf(reading)).toEqual({
      capo: 2,
      strings: ["D4", "C4", "G3", "D3", "G4"].map(midiOfPitch),
    });
    expect(pitchTab(reading)[0]?.notes[0]?.midi).toBe(midiOfPitch("A4"));
    expect(
      pitchTab(reading, standardTuning(5))[0]?.notes[0]?.midi,
      "a tuning passed wins"
    ).toBe(midiOfPitch("G4"));
  });

  it("reads no capo as capo 0, and refuses to pitch without a tuning that fits", () => {
    expect(
      pitchTab({ events, lines: 5, tuning: sawmill })[0]?.notes[0]?.midi
    ).toBe(midiOfPitch("G4"));
    expect(() => pitchTab({ events, lines: 5 })).toThrow("standardTuning(5)");
    expect(() =>
      pitchTab({
        events,
        lines: 5,
        tuning: { ...sawmill, status: "string_count", strings: ["D4"] },
      })
    ).toThrow(RangeError);
    expect(
      tuningOf({
        tuning: { confidence: 1, status: "unknown_name", text: "Open Zeta" },
      })
    ).toBeUndefined();
  });

  it("offers the standard tuning per line count, never applied unasked", () => {
    expect(standardTuning(4).strings).toEqual(
      ["E5", "A4", "D4", "G3"].map(midiOfPitch)
    );
    expect(standardTuning(6)).toEqual({
      capo: 0,
      strings: ["E4", "B3", "G3", "D3", "A2", "E2"].map(midiOfPitch),
    });
  });
});

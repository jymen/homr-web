/**
 * The prerequisites a tab needs before it is a tune (src/tab/tune.ts): what
 * is read, assumed or missing, the needs the app asks the musician for, and
 * resolveTab building the tune once the answers settle every one, on
 * readings built by hand.
 */

import { describe, expect, it } from "vitest";
import type { TabEvent, TabReading, TabSlur } from "../src/tab/read.js";
import type { TabStem } from "../src/tab/rhythm.js";
import {
  barsOf,
  durationOfStem,
  needsOf,
  resolveTab,
  TAB_INSTRUMENTS,
  type TabNeed,
  tabPrerequisites,
} from "../src/tab/tune.js";

const Q: TabStem = { beams: 0, dots: 0, kind: "stem" };
const E: TabStem = { beams: 1, dots: 0, kind: "stem" };
const S: TabStem = { beams: 2, dots: 0, kind: "stem" };
const NONE: TabStem = { kind: "none" };

let at = 0;
const ev = (stem: TabStem, ...notes: [number, number][]): TabEvent => {
  at += 0.02;
  const [first, ...rest] = notes.map(([string, fret]) => ({ fret, string }));
  if (first === undefined) {
    throw new Error("an event has a note");
  }
  return { grace: false, notes: [first, ...rest], stem, x: at };
};
const bar = () => {
  at += 0.01;
  return { repeatEnd: false, repeatStart: false, thick: false, x: at };
};

const BANJO_G = ["D4", "B3", "G3", "D3", "G4"];

const system = (over: Partial<TabReading>): TabReading => ({
  annotations: [],
  barlines: [],
  cx: 0.5,
  cy: 0.3,
  events: [],
  h: 0.05,
  index: 0,
  lines: 5,
  readFrom: "image",
  slurs: [],
  unread: 0,
  w: 0.8,
  ...over,
});

/** Cotton-Eyed Joe's first two bars as the page prints them, with its header. */
function cottonEyedJoe(over: Partial<TabReading> = {}): TabReading {
  at = 0;
  const chord: [number, number][] = [
    [1, 0],
    [2, 0],
    [3, 0],
  ];
  const opening = bar();
  const events = [
    ev(Q, [2, 0]),
    ev(Q, [1, 0]),
    ev(Q, [1, 0]),
    ev(E, ...chord),
    ev(E, [5, 0]),
  ];
  const middle = bar();
  events.push(
    ev(E, [1, 2]),
    ev(E, [1, 0]),
    ev(E, ...chord),
    ev(E, [5, 0]),
    ev(Q, [2, 0]),
    ev(E, ...chord),
    ev(E, [5, 0])
  );
  const closing = bar();
  return system({
    barlines: [
      { ...opening, repeatStart: true, thick: true },
      middle,
      { ...closing, repeatEnd: true },
    ],
    events,
    instrumentName: {
      confidence: 0.99,
      kind: "banjo5",
      text: "Clawhammer Banjo",
    },
    slurs: [{ from: 5, kind: "pull-off", letter: "P", string: 1, to: 6 }],
    timeSignature: { den: 4, num: 4, text: "4/4" },
    tuning: {
      confidence: 0.98,
      source: "text",
      status: "read",
      strings: BANJO_G,
      text: "Tuning: G DGBD",
    },
    ...over,
  });
}

const fieldsOf = (needs: readonly TabNeed[]) => needs.map((n) => n.field);

describe("resolveTab on a page that prints everything", () => {
  it("reads instrument, tuning, meter and durations, assumes capo 0, and makes the tune", () => {
    const resolution = resolveTab([cottonEyedJoe()]);
    const { prerequisites } = resolution;
    expect(prerequisites.instrument).toMatchObject({
      source: "page_text",
      status: "read",
      value: TAB_INSTRUMENTS.banjo5,
    });
    expect(prerequisites.tuning).toMatchObject({
      status: "read",
      value: BANJO_G,
    });
    expect(prerequisites.capo).toEqual({
      evidence: "no capo printed",
      status: "assumed",
      value: 0,
    });
    expect(prerequisites.meter).toMatchObject({
      source: "page_marks",
      status: "read",
      value: { den: 4, num: 4 },
    });
    expect(prerequisites.durations).toMatchObject({ status: "read" });
    expect(prerequisites.bars.map((b) => [b.read, b.fits])).toEqual([
      [4, true],
      [4, true],
    ]);
    if (!resolution.ok) {
      throw new Error(JSON.stringify(resolution.needs));
    }
    const { tune } = resolution;
    expect(tune.tuning.midi).toEqual([62, 59, 55, 50, 67]);
    expect(tune.bars.map((b) => b.events.length)).toEqual([5, 7]);
    expect(tune.bars[0]).toMatchObject({ repeatStart: true });
    expect(tune.bars[1]).toMatchObject({ repeatEnd: true });
    const first = tune.bars[0]?.events[0];
    expect(first?.notes).toEqual([
      expect.objectContaining({ fret: 0, midi: 59, string: 2 }),
    ]);
    expect(first?.duration).toEqual({ dots: 0, value: "quarter" });
    const pull = tune.bars[1]?.events[0]?.notes[0];
    expect(pull).toMatchObject({ fret: 2, midi: 64, techniques: ["pull-off"] });
    expect(tune.bars[1]?.events[3]?.notes[0]).toMatchObject({
      midi: 67,
      string: 5,
    });
  });

  it("raises every string by a printed capo, the fifth included", () => {
    const resolution = resolveTab([
      cottonEyedJoe({ capo: { confidence: 0.9, fret: 2, text: "Capo 2" } }),
    ]);
    expect(resolution.prerequisites.capo).toMatchObject({
      status: "read",
      value: 2,
    });
    expect(
      resolution.ok && resolution.tune.bars[0]?.events[4]?.notes[0].midi
    ).toBe(69);
  });
});

describe("needs, and the answers that settle them", () => {
  it("asks for a tuning the page does not print, offering the instrument's, the standard first", () => {
    const { tuning: _tuning, ...rest } = cottonEyedJoe();
    const resolution = resolveTab([system(rest)]);
    expect(resolution.ok).toBe(false);
    const needs = resolution.ok ? [] : resolution.needs;
    expect(fieldsOf(needs)).toEqual(["tuning"]);
    const [need] = needs;
    expect(need).toMatchObject({
      confidence: 0,
      evidence: "no tuning printed",
      field: "tuning",
      read: null,
      status: "missing",
    });
    expect(need?.field === "tuning" && need.candidates[0]).toEqual(BANJO_G);
    expect(
      need?.field === "tuning" && need.candidates.map((c) => c.join(" "))
    ).toContain("D4 C4 G3 D3 G4");
    expect(resolveTab([system(rest)], { tuning: BANJO_G }).ok).toBe(true);
  });

  it("keeps a tuning for another string count as read text, and refuses an answer that does not fit", () => {
    const reading = cottonEyedJoe({
      tuning: {
        confidence: 0.9,
        source: "text",
        status: "string_count",
        strings: ["D4", "A3", "G3", "D3", "A2", "D2"],
        text: "DADGAD",
      },
    });
    const prerequisites = tabPrerequisites([reading], {
      tuning: ["D4", "B3", "G3", "D3"],
    });
    expect(prerequisites.tuning).toMatchObject({
      confidence: 0.9,
      read: "DADGAD",
      status: "missing",
    });
    expect(
      prerequisites.tuning.status === "missing" && prerequisites.tuning.evidence
    ).toContain("does not fit this tuning");
  });

  it("tells a five-string banjo by its short string when the page names no instrument", () => {
    const { instrumentName: _name, ...rest } = cottonEyedJoe();
    expect(tabPrerequisites([system(rest)]).instrument).toMatchObject({
      status: "read",
      value: TAB_INSTRUMENTS.banjo5,
    });
  });

  it("asks for the instrument when only the line count speaks, with the candidates for that count", () => {
    const { instrumentName: _name, tuning: _tuning, ...rest } = cottonEyedJoe();
    const five = tabPrerequisites([system(rest)]).instrument;
    expect(five).toMatchObject({ read: null, status: "missing" });
    expect(five.status === "missing" && five.candidates).toEqual([
      TAB_INSTRUMENTS.banjo5,
    ]);
    at = 0;
    const four = tabPrerequisites([
      system({ events: [ev(Q, [1, 0])], lines: 4 }),
    ]).instrument;
    expect(
      four.status === "missing" && four.candidates.map((c) => c.kind)
    ).toEqual(["mandolin", "tenorBanjo"]);
    expect(
      tabPrerequisites([system(rest)], { instrument: "mandolin" }).instrument
        .status
    ).toBe("missing");
    expect(
      tabPrerequisites([system(rest)], { instrument: "banjo5" }).instrument
    ).toMatchObject({ source: "musician", status: "read" });
  });

  it("asks for a meter the page does not print, the meters ranked by the bars they fit", () => {
    const { timeSignature: _time, ...rest } = cottonEyedJoe();
    const resolution = resolveTab([system(rest)]);
    const needs = resolution.ok ? [] : resolution.needs;
    expect(fieldsOf(needs)).toEqual(["meter", "durations"]);
    const [meter] = needs;
    expect(meter?.field === "meter" && meter.candidates[0]).toEqual({
      den: 4,
      num: 4,
    });
    expect(meter?.evidence).toContain("4/4 2/2");
    expect(resolveTab([system(rest)], { meter: { den: 4, num: 4 } }).ok).toBe(
      true
    );
  });

  it("asks for durations on a page with no rhythm marks, and takes one value for every note", () => {
    at = 0;
    const plain = cottonEyedJoe();
    const unstemmed = system({
      ...plain,
      events: plain.events.map((e) => ({ ...e, stem: NONE })),
    });
    const needs = needsOf(tabPrerequisites([unstemmed]));
    expect(fieldsOf(needs)).toEqual(["durations"]);
    expect(needs[0]?.evidence).toContain("the page draws no rhythm");
    const [need] = needs;
    expect(
      need?.field === "durations" && need.candidates.map((grid) => grid[0]?.[0])
    ).toEqual([
      { dots: 0, value: "eighth" },
      { dots: 0, value: "quarter" },
    ]);
    const answered = resolveTab([unstemmed], {
      durations: { uniform: { dots: 0, value: "eighth" } },
      meter: { den: 4, num: 4 },
    });
    expect(answered.ok).toBe(false);
    expect(answered.prerequisites.durations.evidence).toContain(
      "bar 1 (system 1) holds 2.5 of 4 quarters"
    );
  });

  it("names the bars that do not sum to the meter, and takes the musician's correction of one note", () => {
    const reading = cottonEyedJoe();
    const wrong = system({
      ...reading,
      events: reading.events.map((e, k) => (k === 1 ? { ...e, stem: E } : e)),
    });
    const needs = needsOf(tabPrerequisites([wrong]));
    expect(fieldsOf(needs)).toEqual(["durations"]);
    expect(needs[0]).toMatchObject({ confidence: 0.5 });
    expect(needs[0]?.evidence).toBe(
      "1 of 2 bars sum to 4/4; not: bar 1 (system 1) holds 3.5 of 4 quarters"
    );
    const fixed = resolveTab([wrong], {
      durations: {
        events: [
          { duration: { dots: 0, value: "quarter" }, event: 1, system: 0 },
        ],
      },
    });
    expect(fixed.ok).toBe(true);
    expect(fixed.prerequisites.durations).toMatchObject({
      source: "musician",
      status: "read",
    });
  });
});

describe("durations from the marks", () => {
  it("reads beams as halvings and dots as half again", () => {
    expect(durationOfStem(Q)).toEqual({ dots: 0, value: "quarter" });
    expect(durationOfStem(E)).toEqual({ dots: 0, value: "eighth" });
    expect(durationOfStem(S)).toEqual({ dots: 0, value: "16th" });
    expect(durationOfStem({ beams: 0, dots: 1, kind: "stem" })).toEqual({
      dots: 1,
      value: "quarter",
    });
    expect(durationOfStem(NONE)).toBeUndefined();
  });

  it("gives a note with no stem what its bar has left: a half, a dotted half, a whole", () => {
    at = 0;
    const events = [ev(Q, [1, 0]), ev(Q, [2, 0]), ev(NONE, [3, 0])];
    const prerequisites = tabPrerequisites([system({ events })], {
      instrument: "banjo5",
      meter: { den: 4, num: 4 },
      tuning: BANJO_G,
    });
    expect(prerequisites.durations).toMatchObject({
      status: "read",
      value: [
        [
          { dots: 0, value: "quarter" },
          { dots: 0, value: "quarter" },
          { dots: 0, value: "half" },
        ],
      ],
    });
  });

  it("lets a first bar of at most half the meter stand as an upbeat, and a last bar complete it", () => {
    at = 0;
    const upbeat = bar();
    const events = [ev(E, [1, 0]), ev(E, [1, 2])];
    const next = bar();
    events.push(ev(Q, [1, 0]), ev(Q, [2, 0]), ev(Q, [3, 0]), ev(Q, [4, 0]));
    const last = bar();
    events.push(ev(Q, [1, 0]), ev(Q, [2, 0]), ev(Q, [3, 0]));
    const prerequisites = tabPrerequisites(
      [system({ barlines: [upbeat, next, last], events })],
      { meter: { den: 4, num: 4 } }
    );
    expect(prerequisites.bars.map((b) => [b.read, b.fits, b.pickup])).toEqual([
      [1, true, true],
      [4, true, false],
      [3, true, false],
    ]);
    expect(prerequisites.durations.status).toBe("read");
  });

  it("lets each repeated part open on an upbeat that the bar closing it completes", () => {
    at = 0;
    const opening = { ...bar(), repeatStart: true };
    const events = [ev(Q, [1, 0])];
    const second = bar();
    events.push(ev(Q, [1, 0]), ev(Q, [2, 0]), ev(Q, [3, 0]), ev(Q, [4, 0]));
    const third = bar();
    events.push(ev(Q, [1, 0]), ev(Q, [2, 0]), ev(Q, [3, 0]));
    const end = { ...bar(), repeatEnd: true };
    const partB = { ...bar(), repeatStart: true };
    events.push(ev(E, [1, 0]), ev(E, [1, 2]));
    const last = bar();
    events.push(ev(Q, [1, 0]), ev(Q, [2, 0]), ev(Q, [3, 0]), ev(Q, [4, 0]));
    const prerequisites = tabPrerequisites(
      [
        system({
          barlines: [opening, second, third, end, partB, last],
          events,
        }),
      ],
      { meter: { den: 4, num: 4 } }
    );
    expect(prerequisites.bars.map((b) => [b.read, b.fits, b.pickup])).toEqual([
      [1, true, true],
      [4, true, false],
      [3, true, false],
      [1, true, true],
      [4, true, false],
    ]);
  });

  it("takes no time for a grace note and strikes it before the next note", () => {
    at = 0;
    const grace = { ...ev(Q, [3, 2]), grace: true };
    const events = [
      ev(Q, [1, 0]),
      grace,
      ev(Q, [3, 3]),
      ev(Q, [4, 0]),
      ev(Q, [5, 0]),
    ];
    const slurs: TabSlur[] = [
      { from: 1, kind: "slide", letter: "S", string: 3, to: 2 },
    ];
    const resolution = resolveTab([system({ events, slurs })], {
      instrument: "banjo5",
      meter: { den: 4, num: 4 },
      tuning: BANJO_G,
    });
    expect(resolution.ok).toBe(true);
    const [, slid] = resolution.ok
      ? (resolution.tune.bars[0]?.events ?? [])
      : [];
    expect(slid?.notes[0]).toMatchObject({ fret: 3, string: 3 });
    expect(slid?.graces).toEqual([
      [expect.objectContaining({ fret: 2, techniques: ["slide"] })],
    ]);
  });

  it("ties a held fret to the next instead of plucking it again", () => {
    at = 0;
    const events = [ev(Q, [1, 2]), ev(Q, [1, 2]), ev(Q, [2, 0]), ev(Q, [3, 0])];
    const resolution = resolveTab(
      [system({ events, slurs: [{ from: 0, kind: "tie", string: 1, to: 1 }] })],
      { instrument: "banjo5", meter: { den: 4, num: 4 }, tuning: BANJO_G }
    );
    const [held, again] = resolution.ok
      ? (resolution.tune.bars[0]?.events ?? [])
      : [];
    expect(held?.notes[0]).toMatchObject({ techniques: [], tieToNext: true });
    expect(again?.notes[0]).toMatchObject({ tieFromPrevious: true });
  });
});

describe("barsOf", () => {
  it("cuts each system at its barlines, drops bars with no note and marks the bar opening a line", () => {
    at = 0;
    const lines = [bar(), bar()];
    const first = [ev(Q, [1, 0]), ev(Q, [1, 2])];
    at = 0;
    const second = [ev(Q, [2, 0])];
    const bars = barsOf([
      system({ barlines: lines, events: first }),
      system({ barlines: [], events: second }),
    ]);
    expect(bars.map((b) => [b.system, b.events, b.systemBreakBefore])).toEqual([
      [0, [0, 1], false],
      [1, [0], true],
    ]);
  });
});

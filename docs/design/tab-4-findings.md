# Tab 4 findings: rhythm, the text layer, and what a tab needs

Step 4 of the line-tablature plan in `.scratch/tab-sketch/REPORT.md`: read a
tab-only page to a playable tune, rhythm included, and when that cannot be
done, say exactly what the musician must supply. Measured 2026-10-17 on arm64
macOS, Node 23.5.0, onnxruntime-web 1.30.0 (wasm, one thread),
@techstark/opencv-js 4.12.0, pdfjs-dist 5.7.284. Every decision below has a
`tab-4` row in `docs/decisions.tsv`.

## What a tab needs

Processing a tab needs prerequisite information by design, not as a fallback.

| | prerequisite | what it is | when nothing is printed |
|---|---|---|---|
| 1 | instrument | string count, a short string (a five-string banjo's fifth), courses | missing |
| 2 | tuning | one pitch per string | missing |
| 3 | capo | a fret | assumed 0, and said to be assumed |
| 4 | meter | the time signature | missing |
| 5 | durations | each note's value, from stems and beams | missing on a page with no rhythm marks |

1 to 3 turn positions into pitches, 4 and 5 turn the order into a tune. The
key (spelling only, derivable from the pitches), the chord symbols and the
playing style (clawhammer, three-finger) are useful and not required; none is
read yet but the instrument names that also name a style.

Each prerequisite is **read** from the page (with its source and the
recogniser's confidence), **assumed** (the capo alone, with the default), or
**missing** (with what was read, how sure, and the candidates). The library
builds the tune only when none is missing; otherwise it answers the needs.
The app asks the musician for exactly those and calls again with the answers,
on the same readings: the page is never read twice.

## The result type

`src/tab/tune.ts`; every name below is exported from `homr-web`.

```ts
type PrerequisiteSource = "page_text" | "page_marks" | "text_layer" | "musician";

type Prerequisite<T> =
  | { status: "read"; value: T; source: PrerequisiteSource; confidence: number; evidence: string }
  | { status: "assumed"; value: T; evidence: string }
  | { status: "missing"; read: string | null; confidence: number; candidates: readonly T[]; evidence: string };

interface TabPrerequisites {
  instrument: Prerequisite<TabInstrument>;      // { kind: "banjo5" | "tenorBanjo" | "mandolin" | "guitar", strings, shortStrings: [{ string, startFret }], courses: 1 | 2 }
  tuning: Prerequisite<readonly string[]>;      // pitch names, top line first: ["D4", "B3", "G3", "D3", "G4"]
  capo: Prerequisite<number>;
  meter: Prerequisite<TimeSignature>;           // { num, den }
  durations: Prerequisite<TabDurations>;        // per system, per event: { value: "quarter" | "eighth" | ..., dots } or "grace"
  bars: readonly TabBarCheck[];                 // { bar, system, expected, read (quarters), open, pickup, fits }
}

type TabNeed = { field: "instrument" | "tuning" | "capo" | "meter" | "durations" } & /* that field's missing variant */;

interface TabAnswers {
  instrument?: TabInstrumentKind;
  tuning?: readonly string[];
  capo?: number;
  meter?: TimeSignature;
  durations?: { uniform: Duration } | { events: { system: number; event: number; duration: Duration }[] };
}

function tabPrerequisites(systems: readonly TabReading[], answers?: TabAnswers): TabPrerequisites;
function needsOf(prerequisites: TabPrerequisites): TabNeed[];          // instrument first, durations last
function resolveTab(systems: readonly TabReading[], answers?: TabAnswers):
  | { ok: true; tune: TabTune; prerequisites: TabPrerequisites }
  | { ok: false; needs: [TabNeed, ...TabNeed[]]; prerequisites: TabPrerequisites };

interface TabTune {
  instrument: TabInstrument;
  tuning: { names: readonly string[]; midi: readonly number[] };   // top line first, before the capo
  capo: number;
  meter: TimeSignature;
  bars: {
    repeatStart: boolean; repeatEnd: boolean; pickup: boolean; systemBreakBefore: boolean;
    events: {
      system: number; event: number;                               // where it was read
      duration: Duration;
      graces: (readonly TabTuneNote[])[];                          // struck before, taking no time
      notes: [TabTuneNote, ...TabTuneNote[]];                      // { string, fret, midi, techniques, tieToNext, tieFromPrevious }
    }[];
  }[];
}
```

Choices behind the shape:

- **One generic `Prerequisite<T>`, five fields.** The app renders one kind of
  question with a typed value; `TabNeed` is derived from it, so a field added
  to `TabPrerequisites` is a need without more code.
- **`evidence` is a sentence**, the thing to show the musician beside the
  question: `"Tuning: G DGBD"`, `no tuning printed`, `"DADGAD" is a 6-string
  tuning over a 5-line tab`, `no time signature printed; bars that fit: 4/4
  8/8, 2/2 8/8, 2/4 0/8, ...`, `7 of 8 bars sum to 4/4; not: bar 4 (system
  1) holds 3.5 of 4 quarters`.
- **Candidates are values the answer can take directly**, most likely first:
  the instruments of that line count, the named tunings for the instrument
  (standard first), the meters ranked by how many bars they fit, and for
  durations the grid as read, or every note an eighth and every note a
  quarter on a page with no stems.
- **An answer that does not fit is not taken** (a tuning of another string
  count, an instrument of another, a capo off the neck): the field stays
  missing and its evidence says the answer did not fit. No exception reaches
  the app from the musician's input.
- **The tune follows MusicXML's conventions**, which the app's
  `ImportedTablature` (src/utils/tablature/import/importedTablature.ts) also
  follows: `Duration` is a value and dots, a connecting technique sits on the
  note it starts from, a tie is `tieToNext` / `tieFromPrevious`.

## Reading the rhythm

`src/tab/rhythm.ts` reads the geometry over a band from 4 spacings above the
tab to 7 below it, lines erased as the fret reader erases them, the frets'
boxes blanked:

- **Stems**: under each fret column, the longest thin vertical run starting
  below its lowest fret and no lower than 1.5 spacings under the bottom line.
  TablEdit, Guitar Pro, LilyPond and Peghead Nation all hang rhythm below
  the tab. No stem is a half note or longer: the bar decides which (half,
  dotted half, whole, shared evenly by the unstemmed notes).
- **Beams and flags**: horizontal runs crossing two columns beside the stem's
  end (0.15 and 0.4 spacing away), each at least 0.1 spacing thick, counted
  on the side with more: 0 a quarter, 1 an eighth, 2 a sixteenth. A flag
  (TablEdit's lone eighth) rises to the right from the stem's end and is
  counted when a column 0.25 spacing away has ink within 0.8 spacing above.
- **Dots**: small round marks right of the stem's end. None of the pages
  measured prints a dotted note; the reader found "dots" only beside the
  slashed stems of Cluck Old Hen's grace notes, which take no time. Dots are
  unmeasured.
- **Barlines**: vertical strokes from the top line to the bottom one that do
  not carry on past them (a stem crossing the tab does), grouped within 1.2
  spacings; thick at 0.25 spacing; two dots within 1.1 spacing after or
  before are a repeat start or end.
- **Arcs**: thin components at least 0.55 spacing wide and twice as wide as
  tall, near the lines, tied to the fret columns at their ends and the string
  they share nearest the arc. The letter beside it is looked for above the tab
  or under the beams, ranked by distance across to the first fret or the
  arc's middle (Peghead Nation prints under the first, LilyPond over the
  middle) plus half the distance from the tab; the first three are read and
  the first known letter wins: H, h a hammer-on, P, p, Po a pull-off, S, s,
  Sl a slide. Two equal frets with no letter are a tie, two different ones a
  `slur`.
- **Grace notes**: frets under 0.88 of the system's median fret height,
  slurred into the next column, take no time.
- **Time signature**: pairs of stacked marks taller than 1.1 times a fret
  before the first fret column (or one component 2.1 spacings tall, two
  digits touching, cut at its thinnest middle row); the recogniser, or the
  text layer, decides which pair reads as two numbers.

Not read yet: **rests** (Old Joe Clark's opening bar holds a quarter and an
eighth rest; its bar is reported short), **tuplets**, stems drawn **above**
the tab, and **two voices** in one tab (Moon over Shanghai's guitar, whose bass
notes stem one way and the melody the other).

## Three reader fixes the Peghead Nation pages needed

The guard found no tab on Cotton-Eyed Joe (`tab-4` rows):

1. **Frets touch their stems and each other.** Peghead Nation hangs each stem
   from the fret's own ink, and the frets stacked in a chord touch once the
   line between them is erased, so every fret component was too tall. On a
   system where at least two components are a fret-sized head hanging a stem
   from its middle, the components too big to be a fret are cut half way
   between the lines and the pieces grown back by one row over each cut.
2. **Beams lie on the lines.** A beam drawn against the bottom line (and, on
   Cluck Old Hen's exercise, against the top two) thickens it out of the line
   rows. Such a line is kept as "hidden" and finishes a group of three or
   four into five, never into four.
3. **"Tuning: G DGBD".** The short string printed apart from the four long
   ones read as a four-string DGBD.

The first two were checked against the 17 pages of the guard's truth and the
three golden pages: unchanged counts, no recogniser call on the golden pages.

## Measurements

`tools/tab-rhythm-accuracy.ts` (local only; `--layer` reads the PDF's text
layer too) on every tab-only page in `test/fixtures/local/tablature/`: the
four Peghead Nation pages copied for this step (Cotton-Eyed Joe, Billy in the
Lowground's two pages, Cluck Old Hen), the earlier banjo pages, and the five
TablEdit and Guitar Pro tab-only exports whose MusicXML gives the truth. Where
a page prints no time signature its bars are measured against the meter that
fits most of them, which is what the needs offer first. Each cell reads image
/ text layer; Cold Frosty Morning has no text layer (its glyphs are outlines).

| page | events | stems | bars that sum to the meter | meter | durations against the truth | resolveTab | time a page |
|---|---|---|---|---|---|---|---|
| Cotton-Eyed Joe (Peghead Nation) | 50 | 50 | 8/8 | 4/4 | 50 of 50 / 50 of 50 (hand) | tune, 8 bars | 8.9 s / 1.4 s |
| Billy in the Lowground p1 (Peghead Nation) | 161 | 161 | 24/24 | 4/4 | - | tune, 24 bars | 21.9 s / 3.7 s |
| Billy in the Lowground p2 (Peghead Nation) | 56 / 55 | 54 / 55 | 7/8 / 8/8 | not printed (4/4 fits best) | - | needs tuning, meter, durations | 8.4 s / 1.2 s |
| Cluck Old Hen (Peghead Nation) | 127 | 125 / 127 | 19/22 / 21/22 | 4/4 | - | needs durations | 20.2 s / 3.8 s |
| Cluck Old Hen (LilyPond) | 44 | 44 | 8/8 | not printed (4/4 fits best) | - | needs meter, durations | 8.2 s / 2.9 s |
| Chicken Reel (TablEdit) | 87 | 87 | 12/13 | 4/4 | - | needs instrument, tuning, durations | 8.9 s / 1.5 s |
| Cripple Creek (TablEdit) | 169 / 167 | 167 | 22/24 / 24/24 | 4/4 | - | needs durations / tune, 24 bars | 19.6 s / 2.0 s |
| Cold Frosty Morning | 101 | 101 | 18/18 | not printed (4/4 fits best) | - | needs meter, durations | 14.8 s |
| Sourwood Mountain | 106 | 106 | 16/16 | 2/4 | - | tune, 16 bars | 14.4 s / 2.1 s |
| Angelina Baker (TablEdit) | 138 | 138 | 20/20 | 2/4 | 138 of 138 / 138 of 138 | tune, 20 bars | 18.0 s / 2.0 s |
| Blackberry Blossom (Guitar Pro) | 90 / 88 | 88 | 11/13 / 13/13 | not printed (4/4 fits best) | 78 of 78 / 88 of 88 | needs meter, durations | 13.7 s / 5.5 s |
| Old Joe Clark (Guitar Pro) | 156 / 155 | 154 / 155 | 22/24 / 24/24 | 2/4 | 142 of 142 / 155 of 155 | needs instrument, tuning, durations / needs instrument, tuning | 19.8 s / 8.2 s |
| Arkansas Traveler, guitar (TablEdit) | 182 / 184 | 181 | 29/30 / 26/30 | not printed (4/4 fits best) / 4/4 | 166 of 166 / 166 of 166 | needs instrument, tuning, meter, durations / needs instrument, tuning, durations | 21.0 s / 0.3 s |
| Moon over Shanghai, guitar (Guitar Pro) | 202 / 195 | 176 / 172 | 2/14 | 4/4 | 136 of 150 / 151 of 177 | needs instrument, tuning, durations | 29.2 s / 5.7 s |

"Durations against the truth" counts, among the events matched to the truth
by their frets, those whose time to the next event equals the truth's (the
hand transcription for Cotton-Eyed Joe, the exported MusicXML otherwise).
"Needs" without a slash is the same in both columns.

In all: **218 of 242 bars** sum to the meter from the image, **224 of 242**
with the text layer (Cold Frosty Morning, which has none, counted the same in
both), and the durations agree with the truth on **710 of 724** matched
events from the image and **748 of 774** with the text layer. Every
disagreement is on Moon over Shanghai, a guitar tab in two voices, bass notes
stemmed one way and the melody the other, which the reader takes for one;
without it, 574 of 574 and 597 of 597, and 216 of 228 and 222 of 228 bars.
Its 2 of 14 bars are the reader saying so.

The bars that do not sum, outside Moon over Shanghai, and why, each checked
on the page:

| page | bar | why |
|---|---|---|
| Billy p2, image | 4 | the fret 10 read as a 1 and a 0, two events with no stem; the text layer reads the bar |
| Cluck Old Hen (Peghead) | 3 | the exercise ends on a chord with a quarter stem: the page's own short bar |
| Cluck Old Hen (Peghead), image | 7, 11 | a slashed grace note read with a stray fret on the line under it, so not taken for a grace |
| Chicken Reel | 13 | the last bar prints three beats of frets and two stems with no fret, which the reader does not count |
| Cripple Creek, image | 9, 22 | TablEdit's slide mark read as a fret 1 with no stem |
| Blackberry Blossom, image | 2, 6 | a mark beside a fret read as a fret 1 with no stem |
| Old Joe Clark, image | 1, 8 | bar 1 opens on a quarter rest and an eighth rest, the eighth rest read as a fret 7; a stray chord in bar 8 |
| Arkansas Traveler | 1 | a quarter rest before three quarters: rests are not read |
| Arkansas Traveler, layer | 22, 26, 30 | a grace note (fret 7, the MusicXML says so) whose slide is drawn as a line, not an arc, so not taken for a grace |

Cold Frosty Morning's A part ends on three beats (bar 9) that complete its
one-beat upbeat, and its B part opens on an upbeat of its own (bar 10); the
bar check takes an upbeat at the start of every repeated part for that.

The fret reading on the earlier pages is unchanged: `tools/tab-accuracy.ts`
gives the tab-2 figures exactly (15/15, 35/35, 33/33 hand truth, Cripple
Creek 161/169, Kildare 0/37), since the cut between lines only runs on a
system whose frets hang their stems, which only Peghead Nation's do.

## The vector-PDF shortcut

pdf.js gives a born-digital page's text with each run's position: on these
pages every fret is its own run, at its exact place. Twelve of the thirteen
tab-only PDFs measured have one (Cold Frosty Morning's glyphs are outlines);
Killdare, a scan with staves, has none worth the name. The question was whether reading the frets, letters and chord
symbols from it beats the recogniser on the image.

Measured with the library's own two paths (`recognizePage` with and without
`textLayer`), against a truth where there is one:

| page | frets matched to the truth: image | text layer |
|---|---|---|
| Cotton-Eyed Joe | 50 of 50 | 50 of 50 |
| Angelina Baker | 138 of 165 (page 1 holds 138) | 138 |
| Blackberry Blossom | 78 of 126 (page 1 holds about 90) | 88 |
| Old Joe Clark | 142 of 181 (page 1 holds about 156) | 155 |
| Arkansas Traveler, guitar | 166 of 406 | 166 |
| Moon over Shanghai, guitar | 150 of 371 | 177 |

On the pages without a truth the two agree but where the image is wrong:
Cluck Old Hen (Peghead) 118 of 127 events in common, every difference a
chord whose stacked frets the image lost on one line; Cripple Creek 156 of
169, the image reading TablEdit's "S'3" (a slide letter glued to its fret) as
13. The text layer was never wrong where the two differ, on the frets
checked by eye.

Speed: the frets cost no recogniser run at all, the header no OCR detector,
and a page takes **0.3 to 8.2 s instead of 8.2 to 29.2 s** in Node (table
above),
the rest being detection and the rhythm geometry.

**Decision: read the text layer when the caller has one, the image
otherwise** (`decisions.tsv`, tab-4). The image stays the path for scans,
photos and outlined PDFs, and the geometry (lines, stems, beams, arcs,
barlines) is always read from the image: pdf.js's operator list would give
the vector paths too, but stems and beams are measured right from the
raster on every page, so parsing paths would add a second geometry for no
measured gain. Where the text layer holds no text for a box (TablEdit draws
its time-signature digits from a music font's private-use characters), the
recogniser reads that box from the image.

A system takes the text layer when it holds at least four frets on its lines,
so a PDF whose text is only its header still reads its frets from the image.

## Cotton-Eyed Joe, bar by bar

From the image and from the text layer alike (test/tab-cotton-local.test.ts
asserts both): instrument read from "Clawhammer Banjo" with five lines,
tuning read from "Tuning: G DGBD" as D4 B3 G3 D3 G4 (top line first), capo
assumed 0 (none printed), meter read 4/4 from the digits at the start of
part A, and every bar sums to 4/4. Strings are numbered from the top line;
pitches are open string plus fret.

| bar | page (string/fret: q quarter, e eighth) | pitches read | quarters | slur |
|---|---|---|---|---|
| 1 (A1) | 2/0:q 1/0:q 1/0:q 1/0+2/0+3/0:e 5/0:e | B3q D4q D4q [D4 B3 G3]e G4e | 4 | |
| 2 (A2) | 1/2:e 1/0:e 1/0+2/0+3/0:e 5/0:e 2/0:q 1/0+2/0+3/0:e 5/0:e | E4e D4e [D4 B3 G3]e G4e B3q [D4 B3 G3]e G4e | 4 | pull-off 1/2 to 1/0 (P) |
| 3 (A3) | 2/0:q 1/0+2/0+3/0:e 5/0:e 3/2:e 3/0:e 1/0+2/0+3/0:e 5/0:e | B3q [D4 B3 G3]e G4e A3e G3e [D4 B3 G3]e G4e | 4 | pull-off 3/2 to 3/0 (P) |
| 4 (A4) | 3/0:q 4/2:q 4/0:q 1/0+2/0+3/0:e 5/0:e | G3q E3q D3q [D4 B3 G3]e G4e | 4 | |
| 5 (B1) | 4/0:e 4/2:e 1/0+2/0+3/0:e 5/0:e 3/0:q 1/0+2/0+3/0:e 5/0:e | D3e E3e [D4 B3 G3]e G4e G3q [D4 B3 G3]e G4e | 4 | hammer-on 4/0 to 4/2 (H) |
| 6 (B2) | 3/2:e 3/0:e 1/0+2/0+3/0:e 5/0:e 2/0:q 1/0+2/0+3/0:e 5/0:e | A3e G3e [D4 B3 G3]e G4e B3q [D4 B3 G3]e G4e | 4 | pull-off 3/2 to 3/0 (P) |
| 7 (B3) | as bar 3 | as bar 3 | 4 | pull-off 3/2 to 3/0 (P) |
| 8 (B4) | as bar 4 | as bar 4 | 4 | |

Bars 1 and 5 open a repeat, bars 4 and 8 close one. 50 events, 50 stems, 5
slurs, nothing unread, nothing missing: `resolveTab` answers the tune on the
first call. Every quarter here is Peghead Nation's short stem ending inside or
just under the tab; every eighth hangs to a beam under it. The page has no
note without a stem: the "long notes" of the brief carry quarter stems too.

## Needs exercised

On the measured pages, and in `test/tab-tune.test.ts`:

| case | what the result says |
|---|---|
| tuning not printed (Billy p2, Chicken Reel, Old Joe Clark) | `{ field: "tuning", read: null, confidence: 0, evidence: "no tuning printed", candidates: [gDGBD, gCGCD, gDGCD, aDADE, f#DF#AD, gCGBD, aEAC#E] }` for a banjo, the standard first |
| tuning for another string count | `read: "DADGAD"`, the recogniser's confidence, `evidence: "\"DADGAD\" is a 6-string tuning over a 5-line tab"` |
| instrument from line count alone (Chicken Reel, Old Joe Clark: five lines; Arkansas Traveler: six) | `read: null`, `candidates: [banjo5]` or `[guitar]`; four lines offer mandolin and tenor banjo |
| instrument from the short string (Cripple Creek: no name, gDGBD) | read: five lines and a tuning whose bottom line is its highest string |
| capo not printed (Cotton-Eyed Joe) | `{ status: "assumed", value: 0, evidence: "no capo printed" }` |
| meter not printed (Cluck Old Hen, LilyPond) | `candidates: [4/4, 2/2, ...]`, `evidence: "no time signature printed; bars that fit: 4/4 8/8, 2/2 8/8, 2/4 0/8, ..."` |
| meter missing, so durations unchecked | durations missing: "durations are checked against the meter, which is missing" |
| no rhythm marks | `evidence: "stems under 0 of 12 notes: the page draws no rhythm; ..."`, `candidates`: every note an eighth, every note a quarter |
| bars that do not sum (Cripple Creek from the image) | `confidence: 22/24`, `evidence: "22 of 24 bars sum to 4/4; not: bar 9 (system 2) holds 4 of 4 quarters and a note with no stem, ..."`, `candidates: [the durations as read]` |
| answer of the wrong string count | not taken; the need stays, its evidence ends "the answer [...] does not fit this tuning" |
| answers that settle | a tuning, a meter, `{ uniform }` durations or `{ events: [...] }` corrections: the second call answers the tune |

Billy in the Lowground's two pages read together (page 1 then page 2) take
page 1's tuning and meter for page 2: from the image they need only
durations, "31 of 32 bars sum to 4/4; not: bar 28 (system 7) holds 3.5 of 4
quarters and 2 notes with no stem" (the fret 10 read as 1 and 0); with the
text layer they resolve to a tune of 32 bars in gCGCD.

## What the app must do

1. **Hand the text layer over** when the page comes from a PDF: pdf.js's
   `getTextContent()` items as `PageTextItem`s (README, "A PDF's text
   layer"). Leave it out for an image or a scan.
2. **Keep the readings of every page of the tune**: `result.tablature` on a
   `tablature_only` failure, and on success when a page has both staves and
   tabs. Concatenate them in page order.
3. **Call `resolveTab(readings)`.** If `ok`, go to 5.
4. **Ask for each need**, in the order given, from its `field`,
   `candidates`, `read` and `evidence`: a picker of instruments; a picker of
   tunings with free entry (pitch names, top line first); a meter picker; for
   durations either "every note an eighth / a quarter" (`{ uniform }`) or the
   bars named in `prerequisites.bars` with `fits: false` to correct
   (`{ events: [{ system, event, duration }] }`). Show the capo as assumed.
   Call `resolveTab(readings, answers)` again; repeat until `ok`.
5. **Write the ABC through the app's own tablature import.** `TabTune` maps
   onto `ImportedTablature` (src/utils/tablature/import/importedTablature.ts)
   field for field, and `toAbc.ts` then writes the ABC with its `%abcms:tab`
   settings and positions lines:
   - one `ImportedTrack`: `strings` from `tune.tuning.midi` (top line first,
     the same order), `capo`, `instrument` "Banjo" for `banjo5` (which
     `decideInstrument` turns into `banjo5`, octave up), "Tenor Banjo",
     "Mandolin", "Guitar";
   - one `ImportedBar` per `tune.bars` entry: `time` = `tune.meter`,
     `repeatStart`, `repeatEnd` (2 when true), `systemBreakBefore`, key C
     major (the key is not read; the app's spelling pass can set it from the
     pitches), no endings, tempo or directions;
   - one voice of beats per bar: each event a `notes` beat with its
     `duration`, its `graces` as `ImportedGrace` (value "eighth", not on the
     beat), and each note's `string`, `fret`, `tieToNext`, `tieFromPrevious`
     and `techniques` (`hammer-on`, `pull-off` as they are; `slide` as `{
     kind: "slide", style: "legato" }`; `slur`, an arc with no letter, as a
     hammer-on when the fret rises and a pull-off when it falls, which is what
     an unlettered slur between two frets means in tab);
   - `ImportFormat` gains a value for a page read by homr-web, and `software`
     says so.
6. **Leave `TabTuneNote.midi` for display**: the app computes pitches from
   strings and capo itself, and the two agree by construction.

## What remains

- Rests, tuplets, two voices in one tab, stems above the tab (all reported
  as bars that do not fit, never guessed).
- Chord symbols: in the text layer they are plain runs above the tab; they
  are not attached to events yet.
- Several pages of one tune resolve together when their readings are passed
  in page order (Billy in the Lowground's page 2 prints no tuning and takes
  page 1's); the app has to keep page 1's readings for that.
- Scans and photos: unchanged from tab-2 and tab-3.

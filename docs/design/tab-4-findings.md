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

## A handwritten scan: Weeping Willow (2026-10-25)

The owner's `weeping-weelow.png`: a scanned page of five-string banjo tab
written by hand, five tab lines and an empty one, eighth notes beamed under
the tab, chord names, a start repeat, first and second endings. homr-web
0.5.1 found two tab systems of the six; homr read the other four as treble
staves, so the app took the staff path and made a tune of tab lines read as
notes, with every fret digit a `T:` line. Four causes, three fixed:

1. **The scan is turned a third of a degree.** Each line of systems 3 to 6
   spreads over 5 to 7 rows, more than `maxThickness`, and `findLines`
   dropped it. A cluster up to `maxSlope` (0.006 rows per column) too tall
   is now a line when its long-run ink is at most a line's thickness over
   its length and its median inked column no thicker than the page's level
   lines (`spreadThickness`). The second test is what keeps Moon over
   Shanghai page 3's slanting sixteenth beams (5 rows a column, lines 3 or
   4) from making a sixth line over a staff.
2. **A hand writes its digits larger than a line space.** Zeros 1.05 to 1.3
   spacings wide, a 4 and a drone 0 1.2 tall, all over `maxMarkWidth` and
   `maxMarkHeight`. Reading a system already found to be a tab now takes
   marks up to `READ_MARK` (1.3 tall, 1.35 wide); deciding whether a group
   is a tab keeps the printed bounds, under a staff's noteheads, and so does
   a system whose frets hang their stems (Peghead Nation: the larger bound
   there let the TAB clef letters and fret-plus-stem pieces through, and
   Cotton-Eyed Joe lost its time signature), and a mark taller than a
   printed fret must be wider than a stroke (a piece of TablEdit stem
   crossing Arkansas Traveler's bottom line read as a 1). The recogniser
   reads a handwritten zero as "θ", now a 0.
3. **A pen's stem slants.** The straight run down one column ended two or
   three spacings below the fret, above the beam, so every eighth read as a
   quarter. `stemPath` follows the ink a column aside, leaning at most
   `stemLean` (0.15), and the beams are looked for at the stem's lower end.
4. **Not fixed: the hand itself.** The guard still finds the endings line
   too few printed-size digits to call it a tab (4 marks, "日" for a zero),
   the recogniser reads a handwritten 4 as "u" and a 1 drawn as a tick not
   at all, a 5 or 2 written across a line is cut in two by the line
   erasure, and the top of a stem drawn through the tab reads as a "1" on
   the line it starts at. Mapping "u" to 4 would be guessing; it was not
   done.

On lines 1 and 2, 17 of the page's 59 events came out in order before, 45
after, and the stems give 45 of the 47 events read there their printed
value, against 12 of 19 before. The page still needs the bar fixer: about a
fret in four is missing or extra, and with six of eight eighths read per bar
the best-fitting meter offered first is 3/4.

Rechecked: `tools/tab-rhythm-accuracy.ts` on its 14 pages moves from 218 to
221 of 242 bars summing to the meter and from 710 to 715 of 724 durations
agreeing (Moon over Shanghai's slanting stems, 176 to 184 stemmed events;
Old Joe Clark one more), every other page unchanged; the 21 pages of
`tab-local.test.ts` and Cotton-Eyed Joe from the image and the text layer
pass; and the app's harness reads five sample PDFs and two staff photos to
the same events, stems and ABC.

## A digit classifier for a hand, and staves the guard took (2026-10-31)

### The tab digit classifier

RapidOCR reads print well and a hand badly: on Weeping Willow it reads a
handwritten 4 as "u", a 0 as "9" or "6", and a 1 drawn as a tick not at all.
homr-web 0.5.3 adds a second reader for the frets of a system read off the
page's ink: a small convolutional classifier (`src/tab/digits.ts`, the
`tabDigits` model role, `models/tab-digits-037fac028e12.onnx`, 471 919 bytes,
about 117 000 weights, fp32, opset 17, input `x` [n, 1, 28, 28], output
`logits` [n, 11]). It answers ten digits and an eleventh class, not a digit.
Its input is the mark's tight ink box scaled so its longer side is 20 pixels
and centred in 28 x 28, MNIST's own form (`digitInput`, the one definition).

**Data.** MNIST (LeCun, Cortes and Burges, Creative Commons
Attribution-Share Alike 3.0; the weights carry the same licence, NOTICE)
and crops of the pages typeset for this repository, nothing else:

- MNIST digits drawn by `tools/tab-digits-data.ts` on synthetic tab systems
  at line spacings of 10 to 22 pixels, 0.9 to 1.8 spacings tall, slanted,
  with pen stems hanging from them, lone stems through the tab, clawhammer
  brush arrows and slur arcs among them, then put through the reader's own
  line erasure, mark finding and cropping. A mark is labelled with the digit
  it came from, or "not a digit" when it covers less than 0.6 of a digit's
  box (a stem stub, a piece of arc, half a digit the line erasure cut).
  Four training sets of 10 000 pages (46 862 crops), one held-out set.
- The 14 typeset pages of `test/fixtures/tab/` read the same way and
  labelled from `truth.json`: 316 crops, the printed digits and, as not a
  digit, the TAB clef's letters, Sl, Po, H, R, h, p, x and slur pieces.
- MNIST digits under a pen-on-tab augmentation, fragments of them and
  curved strokes, made by the trainer.

`tools/train-tab-digits.sh` rebuilds everything from scratch (a venv in
`.venv-digits`, MNIST in `.digits/`, 8000 CPU steps, every seed fixed) and
names the result by its hash. Regenerating the crops reproduced all three
sets byte for byte, and retraining from them gave the same file, SHA-256 for
SHA-256 (torch 2.15.0 on an Apple M-series CPU; another platform may differ
in the last bits). The first generator drew its random numbers from an LCG
computed in doubles, which loses its low bits past 2^53 and fell into a
short cycle (one training set held 4 nines and 11 063 non-digits in 15 415
crops); it is mulberry32 now.

**Accuracy.** 88.5% on 5000 MNIST test digits under the tab augmentation, a
line through 40% of them (1s are the weakest, 0.89: a pen's 1 is the stem
the not-a-digit class also learns), 89.8% on the held-out synthetic crops,
316 of 316 on the fixture crops (seen in training), and 29 of 30 on digits
drawn in OpenCV's Hershey script face, which it never saw
(`test/tab-digits.test.ts`).

**The rule (`fuseDigit`).** The classifier reads every digit-height mark of
a system read off the ink, beside RapidOCR. Where it is at least 0.95 sure of
a digit and RapidOCR read either nothing the reader can use or another single
digit, its digit stands. A two-digit fret, a technique letter, a fret with
letters stuck to it and every short mark are RapidOCR's alone, and "not a
digit" never removes a read. Measured on Weeping Willow against a hand
transcription of its five lines (notes in order, longest common
subsequence), and on 16 printed tab pages (2189 notes) against RapidOCR's
own reading:

| rule | Weeping Willow | printed notes changed |
|---|---|---|
| RapidOCR alone | 80/129 | - |
| fill where RapidOCR read nothing usable, 0.9 | 90/129 | 0 |
| fill, 0.7 | 92/129 | 6 |
| fill, 0.5 | 92/129 | 11 |
| fill and replace any one-character read, 0.9 | 94/129 | 2 (Sourwood Mountain's S of Sl read as a 5) |
| classifier first, 0.9 | 94/129 | 25 |
| **fill and replace a one-digit read, 0.95 (taken)** | **94/129** | **0** |
| fill and replace a one-digit read, 0.8 | 96/129 | 4 (new frets on Arkansas Traveler and Borus March) |

Per line, RapidOCR alone against the rule: line 1 20 to 23 of 27, line 2 26
to 27 of 32, line 3 21 to 30 of 32, line 4 13 to 14 of 31, and the endings
line 0 of 7 both ways (the guard does not find it). The earlier experiment's
other modes (a CTC decode constrained to an alphabet, a classifier vote in
the guard, the short marks read too) and its environment switches are gone.
The model is fetched like the other nine, from `${baseUrl}/${sha256}/` with
its hash checked, and only on a page with a tab read off its ink.

Rechecked: the three typeset pages still read every event exactly with the
classifier on (`test/tab-read.test.ts`); `tools/tab-rhythm-accuracy.ts`
reads its 14 pages to 221 of 242 bars summing to the meter and 715 of 724
durations agreeing, with the classifier and without it (`--no-digits`), the
same events on every page; `npm test` passes, the 21 pages of
`test/tab-local.test.ts` included (79 tab systems found).

### Staves the guard took for tabs: Reel de Mattawa

An engraved page of four staff systems in sixteenths (the app's sample
`Reel_de_mattawa.pdf`). homr-web 0.5.2 logged `Tab guard: 2 tablature systems
(4, 4 lines) painted out` and `Tab reader: 0, 0 events` at 300 and 400 dpi,
and homr read bars 10 to 18 only. On system 1 a beam row two pixels under
the top line (11% of the page wide) was kept as a line: the top line could
not start a group (its gap to the beam under `minGap`), the beam row could
not either (its break far over `maxBreak`), so the group started at the
second line with four, which passes for a mandolin tab on its count. System 2
has beam rows on both sides of its top line, one cluster ten rows tall that
`findLines` keeps as neither a line nor a hidden one. Three changes, the
third proposed and not taken:

1. **Only a system read to an event is painted out** (`recognizePage`). A
   system the guard took and read nothing on stays on the page for homr and
   in `tablature` with no events; the log says `N painted out, M with no
   event left to the staff reader`.
2. **A line closer than `minGap` to the one before it is a beam lying on
   it** (`withoutBeamRows`): of the two, the one with the shorter longest
   break is kept. This alone gives system 1 its five lines; system 2's top
   line is restored by the faint-ink completion made for Marche des élèves
   (`withFaintLine`, below), which finds the line's row covering 0.8 of the
   group's width.
3. **Reading four- and six-line groups' marks as well, not taken.** It
   loses two real tabs: Killdare Fancy's first system, whose top line runs
   through its digits so every mark is half a digit (RapidOCR reads "上",
   "U", "ど", the line-erased crops no better), and Moon over Shanghai p3's
   last, a one-bar system holding one "0". Rejecting a four- or six-line
   group with under four marks would have caught Mattawa (its groups hold 0
   and 1) and loses that same one-bar system. Mattawa's groups never reach
   a digit test anyway.

On the 0.5.2 code, (2) alone keeps system 1 and not system 2; on the
working tree, with the faint-ink completion, both systems are staves at 200,
300 and 400 dpi and homr reads all four (`test/tab-staves-local.test.ts`,
local only). The guard finds the same 79 tab systems on the 21 pages of
`test/tab-local.test.ts`.

**Marche des élèves (2026-10-28, kept).** The working tree also holds the
fix for the app's 2711 x 3508 enlargement of an 820 x 1061 web image, where
Otsu leaves a staff line spread over five pale rows and a pale stem out of
the ink: `faintInkOf` (half way between the threshold and the paper),
`withFaintLine` (a four-line group made five by a faint line a gap away) and
`crossesAsStroke` (a mark where a pale stroke crosses a line is no fret).
The same test reads its nine staves with no tab.

**A handwritten second chance (2026-10-26, kept).** A five-line group with
too few printed-size marks is looked at again at a hand's size
(`HAND_MARK`, 2.2 spacings tall, 2.4 wide) after its stems are erased
(`eraseStems`), and goes to the digit test when those marks sit on its
lines, at most a third as many in its spaces (a staff's noteheads sit in
both): a hand's circled zeros, 1.4 to 2.1 spacings, as on the owner's
handwritten Sourwood Mountain. Such a tab is read at that size. Its debug
dumps are gone; it changes nothing on the test pages above.

## Weeping Willow: the endings line, pen stems and jogs (2026-11-01)

Three more reader fixes for the owner's handwritten banjo page, each
measured on Weeping Willow against the hand transcription (notes in order,
longest common subsequence), on the 16 printed tab pages (2189 notes)
against 0.5.3's own reading, and on the staff-only pages (Reel de Mattawa at
200, 300 and 400 dpi, Marche des élèves: no tab, as before):

| change | Weeping Willow | printed notes changed |
|---|---|---|
| 0.5.3 | 94/129 (23, 27, 30, 14, 0 of 27, 32, 32, 31, 7) | - |
| 1. five-line groups drawn like the page's tabs | 101/129 (endings 7/7) | 0 |
| 2. a pen's stem has no head | 109/129 (line 4 14 to 22) | 0 |
| 3. a stroke turning along a line keeps it | 119/129 (26, 28, 30, 28, 7) | 4 (Killdare Fancy, below) |

1. **The endings line.** It holds four printed-size marks, so it went to the
   digit test with them, one zero read "日", failed, and never reached the
   hand's second chance (which looks only at groups under four marks). Once
   a page has two five-line tabs read as numbers, a five-line group the test
   refused is taken too when its line spacing is within 10% of every one of
   theirs and its marks, at a hand's size with the stems erased, stand on
   its lines rather than in its spaces (at most a third as many in the
   spaces, the hand test's own share): `alikeFiveLineTabs`. Measured first
   on every score-and-tab page here, a program draws the staff smaller than
   its tab (17.7 against 26.6 px on Moon over Shanghai, 18.75 against 28.1
   on Amazing Grace, 21 to 22 against 32.9 on Old Joe Clark, 24 against 42
   on the TablEdit pages, 27.1 against 40.7 on Blackberry Blossom), a third
   smaller or more; Weeping Willow's tabs are 18.40 to 18.52 and its endings
   line 18.71. A staff drawn at its tab's size would still have its
   noteheads in its spaces, which the placement test refuses (a test page
   draws one). A group taken this way and read to nothing is left to homr,
   as any other (2026-10-31).
2. **A stem drawn through the tab.** `hangsAStem` took two of line 4's pen
   stems for frets hanging their stems: rows 4 to 7 px wide over the stem's
   first spacing count as "not thin" against 0.2 spacings (3.7 px), so the
   stem's top passed for a head. Two such components switch the system to
   Peghead Nation's cutting between the lines, which cut every stem into
   pieces, and the top pieces, a pen's width and a spacing tall, read as 1s
   (5/1 seven times). A head is now at least twice as wide as the stem's
   median row. The printed pages read as before, Peghead Nation's
   stem-hanging frets included.
3. **A 5 or a 2 written across a line.** The line erasure keeps a line pixel
   only where ink touches the line from above or below in the same column.
   A hand's 5 comes down to the line and starts its bowl four pixels to the
   right, so the columns between were erased and the 5 fell in two, its top
   a short mark and its bowl read as nothing; a 2 meets the line with its
   diagonal the same way. Reading a system, the line is now kept between a
   stroke leaving it upwards and one leaving it downwards when they are at
   most two of the narrower stroke's widths apart and under 0.4 spacings
   (`bridgeJogs`). Knocked-out systems are read as before, since a
   knocked-out fret never runs along a line: on Cripple Creek the rule
   joined a slide letter to the line's stub and lost a 2/0, and on Moon over
   Shanghai it moved two frets. Deciding whether a group is a tab is
   unchanged.

   **Killdare Fancy changes.** Its digits are printed over the lines, as a
   hand's are, and its slanting strokes cross them the same way. System 4
   read `2/5 2/1 2/5 3/5 2/1 2/5 2/5 2/1 2/1` and now reads `2/5 2/5 3/5 2/5
   2/5`: the four 2/1 were pieces of glyphs the line erasure had cut (a 4's
   upright read alone as "1"; the foot of a 2 read "と"), and no fret read
   before is lost. It is the only
   printed change, and it breaks the bar set for this work (no printed note
   changed), so it is the owner's to accept or not; passing `false` for
   `jogs` in `placementByRecogniser` (src/tab/read.ts) undoes step 3 alone.

Rechecked on the final tree: `npm test` (62 files, 1144 tests, the local
suites included: 79 tab systems on the 21 pages of `tab-local.test.ts`, no
tab on the staff pages, Cotton-Eyed Joe from the image and the text layer);
`tools/tab-rhythm-accuracy.ts` 221/242 bars summing to the meter and
715/724 durations agreeing, as before; the owner's handwritten Shortnin'
Bread reads the same events (15, 17, 18, 19).

What Weeping Willow still loses (10 notes): three 5s on the third string
(lines 2 and 3), a 1 drawn as a tick on the second string (lines 1 and 4),
two 2s on the first string (lines 2 and 4), a 3/2 on line 3, and two 4s on
the third string read "11" and "1"; a stem crossing the bottom line still
reads 5/1 once (line 2, inside a fret's column). Against 4/4, 4 of 18
bars sum: seven are an eighth short or long, a fret not read or a beam
missed (a quarter for an eighth), and seven hold notes with no stem read
(the pickup and the endings among them).

## Notes whose fret was not read, and a pen's stems and beams (2026-11-01)

The owner chose, for a bar with a fret missing: report the note with its
duration and let the app show "?" (option (a)). Evidence on the page only: a
stem with its beam under the tab and no fret read above it.

**The shape.** `TabReading.unreadFrets?: TabUnreadFret[]`, each `{ x, stem,
reason: "stem_without_fret", string?, text? }`, apart from `events`:
`TabPlacedEvent.notes` and `TabTuneEvent.notes` are non-empty tuples and the
answers name events by index, so an event without a fret could not join them
without breaking both. `string` and `text` come from a mark standing over the
stem that read as no fret (an unread mark, a technique letter, a short mark):
Weeping Willow's lost 1s carry "1" and "f", its 2 on the first string "Z", a
5 "S". Bars count an unread fret's time (`TabBarCheck.unreadFrets`), and the
tune holds it in `TabTuneBar.unread`, `{ before, duration, index, system,
string?, text? }`, standing before the bar's `events[before]`. Everything is
optional: a reading without unread frets resolves exactly as before.

**Finding the stems** (`strayStems`, `ownerOf`). A thin stroke crossing the
row one spacing under the bottom line that no fret column's stem took, at
least 1.5 spacings long, starting between the top line and 0.75 spacing
under the bottom one, with a beam or flag at its end. Next to one stemless
fret column it is that column's stem (a pen hangs it from a wide zero's
side, out of `stemReach`); next to a stemmed column, or to two columns, it is
nobody's; near none, it is an unread fret, unless a dead note's `x` stands
over it. Each condition was earned: without the length and start, the
letters of "DADGAD" under Brian Boru's March's clef made seven notes; without
the `x` rule, Moon over Shanghai's two dead notes; without the two-column
rule, a stem went to the wrong voice of a Moon over Shanghai chord.

**A pen's stems and beams.** A stem's path now skips one empty row (a pixel's
break in the stroke), a beam may lie up to 0.4 spacing under the stem's end
(Weeping Willow's stems stop up to six rows short of it), and a beam counts
when `beamMinThickness` of its rows at the far column are ink (its end
frays).

| Weeping Willow, against 4/4 | bars that sum |
|---|---|
| after steps 1 to 3 | 4/18 |
| unread frets reported | 11/18 |
| a pen's stems and beams | 15/18 |

The three bars left hold notes the page draws with no stem at all (the
pickup's three, the last bar's first note, two notes of the first ending):
nothing to read, so they stay to the musician. Notes read stay 119 of 129;
the eight unread frets stand at eight of the ten notes lost, each with its
duration.

Printed pages: every fret and every stem as before on the 16 pages. One
unread fret appears, on Moon over Shanghai's first system: a stem and double
beam under no fret where Guitar Pro leaves out a tied note's fret, which
makes that bar sum (`tools/tab-rhythm-accuracy.ts` 221 to 222 of 242 bars,
715 of 724 durations as before). Staff pages: no tab, as before.
`--layer` could not be rerun: its source PDFs are no longer where
`tab-rhythm-truth.json` points (tab-import-samples/Angelina baker ...).

## Weeping Willow: the last lost frets (2026-11-07)

Each of the ten notes the page still lost (and the one extra 5/1) traced to
where it went wrong, with the marks' boxes, RapidOCR's text and the
classifier's guess dumped per system (`.scratch/ww/t5-marks.ts`, local):

| lost mark | where it went wrong | fixed by |
|---|---|---|
| two 4s on the third string, read "11" (line 2) and "1" (line 4) | line erasure: the 4's crossbar lies along the line and went with it, so the 4 fell in two, its diagonal and its upright; on line 2 the two pieces merged and read "11" (the classifier said 4 at 0.99, but a two-digit read is never replaced), on line 4 they stood 4 px apart, past `mergeGap`, a short piece and a "1" | 1. a pen over printed rules |
| the 3/2 on line 3 | line erasure: the 2's diagonal crosses the line where its hook's tail stands over it, so neither side was one stroke leaving the line (bridgeJogs); the hook stayed alone, a short mark read "1" | 1. |
| two 5s on the third string, line 2 | segmentation: 1.47 by 1.36 and 1.30 by 1.36 spacings with their tails, over `READ_MARK`'s 1.35 by 1.3; no mark at all | 2. `READ_MARK` 1.45 tall, 1.55 wide |
| the 2 on the first string, line 2 | segmentation: 24 px, 1.30 spacings tall, a pixel over the bound | 2. |
| the 2 on the first string, line 4 | the trust rule: RapidOCR reads "Z", the classifier 2 at 0.91, under `DIGIT_SURE` (0.95) | 4. a fret a stem proves |
| the 1 drawn as a tick on the second string, line 1 | segmentation: 14 px tall against a median of 19, under `minMedianHeight`, so a short mark, never a fret (RapidOCR "1", the classifier 1 at 0.45) | 4. |
| the 1 drawn as a tick on the second string, line 4 | the trust rule: RapidOCR "f", the classifier 1 at 0.81 | 4. |
| the 5/1 of a stem crossing the bottom line (line 2) | segmentation: the band read for marks ends 0.75 spacing under the bottom line and cut a pen's stem starting 0.3 spacing above it into a stroke 1.08 spacings tall, a "1" (the classifier said not a digit, 0.82) | 3. a stroke the band's edge cuts |
| the third 5 on the third string, line 3 | segmentation: the 5's flag is a separate piece, one pixel off its body in a pen's thin stroke; the body reads "S" (a slide letter), the classifier 5 at 0.59 | not fixed |

The classifier was not the limit: where a mark reached it whole it read the
hand (4 at 0.98 to 1.00, 2 at 0.91, 1 at 0.81); what failed was the line
erasure, the size bounds, a band edge and the trust rule. No model was
retrained, and nothing changes for the owner's hosts.

1. **A pen over printed rules** (`penLevelOf`, the `pen` option of
   `bandWithoutLines`, `marksOf` and `inkWithoutLines`). On the owner's
   scan the rules are gray, 120 to 130 at their darkest across a line, and
   the pen black, 0 to 30. When a group's lines are at least `penLineGray`
   (64) at their darkest (the median over their columns), a pixel on a line
   darker than `penShare` (half) of that is the pen's and the erasure keeps
   it. A printed page's rules are as black as its digits, the level is
   undefined and the erasure is as before. Weeping Willow's levels: 62.5 to
   66.5. A first try generalised bridgeJogs instead (a stroke crossing the
   line beside one leaving it), which joined a printed 5 on Killdare Fancy
   to its neighbour and lost it; it was dropped.
2. **`READ_MARK` 1.45 by 1.55 spacings** (was 1.3 by 1.35), reading a
   system already found to be a tab; the guard's printed bound is unchanged,
   and a mark taller than a printed fret must still be wider than a stroke.
3. **A stroke the band's edge cuts is no mark** (`marksOf`): narrower than
   `strokeWidth`, touching the band's first or last row, its ink running on
   past it. A fret on the outer line stops short of the edge.
4. **A fret a stem proves** (`stemFretOf`, read.ts). A stray stem (a beamed
   stem under no fret, 2026-11-01) is a note. The mark over it that read as
   no fret becomes its fret when the classifier names a digit at
   `STEM_DIGIT_SURE` (0.8) or more, or, as a 1, when it is a tick: one
   stroke per row, its median row no wider than 0.4 spacing, at least 0.6
   spacing tall and no wider than 0.6 of its height (Weeping Willow's ticks
   lean 0.36 and 0.47; a hand's 5, also one stroke per row, 0.72). A
   technique letter stays a letter. The system's columns and rhythm are then
   read again with those frets in. Away from a stray stem nothing changes:
   `DIGIT_SURE` still holds for every other mark.

| step | Weeping Willow notes | per line | printed notes changed (16 pages, 2185) |
|---|---|---|---|
| aa058be | 119/129 | 26, 28, 30, 28, 7 | - |
| 1. pen over printed rules | 122/129 | 26, 29, 31, 29, 7 | 0 |
| 2. and 3. size, band edge | 125/129 | 27, 32, 31, 29, 7 | 118 (below) |
| 4. a fret a stem proves | 128/129 | 27, 32, 31, 31, 7 | as 2. and 3. |

Unread frets left: one, line 3's "S" over its stem, with its time. Bars
summing to 4/4: 15 of 18, as before (the three left hold notes drawn with
no stem: the pickup, the last bar's first note, two notes of the first
ending).

**The printed pages change, each change a correction.** The bar for this
work was no printed note changed; 118 change, and every one was checked
against a truth:

- *Killdare Fancy*, all four systems (99 notes): its digits are printed 1.41
  spacings tall on close lines, over the old bound, so its systems read 5,
  7, 9 and 5 notes, many of them pieces ("2/1"). Now 31, 26, 29 and 26, and
  every right note read before is still read. System 1 against
  `tab-truth.json`: all 31 read are in the truth's order (37 notes; 3
  before). Systems 2 and 4 checked by eye: every note read is right; what
  they lose are 5s with the line through them and one 3/5. System 3 was not
  checked note by note. This is the page
  "Reading four- and six-line groups' marks as well" (2026-10-31) could not
  read.
- *Blackberry Blossom, Old Joe Clark, Moon over Shanghai* (Guitar Pro, 19
  notes): false 5/1 and 6/1, the piece of a stem the band's edge cut under
  the bottom line, joined to real columns as a chord (the classifier called
  them not a digit at 0.95). They are gone; `tools/tab-rhythm-accuracy.ts`
  matches more events to the MusicXML: Blackberry Blossom 78 to 86, Old Joe
  Clark 142 to 148, Moon over Shanghai 150 to 155.
- *Shortnin' Bread* (the owner's other hand page) gains its first note, a
  3/0 drawn 1.35 spacings tall (step 2).

Rechecked: `tools/tab-rhythm-accuracy.ts` 222 to 223 of 242 bars summing to
the meter, durations agreeing on 734 of 743 matched events (715 of 724
before: 19 more events matched, the same 9 disagreeing); Reel de Mattawa at
200, 300 and 400 dpi and Marche des élèves: no tab, as before; Sourwood
Mountain unchanged; `npm test` passes (68 files, 1191 tests, the local
suites included: `tab-local.test.ts`'s 21 pages, the button tab's fixtures
38/38 straight and turned, and the Loudéac photo 136/136 through
`BUTTON_TAB_READ_PAGES`).

## What remains

- Rests, tuplets, two voices in one tab, stems above the tab (all reported
  as bars that do not fit, never guessed).
- Chord symbols: in the text layer they are plain runs above the tab; they
  are not attached to events yet.
- Several pages of one tune resolve together when their readings are passed
  in page order (Billy in the Lowground's page 2 prints no tuning and takes
  page 1's); the app has to keep page 1's readings for that.
- Scans and photos: unchanged from tab-2 and tab-3, but for a turned scan's
  lines, a hand's large digits and slanting stems, and a hand's digits read
  by the classifier (Weeping Willow, above).
- Weeping Willow loses 1 of 129 notes, a 5 whose flag a thin pen stroke
  left a pixel off its body, reported as an unread fret with its time; 15 of
  its 18 bars sum to 4/4, the other three holding notes drawn with no stem.
- Dead notes ("x") are annotations, not events, so their time is not
  counted; and a tied note's fret Guitar Pro leaves out comes back as an
  unread fret, not as the fret its tie gives.

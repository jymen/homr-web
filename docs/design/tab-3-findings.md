# Tab 3 findings: tuning and capo from the page

Step 3 of the line-tablature plan in `.scratch/tab-sketch/REPORT.md`: the
text a tab page prints about its tuning and capo, read with the RapidOCR port
the library already ships, parsed into a typed result and tied to each tab
system. Measured 2026-10-09 on arm64 macOS, Node 23.5.0, onnxruntime-web
1.30.0 (wasm, one thread), @techstark/opencv-js 4.12.0. Every decision below
has a `tab-3` row in `docs/decisions.tsv`.

## What the pages print

The prototype's report said nine of its ten pages print a tuning or a capo.
Five do. The other five print nothing about tuning: a photographed DADGAD
page whose tuning is only in its file name, a photographed standard-tuning
page, a mandolin page, a scanned mandolin page that labels its tab
"Mandolin", and a banjo page the prototype assumed was in open G. Over the 23
local pages measured (the prototype's ten, three guitar pages and the ten
TablEdit and Guitar Pro PDFs), 10 print a tuning and 5 a capo, in four places:

| where | example |
|---|---|
| the header, above the first tab | "aDADE tuning, Brainjo level 3", "gDGBD" under the title, "Key Of A (Capo 2)", "Capo 2 Sawmill tuning" |
| just above the first bar | "Capo 2" in italics over bar 1 (Guitar Pro) |
| the left margin, turned a quarter | "gDGBD" beside the TAB clef, centred on staff and tab together |
| under the first system | "DADGAD" at the left |

## The result type

```ts
interface TabReading extends TabSystem {
  // ...events, annotations, unread as in 0.4.0
  readonly tuning?: TabTuning;
  readonly capo?: TabCapo; // { fret, text, confidence }
}

type TabTuning =
  | { status: "read" | "string_count"; source: "text" | "named";
      strings: readonly string[]; text: string; confidence: number }
  | { status: "unknown_name"; text: string; confidence: number };
```

- **Per system, not per page.** `pitchTab` works on one reading, line counts
  live on the system, and a margin label speaks for its own system. A header
  speaks for every system, and is resolved against each system's own line
  count, so a page with a four-line and a six-line tab gets the right answer
  on each, or `string_count` on the one it does not fit.
- **Optional fields, so 0.4.0 needs no other change.** Absent means the page
  says nothing. Nothing is ever filled in: `standardTuning(lines)` is an
  exported helper a caller applies on purpose.
- **`strings` are pitch names, top line first**, ready for `pitchTab` and
  for display. `text` and `confidence` let the app show what was read and how
  sure the recogniser was, since a wrong tuning moves every pitch.
- **`pitchTab(reading)`** with no second argument uses the read tuning and
  capo (no printed capo is capo 0) and throws when no fitting tuning was
  read; a tuning passed in wins, which is how a user's correction applies.
  `tuningOf(reading)` gives the read tuning in `pitchTab`'s terms.

## The parser

`src/tab/tuning.ts` is pure and table-driven: one recognised line in, a
`ParsedTuning` and a capo out, then `resolveTuning` ties it to a line count.

- **Letters are printed low to high, bottom line first.** A guitar's EADGBE,
  a mandolin's GDAE, a banjo's gDGBD whose first letter is the fifth string,
  the bottom line in banjo tab (MusicXML exported from the Guitar Pro sample
  confirms it: line 1, the bottom, is G4). The resolver reverses to top line
  first.
- **Octaves come from the standard tuning for that string count**: each
  letter takes the octave nearest the same string of EADGBE, gDGBD or GDAE.
  aEAC#E becomes A4 E3 A3 C#4 E4 and f#DF#AD F#4 D3 F#3 A3 D4, as players
  read them.
- **Forms**: compact letters ("gDGBD", "aEAC#E", "gDGBbD"; first letter any
  case, the rest capitals, # and b, 4 to 6 notes), spaced letters ("D A D G
  B E", "D-A-D-G-A-D", "Ré La Ré Sol La Ré" after a tuning word), string
  assignments ("(6)=D", "6=D 5=G", "⑥=D", the rest standard, as the
  convention means), names, and per-line labels in the margin (one note name
  beside each line, top line first).
- **Names go through one registry per string count**: "Open G" is DGDGBD on
  six lines and gDGBD on five. English and French ("sol ouvert", "Double do",
  "Drop ré"). A name is looked up for the system's count first, then the
  others, which gives `string_count` for "Sawmill" over a guitar tab.
- **Guards against titles**: spaced letters and names count only beside a
  tuning word (tuning, accord, accordage) or alone on their line, so "Jazz
  Standard" and "A B C D" in a sentence are not tunings. A tuning word with
  nothing the parser knows is `unknown_name`, kept as text.
- **Capo**: "Capo 2", "Capo II", "capo on 2nd fret", "Capo 3rd", "Capodastre
  en 3e case", "Capo : 5", "No capo" and "sans capodastre" (fret 0), frets 0
  to 12.
- **Normalisation**: accents and stray diacritics dropped (the recogniser
  read "Standard" as "Štandard" on one page), ♯ and ♭ as # and b, circled
  string numbers as "(6)".

`test/tab-tuning.test.ts` holds 77 cases.

## Reading the text

`src/tab/text.ts` reads three kinds of region with `RapidOcr.read`:

- **the header**, the full width from the page top to 0.75 spacing above the
  first tab;
- **the line under the first system**, 0.75 to 3.5 spacings below it;
- **each system's left margin**, from 5 spacings above its top line (or the
  system above) to 1 below its bottom line. A label centred on a staff and
  its tab starts about three spacings above the tab; with the first reach of
  1 spacing, Blackberry Blossom's label was cut in two and read in neither
  half.

Each line read is parsed. The best tuning for a system ranks, in order: fits
the line count, from a margin, printed letters over a name, the recogniser's
score. A margin label holds for the systems after it until the next one, as
an instrument label does: Guitar Pro prints the tuning beside the first
system only.

**Detection limit.** RapidOCR's default brings the short side up to 736. A
2480 by 102 strip became 18 000 by 736 and took 2.3 s with nothing on it. The
regions are read with `limit_type` "max" and a side of 1280, an option added
to `RapidOcr.read` whose default stays RapidOCR's, so the chord strips and
the golden OCR tests are unchanged:

| detection long side | readings on 23 pages | text time a page (Node) |
|---|---|---|
| 2000 | all correct | 2.4 to 7.2 s |
| 1280 (chosen) | all correct | 1.3 to 5.2 s |
| 960 | all correct | 1.0 to 3.5 s |

1280 keeps header text at 300 dpi 20 px tall or more for the detector; 960
gave no reading worse but leaves 14 px, near what the detector finds.

## In the pipeline

`recognizePage` reads the text right after the frets, on a page with at least
one tab, opening the OCR detector and classifier (the recogniser is already
open). A failure there logs a line and leaves every system without a tuning,
as the chord strips fail alone; a cancel still cancels. The `tab` progress
stage counts the systems plus one for the text. It runs whether `ocr` is
true or false: the text belongs to the tab reading, not to the chord strips.
A page without a tab never reaches it, so the golden pages and their
model-opening order are untouched.

## Accuracy

`tools/tab-accuracy.ts --text` on 23 local pages, the systems' most common
reading per page. A printed value read right is correct; nothing printed and
nothing read is none.

| page | tuning | capo |
|---|---|---|
| Cold Frosty Morning | correct, from the margin "GDGBD" (the header's "Standard Open G Tuning" agrees) | none |
| Chicken reel | none | none |
| Cluck Old Hen | correct, "gDGCD" (and "Sawmill") | correct, 2 |
| Cripple Creek, tab only | correct, "gDGBD" | correct, 2 |
| Cripple Creek, with score | correct | correct, 2 |
| Sourwood Mountain | correct, "aEAC#E tuning" | none |
| 'Nuff Said (photo) | none (no tab detected) | none |
| La foule (photo) | none (no tab detected) | none |
| Arkansas Traveler (mandolin) | none | none |
| Kildare Fancy (scan) | none | none |
| Amazing Grace p1 | none | none |
| Moon over Shanghai p1 | none | none |
| Brian Boru's March | correct, "DADGAD" under the first system | none |
| Angelina Baker, tab only | correct, "aDADE tuning" | none |
| Angelina Baker, tab and score | correct | none |
| Arkansas Traveler (guitar), tab only | none | none |
| Arkansas Traveler (guitar), tab and score | none | none |
| Blackberry Blossom, tab only | correct, margin "gDGBD" | none |
| Blackberry Blossom, tab and score | correct, margin "gDGBD" | none |
| Moon over Shanghai, tab only | none | none |
| Moon over Shanghai, tab and score | none | none |
| Old Joe Clark, tab only | none | correct, "Capo 2" over bar 1 |
| Old Joe Clark, tab and score | none | correct, 2 |

Tunings: 10 found, 10 correct, 0 wrong, 0 missing; 13 pages print none and
none was read. Capos: 5 found, 5 correct, 0 wrong, 0 missing; 18 none.

The typeset fixtures (`test/tab-text.test.ts`, eleven header pages and the
three fret pages) cover what the local pages lack: French, guitar and
mandolin text, Roman and ordinal capos, string assignments, a named tuning,
per-line labels, a tuning for the wrong string count and an unknown name.
All fourteen read as written.

## Speed

Node, one wasm thread, models open, text reading alone (the frets and the
detection are as in tab-2):

| page | detect | text |
|---|---|---|
| Brian Boru's March | 0.04 s | 1.3 s |
| Chicken reel | 1.3 s | 1.5 s |
| Cluck Old Hen | 0.9 s | 1.7 s |
| Cripple Creek | 2.4 s | 2.1 s |
| Cold Frosty Morning | 2.7 s | 2.2 s |
| Kildare Fancy | 0.04 s | 1.8 s |
| Arkansas Traveler (mandolin) | 0.04 s | 2.5 s |
| Blackberry Blossom, tab only | 1.7 s | 3.0 s |
| Cripple Creek, with score | 1.8 s | 3.7 s |
| Amazing Grace p1 | 0.04 s | 5.2 s |

1.3 to 5.2 s a page with tabs, most of it detection over the header band;
the slowest is a page whose header band holds a voice staff, lyrics and
chord diagrams above the first tab. A page without a tab pays nothing.

## Failure analysis

- **None wrong, none missing on these pages**, so the failures are of the
  guard, met on the way. On the Guitar Pro tab-and-score PDFs it takes
  staves for tabs: Blackberry Blossom reports 5 tabs for 3 (a staff whose
  top line it missed, read as four lines, and a staff whose fingering digits
  read as frets), Old Joe Clark 7 for 4. The text reader then gives the false
  first system nothing, since the margin label sits beside the real first
  tab. The accuracy table scores the systems' most common reading for that
  reason. These pages were not in step 1's 15-page truth; they belong to it.
- **The guard finds no tab on the two phone photographs**, so their text is
  never read; neither prints a tuning.
- **A label split across regions is lost.** The header band and a margin
  overlap on purpose; a label longer than the margin's reach of 5 spacings
  above the tab would still be cut.
- **Kildare's "Mandolin" label is not read as standard mandolin tuning.** An
  instrument name implies a tuning to a player, but the library leaves that
  to the caller (`standardTuning(4)`).

## What remains

- **The app's correction UI.** A wrong tuning moves every pitch; the result
  carries `text` and `confidence` so the app can show what was read and let
  the musician change it, passing their own tuning to `pitchTab`.
- **The guard's false tabs on Guitar Pro score-and-tab pages** (above), a
  step-1 fix.
- **Step 4, rhythm**, and **step 5, scans**, unchanged from tab-2.

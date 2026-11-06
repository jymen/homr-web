# Button tab findings: a page x per note event

Question (plan A1 in the app's `docs/homr-web-plan/button-tab.md`): can
homr-web give each note it reads a page-normalised x, accurate enough to tie
a printed button number, written under the note head, to its note?

Measured on 2026-11-05 with a throwaway script outside the repo, on two
pages: `the-kesh-300dpi.png` (typeset, 4 staves) and a private phone photo
of a commercial melodeon book (4 staves, shot at an angle, not
redistributable, described here by aggregate numbers only).

## Method

For each staff, the script runs `prepareStaffImage`'s steps while keeping
the geometry (scaling, both crops' offsets, the dewarp transform, the resize
onto the 1280 by 256 canvas), decodes the canvas, and compares two things in
canvas space:

- each decoded note's `coordinates.x`, the decoder's cross-attention point;
- the noteheads segnet found on that staff (`staff.symbols` of kind `note`),
  mapped into canvas space through the same crop, dewarp and resize, then
  grouped into columns (heads closer than 0.8 unit size).

Errors are in units of the local note spacing (the smaller gap to the
neighbouring column). On both pages the decoded note count equals the
notehead column count on every staff, and the decoded pitch of the k-th note
equals the staff position of the k-th column on every note (offset 29 steps
for a G clef), so pairing the k-th note with the k-th column is the truth
for these pages.

## Numbers

| | Kesh | Photo |
|---|---|---|
| note events / notehead columns | 80 / 80 | 136 / 136 |
| symbols with null coordinates | 0 | 0 |
| backward steps of attention x along a staff | 11 | 8 |
| attention to the right column, median | 0.46 spacing | 0.21 |
| same, 90th percentile | 1.79 | 1.06 |
| same, max | 4.21 | 6.80 |
| notes more than half a spacing off | 39 | 25 |
| nearest column is the right one | 44 / 80 | 114 / 136 |
| staves where nearest-column snapping is one-to-one and in order | 0 / 4 | 0 / 4 |

The attention point lags: on the Kesh page its signed median is -0.46
spacing, and it drifts by several spacings inside runs of eighths. The
dewarp moves no notehead's x at all on either page (0.000 canvas px: it
only straightens in y), so it is not the cause.

## Alignment instead of snapping

An order-preserving alignment (Needleman-Wunsch, gap 1) of note events to
columns, cost = pitch against staff position (0 same, 0.5 one step off, 1
otherwise) + w times the attention distance in spacings (capped at 3).
Robustness was tested by removing columns (missed noteheads) and half as
many events at random, 20 draws per rate:

| cost | drop | Kesh right / wrong | Photo right / wrong |
|---|---|---|---|
| attention only | 0 | 45% / 44% | 88% / 7% |
| pitch only | 0 | 100% / 0% | 100% / 0% |
| pitch + 0.25 attention | 0 | 100% / 0% | 100% / 0% |
| pitch only | 10% | 97% / 5% | 93% / 9% |
| pitch + 0.25 attention | 10% | 99% / 3% | 99% / 3% |
| pitch + 0.25 attention | 20% | 96% / 9% | 96% / 11% |
| pitch + 1.0 attention | 0 | 83% / 6% | 92% / 3% |

(Right is a share of the pairs that still exist after the drop; wrong pairs
are counted against the same base.)

## Conclusion

Neither attention alone nor attention snapped to the nearest notehead is
enough: the attention point is off by more than half a spacing on a fifth to
a half of the notes. What works is the notehead's own x, tied to its note
by an order-preserving alignment whose cost is mainly the decoded pitch
against the notehead's staff position, with attention as a weak tie-break
(w = 0.25). On both pages that gives every note its column; with a tenth of
the noteheads missing it still pairs 99% correctly.

## What H4 would add

`RecognizeSuccess.noteColumns: readonly (number | null)[]`, one entry per
`<note>` element of `musicXml` in document order (chord members and rests
included), each the page-normalised x of the notehead column tied to that
note, `null` when the alignment left it unpaired (a rest, a missed head).
Normalised like `staves` (by the preprocessed page's width), so it lines up
with `StaffBox` and, through the same autocrop, with the button marks.

Where it is computed:

1. `prepareStaffImage` also returns the canvas-to-page x map (scaling, the
   two crop offsets, `size.width / cleaned.width`); attention is mapped to
   page x with it. The dewarp can be ignored for x.
2. `parseStaffCanvas`'s caller in `parseStaffs` aligns the staff's decoded
   notes (chord members grouped into one event) with `staff.symbols`' notes
   (page space, with `position`), as above, and writes the column's page x
   on each note symbol: a new optional field on `EncodedSymbol`, carried
   like `coordinates` through `joinVoice` and `removeDuplicatedSymbols`.
3. `generateMusicXml` collects that field from `buildNoteOrRest` in emission
   order (the generator sorts chords and splits positions, so the symbol
   order is not the `<note>` order), and returns it beside the XML.

The clef offset (29 for G2, 17 for F4) comes from the staff's decoded clef;
a lift (sharp, flat) does not change the staff position.

# H0: typeset CADB fixtures (2026-11-05)

`tools/typeset-tab.mjs` now also writes `test/fixtures/button-tab/`: a
straight page and the same page turned 0.3 degrees, three systems of an
original 3/4 tune, and `truth.json` with one `ButtonTabReading` per staff,
written in the same run. Decisions are `btab-0` in `docs/decisions.tsv`;
the file format is in the folder's `SOURCE.md`.

How it is built: the staff is abcjs 6.6.4's, rendered by headless Chrome
inside the script (each line's markup, bars and note head boxes read back
from the DOM), scaled to the 2080 px system and placed per system so the
bands fit under it. The bands, numbers and letters are the script's own
SVG, measured and placed on whole pixels like the string pages' frets. The
string pages come out byte-identical, and two runs give identical button
pages.

What was checked:

- By eye on crops of both PNGs: every number sits under its note head in the
  band of its bellows, suffixes legible (`7'`, `3''`, `8'-`, `10'`), the
  rule, band bars and margin P/T aligned with the staff's ends and bars, the
  letter line under T, the turn visible on the rotated page.
- `test/button-tab-truth.test.ts`: the truth parses into the H2 shape with no
  extra key, both pages carry the same marks, every x lies in its staff's
  extent, and each mark has ink at its x inside its band.

What the renderer does not do, and what H1/H2 should know:

- One note head per note: the script refuses chords and grace notes
  (abcjs's grace heads are not `.abcjs-notehead` in a note group). The
  "note without a number" is an ordinary note.
- `''` is two ASCII apostrophes, which Roboto draws like a straight `"`: the
  reader must take that glyph as row 2.
- The band bars are separate from the staff's (a 3.4-spacing gap); the staff
  itself has no left barline, as abcjs draws a single staff.
- An `11`'s ink centre is the white gap between its digits, so anything that
  samples a narrow column at a mark's x finds nothing there.
- Only one font (Roboto) and one size; no knock-outs, no noise, no
  perspective. The Loudéac photo stays the hard case.

# H4: note columns (2026-11-06)

`RecognizeSuccess.noteColumns` now gives, per `<note>` of `musicXml` in
document order, the page-normalised x of the notehead tied to it, or null.
Built as "What H4 would add" planned; the decisions are `btab-1` in
`docs/decisions.tsv`.

- `prepareStaffImageAndPageX` returns the canvas with a linear
  canvas-to-page x map; `prepareStaffImage` is unchanged.
- `src/pipeline/note-columns.ts` holds the alignment (pure, unit-tested in
  `test/note-columns.test.ts`) and `withNoteColumns`, which `parseStaffs`
  runs on each staff right after decoding. It aligns in page space.
- The column rides on `EncodedSymbol.column` and is collected in
  `buildNoteOrRest`; `generateMusicXmlWithColumns` returns it beside the XML,
  `generateMusicXml` is unchanged.
- `noteColumns` is optional in the type, so hand-built results in callers
  still compile; `recognizePage` always sets it.

Measured with the library code itself (`test/note-columns-page.test.ts`,
the photo passed through `NOTE_COLUMN_PAGES` and never stored):

| | Kesh | Photo |
|---|---|---|
| right pairs per staff | 21/21, 18/18, 19/19, 22/22 | 31/31, 32/32, 36/36, 37/37 |
| total | 80/80 | 136/136 |

As the throwaway measurement predicted: on every staff the alignment pairs
the k-th note with the k-th column, and every pair's pitch sits on its head.

On the typeset CADB fixtures (`test/fixtures/button-tab/`), every one of the
39 notes per page gets a column, and every mark and left-hand letter of
`truth.json` lies within 1.6 px of a column on the straight page and 2.4 px
on the turned one (a note head is about 25 px wide). The turned page's
extra pixel is probably the 0.3 degree turn itself: the truth's x is taken
on the band row, some 2 spacings below the head. Not checked further.

Not covered: a grand staff gets null for every note (its positions span
both staffs with an unknown gap); the drop robustness above (99% at a tenth
of the heads missing) was simulated, not seen on a real page.

# H1: bands (2026-11-06)

`src/tab/button.ts` finds the push/pull bands under each staff and paints
them out, with the letter line under them, before homr reads the page.
`recognizePage` runs it after the string-tab guard; when it finds nothing
the page is left as it was and no log line is added. Decisions are
`btab-2` in `docs/decisions.tsv`.

How a band is found:

1. Ink against the local mean, not Otsu: the photo's paper darkens toward
   its bottom, and Otsu took the last system's paper for ink.
2. Lines traced strip by strip (1/96 of the page width) and chained left to
   right on their own slope. detect.ts's whole-row cover finds no staff at
   all on the photo, which is turned about 3 degrees and bends.
3. A rule is a long traced line with a staff above it (five evenly spaced
   lines, the lowest 1.5 to 10 spacings up) and no line close below it. Its
   ends are followed to the pixel along its ink, through the barlines.
4. Only then is the recogniser asked: among the two leftmost letter-sized
   marks above the rule's left end, one must read P; below it, one must read
   T. The letters stand left of the rule on the typeset page and right of
   the left barline on the book page; both are searched.
5. Each band is twice its letter's centre away from the rule. The letter
   line is the first letter-tall run of ink under the T band, found in
   chunks of 10 spacings along the rule (bars coming out of the band are
   dropped first), and painted to its end.

`withStaves` gives each band the index of the staff above it among homr's
staves, once homr has read the page. In H1 it only reaches the log
(`Button tab guard: N push/pull bands painted out, under staves ...`);
`RecognizeSuccess` is unchanged until H4.

Measured (`test/button-tab-guard.test.ts`; the photo through
`BUTTON_TAB_PAGES` and never stored):

| | straight | turned 0.3° | book photo |
|---|---|---|---|
| bands found / systems | 3 / 3 | 3 / 3 | 4 / 4 |
| bandTop, rule, bandBottom against truth.json | within 1.1 px | within 1.1 px | no band truth |
| x0, x1 against truth.json | within 0.9 px | within 0.9 px | |
| recogniser calls | 3 | 3 | 4 |

No band on 62 other pages (the public staff fixtures, the string-tab
fixtures, and the local staff and tablature pages), and the recogniser is
not asked at all on Mattawa, Gavotte Koun, Flatworld, Weeping Willow or the
three public staff pages: no long line there lacks a neighbour a spacing
away. Marche des élèves asks once (two crops, read ")" and "1").

On the photo the band heights grow from top to bottom, 0.054, 0.058, 0.060
and 0.064 of the page height (P and T bands together) with line spacings
of 21 to 25 px: in the upright 3024 by 4032 image the camera was nearer the
page's foot. The plan's 0.045 to 0.031 was measured another way; the guard
takes each band's own size from its letters, so either holds.

Through `recognizePage` (CPU, with the chord OCR), before and after:

| | Mattawa, Gavotte Koun, Flatworld, Weeping Willow | straight, turned | book photo |
|---|---|---|---|
| result (musicXml, tablature, texts, staves, log) | byte-identical | | |
| musicXml | | identical | 136 notes before and after; two grace flags differ |
| chord texts | | 20 before (the letter lines) / 1 and 0 after | 20 before (16 of them letters) / 4 after |

The two photo notes: in system 1's last bar the grace B is now read as one
(it was an eighth), in system 2's first bar the grace B is now an eighth
(it was a grace). The staff crop no longer holds the bands, and the
decoder moves on that boundary; no note is gained or lost and every note
still gets its column. The one text left on the straight page is an "e"
read off the ledger-line notes above staff 1, which the letter line used
to hide in the same strip.

What the guard does not do:

- Read anything in the bands: the numbers are H2, the letter line H3.
- Find a band whose rule is shorter than a fifth of the page, or whose P or
  T the recogniser misreads; a page with button tab and no letters is not
  recognised as one.
- Look for Corgeron tab (one band per row, several rules), which the plan
  leaves for later.

# H2: numbers (2026-11-06)

`src/tab/button-read.ts` reads the numbers in the bands the guard found:
per tab a `ButtonTabReading` (the plan's type, `leftHand` left empty for
H3), each mark with its bellows, button, row (`'`, `''`), held (`-`) and
page-normalised x. `recognizePage` runs it once homr has placed the bands
under their staves, and `RecognizeSuccess.buttonTabs` carries the result
(optional in the type, always set; empty on a page without bands). The
decisions are `btab-3` in `docs/decisions.tsv`.

How a band is read:

1. **Straightened** along its traced rule, one page column at a time (page
   x kept, rows counted from the band's top), so the turned, bent photo
   reads like the typeset page. The rule is then erased with detect.ts's
   `bandWithoutLines`.
2. **Suffix pieces taken out first**, by shape: apostrophes (tall, thin,
   top-aligned with a number) and dashes (flat, at a number's middle). Each
   goes to the nearest number on its left; `''` counts its strokes, so the
   fixture's `"`-like glyph is row 2. A suffix touching its digit (two
   apostrophes joined to a 7's bar on the photo) is split off at the mark's
   thinnest column when the mark is 1.2 to 1.7 median marks wide.
3. **Marks** from `marksInBand`, the part of `marksOf` after the band is
   built (split out, `marksOf` unchanged), on three lines: the P band's
   middle, the rule, the T band's middle. Its spacing is 2.2 digit heights,
   which makes its merge gap join "10" and "11" and keep two numbers apart.
   Marks on the rule and over the margin P and T are dropped.
4. **Digits** read by the tab digit classifier alone when it is trusted
   (btab-4: one piece, at most 1.2 median marks wide, a digit 1 to 9 at
   probability 0.8 or more); every other mark (10, 11, a doubtful digit)
   by the recogniser, as frets are, with the classifier through
   `fuseDigit`. `readButton` keeps 1 to 12. The recogniser's own `'` or
   `-` is used only when the shape found none.

Measured (`test/button-tab-read.test.ts`; the photo and its hand-written
truth through `BUTTON_TAB_READ_PAGES`, never stored; the truth has no x, so
each system's reading is aligned with it in order):

| | straight | turned 0.3° | book photo |
|---|---|---|---|
| marks read / truth | 38 / 38 | 38 / 38 | 136 / 136 |
| exact (bellows, button, row, held) | all | all | all |
| x against truth.json | within 3 px | within 3 px | no x in its truth |
| unread | 0 | 0 | 0 |

Per system on the photo: 31/31, 32/32, 36/36 and 37/37, no miss of any
class (digit, apostrophe, dash, bellows, unread, missed, extra). Two steps
were needed to get there, both measured on this page first:

- the first pass read 134/136 with five false apostrophes on system 4 (a
  number took the next one's apostrophe), fixed by giving a piece to the
  nearest number on its left only;
- the two remaining misses were apostrophes touching their 7 (one read as
  `7` with no row, one as `T`, unread), fixed by `attachedSuffix`, which
  split exactly those two marks and no other on the three pages.

One page of one book, so these numbers say the method holds on an angled
photo, not that every photo reads clean. A lost apostrophe or a misread
digit is what A2's staff cross-check exists for.

Mark x against `noteColumns`: on the photo every mark lies within 10.5 px
of a note column (median 0.8 px; a note spacing is about 60 px), on the
fixtures within 2.3 px. The two are normalised differently, marks by the
page given and note columns by the autocropped page; autocrop left these
pages' width unchanged, but a photo it crops would shift one against the
other, so A1 has to bring them into one frame (see the open point below).

Time, CPU, one wasm thread:

| | straight | turned | book photo |
|---|---|---|---|
| reading the numbers | 2.7 s | 2.7 s | 9.4 to 9.7 s |
| whole `recognizePage`, H1 → H2 | 46.4 → 49.7 s | 46.0 → 49.1 s | 53.4 → 63.9 s |

Those times are the first version, recogniser on every mark: about 70 ms
a mark, as RapidOCR pads every crop to its 320 by 48 line. The classifier
takes under 1 ms a mark and gave the recogniser's digit on every single
digit here, so after the owner's go (2026-11-06) single digits are read by
the classifier alone under a trust rule (btab-4), and only the rest by the
recogniser:

| | straight | turned | book photo |
|---|---|---|---|
| reading the numbers, recogniser on every mark | 2.7 s | 2.7 s | 9.5 s |
| classifier first (btab-4) | 0.3 s | 0.3 s | 0.37 s |
| marks left to the recogniser | 4 of 38 | 4 of 38 | 3 of 136 |
| whole `recognizePage`, classifier first | 46.5 s | 46.3 s | 53.8 s |

The whole page is back to H1's time (46.4, 46.0 and 53.4 s), and its
result is the same as with the recogniser on every mark, field for field.

The readings are the same: 38/38, 38/38 and 136/136. The marks left to the
recogniser are the fixtures' four two-digit numbers, and on the photo its
two 10s and one digit the classifier gave under 0.8. At a 0.5 or 0.9
threshold the readings are the same too (0.9 sends 9 and 6 marks to the
recogniser).

The largest button stays 12 (owner, 2026-11-06): a melodeon numbers its
buttons per row and only a three-row instrument reaches 12; a concertina
seldom goes above 10. A larger number, or a 0, is counted unread.

Other samples, through `recognizePage` against H1's results: Mattawa,
Gavotte Koun, Flatworld and Weeping Willow are byte-identical in every field
(`musicXml`, `staves`, `texts`, `tablature`, `noteColumns`, `log`, `error`),
the only addition being `buttonTabs: []` on the three that succeed. On the
fixtures and the photo `musicXml` is identical to H1's and the log gains
one line, `Button tab reader: 31, 32, 36, 37 marks, 0 unread`.

What H2 does not do:

- Read the letter line (`leftHand` stays empty): H3.
- Tie a mark to its note or check it against the staff: A1 and A2 in the app.
- Read a number above 12, or a 0: counted unread (owner, btab-4).
- Read a page whose bands the guard did not find.

Open points:

- **One frame for marks and note columns.** `buttonTabs` and `tablature`
  are normalised by the page given, `noteColumns` and `staves` by the
  autocropped page. Before A1, either homr-web gives the button marks in the
  autocropped frame too, or the result carries the autocrop rectangle.

# Phase 9 findings: the MusicXML writer and process_image

What the pinned homr 0.7.0 does between the voices `parse_staffs` returns and
the files the server reads, from `music_xml_generator.py`, `main.py`,
`circle_of_fifths.py`, `transformer/vocabulary.py`,
`staff_position_save_load.py` and the `musicxml` 1.4 package in the oracle's
`.venv`. The server runs `homr --write-staff-positions <page.png>`
(`AbcGoDb_V03/abcsql/omr.go`), so every `XmlGeneratorArguments` field is
`None`.

## What is on the inference path

- **`relieur.py` does not exist in 0.7.0.** There is no multi-page
  concatenation to port. A multi-page PDF is one `process_image` per page,
  and joining them is the caller's.
- **`circle_of_fifths.py` is not called.** Nothing imports
  `maintain_accidentals_during_measure` or the key classes; the file is dead
  code in 0.7.0, and with it `EncodedSymbol.change_lift`, its only caller.
- **Of the `vocabulary.py` methods phase 7 deferred, only
  `sort_token_chords` and `EncodedSymbol.__lt__` are reached.**
  `add_articulations`, `add_slurs` and `is_valid` have callers only in
  training code and in `vocabulary.py`'s `__main__`; `strip_articulations`
  and `strip_slurs` are phase 7's. None is ported.
- **`process_image` without file IO, title and debug** is: detect, then
  `parse_staffs`, then `generate_xml(args, result_staffs, title)`, then
  `save_staff_positions(multi_staffs, image.shape, …)` when
  `--write-staff-positions`. The title comes from a RapidOCR future; the port
  passes `""`, which is also what `dump-golden.py` passes, so
  `page.musicxml` holds `<work-title />`. The server's file holds whatever
  title the OCR read. The app takes titles from its own text layer and OCR,
  so the difference is one element the app does not read.
- **`staves` is the server's parse of the staff-positions file**:
  `parseOmrStaves` keeps lines of five numbers, stable-sorts them by `cy` and
  numbers them after the sort. The file prints Python's `str(float)`, which
  is the shortest round-tripping repr, and Go's `ParseFloat` is correctly
  rounded, so the parsed numbers are the float64 values homr computed. The
  port can therefore build `staves` from the same numbers it formats, without
  a string round trip, and still match the server exactly.

## The `musicxml` package reorders children

`XMLElement.add_child` places a child where the MusicXML XSD sequence puts
it, not where it was added. homr adds a note's children in the order chord,
pitch, type, duration, staff, voice, dot, time-modification, notations, and
the file has duration, voice, type, dot, time-modification, staff,
notations. Four elements written by the generator have a sequence that
differs from homr's insertion order:

| element | schema order used |
|---|---|
| `note` | grace, chord, pitch or rest, duration, voice, type, dot, time-modification, staff, notations |
| `pitch` | step, alter, octave (homr adds alter last) |
| `attributes` | divisions, key, time, staves, part-symbol, clef, measure-style (a clef group comes first in the tokens) |
| `barline` | bar-style, ending, repeat |

`measure`, `notations`, `articulations`, `ornaments` and `part-list` are
choice groups and keep insertion order. The vectors confirm both kinds.

The package also validates values: a `<duration>` of 0 raises
`ValueError: XMLDuration.value '0' must be greater than '0'`. It is reachable:
a chord of a triplet half and a quarter (`note_3` with `note_4`) has a
shortest duration of 1/4, so the division is 4 and the final backup is
`int((1/3 - 1/4) * 4) = 0`. The port throws `MusicXmlError` there, which makes
the page `ok: false` as it is on the server.

## Rounding and ordering traps

- **`np.median` of Fractions is exact.** numpy 2.5 on an object array returns
  the Fraction mean of the two middle values for an even count (11/16 for
  3/4, 3/4, 1/2, 5/8). `_get_typical_duration_of_measures` in phase 7 takes
  the upper middle instead; the two functions differ and both are ported as
  they are.
- **`int()` of a Fraction truncates.** The time signature's beats are
  `int(nominator * denominator)` (11/16 of an 8 is 5), floored at 1; every
  `<duration>` and `<backup>` is `int(fraction * division)`.
- **`division` is the LCM of chord durations only**, seeded with 1/4, so the
  longer notes of a chord can truncate (above). `<divisions>` is
  `division // 4`, exact because 4 always divides the LCM.
- **`sort_token_chords` sorts descending by text.** `__lt__` is
  `str(self) > str(other)`, and `sorted` is stable on `__lt__` alone. ASCII
  tokens compare the same in JavaScript.
- **`_group_notes` keys a dict by Fraction and iterates `sorted(by_duration)`**,
  ascending. A grace note's key is 0, a whole-measure rest's (fraction
  numerator 0) is the chord's longest.
- **`into_positions` swaps when the lower staff holds only rests**, and also
  when there is no lower symbol at all, where the swap is invisible because the
  empty side is dropped.
- **`rebalance_measure_voices` sorts events by `(start, end)`.** Within one
  measure the events of a staff arrive in ascending end for equal starts, so
  the `end` key never reorders on the vectors; a mutation that drops it
  survives every test. It is kept as homr has it.
- **Logging.** Every `eprint` in the generator, including the duration
  warning `get_duration` prints for a non-note on each call, is a log line in
  the port, and the vector test compares the lines.

## Bugs ported as they are

- `repeatStart` closes the measure and puts a **forward repeat on the new
  measure's right barline**.
- A non-note, non-grace symbol inside a chord (a clef chorded with a note)
  gets `<type>breve</type>` and a `<duration>` of `state.beats`, which is 64
  until a time signature sets it to the beat count.
- The second `fermata` and `arpeggiate` branches of `build_articulations` are
  unreachable.
- `"spiccato"` is in the vocabulary and raises `Unsupported articulation`.
- A grace note whose kern is not a power of two (`note_3G`) raises
  `KeyError: 3` in `DURATION_NAMES`.

## The oracle

`tools/dump-vectors.py musicxml` runs `generate_xml` on 32 hand-built voice
lists, one per branch, and records the XML, the stderr lines and the error.
`test/musicxml.test.ts` compares them canonically (attributes sorted, text
trimmed, children in order) together with the log, and generates both pages'
`page.musicxml` from their `voices.json`, which come out byte for byte.
Seven single wrong readings of the port were applied one at a time: the
upper-middle median, the lower-rest swap, the active-voice filter, the chord
order, the sort direction, the volta count and truncation versus rounding
each fail a test; the `(start, end)` sort key does not, for the reason above.

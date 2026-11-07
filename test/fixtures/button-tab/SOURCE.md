# Button tab (CADB) fixture sources

These pages are melodeon button tab typeset for this repository: an original
waltz written for it on 2026-11-05, so they carry no third-party rights and
fall under the repository's rule for public fixtures. They exist to test the
button tab reader (docs/homr-web-plan/button-tab.md in the app, H1 to H3)
against exact truth.

`tools/typeset-tab.mjs` draws them in the same run as the string tab pages,
from `BUTTON_TUNE`, and writes the PNGs and `truth.json` together. Neither
is edited by hand: change the spec and rerun.

| File | Content |
|---|---|
| `gc-cadb-straight.png` | Three systems of a treble staff in 3/4, four bars each, with two bands under each staff: P (push) above T (pull), split by one rule, the letters P and T bold in the left margin, a bar segment through both bands under each staff bar. One button number per note in its bellows' band, under the note head: 38 marks, 18 pushed and 20 pulled, rows unmarked (8), `'` (28) and `''` (2), four two-digit numbers (10, 11, 10', 10), three held (`8'-`, `7'-`, `6'-`), and one note with no number (system 3, bar 3, the G). Under the T band, 30 left-hand letters on note columns, set on one baseline as print sets them: 14 capitals for the basses and 16 lower case for the chords (chords `a`, `b`, `c`, `d`, `e`, `f`, `g`; `c` 5 times beside `C` 5 times), the melodeon convention (owner, 2026-11-07). |
| `gc-cadb-rotated.png` | The same page turned 0.3 degrees clockwise about its centre. |

Every page is 2480×3508 (A4 at 300 dpi), 8-bit grayscale. The staff, clef,
notes and bars are abcjs 6.6.4's, rendered in headless Chrome and scaled to
the 2080 px system width (line spacing about 23 px); the bands, numbers and
letters are Roboto, drawn by the script on whole-pixel anchors. Every button
sounds its staff note on a three-row G/C (`melodeonGC3` in the app), so the
page is also a known case for keyboard inference (plan A2).

## Truth

`truth.json` has `keyboard` and one entry per page with `rotation`,
`staves` and `buttonTabs`. Each `buttonTabs` entry is the plan's H2
`ButtonTabReading`: `staff`, `marks` (`x`, `bellows`, `button` as printed,
`row` 0/1/2 for no mark, `'`, `''`, `held` for a trailing `-`), `leftHand`
(`x`, `text`) and `unread` 0. Coordinates are page-normalised.

- A mark's `x` is the ink centre of its digits alone; the suffix trails to
  the right. An 11's centre therefore falls in the gap between its digits.
- A letter's `x` is its ink centre.
- `staves` holds, per staff, `top` and `bottom` (its outer lines), `bandTop`,
  `rule` and `bandBottom`, and `x0`/`x1`, the system's ends.
- On the turned page, every x is the drawn point rotated (a mark from its
  band's centre row), `x0`/`x1` are taken at the bands' middle row and the y
  fields at the system's middle column; the bands drift up to 5.4 px in y
  across a system.

`test/button-tab-truth.test.ts` checks the shape, that every mark and letter
lies inside its staff's `x0..x1`, and that each mark has ink at its x in its
band on both pages.

## Regenerating

Needs everything the string pages need, plus Google Chrome (or `CHROME`) and
abcjs 6.6.4's `dist/abcjs-basic-min.js` (`ABCJS_FILE`, default the app
checkout beside this repository):

```bash
TAB_FONT_DIRS=/path/to/texgyrebonum \
ABCJS_FILE=/path/to/abcjs/dist/abcjs-basic-min.js \
node tools/typeset-tab.mjs
```

The output is byte-identical from run to run on one machine.

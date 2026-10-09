# Tablature fixture sources

These three pages are line tablature typeset for this repository, so they
carry no third-party rights and fall under the repository's own rule for
public fixtures: only pages we typeset ourselves are committed. They exist to
test the tab reader against exact truth, which a scan cannot give.

`tools/typeset-tab.mjs` draws every page from the column specs it holds,
rasterises it with `rsvg-convert` and writes the PNG and `truth.json`
together. Neither is edited by hand: change the specs and rerun.

| File | Content |
|---|---|
| `banjo-5-lines.png` | Three five-line systems, line spacing 42 px, Roboto, a white knock-out behind every number and letter. Single notes on every line, frets 0 to 12, two-digit frets with touching digits, three-note chords, the technique letters Sl, Po, H and R standing alone on a line, one slur arc, and a TAB clef. |
| `guitar-6-lines.png` | Two six-line systems, spacing 31 px, TeX Gyre Bonum, knock-outs. Frets to 12, chords of two to six notes, h, p and x standing alone on a line, Harm. once, and a TAB clef. |
| `mandolin-4-lines.png` | Two four-line systems, spacing 36 px, Roboto. The first has knock-outs; the second has none, so the lines run through the digits. Two-digit frets on both. |

Every page is 2480×3508 (A4 at 300 dpi), 8-bit grayscale, black on white,
with a title line at the top and a bar line at each end of every system.

## Truth

`truth.json` lists, per system, its line count, spacing and `top`, then its
events, technique letters and arc count. Coordinates are continuous page
pixels: a line `top` of 520 is drawn 2 px thick over rows 519 and 520, so a
detector that reports row indices sees its centre at 519.5. An event's `x` is
the centre of its numbers' ink, read from the layout that drew them; each
glyph is placed on whole pixels, so `x` is exact to the half pixel. `string`
1 is the top line. `technique` is one of `Sl`, `Po`, `H`, `R`, `p`, `h`, `x`,
`Harm.`.

Two properties a reader test can rely on, measured on the pages: fret digits
are 0.78 to 0.81 of the line spacing tall, and the slur's ends stop 0.2
spacing above the line, inside the reader's on-line band. Lowercase `x` in
Bonum is only 0.68 of the digit height.

## Fonts

| Font | Used on | Licence |
|---|---|---|
| Roboto (Google) | banjo, mandolin, titles | Apache License 2.0 |
| TeX Gyre Bonum (GUST) | guitar | GUST Font License |

Neither font is committed; only the rasterised pages are.

## Regenerating

Needs `rsvg-convert` and fontconfig (Homebrew: `brew install librsvg`),
Roboto installed, and TeX Gyre Bonum installed or in a directory named by
`TAB_FONT_DIRS`. The script renders through pango's fontconfig backend, so a
font in that directory is found without installing it; it stops with a
message if either family does not resolve.

```bash
TAB_FONT_DIRS=/path/to/texgyrebonum node tools/typeset-tab.mjs
```

The output is byte-identical from run to run on one machine. Another
version of rsvg, pango or FreeType may move antialiasing by a pixel; `git
diff --stat` after a rerun says whether it did.

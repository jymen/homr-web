# Fixture sources

**Rule (owner, 2026-09-25): only pages the app typeset itself may enter this
public repository.** A page typeset from ABC by AbcMusicStudio's own score
renderer carries no third-party rights, so it can be redistributed under
this repository's licence. Every other page, whether a scan of a printed
score, a page from a published collection such as the Nantes page, or a
phone photograph, is private and never committed. Those go in
`test/fixtures/local/`, which git ignores, and their golden data is written
to `test/golden/local/<name>/`, also ignored. The test suite picks both
directories up when they exist and skips them silently when they do not, so
CI runs on the public pages alone and a developer's machine runs on all.

| File | Origin | Rights |
|---|---|---|
| `the-kesh-300dpi.png` | The Kesh Jig, a traditional Irish tune (public domain), typeset from ABC by AbcMusicStudio's own score renderer and exported to PDF, rasterised at 300 dpi (2481×3509). Same file as `static/tests/pdf2abc/the-kesh-300dpi.png` in the AbcMusicStudio repository. | Public-domain tune, rendering produced by the owner of this repository. |
| `grand-staff-300dpi.png` | "Grand Staff Study", sixteen bars for piano written for this repository on 2026-10-02: four systems of two braced staffs, treble and bass. Typeset from the ABC below by abcjs 6.6.4, the app's score renderer, loaded from the app's `node_modules` into a standalone HTML page (`ABCJS.renderAbc` with `staffwidth: 700` and `responsive: "resize"` in a 2481×3509 px page padded 150 px at the top and 140 px at each side) and captured by a headless browser screenshot at that size. Not exported through the app's PDF path, unlike the Kesh page. | Music and rendering produced for this repository; no third-party material. |

```abc
X:1
T:Grand Staff Study
M:4/4
L:1/8
%%score {RH | LH}
V:RH clef=treble
V:LH clef=bass
K:C
V:RH
c2 e2 g2 e2 | d2 f2 a2 f2 | e2 g2 c'2 g2 | f2 d2 B2 G2 |
V:LH
C,4 G,4 | D,4 A,4 | E,4 C,4 | G,,4 G,4 |
V:RH
e2 c2 G2 c2 | f2 d2 A2 d2 | g2 e2 c2 e2 | d2 B2 G4 |
V:LH
C,2 E,2 G,2 E,2 | D,2 F,2 A,2 F,2 | E,2 G,2 C2 G,2 | G,,2 B,,2 D,4 |
V:RH
g4 e4 | a4 f4 | g2 f2 e2 d2 | c2 B2 A2 G2 |
V:LH
C,2 G,2 E,2 G,2 | F,,2 C,2 A,,2 C,2 | G,,2 D,2 B,,2 D,2 | E,2 D,2 C,2 B,,2 |
V:RH
c2 d2 e2 f2 | g2 a2 g2 e2 | f2 d2 B2 d2 | c8 |]
V:LH
A,,4 E,4 | E,,4 C,4 | G,,4 G,4 | C,8 |]
```

The piano page is the fixture with braces: its dump has 25 `brace_dot` boxes
and four grand staffs, where the Kesh page has neither.

Adding a public page: typeset it in the app, export the PDF, rasterise at
300 dpi, add a row here, run `npm run golden`. Adding a private page: drop
the PNG in `test/fixtures/local/` and run `npm run golden`; nothing else
changes, and `git status` must stay clean.

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

Adding a public page: typeset it in the app, export the PDF, rasterise at
300 dpi, add a row here, run `npm run golden`. Adding a private page: drop
the PNG in `test/fixtures/local/` and run `npm run golden`; nothing else
changes, and `git status` must stay clean.

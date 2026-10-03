# Phase 7 findings: encoder, decoder loop, vocabulary

What the pinned homr 0.7.0 does between a staff canvas and `voices.json`, read
from `staff2score.py`, `encoder_inference.py`, `decoder_inference.py`,
`vocabulary.py`, `staff_parsing_tromr.py`, `staff_parsing.py` and `configs.py`
in the oracle's `.venv`. The oracle runs with `use_gpu_inference = False` and
`use_coreml_encoder = False`, so both models are the fp32 files on the CPU EP.

## The call chain

`parse_staffs` loops voice-major, as phase 6 found. Per staff,
`parse_staff_image` calls `prepare_staff_image`, then
`parse_staff_tromr(staff_image, transformed_staff)`, which is `predict_best`:

1. `Staff2Score.predict(canvas)`: `ConvertToArray`, the encoder, the decoder.
2. Unless `staff.is_grandstaff`, drop every symbol whose `position == "lower"`.

`parse_staffs` then skips a staff whose list is empty, appends
`EncodedSymbol("newline")` to each non-empty one, concatenates the staffs of a
voice and runs `remove_duplicated_symbols` once per voice.

The `staff` that reaches `predict_best` is `staff_in_crop`; only
`is_grandstaff` is read off it, so phase 6's question of whether the staff is
in canvas coordinates does not matter here.

`tokens-<n>.json` is the output of step 2, after the position filter. On Kesh
no token has `lower`, and every staff is single, so the filter is proven by the
grand-staff page's lower tokens surviving (9, 16, 17 and 8) and by vectors.

## `ConvertToArray`

`np.array(image) / 255` is float64, `(arr - 0.7931) / 0.1738` is float64
(the mean and std are float64 arrays), then `.astype(np.float32)`. The result
is `[1, 1, 256, 1280]`. Because a pixel is one of 256 values, the port
precomputes a 256-entry table: `Math.fround((v / 255 - 0.7931) / 0.1738)`.
JavaScript's float64 operations are IEEE, so the table equals numpy's.

On the fp16 encoder `Encoder.generate` casts that float32 array with
`astype(np.float16)`: a second rounding, float32 to half, ties to even. The
port's table for that case is `float16FromFloat32` of the float32 table.

## The encoder

One run, input `input` `[1, 1, 256, 1280]`, output `output`
`[1, 1280, 512]`. `predict` casts the output to the decoder's dtype when they
differ; the port's `handoff` already does that cast (phase 2).

## The decoder loop (`ScoreDecoder.generate`)

Config: `max_seq_len = 608`, `eos_token = 2`, `decoder_depth = 8`,
`decoder_heads = 8`, `decoder_dim = 512`, so `head_dim = 64`.

Inputs per step, all bound fresh each step in Python:

| Input | Shape | Value |
|---|---|---|
| `rhythms` | `[1, 1]` int64 | the previous rhythm id; 1 (BOS) at step 0 |
| `pitchs`, `lifts`, `articulations`, `slurs` | `[1, 1]` int64 | the previous id of that head; 0 (`.`) at step 0 |
| `context` | `[1, 1280, 512]` at step 0, `[1, 1, 512]` after | the encoder output, then `context[:, :1]`, its first row |
| `cache_len` | `[1]` int64 | the step number |
| `cache_in0` to `cache_in31` | `[1, 8, step, 64]` | zeros of length 0 at step 0, then the previous step's `cache_out<i>` |

There is no `positions` input: the position head is read but never fed back.

Outputs: `out_rhythms [1,1,259]`, `out_pitchs [1,1,72]`, `out_lifts [1,1,7]`,
`out_positions [1,1,3]`, `out_articulations [1,1,54]`, `out_slurs [1,1,5]`,
`attention [2]`, and `cache_out0` to `cache_out31` `[1, 8, step + 1, 64]`.
Python takes the caches as `outputs[7:]`, positionally; the port takes them by
the manifest's `DECODER_CACHE_OUT` names, which the manifest test pins index
for index against `DECODER_CACHE_IN`.

Per head, `logits[:, -1, :].argmax()`: numpy's argmax, which returns the
**first** index on a tie. The port scans with a strict `>`.

EOS: the rhythm id is compared with 2 **after** every head is decoded and
**before** the symbol is built, so the EOS step contributes nothing. When 608
steps pass without EOS the loop simply ends.

`detokenize` filters `"[BOS]"`, `"[EOS]"` and `"[PAD]"`, strings no table
contains, so it filters nothing: a decoded `BOS` or `PAD` rhythm would become a
symbol. Ported as a plain lookup.

The symbol's `coordinates` is the whole `attention` array, two float32 values
`[x, y]` in canvas pixels. The dumper writes them with `float(c)`, so the
golden values are the float32 values exactly.

## `remove_duplicated_symbols(symbols, cleanup_tuplets=True)`

1. `_group_into_chords`: a `chord` symbol marks that the next symbol joins the
   last group. A `chord` that is the very first symbol is swallowed and the
   next symbol starts a group of its own (`len(chords) > 0` fails, and
   `is_in_chord` stays true until a later symbol joins).
2. `_fix_over_eager_tuplets`: groups into measures (a group ends a measure
   when its first symbol's rhythm contains `barline` or `repeat`); a measure's
   duration is the sum, over its chords, of the shortest positive duration of
   the chord's notes and rests; the typical duration is
   `sorted(durations)[len // 2]`; every measure strictly shorter has
   `remove_tuplet` applied to each symbol.
3. `_only_keep_lower_staff_if_there_is_a_clef`: until a chord with index below
   5 holds a `clef` with position `lower`, every symbol is moved to `upper`;
   from that clef on, symbols are kept as they are.
4. `_remove_duplicated_piches` per chord: only when the chord has more than
   one symbol and the first is a note or rest. The key is `pitch + " " +
   position`. A longer duplicate is written under `by_pitch[symbol.pitch]`,
   **not** under the key: a homr bug, so a longer duplicate never replaces the
   first. Ported as it is.
5. `_remove_redudant_clefs_keys_and_time_signatures`: a clef is dropped when
   equal to the last clef of its position (`upper`, else the lower slot); key
   and time signatures when equal to the last one. The state runs over the
   whole voice.
6. `_flatten_chords`: a `chord` symbol before every symbol after the first of
   a group. A group emptied by step 5 contributes nothing.

Durations (`kern_to_symbol_duration`), exact rationals:

- The multirest branch (`kern.endswith("m")`) builds a value and discards it:
  no `return`. So `2m` falls through and reads as base 2, a half note. Ported
  as it is.
- digits, then `dots = rest.count(".")`; `G` anywhere gives a grace note of
  duration 0; base 0 gives a whole; a power of two gives `1 / base`; anything
  else is a tuplet with `normal = prior_power_of_two(base)` and the fraction
  `(1 / normal) * normal / base` after dots.
- Dots add `d / 2`, then half of that, per dot.
- `get_duration` on a symbol that is neither note nor rest returns 0.

`remove_tuplet`: `(note|rest)_(\d+)(.*)`; a duration divisible by 3 becomes
`d // 3 * 2`, else by 5 becomes `d // 5 * 4`, else by 7 becomes `d // 7 * 4`;
otherwise the symbol is unchanged. Only the rhythm changes.

`newline` is not a note or rest, so it never contributes a duration and never
ends a measure.

## Rounding and float32 sites

| Site | Python | TypeScript |
|---|---|---|
| normalisation | float64, then `astype(float32)` | float64, then `Math.fround`, by table |
| fp16 encoder input | `astype(float16)` of the float32 | `float16FromFloat32` |
| encoder output to decoder | `astype` when dtypes differ | `handoff` (phase 2) |
| argmax | numpy, first on ties | strict `>` scan |
| `remove_tuplet` | `//` on positive ints | `Math.floor` |
| durations | `Fraction` | a small exact rational over safe integers |
| typical duration | `sorted(...)[len // 2]` | stable sort by rational, `Math.floor` |

Nothing else in the path rounds. Tokens are integer ids; only the coordinates
carry float32, and they come straight out of the model.

## What can differ, and where

The token ids are an argmax over fp32 logits computed by onnxruntime's CPU EP
natively and by its WebAssembly build. Different SIMD reductions can move a
logit in its last bits; a flipped argmax needs two logits within that
distance, which is why exact equality is the target and a difference is a port
bug until a logit measurement says otherwise. The attention coordinates are
float32 outputs of the same graph and are compared within a tolerance.

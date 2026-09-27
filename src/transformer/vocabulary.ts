/**
 * The six vocabularies of homr's transformer/vocabulary.py, in Python's
 * insertion order. The position of a token in its table is the decoder's
 * output column for that token, so the order is a contract with the ONNX
 * file, not a style choice. The tables between the markers are generated
 * by tools/gen-vocabulary.mjs from test/golden/vocabulary.json, which the
 * golden dumper writes from the installed homr; a re-pin regenerates them
 * and shows as a diff. test/golden.test.ts compares them index for index.
 *
 * Phase 7 adds the rest of vocabulary.py (durations, tuplet cleanup,
 * remove_duplicated_symbols) beside this file.
 */

/** The "nonote" sentinel: the head has nothing to say for this symbol. */
export const NONOTE = ".";
/** The "empty" sentinel: the head applies but carries no decoration. */
export const EMPTY = "_";

// BEGIN GENERATED (tools/gen-vocabulary.mjs)
/** 259 tokens, index = decoder output column. */
export const RHYTHM_TOKENS = [
  "PAD",
  "BOS",
  "EOS",
  "chord",
  "barline",
  "doublebarline",
  "bolddoublebarline",
  "repeatStart",
  "repeatEnd",
  "repeatEndStart",
  "voltaStart",
  "voltaStop",
  "voltaDiscontinue",
  "clef_F3",
  "clef_F4",
  "clef_F5",
  "clef_C1",
  "clef_C2",
  "clef_C3",
  "clef_C4",
  "clef_C5",
  "clef_G1",
  "clef_G2",
  "keySignature_-7",
  "keySignature_-6",
  "keySignature_-5",
  "keySignature_-4",
  "keySignature_-3",
  "keySignature_-2",
  "keySignature_-1",
  "keySignature_0",
  "keySignature_1",
  "keySignature_2",
  "keySignature_3",
  "keySignature_4",
  "keySignature_5",
  "keySignature_6",
  "keySignature_7",
  "timeSignature/1",
  "timeSignature/2",
  "timeSignature/3",
  "timeSignature/4",
  "timeSignature/6",
  "timeSignature/8",
  "timeSignature/12",
  "timeSignature/16",
  "timeSignature/32",
  "timeSignature/48",
  "rest_2m",
  "rest_3m",
  "rest_4m",
  "rest_5m",
  "rest_6m",
  "rest_7m",
  "rest_8m",
  "rest_9m",
  "rest_10m",
  "note_0",
  "note_0.",
  "note_0..",
  "note_0G",
  "note_0G.",
  "note_0G..",
  "note_1",
  "note_1.",
  "note_1..",
  "note_1G",
  "note_1G.",
  "note_1G..",
  "note_2",
  "note_2.",
  "note_2..",
  "note_2G",
  "note_2G.",
  "note_2G..",
  "note_3",
  "note_3.",
  "note_3..",
  "note_3G",
  "note_3G.",
  "note_3G..",
  "note_4",
  "note_4.",
  "note_4..",
  "note_4G",
  "note_4G.",
  "note_4G..",
  "note_5",
  "note_5.",
  "note_5..",
  "note_5G",
  "note_5G.",
  "note_5G..",
  "note_6",
  "note_6.",
  "note_6..",
  "note_6G",
  "note_6G.",
  "note_6G..",
  "note_8",
  "note_8.",
  "note_8..",
  "note_8G",
  "note_8G.",
  "note_8G..",
  "note_10",
  "note_10.",
  "note_10..",
  "note_10G",
  "note_10G.",
  "note_10G..",
  "note_12",
  "note_12.",
  "note_12..",
  "note_12G",
  "note_12G.",
  "note_12G..",
  "note_16",
  "note_16.",
  "note_16..",
  "note_16G",
  "note_16G.",
  "note_16G..",
  "note_32",
  "note_32.",
  "note_32..",
  "note_32G",
  "note_32G.",
  "note_32G..",
  "note_64",
  "note_64.",
  "note_64..",
  "note_64G",
  "note_64G.",
  "note_64G..",
  "note_128",
  "note_128.",
  "note_128..",
  "note_128G",
  "note_128G.",
  "note_128G..",
  "note_7",
  "note_11",
  "note_13",
  "note_18",
  "note_20",
  "note_21",
  "note_22",
  "note_24",
  "note_26",
  "note_28",
  "note_30",
  "note_34",
  "note_36",
  "note_40",
  "note_48",
  "note_56",
  "note_96",
  "rest_0",
  "rest_0.",
  "rest_0..",
  "rest_0G",
  "rest_0G.",
  "rest_0G..",
  "rest_1",
  "rest_1.",
  "rest_1..",
  "rest_1G",
  "rest_1G.",
  "rest_1G..",
  "rest_2",
  "rest_2.",
  "rest_2..",
  "rest_2G",
  "rest_2G.",
  "rest_2G..",
  "rest_3",
  "rest_3.",
  "rest_3..",
  "rest_3G",
  "rest_3G.",
  "rest_3G..",
  "rest_4",
  "rest_4.",
  "rest_4..",
  "rest_4G",
  "rest_4G.",
  "rest_4G..",
  "rest_5",
  "rest_5.",
  "rest_5..",
  "rest_5G",
  "rest_5G.",
  "rest_5G..",
  "rest_6",
  "rest_6.",
  "rest_6..",
  "rest_6G",
  "rest_6G.",
  "rest_6G..",
  "rest_8",
  "rest_8.",
  "rest_8..",
  "rest_8G",
  "rest_8G.",
  "rest_8G..",
  "rest_10",
  "rest_10.",
  "rest_10..",
  "rest_10G",
  "rest_10G.",
  "rest_10G..",
  "rest_12",
  "rest_12.",
  "rest_12..",
  "rest_12G",
  "rest_12G.",
  "rest_12G..",
  "rest_16",
  "rest_16.",
  "rest_16..",
  "rest_16G",
  "rest_16G.",
  "rest_16G..",
  "rest_32",
  "rest_32.",
  "rest_32..",
  "rest_32G",
  "rest_32G.",
  "rest_32G..",
  "rest_64",
  "rest_64.",
  "rest_64..",
  "rest_64G",
  "rest_64G.",
  "rest_64G..",
  "rest_128",
  "rest_128.",
  "rest_128..",
  "rest_128G",
  "rest_128G.",
  "rest_128G..",
  "rest_7",
  "rest_11",
  "rest_13",
  "rest_18",
  "rest_20",
  "rest_21",
  "rest_22",
  "rest_24",
  "rest_26",
  "rest_28",
  "rest_30",
  "rest_34",
  "rest_36",
  "rest_40",
  "rest_48",
  "rest_56",
  "rest_96",
] as const;

/** 72 tokens, index = decoder output column. */
export const PITCH_TOKENS = [
  ".",
  "_",
  "B9",
  "A9",
  "G9",
  "F9",
  "E9",
  "D9",
  "C9",
  "B8",
  "A8",
  "G8",
  "F8",
  "E8",
  "D8",
  "C8",
  "B7",
  "A7",
  "G7",
  "F7",
  "E7",
  "D7",
  "C7",
  "B6",
  "A6",
  "G6",
  "F6",
  "E6",
  "D6",
  "C6",
  "B5",
  "A5",
  "G5",
  "F5",
  "E5",
  "D5",
  "C5",
  "B4",
  "A4",
  "G4",
  "F4",
  "E4",
  "D4",
  "C4",
  "B3",
  "A3",
  "G3",
  "F3",
  "E3",
  "D3",
  "C3",
  "B2",
  "A2",
  "G2",
  "F2",
  "E2",
  "D2",
  "C2",
  "B1",
  "A1",
  "G1",
  "F1",
  "E1",
  "D1",
  "C1",
  "B0",
  "A0",
  "G0",
  "F0",
  "E0",
  "D0",
  "C0",
] as const;

/** 7 tokens, index = decoder output column. */
export const LIFT_TOKENS = [".", "_", "#", "##", "N", "b", "bb"] as const;

/** 54 tokens, index = decoder output column. */
export const ARTICULATION_TOKENS = [
  ".",
  "_",
  "accent",
  "accent_arpeggiate",
  "accent_arpeggiate_fermata",
  "accent_arpeggiate_staccato",
  "accent_arpeggiate_tenuto",
  "accent_breathMark",
  "accent_breathMark_fermata",
  "accent_fermata",
  "accent_fermata_staccato",
  "accent_staccatissimo",
  "accent_staccato",
  "accent_staccato_tenuto",
  "accent_tenuto",
  "accent_tremolo",
  "accent_trill",
  "accent_fermata_trill",
  "arpeggiate",
  "arpeggiate_breathMark_fermata",
  "arpeggiate_fermata",
  "arpeggiate_fermata_staccato",
  "arpeggiate_staccatissimo",
  "arpeggiate_staccato",
  "arpeggiate_staccato_tenuto",
  "arpeggiate_tenuto",
  "arpeggiate_tremolo",
  "arpeggiate_trill",
  "breathMark",
  "breathMark_fermata",
  "breathMark_fermata_tenuto",
  "breathMark_staccato",
  "breathMark_tenuto",
  "breathMark_trill",
  "breathMark_tremolo",
  "breathMark_staccato_tenuto",
  "fermata",
  "fermata_staccato",
  "fermata_staccato_tenuto",
  "fermata_tenuto",
  "fermata_tremolo",
  "fermata_trill",
  "fermata_turn",
  "spiccato",
  "staccatissimo",
  "staccato",
  "staccato_tenuto",
  "staccato_tremolo",
  "staccato_trill",
  "staccato_turn",
  "tenuto",
  "tremolo",
  "trill",
  "turn",
] as const;

/** 5 tokens, index = decoder output column. */
export const SLUR_TOKENS = [
  ".",
  "_",
  "slurStart_slurStop",
  "slurStart",
  "slurStop",
] as const;

/** 3 tokens, index = decoder output column. */
export const POSITION_TOKENS = [".", "upper", "lower"] as const;
// END GENERATED

export const VOCABULARIES = {
  articulation: ARTICULATION_TOKENS,
  lift: LIFT_TOKENS,
  pitch: PITCH_TOKENS,
  position: POSITION_TOKENS,
  rhythm: RHYTHM_TOKENS,
  slur: SLUR_TOKENS,
} as const;

export type Head = keyof typeof VOCABULARIES;
export type TokenOf<H extends Head> = (typeof VOCABULARIES)[H][number];

export type RhythmToken = TokenOf<"rhythm">;
export type PitchToken = TokenOf<"pitch">;
export type LiftToken = TokenOf<"lift">;
export type ArticulationToken = TokenOf<"articulation">;
export type SlurToken = TokenOf<"slur">;
export type PositionToken = TokenOf<"position">;

/**
 * The order of the decoder's first six outputs (decoder_inference.py:
 * out_rhythms, out_pitchs, out_lifts, out_positions, out_articulations,
 * out_slurs). Positions come third, not sixth; the loop in phase 7 reads
 * the outputs by this list, never by guessing from the head names.
 */
export const DECODER_OUTPUT_HEADS = [
  "rhythm",
  "pitch",
  "lift",
  "position",
  "articulation",
  "slur",
] as const satisfies readonly Head[];

declare const tokenIdBrand: unique symbol;

/**
 * A column index into one head's table. Branded per head so that a rhythm
 * id cannot be written into the pitch input tensor: the decoder loop feeds
 * six near-identical int64 inputs per step and the compiler is the only
 * thing that would notice two of them swapped.
 */
export type TokenId<H extends Head> = number & { readonly [tokenIdBrand]: H };

export class VocabularyError extends Error {}

const INDEX: { readonly [H in Head]: ReadonlyMap<string, number> } = {
  articulation: new Map(ARTICULATION_TOKENS.map((t, i) => [t, i])),
  lift: new Map(LIFT_TOKENS.map((t, i) => [t, i])),
  pitch: new Map(PITCH_TOKENS.map((t, i) => [t, i])),
  position: new Map(POSITION_TOKENS.map((t, i) => [t, i])),
  rhythm: new Map(RHYTHM_TOKENS.map((t, i) => [t, i])),
  slur: new Map(SLUR_TOKENS.map((t, i) => [t, i])),
};

/** Index of a token in its head's table; throws on a token the head does not know. */
export function tokenIndex<H extends Head>(
  head: H,
  token: TokenOf<H>
): TokenId<H> {
  const index = INDEX[head].get(token);
  if (index === undefined) {
    throw new VocabularyError(
      `${head} vocabulary has no token ${JSON.stringify(token)}`
    );
  }
  return index as TokenId<H>;
}

/** Token at a column index; throws when the index is outside the table. */
export function tokenAt<H extends Head>(head: H, id: TokenId<H>): TokenOf<H> {
  const token = VOCABULARIES[head][id] as TokenOf<H> | undefined;
  if (token === undefined) {
    throw new VocabularyError(`${head} vocabulary has no column ${id}`);
  }
  return token;
}

/** Type guard used by the golden decoder and by anything that reads a token from a string. */
export function isToken<H extends Head>(
  head: H,
  value: string
): value is TokenOf<H> {
  return INDEX[head].has(value);
}

/**
 * staff_parsing.py's parse_staffs and parse_staff_image, with
 * staff_parsing_tromr.py's predict_best between them: canvas, encoder,
 * decoder, and the lower-position filter for a single staff.
 */

import type { OpenCv } from "../cv/opencv.js";
import type { GrayImage } from "../image/plane.js";
import type { StaffCanvas } from "../model/pipeline.js";
import type { MultiStaff } from "../model/staff.js";
import type { ModelSession } from "../models/session.js";
import {
  type DecodeOptions,
  runDecoder,
  type TokenSequence,
} from "../transformer/decoder.js";
import { encodeCanvas } from "../transformer/encoder.js";
import { removeDuplicatedSymbols } from "../transformer/remove-duplicated-symbols.js";
import {
  createEncodedSymbol,
  type DecodedSymbol,
  type EncodedSymbol,
  NEWLINE,
} from "../transformer/symbol.js";
import {
  ensureSameNumberOfStaffs,
  prepareStaffImage,
  staffRegions,
} from "./staff-image.js";

export interface TransformerSessions {
  readonly decoder: ModelSession;
  readonly encoder: ModelSession;
}

/** predict_best: a single staff keeps no symbol placed on a lower staff. */
export function filterPositions(
  symbols: TokenSequence,
  isGrandstaff: boolean
): DecodedSymbol[] {
  return isGrandstaff
    ? [...symbols]
    : symbols.filter((symbol) => symbol.position !== "lower");
}

/** parse_staff_tromr on a prepared canvas: what tokens-<n>.json holds. */
export async function parseStaffCanvas(
  sessions: TransformerSessions,
  canvas: StaffCanvas,
  options: DecodeOptions = {}
): Promise<DecodedSymbol[]> {
  const context = await encodeCanvas(
    sessions.encoder,
    sessions.decoder,
    canvas.image
  );
  try {
    const symbols = await runDecoder(sessions.decoder, context, options);
    return filterPositions(symbols, canvas.staff.isGrandstaff);
  } finally {
    context.dispose();
  }
}

/**
 * The per-voice tail of parse_staffs: an empty staff is skipped, every other
 * staff gains a newline, and the voice is cleaned once.
 */
export function joinVoice(
  staffs: readonly (readonly EncodedSymbol[])[]
): EncodedSymbol[] {
  const voice: EncodedSymbol[] = [];
  for (const staff of staffs) {
    if (staff.length > 0) {
      voice.push(...staff, createEncodedSymbol(NEWLINE));
    }
  }
  return removeDuplicatedSymbols(voice);
}

/** parse_staffs: one symbol list per voice, staffs parsed one at a time in homr's order. */
export async function parseStaffs(
  cv: OpenCv,
  sessions: TransformerSessions,
  multiStaffs: readonly MultiStaff[],
  page: GrayImage,
  options: DecodeOptions = {}
): Promise<EncodedSymbol[][]> {
  const systems = ensureSameNumberOfStaffs(multiStaffs, page.height);
  const regions = staffRegions(systems);
  const voices: EncodedSymbol[][] = [];
  for (let voice = 0; voice < (systems[0]?.staffs.length ?? 0); voice += 1) {
    const staffs: DecodedSymbol[][] = [];
    for (const system of systems) {
      const staff = system.staffs[voice];
      if (staff !== undefined) {
        const canvas = prepareStaffImage(cv, staff, page, regions);
        // biome-ignore lint/performance/noAwaitInLoops: one staff at a time, as homr does; the sessions share one WebAssembly arena
        staffs.push(await parseStaffCanvas(sessions, canvas, options));
      }
    }
    voices.push(joinVoice(staffs));
  }
  return voices;
}

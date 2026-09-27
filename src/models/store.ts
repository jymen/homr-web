/**
 * The catalogue of artifacts over one realm's runtime: it resolves a role to a
 * plan, gets that plan's bytes past two boundaries, and reports what it did.
 * Everything it touches outside the library is injected, so this module
 * contains no environment test and no I/O of its own.
 *
 * The one guarantee worth stating twice: bytes are verified against the
 * manifest's digest on *every* load, cache hits included, and nothing
 * unverified is ever written to the cache. So the cache holds complete verified
 * artifacts or nothing, an interrupted download leaves no trace, and a poisoned
 * entry heals itself once.
 */

import type { ModelRuntime } from "./backend.js";
import { placementOf } from "./backend.js";
import type { ModelCache, ModelCacheKind } from "./cache.js";
import { asBufferSource } from "./cache.js";
import { ModelError } from "./errors.js";
import type {
  ArtifactId,
  ArtifactRecord,
  ExecutionProvider,
  ModelCatalog,
  ModelRole,
  Placement,
  ResolvedModel,
} from "./manifest.js";
import { DEFAULT_CATALOG, resolveRole } from "./manifest.js";
import type { ModelSession, SessionTuning } from "./session.js";
import { openModelSession } from "./session.js";

export interface FetchRequest {
  /** `total` is 0 where the response declared no length, which a progress bar reads as unknown rather than as zero bytes. */
  readonly onProgress: ((received: number, total: number) => void) | undefined;
  readonly signal: AbortSignal | undefined;
  readonly url: string;
}

export type FetchBytes = (request: FetchRequest) => Promise<Uint8Array>;

/**
 * `globalThis.fetch`, streamed so progress is reported. One implementation for
 * Node 22+ and the browser: fetch is a global, not an environment branch.
 *
 * This is the boundary the network sits behind, so it is also where a bad
 * response becomes a ModelError. Errors from an injected fetch are left alone:
 * that port is its own boundary.
 */
export function httpFetch(): FetchBytes {
  return async ({ onProgress, signal, url }) => {
    const response = await globalThis.fetch(
      url,
      signal === undefined ? {} : { signal }
    );
    if (!response.ok) {
      const status = `${response.status} ${response.statusText}`.trim();
      throw new ModelError("fetch-failed", `${url} answered ${status}`, {
        actual: status,
        id: url,
      });
    }
    const { body } = response;
    if (body === null || onProgress === undefined) {
      return new Uint8Array(await response.arrayBuffer());
    }
    const total = Number(response.headers.get("content-length") ?? 0);
    let received = 0;
    // Counted through a transform rather than a read loop, so the whole body is
    // still assembled by the platform.
    const counted = body.pipeThrough(
      new TransformStream<Uint8Array, Uint8Array>({
        transform: (chunk, controller) => {
          received += chunk.length;
          onProgress(received, total);
          controller.enqueue(chunk);
        },
      })
    );
    return new Uint8Array(await new Response(counted).arrayBuffer());
  };
}

/**
 * Lower-case hex, which is the form tools/gen-manifest.mjs writes into
 * ARTIFACTS. That generator hashes with node:crypto and cannot import a .ts
 * file, so the two implementations are held together by a test over a real
 * artifact rather than by a shared import.
 */
export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", asBufferSource(bytes));
  return Array.from(new Uint8Array(digest), (byte) =>
    byte.toString(16).padStart(2, "0")
  ).join("");
}

export type ModelEvent =
  | {
      readonly artifact: string;
      readonly bytes: number;
      readonly from: ModelCacheKind;
      readonly kind: "cached";
    }
  | {
      readonly artifact: string;
      readonly kind: "download";
      readonly received: number;
      readonly total: number;
    }
  | {
      readonly artifact: string;
      readonly kind: "verified";
      readonly ms: number;
    }
  /** A verification failure. source "cache" is the self-heal path and is followed by one re-fetch; source "network" is the failure. */
  | {
      readonly actual: string;
      readonly artifact: string;
      readonly expected: string;
      readonly kind: "rejected";
      readonly source: "cache" | "network";
    }
  | {
      readonly artifact: string;
      readonly kind: "opened";
      readonly ms: number;
      readonly provider: ExecutionProvider;
      readonly role: ModelRole;
    };

export interface ModelStoreOptions {
  /** Where the artifacts are served; urlPath is appended. Must be same-origin, or carry Cross-Origin-Resource-Policy: same-origin, when the page is COEP require-corp. */
  readonly baseUrl: string;
  readonly cache: ModelCache;
  readonly catalog?: ModelCatalog;
  readonly fetchBytes?: FetchBytes;
  readonly onEvent?: (event: ModelEvent) => void;
  /** Defaults to placementOf(runtime). Given explicitly by the fp16 golden test and by the bench's precision A/B. */
  readonly placement?: Placement;
  readonly runtime: ModelRuntime;
}

/** Why these bytes are not the artifact's, or how long it took to prove that they are. */
type Digest =
  | { readonly kind: "ok"; readonly ms: number }
  | {
      readonly actual: string;
      readonly expected: string;
      readonly kind: "wrong";
      readonly why: string;
    };

/**
 * Length before digest, so a 404 HTML page served in place of a model is
 * refused by a message naming the length rather than by two unreadable digests.
 */
async function digestAgainst(
  artifact: ArtifactRecord,
  bytes: Uint8Array
): Promise<Digest> {
  if (bytes.length !== artifact.bytes) {
    return {
      actual: `${bytes.length} bytes`,
      expected: `${artifact.bytes} bytes`,
      kind: "wrong",
      why: `holds ${bytes.length} bytes, the manifest says ${artifact.bytes} bytes`,
    };
  }
  const started = performance.now();
  const actual = await sha256Hex(bytes);
  const ms = performance.now() - started;
  if (actual !== artifact.sha256) {
    return {
      actual,
      expected: artifact.sha256,
      kind: "wrong",
      why: `hashes to ${actual}, the manifest says ${artifact.sha256}`,
    };
  }
  return { kind: "ok", ms };
}

/** A cache write or drop that fails (quota, private mode, a revoked handle) must not fail the load. */
async function bestEffort(work: Promise<void>): Promise<void> {
  await work.catch(() => undefined);
}

/** Tolerates a baseUrl with no trailing slash rather than silently fetching `/modelsHASH/…`. */
const urlOf = (baseUrl: string, urlPath: string): string =>
  baseUrl.endsWith("/") ? `${baseUrl}${urlPath}` : `${baseUrl}/${urlPath}`;

/** A class, not a factory: it owns mutable state and lives in a Worker, per the project's Worker-state convention. */
export class ModelStore {
  readonly placement: Placement;
  readonly runtime: ModelRuntime;

  readonly #baseUrl: string;
  /** In-flight loads only, keyed on sha256. See #loadBytes for why an entry never outlives its load. */
  readonly #bytes = new Map<string, Promise<Uint8Array>>();
  readonly #cache: ModelCache;
  readonly #catalog: ModelCatalog;
  /**
   * Whether this store is closed, and the abort that makes it so: one piece of
   * state rather than a flag beside a controller, so a load cannot read the two
   * disagreeing. Aborting also releases a 57 MB download's socket instead of
   * leaving it to finish into nothing.
   */
  readonly #closing = new AbortController();
  readonly #fetchBytes: FetchBytes;
  readonly #onEvent: ((event: ModelEvent) => void) | undefined;
  /** Open sessions, keyed on role and batch. Nothing else can vary: the provider, the output locations and the graph options are policy derived from (role, placement), so there is no option bag to hash. */
  readonly #sessions = new Map<string, Promise<ModelSession>>();

  constructor(options: ModelStoreOptions) {
    const placement = options.placement ?? placementOf(options.runtime);
    // The dangerous half of the placement axis: the fp16 artifacts run happily
    // on the wasm execution provider, so artifactsFor is left free, but the
    // webgpu provider on a runtime that never configured a GPU device cannot
    // produce a session at all.
    if (
      placement.provider === "webgpu" &&
      options.runtime.backend !== "webgpu"
    ) {
      throw new ModelError(
        "bad-placement",
        `placement asks for the webgpu execution provider on a ${options.runtime.backend} runtime`,
        { actual: options.runtime.backend, expected: "webgpu" }
      );
    }
    this.#baseUrl = options.baseUrl;
    this.#cache = options.cache;
    this.#catalog = options.catalog ?? DEFAULT_CATALOG;
    this.#fetchBytes = options.fetchBytes ?? httpFetch();
    this.#onEvent = options.onEvent;
    this.placement = placement;
    this.runtime = options.runtime;
  }

  /** What this role would use, without fetching anything. The app asks before deciding whether to pull 160 MB over a phone connection. */
  plan(role: ModelRole): ResolvedModel {
    return resolveRole(this.#catalog, role, this.placement);
  }

  /** Verified bytes in the cache for these roles, and no sessions. The transcriber page calls this while the musician is still choosing a file. Cheap and safe to call twice. */
  async prefetch(
    roles: readonly ModelRole[],
    signal?: AbortSignal
  ): Promise<void> {
    await Promise.all(
      roles.map((role) => this.#loadBytes(this.plan(role), signal))
    );
  }

  /**
   * The session for this role. Opening the same role with the same tuning
   * returns the same session, and the store owns it until close().
   *
   * The verified bytes are a local of #openSession and nothing outlives it, so
   * ort has copied them into wasm memory by the time the last reference goes:
   * 57 MB held for nothing is a real cost on iOS, where a tab dies near 1 GB.
   */
  async open(
    role: ModelRole,
    tuning: SessionTuning = {}
  ): Promise<ModelSession> {
    const key = `${role}@${tuning.batch ?? 0}`;
    const inFlight = this.#sessions.get(key);
    if (inFlight !== undefined) {
      return await inFlight;
    }
    const plan = this.plan(role);
    this.#assertOpen(plan.artifactId);
    const opening = this.#openSession(plan, tuning);
    this.#sessions.set(key, opening);
    return await opening.catch((cause: unknown) => {
      if (this.#sessions.get(key) === opening) {
        this.#sessions.delete(key);
      }
      throw cause;
    });
  }

  /** Idempotent, and safe during a load or an open, which then reject with store-closed. */
  async close(): Promise<void> {
    this.#bytes.clear();
    this.#closing.abort();
    const opened = [...this.#sessions.values()];
    this.#sessions.clear();
    // A session that never opened has nothing to release, and its rejection is
    // already the caller's.
    await Promise.all(
      opened.map((pending) =>
        pending.then(
          (session) => session.close(),
          () => undefined
        )
      )
    );
  }

  async #openSession(
    plan: ResolvedModel,
    tuning: SessionTuning
  ): Promise<ModelSession> {
    const bytes = await this.#loadBytes(plan, undefined);
    this.#assertOpen(plan.artifactId);
    const started = performance.now();
    const session = await openModelSession(plan, bytes, tuning);
    this.#emit({
      artifact: plan.artifactId,
      kind: "opened",
      ms: performance.now() - started,
      provider: plan.provider,
      role: plan.role,
    });
    return session;
  }

  /**
   * Single flight on the artifact's digest, so two concurrent loads of segnet
   * in one realm share one 57 MB download. Across realms (phase 3's Worker and
   * phase 7's are separate realms and cannot share a Map) the content-keyed
   * cache makes the duplicate harmless: two writers put identical bytes under
   * one key.
   *
   * The entry is dropped as soon as the load settles, in either direction. It
   * deliberately does not outlive the load: holding a fulfilled 57 MB
   * Uint8Array for a caller who may never come is the cost this phase refuses
   * on a tab that dies near 1 GB, which is the same reason a session drops its
   * bytes the moment ort has copied them. A later caller reads the verified
   * entry back out of the cache and re-digests it, measured at 33 ms for
   * segnet.
   */
  async #loadBytes(
    plan: ResolvedModel,
    signal: AbortSignal | undefined
  ): Promise<Uint8Array> {
    const { sha256 } = plan.artifact;
    const inFlight = this.#bytes.get(sha256);
    if (inFlight !== undefined) {
      return await inFlight;
    }
    this.#assertOpen(plan.artifactId);
    const load = this.#verifiedBytes(plan, signal);
    this.#bytes.set(sha256, load);
    return await load.finally(() => {
      if (this.#bytes.get(sha256) === load) {
        this.#bytes.delete(sha256);
      }
    });
  }

  async #verifiedBytes(
    plan: ResolvedModel,
    signal: AbortSignal | undefined
  ): Promise<Uint8Array> {
    const { artifact, artifactId } = plan;
    const cached = await this.#readCache(artifact.sha256);
    if (cached !== undefined) {
      const digest = await digestAgainst(artifact, cached);
      this.#assertOpen(artifactId);
      if (digest.kind === "ok") {
        this.#emit({
          artifact: artifactId,
          bytes: cached.length,
          from: this.#cache.kind,
          kind: "cached",
        });
        this.#emit({ artifact: artifactId, kind: "verified", ms: digest.ms });
        return cached;
      }
      // Dropped and re-fetched once. The network's answer is then final: there
      // is no loop and no third attempt.
      this.#emit({
        actual: digest.actual,
        artifact: artifactId,
        expected: digest.expected,
        kind: "rejected",
        source: "cache",
      });
      await bestEffort(this.#cache.drop(artifact.sha256));
    }
    this.#assertOpen(artifactId);
    const fetched = await this.#download(plan, signal);
    this.#assertOpen(artifactId);
    const digest = await digestAgainst(artifact, fetched);
    if (digest.kind === "wrong") {
      this.#emit({
        actual: digest.actual,
        artifact: artifactId,
        expected: digest.expected,
        kind: "rejected",
        source: "network",
      });
      throw new ModelError(
        "digest-mismatch",
        `${artifactId} from the network ${digest.why}`,
        { actual: digest.actual, expected: digest.expected, id: artifactId }
      );
    }
    this.#emit({ artifact: artifactId, kind: "verified", ms: digest.ms });
    await bestEffort(this.#cache.write(artifact.sha256, fetched));
    return fetched;
  }

  /** Any throw is a miss, so an unreadable cache costs a download rather than a load. */
  async #readCache(sha256: string): Promise<Uint8Array | undefined> {
    return await this.#cache.read(sha256).catch(() => undefined);
  }

  async #download(
    plan: ResolvedModel,
    signal: AbortSignal | undefined
  ): Promise<Uint8Array> {
    const { artifact, artifactId } = plan;
    const url = urlOf(this.#baseUrl, artifact.urlPath);
    const signals =
      signal === undefined
        ? [this.#closing.signal]
        : [signal, this.#closing.signal];
    try {
      return await this.#fetchBytes({
        onProgress:
          this.#onEvent === undefined
            ? undefined
            : (received, total) => {
                this.#emit({
                  artifact: artifactId,
                  kind: "download",
                  received,
                  total,
                });
              },
        signal: AbortSignal.any(signals),
        url,
      });
    } catch (cause) {
      // A close() during the download is the reason it failed, and a caller
      // acts on that rather than on an AbortError from a socket.
      this.#assertOpen(artifactId);
      throw cause;
    }
  }

  #assertOpen(artifactId: ArtifactId): void {
    if (this.#closing.signal.aborted) {
      throw new ModelError(
        "store-closed",
        `the store was closed while loading ${artifactId}`,
        { id: artifactId }
      );
    }
  }

  #emit(event: ModelEvent): void {
    this.#onEvent?.(event);
  }
}

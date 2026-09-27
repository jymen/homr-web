/**
 * Where verified model bytes live between loads, and the only module in the
 * library that asks what it is running on. browserCache picks an
 * implementation here, so the store's own logic contains no environment test
 * and Node needs no branch at all: it passes memoryCache().
 *
 * Every entry is keyed on the artifact's SHA-256 and on nothing else. A
 * manifest bump therefore cannot read a stale entry, two homr-web versions
 * share a byte-identical model instead of holding a copy each, and there is no
 * version field to keep in step with anything.
 */

export type ModelCacheKind = "cache-api" | "memory" | "opfs";

/**
 * May throw, and the store treats any throw from `read` as a miss so that
 * policy lives in one place. `write` and `drop` are best effort for the same
 * reason: a full disk must not fail a load.
 */
export interface ModelCache {
  readonly drop: (sha256: string) => Promise<void>;
  readonly kind: ModelCacheKind;
  readonly read: (sha256: string) => Promise<Uint8Array | undefined>;
  readonly write: (sha256: string, bytes: Uint8Array) => Promise<void>;
}

/** One namespace across every version of the library, because the key is already the bytes' digest. */
export const MODEL_CACHE_NAMESPACE = "homr-web-models";

/**
 * The same bytes where a `BufferSource` is asked for, as a view and not a copy.
 *
 * TypeScript 5.7 made `Uint8Array` generic over its buffer and `BufferSource`
 * admits only an ArrayBuffer-backed view, so a plain `Uint8Array` does not
 * reach WebCrypto, `Response` or an OPFS write on its own. The narrowing here is
 * real rather than asserted, and it is deliberately a positive test on
 * `ArrayBuffer`: `SharedArrayBuffer` is absent on a page that is not
 * cross-origin isolated, so naming it would throw where this is needed most.
 * Shared memory is the one case that cannot be viewed, and nothing in this
 * library produces one.
 */
export function asBufferSource(bytes: Uint8Array): BufferSource {
  const { buffer, byteLength, byteOffset } = bytes;
  return buffer instanceof ArrayBuffer
    ? new Uint8Array(buffer, byteOffset, byteLength)
    : new Uint8Array(bytes);
}

/**
 * Holds the caller's buffer rather than a copy: this is what the Node tests and
 * the memory fallback use, both of which hand the bytes straight to a session
 * and never write to them.
 */
export function memoryCache(): ModelCache {
  const held = new Map<string, Uint8Array>();
  return {
    drop: (sha256) => {
      held.delete(sha256);
      return Promise.resolve();
    },
    kind: "memory",
    read: (sha256) => Promise.resolve(held.get(sha256)),
    write: (sha256, bytes) => {
      held.set(sha256, bytes);
      return Promise.resolve();
    },
  };
}

/**
 * A request-shaped key under a reserved `.invalid` host (RFC 2606), so a cache
 * entry can never collide with, or be mistaken for, the real request for the
 * same artifact from the store's baseUrl.
 */
const urlFor = (namespace: string, sha256: string): string =>
  `https://models.homr-web.invalid/${namespace}/${sha256}`;

/** Throws where there is no Cache API, which is what lets browserCache fall through to the next one. */
export async function cacheApiCache(
  namespace: string = MODEL_CACHE_NAMESPACE
): Promise<ModelCache> {
  if (typeof caches === "undefined") {
    throw new Error("this host has no Cache API");
  }
  const store = await caches.open(namespace);
  return {
    drop: async (sha256) => {
      await store.delete(urlFor(namespace, sha256));
    },
    kind: "cache-api",
    read: async (sha256) => {
      const hit = await store.match(urlFor(namespace, sha256));
      return hit === undefined
        ? undefined
        : new Uint8Array(await hit.arrayBuffer());
    },
    write: async (sha256, bytes) => {
      await store.put(
        urlFor(namespace, sha256),
        new Response(asBufferSource(bytes))
      );
    },
  };
}

/** Safari's private mode has no Cache API but does have this, which is the only reason the second fallback exists. */
export async function opfsCache(
  namespace: string = MODEL_CACHE_NAMESPACE
): Promise<ModelCache> {
  if (
    typeof navigator === "undefined" ||
    typeof navigator.storage?.getDirectory !== "function"
  ) {
    throw new Error("this host has no origin-private file system");
  }
  const root = await navigator.storage.getDirectory();
  const dir = await root.getDirectoryHandle(namespace, { create: true });
  return {
    // Removing an entry that is not there is the state the caller asked for.
    drop: async (sha256) => {
      await dir.removeEntry(sha256).catch(() => undefined);
    },
    kind: "opfs",
    read: async (sha256) => {
      const handle = await dir.getFileHandle(sha256).catch(() => undefined);
      if (handle === undefined) {
        return;
      }
      const file = await handle.getFile();
      return new Uint8Array(await file.arrayBuffer());
    },
    // An interrupted write leaves a short file rather than none, and the
    // store's length check refuses it and re-fetches, so there is no
    // temporary-file dance here.
    write: async (sha256, bytes) => {
      const handle = await dir.getFileHandle(sha256, { create: true });
      const stream = await handle.createWritable();
      await stream.write(asBufferSource(bytes));
      await stream.close();
    },
  };
}

/**
 * Cache API, then OPFS, then memory. The chain *is* the environment test: each
 * factory refuses a host it cannot serve, so nothing outside this module has to
 * ask, and a browser that offers no persistence still gets a working store with
 * a per-session cache.
 */
export async function browserCache(
  namespace: string = MODEL_CACHE_NAMESPACE
): Promise<ModelCache> {
  return await cacheApiCache(namespace)
    .catch(() => opfsCache(namespace))
    .catch(() => memoryCache());
}

"""In-memory LRU cache with optional per-entry TTL.

Thread-safe (a single `threading.Lock` per instance) since O1 Component 4's
parallel retrieval executor calls into a shared cache instance from multiple
worker threads. Eviction is pure LRU (`OrderedDict.move_to_end` /
`popitem(last=False)`) — no probabilistic/approximate eviction, since the
cache sizes this project needs (hundreds to low thousands of entries) don't
warrant one.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, Protocol, TypeVar

from app.caching.models import CacheStats

K = TypeVar("K")
V = TypeVar("V")


class Cache(Protocol[K, V]):
    def get(self, key: K) -> V | None: ...
    def set(self, key: K, value: V, ttl: float | None = None) -> None: ...
    def clear(self) -> None: ...
    @property
    def stats(self) -> CacheStats: ...


class LRUTTLCache(Generic[K, V]):
    def __init__(self, max_size: int = 256, default_ttl: float | None = None, clock: Callable[[], float] = time.monotonic):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._clock = clock
        self._store: OrderedDict[K, tuple[V, float | None]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K) -> V | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            value, expires_at = entry
            if expires_at is not None and self._clock() >= expires_at:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: K, value: V, ttl: float | None = None) -> None:
        with self._lock:
            effective_ttl = ttl if ttl is not None else self._default_ttl
            expires_at = self._clock() + effective_ttl if effective_ttl is not None else None
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, expires_at)
            if len(self._store) > self._max_size:
                self._store.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(hits=self._hits, misses=self._misses, evictions=self._evictions, size=len(self._store), max_size=self._max_size)


def cache_key(*parts: Any) -> str:
    """A stable string cache key from arbitrary hashable-ish parts — used
    instead of raw tuples so keys stay readable in logs/debugging and so
    callers don't need to worry about tuple-hashing edge cases (e.g. a list
    argument) themselves."""
    return "|".join(str(p) for p in parts)

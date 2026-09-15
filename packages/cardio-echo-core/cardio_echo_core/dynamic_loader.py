"""Dynamic model loader — LRU cache with idle eviction.

Solves the "6 models don't fit on one GPU" problem by keeping at most N
models resident at any time. Models are loaded on demand and evicted after
a configurable idle timeout.

Usage:
    from cardio_echo_core.dynamic_loader import DynamicModelCache

    cache = DynamicModelCache(max_resident=2, idle_timeout_s=300)
    model = await cache.get_or_load(
        key="echonet-dynamic",
        loader_fn=lambda: EchoNetDynamicModel(),
    )
    # Use model...
    # After 5 min idle, the model is evicted from memory.

For multi-process deployments (e.g. one uvicorn worker per service), each
process has its own cache — the cache is per-process, not shared.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    model: Any
    last_used: float = field(default_factory=time.time)
    load_count: int = 0


class DynamicModelCache:
    """LRU cache with idle eviction for model instances.

    Thread-safe (uses an asyncio.Lock for async loaders; for sync loaders,
    the GIL is sufficient).
    """

    def __init__(
        self,
        max_resident: int = 2,
        idle_timeout_s: float = 300.0,
        eviction_callback: Optional[Callable[[str, Any], None]] = None,
    ):
        self.max_resident = max_resident
        self.idle_timeout_s = idle_timeout_s
        self.eviction_callback = eviction_callback
        self._cache: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._lock = asyncio.Lock()
        self._last_eviction_check = time.time()

    async def get_or_load(
        self,
        key: str,
        loader_fn: Callable[[], Any],
        is_async_loader: bool = False,
    ) -> Any:
        """Get a model from cache, loading it if necessary.

        Args:
            key: unique key (e.g. service name).
            loader_fn: callable that returns the model. May be async.
            is_async_loader: if True, await the loader_fn.

        Returns:
            The model instance.
        """
        async with self._lock:
            self._maybe_evict_idle()

            if key in self._cache:
                entry = self._cache.pop(key)
                entry.last_used = time.time()
                entry.load_count += 1
                self._cache[key] = entry
                logger.debug("Cache hit for %s (load_count=%d)", key, entry.load_count)
                return entry.model

            # Cache miss — load
            if len(self._cache) >= self.max_resident:
                self._evict_lru()

            logger.info("Loading model %s (cache miss)...", key)
            if is_async_loader:
                model = await loader_fn()
            else:
                model = loader_fn()
            self._cache[key] = CacheEntry(model=model)
            logger.info("Model %s loaded. Cache size: %d/%d",
                        key, len(self._cache), self.max_resident)
            return model

    def _maybe_evict_idle(self) -> None:
        """Evict entries that have been idle for longer than idle_timeout_s."""
        now = time.time()
        if now - self._last_eviction_check < 10:  # check at most every 10s
            return
        self._last_eviction_check = now

        to_evict = []
        for key, entry in self._cache.items():
            if now - entry.last_used > self.idle_timeout_s:
                to_evict.append(key)

        for key in to_evict:
            entry = self._cache.pop(key)
            logger.info(
                "Evicted idle model %s (idle for %.1fs)",
                key, now - entry.last_used,
            )
            if self.eviction_callback:
                try:
                    self.eviction_callback(key, entry.model)
                except Exception as e:
                    logger.warning("Eviction callback for %s failed: %s", key, e)

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return
        key, entry = self._cache.popitem(last=False)
        logger.info("Evicted LRU model %s to make room", key)
        if self.eviction_callback:
            try:
                self.eviction_callback(key, entry.model)
            except Exception as e:
                logger.warning("Eviction callback for %s failed: %s", key, e)

    def stats(self) -> dict:
        """Return cache stats for monitoring."""
        now = time.time()
        return {
            "size": len(self._cache),
            "max_resident": self.max_resident,
            "entries": {
                key: {
                    "last_used_s_ago": round(now - e.last_used, 1),
                    "load_count": e.load_count,
                }
                for key, e in self._cache.items()
            },
        }

    def clear(self) -> None:
        """Clear all cached models."""
        for key, entry in list(self._cache.items()):
            if self.eviction_callback:
                try:
                    self.eviction_callback(key, entry.model)
                except Exception:
                    pass
        self._cache.clear()


# Default global cache — 2 resident models, 5 min idle timeout
_default_cache: Optional[DynamicModelCache] = None


def get_default_cache() -> DynamicModelCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = DynamicModelCache(
            max_resident=2,
            idle_timeout_s=300.0,
            eviction_callback=lambda key, model: logger.info(
                "Default cache evicted %s", key
            ),
        )
    return _default_cache

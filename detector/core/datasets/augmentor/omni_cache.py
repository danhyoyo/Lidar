"""Bounded In-Memory LRU Point Cache for DataLoader Worker Acceleration."""

from __future__ import annotations
import collections
import os
import numpy as np


class BoundedPointCache:
    """Worker-local bounded LRU memory cache for decoded GT object point clouds.
    
    Automatically invalidates on process fork to prevent memory corruption
    or stale shared handles across PyTorch DataLoader workers.
    """

    def __init__(self, max_size_mb: float = 128.0):
        self.max_bytes = int(max_size_mb * 1024 * 1024)
        self.current_bytes = 0
        self.pid = os.getpid()
        self._cache: collections.OrderedDict[str, np.ndarray] = collections.OrderedDict()

    def _check_fork(self) -> None:
        current_pid = os.getpid()
        if current_pid != self.pid:
            self._cache.clear()
            self.current_bytes = 0
            self.pid = current_pid

    def get(self, key: str) -> np.ndarray | None:
        self._check_fork()
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key].copy()

    def put(self, key: str, points: np.ndarray) -> None:
        self._check_fork()
        if key in self._cache:
            self.current_bytes -= self._cache[key].nbytes
            del self._cache[key]

        pts_bytes = points.nbytes
        while self.current_bytes + pts_bytes > self.max_bytes and self._cache:
            _, evicted = self._cache.popitem(last=False)
            self.current_bytes -= evicted.nbytes

        self._cache[key] = points.copy()
        self.current_bytes += pts_bytes

    def __len__(self) -> int:
        self._check_fork()
        return len(self._cache)

    def __getstate__(self) -> dict:
        """Exclude cache memory during pickle serialization."""
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state: dict) -> None:
        self.max_bytes = state["max_bytes"]
        self.current_bytes = 0
        self.pid = os.getpid()
        self._cache = collections.OrderedDict()

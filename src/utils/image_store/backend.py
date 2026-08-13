"""Storage backends for the image store.

Provides:
- ``StorageBackend``: abstract base class
- ``MemoryBackend``: LRU + spill-to-disk in-memory cache
- ``FileBackend``: file-system storage, organized by category

The ``MemoryBackend`` keeps up to ``max_memory_items`` ndarray entries in an
``OrderedDict`` (LRU order). When the limit is exceeded, the least-recently
used entry is evicted from memory and spilled to disk via an internal
``FileBackend``. On ``load``, memory is checked first; on miss, disk is
read. The caller sees the same four methods (``save`` / ``load`` / ``exists``
/ ``list``) regardless of where the bytes live.
"""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np


class StorageBackend(ABC):
    """Abstract storage backend protocol."""

    @abstractmethod
    def save(self, image: np.ndarray, category: str, filename: str) -> None:
        """Persist an ndarray under ``(category, filename)``."""

    @abstractmethod
    def load(self, category: str, filename: str) -> np.ndarray:
        """Load the ndarray at ``(category, filename)``; raise if missing."""

    @abstractmethod
    def exists(self, category: str, filename: str) -> bool:
        """Check whether ``(category, filename)`` is currently stored."""

    @abstractmethod
    def list(self, category: Optional[str] = None) -> List[str]:
        """List filenames. If ``category`` is given, filter to that category;
        otherwise return all filenames across all categories."""


class FileBackend(StorageBackend):
    """File-system backend rooted at ``root_dir``.

    Layout::

        root_dir/
          <category>/
            <filename.png>
            ...

    Encoding: PNG via ``imageio.v3.imwrite`` / ``imageio.v3.imread``.
    """

    def __init__(self, root_dir: Path) -> None:
        self.root_dir: Path = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, category: str, filename: str) -> Path:
        return self.root_dir / category / filename

    def save(self, image: np.ndarray, category: str, filename: str) -> None:
        path = self._path_for(category, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Lazy import keeps the rest of the module usable when imageio
        # is not installed (e.g. partial dev envs).
        import imageio.v3 as iio

        iio.imwrite(path, image)

    def load(self, category: str, filename: str) -> np.ndarray:
        path = self._path_for(category, filename)
        import imageio.v3 as iio

        return np.asarray(iio.imread(path))

    def exists(self, category: str, filename: str) -> bool:
        return self._path_for(category, filename).exists()

    def list(self, category: Optional[str] = None) -> List[str]:
        if category is None:
            # list across all categories
            results: List[str] = []
            for sub in sorted(self.root_dir.iterdir()):
                if sub.is_dir():
                    results.extend(sorted(p.name for p in sub.iterdir() if p.is_file() and p.suffix == ".png"))
            return results
        sub = self.root_dir / category
        if not sub.exists():
            return []
        return sorted(p.name for p in sub.iterdir() if p.is_file() and p.suffix == ".png")


class MemoryBackend(StorageBackend):
    """LRU + spill-to-disk in-memory cache.

    Keeps the most-recently-used ``max_memory_items`` ndarrays in an
    ``OrderedDict``. On overflow, the least-recently-used entry is written
    to a private ``FileBackend`` rooted at ``spill_dir`` before being
    dropped from memory.

    Public surface mirrors ``StorageBackend`` exactly; the caller does not
    need to know whether a value lives in memory or on disk.
    """

    def __init__(
        self,
        max_memory_items: int = 10,
        spill_dir: Path = Path("data/images"),
    ) -> None:
        if max_memory_items <= 0:
            raise ValueError(
                f"max_memory_items must be positive, got {max_memory_items}"
            )
        self._max_memory_items: int = max_memory_items
        self._spill_dir: Path = Path(spill_dir).resolve()
        self._memory: "OrderedDict[tuple[str, str], np.ndarray]" = OrderedDict()
        # Internal spill target is a FileBackend rooted at spill_dir.
        self._spill: FileBackend = FileBackend(self._spill_dir)

    # ---- introspection helpers (used by tests) ----

    @property
    def max_memory_items(self) -> int:
        return self._max_memory_items

    @property
    def spill_dir(self) -> Path:
        return self._spill_dir

    def __len__(self) -> int:
        return len(self._memory)

    # ---- StorageBackend interface ----

    def save(self, image: np.ndarray, category: str, filename: str) -> None:
        key = (category, filename)
        self._memory[key] = image
        self._memory.move_to_end(key)
        # Spill oldest entries while over the soft cap.
        while len(self._memory) > self._max_memory_items:
            old_key, old_image = self._memory.popitem(last=False)
            old_cat, old_fn = old_key
            self._spill.save(old_image, old_cat, old_fn)

    def load(self, category: str, filename: str) -> np.ndarray:
        key = (category, filename)
        if key in self._memory:
            self._memory.move_to_end(key)
            return self._memory[key]
        # Disk fallback.
        return self._spill.load(category, filename)

    def exists(self, category: str, filename: str) -> bool:
        key = (category, filename)
        if key in self._memory:
            return True
        return self._spill.exists(category, filename)

    def list(self, category: Optional[str] = None) -> List[str]:
        if category is None:
            # merge memory keys + spill list
            mem_filenames = [fn for (_c, fn) in self._memory.keys()]
            disk_filenames = self._spill.list(None)
            seen: set[str] = set()
            results: List[str] = []
            for fn in mem_filenames + disk_filenames:
                if fn not in seen:
                    seen.add(fn)
                    results.append(fn)
            return results

        mem_filenames = [fn for (c, fn) in self._memory.keys() if c == category]
        disk_filenames = self._spill.list(category)
        seen = set()
        results: List[str] = []
        for fn in mem_filenames + disk_filenames:
            if fn not in seen:
                seen.add(fn)
                results.append(fn)
        return results

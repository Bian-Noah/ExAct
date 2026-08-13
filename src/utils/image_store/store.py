"""ImageStore: high-level facade combining URL scheme + storage backend.

The caller uses ``save`` / ``load`` / ``exists`` / ``list`` and receives
``img://...`` URLs. They never see the underlying backend.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np

from utils.image_store.backend import StorageBackend
from utils.image_store.url_scheme import build_url, parse_url


class ImageStore:
    """High-level image store facade."""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        # Per-instance monotonic counter for filename generation
        self._counter: int = 0
        self._counter_lock = Lock()

    # ---- filename generation ----

    def _generate_filename(self) -> str:
        """Generate a unique ``YYYYMMDD_HHMMSS_NNN.png`` filename."""
        with self._counter_lock:
            self._counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{ts}_{self._counter:03d}.png"

    # ---- public API ----

    def save(
        self,
        image: np.ndarray,
        category: str = "observations",
    ) -> str:
        """Save an image; return an ``img://...`` URL.

        Args:
            image: ndarray to persist.
            category: logical bucket, default ``"observations"``.

        Returns:
            A URL string of the form ``img://{category}/{filename}.png``.
        """
        filename = self._generate_filename()
        self._backend.save(image, category, filename)
        return build_url(category, filename)

    def load(self, url: str) -> np.ndarray:
        """Load an image by URL.

        Raises:
            ValueError: if the URL is malformed.
            FileNotFoundError / KeyError: if the URL is well-formed but no
                such image exists.
        """
        category, filename = parse_url(url)
        return self._backend.load(category, filename)

    def exists(self, url: str) -> bool:
        """Return whether the URL points to a stored image."""
        category, filename = parse_url(url)
        return self._backend.exists(category, filename)

    def list(self, category: Optional[str] = None) -> List[str]:
        """Return URLs of stored images, optionally filtered by category.

        The output is a list of URL strings (``img://...``), one per
        stored file in the requested scope.
        """
        filenames = self._backend.list(category)
        if category is None:
            # Need to reconstruct full URLs with category info
            # The backend's list() with None returns filenames across all
            # categories — we need to inspect memory keys / disk layout
            # to recover category info. For simplicity here, delegate to
            # backend.list per-category and union.
            categories = self._discover_categories()
            results: List[str] = []
            seen: set[str] = set()
            for cat in categories:
                for fn in self._backend.list(cat):
                    url = build_url(cat, fn)
                    if url not in seen:
                        seen.add(url)
                        results.append(url)
            return results

        return [build_url(category, fn) for fn in filenames]

    def _discover_categories(self) -> List[str]:
        """Return the union of categories known to the backend."""
        if hasattr(self._backend, "_memory"):
            # MemoryBackend: scan memory keys
            cats = sorted({cat for (cat, _fn) in self._backend._memory.keys()})
            # Merge with spill categories if any
            try:
                for sub in sorted(self._backend._spill_dir.iterdir()):
                    if sub.is_dir() and sub.name not in cats:
                        cats.append(sub.name)
            except (FileNotFoundError, OSError):
                pass
            return cats

        if hasattr(self._backend, "root_dir"):
            # FileBackend: list root_dir subdirs
            root = self._backend.root_dir
            try:
                return sorted(p.name for p in root.iterdir() if p.is_dir())
            except (FileNotFoundError, OSError):
                return []

        return []

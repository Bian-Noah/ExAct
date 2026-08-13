"""ImageStoreConfig: configuration for the image store.

Three fields:
- ``backend``: ``"memory"`` (default LRU+spill) or ``"file"`` (pure disk)
- ``file_dir``: relative path used both as spill dir (memory) and root_dir (file)
- ``max_memory_items``: in-memory cap for the memory backend
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImageStoreConfig:
    backend: str = "memory"
    file_dir: str = "data/images"
    max_memory_items: int = 10

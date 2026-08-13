"""Public API for the image_store package."""

from __future__ import annotations

from pathlib import Path

from utils.image_store.backend import (
    FileBackend,
    MemoryBackend,
    StorageBackend,
)
from utils.image_store.store import ImageStore
from utils.image_store.url_scheme import (
    build_url,
    is_valid_url,
    parse_url,
)

__all__ = [
    "FileBackend",
    "ImageStore",
    "MemoryBackend",
    "StorageBackend",
    "build_url",
    "create_image_store",
    "is_valid_url",
    "parse_url",
]


def _resolve_dir(file_dir: str) -> Path:
    """Resolve a possibly-relative path to an absolute Path.

    Relative paths are anchored at the current working directory. Callers
    who want project-root anchoring should pass an already-absolute path.
    """
    p = Path(file_dir)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def create_image_store(cfg) -> ImageStore:
    """Construct an ``ImageStore`` from an ``ImageStoreConfig``.

    Args:
        cfg: ``ImageStoreConfig`` dataclass (or duck-typed object with
            ``backend`` / ``file_dir`` / ``max_memory_items`` attributes).

    Returns:
        An ``ImageStore`` wrapping the appropriate backend.

    Raises:
        ValueError: if ``cfg.backend`` is not ``"memory"`` or ``"file"``.
    """
    spill_dir = _resolve_dir(cfg.file_dir)

    if cfg.backend == "memory":
        backend: StorageBackend = MemoryBackend(
            max_memory_items=cfg.max_memory_items,
            spill_dir=spill_dir,
        )
    elif cfg.backend == "file":
        backend = FileBackend(root_dir=spill_dir)
    else:
        raise ValueError(
            f"Unknown image_store backend {cfg.backend!r}: expected 'memory' or 'file'"
        )

    return ImageStore(backend)

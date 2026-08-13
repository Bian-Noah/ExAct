"""Unit tests for create_image_store factory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config.image_store_config import ImageStoreConfig
from src.utils.image_store import (
    FileBackend,
    ImageStore,
    MemoryBackend,
    create_image_store,
)


# ---------- backend dispatch ----------

class TestFactoryBackendDispatch:
    def test_memory_backend_returns_memorybackend(self, tmp_path: Path):
        cfg = ImageStoreConfig(
            backend="memory", file_dir=str(tmp_path / "spill"), max_memory_items=10
        )
        store = create_image_store(cfg)
        assert isinstance(store, ImageStore)
        assert isinstance(store._backend, MemoryBackend)

    def test_file_backend_returns_filebackend(self, tmp_path: Path):
        cfg = ImageStoreConfig(
            backend="file", file_dir=str(tmp_path / "files"), max_memory_items=10
        )
        store = create_image_store(cfg)
        assert isinstance(store._backend, FileBackend)
        assert store._backend.root_dir.is_absolute()

    def test_invalid_backend_raises_value_error(self, tmp_path: Path):
        cfg = ImageStoreConfig(
            backend="redis", file_dir=str(tmp_path), max_memory_items=10
        )
        with pytest.raises(ValueError, match="backend"):
            create_image_store(cfg)


# ---------- defaults ----------

class TestFactoryDefaults:
    def test_default_config_values(self):
        cfg = ImageStoreConfig()
        assert cfg.backend == "memory"
        assert cfg.file_dir == "data/images"
        assert cfg.max_memory_items == 10


# ---------- path resolution ----------

class TestFactoryPathResolution:
    def test_relative_path_resolved_to_absolute(self, tmp_path: Path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = ImageStoreConfig(backend="file", file_dir="imgs", max_memory_items=10)
        store = create_image_store(cfg)
        assert isinstance(store._backend, FileBackend)
        # Should resolve to tmp_path/imgs
        assert store._backend.root_dir == (tmp_path / "imgs").resolve()

    def test_absolute_path_used_as_is(self, tmp_path: Path):
        abs_dir = (tmp_path / "abs").resolve()
        cfg = ImageStoreConfig(backend="file", file_dir=str(abs_dir), max_memory_items=10)
        store = create_image_store(cfg)
        assert store._backend.root_dir == abs_dir


# ---------- max_memory_items propagation ----------

class TestFactoryMaxMemoryItems:
    def test_max_memory_items_propagated_to_memory_backend(self, tmp_path: Path):
        cfg = ImageStoreConfig(
            backend="memory", file_dir=str(tmp_path), max_memory_items=5
        )
        store = create_image_store(cfg)
        assert isinstance(store._backend, MemoryBackend)
        assert store._backend.max_memory_items == 5


# ---------- bad input ----------

class TestFactoryBadInput:
    def test_invalid_backend_raises_value_error_via_bad_input(self):
        # Duck-typed object that quacks like a config with the wrong backend.
        class BadCfg:
            backend = "redis"
            file_dir = "x"
            max_memory_items = 10

        with pytest.raises(ValueError, match="backend"):
            create_image_store(BadCfg())


# ---------- smoke roundtrip ----------

class TestFactorySmoke:
    def test_save_load_roundtrip_via_factory(self, tmp_path: Path):
        cfg = ImageStoreConfig(
            backend="memory", file_dir=str(tmp_path), max_memory_items=10
        )
        store = create_image_store(cfg)
        img = np.full((8, 8, 3), 1, dtype=np.uint8)
        url = store.save(img, "obs")
        loaded = store.load(url)
        assert np.array_equal(loaded, img)

    def test_file_backend_roundtrip_via_factory(self, tmp_path: Path):
        cfg = ImageStoreConfig(
            backend="file", file_dir=str(tmp_path / "files"), max_memory_items=10
        )
        store = create_image_store(cfg)
        img = np.full((8, 8, 3), 2, dtype=np.uint8)
        url = store.save(img, "obs")
        loaded = store.load(url)
        assert np.array_equal(loaded, img)

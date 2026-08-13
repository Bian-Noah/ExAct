"""Unit tests for ImageStore facade."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.utils.image_store.backend import (
    FileBackend,
    MemoryBackend,
    StorageBackend,
)
from src.utils.image_store.store import ImageStore
from src.utils.image_store.url_scheme import parse_url


def _img(value: int = 0, h: int = 8, w: int = 8, channels: int = 3) -> np.ndarray:
    arr = np.zeros((h, w, channels), dtype=np.uint8)
    arr[..., 0] = value
    return arr


# Mock backend for verifying facade contract
class MockBackend(StorageBackend):
    """Records every call; lets the test inspect interaction."""

    def __init__(self) -> None:
        self.store: dict = {}
        self.save_calls: List = []
        self.load_calls: List = []
        self.exists_calls: List = []
        self.list_calls: List = []

    def save(self, image, category, filename):
        self.save_calls.append((category, filename))
        self.store[(category, filename)] = image

    def load(self, category, filename):
        self.load_calls.append((category, filename))
        if (category, filename) not in self.store:
            raise KeyError((category, filename))
        return self.store[(category, filename)]

    def exists(self, category, filename):
        self.exists_calls.append((category, filename))
        return (category, filename) in self.store

    def list(self, category=None):
        self.list_calls.append(category)
        if category is None:
            return sorted(fn for (_c, fn) in self.store.keys())
        return sorted(fn for (c, fn) in self.store.keys() if c == category)


# ---------- URL format ----------

class TestImageStoreUrlFormat:
    def test_save_returns_parseable_url(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        url = store.save(_img(), "observations")
        cat, fn = parse_url(url)
        assert cat == "observations"
        assert fn.endswith(".png")

    def test_save_url_includes_category(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        url = store.save(_img(), "actions")
        cat, _ = parse_url(url)
        assert cat == "actions"


# ---------- save → load roundtrip ----------

class TestImageStoreRoundtrip:
    def test_roundtrip_with_memory_backend(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        img = _img(42)
        url = store.save(img, "observations")
        loaded = store.load(url)
        assert np.array_equal(loaded, img)

    def test_roundtrip_with_file_backend(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        store = ImageStore(fb)
        img = _img(7)
        url = store.save(img, "obs")
        loaded = store.load(url)
        assert np.array_equal(loaded, img)


# ---------- exists ----------

class TestImageStoreExists:
    def test_exists_true_after_save(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        url = store.save(_img(), "obs")
        assert store.exists(url) is True

    def test_exists_false_for_missing(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        assert store.exists("img://obs/missing.png") is False


# ---------- list ----------

class TestImageStoreList:
    def test_list_single_category_returns_urls(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        store.save(_img(0), "obs")
        store.save(_img(1), "obs")
        store.save(_img(2), "actions")
        listing = store.list("obs")
        assert len(listing) == 2
        for url in listing:
            cat, fn = parse_url(url)
            assert cat == "obs"

    def test_list_none_returns_all_urls(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        store.save(_img(), "obs")
        store.save(_img(), "actions")
        listing = store.list(None)
        cats = {parse_url(u)[0] for u in listing}
        assert cats == {"obs", "actions"}


# ---------- default category ----------

class TestImageStoreDefaults:
    def test_default_category_is_observations(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        url = store.save(_img())
        cat, _ = parse_url(url)
        assert cat == "observations"


# ---------- error paths ----------

class TestImageStoreErrors:
    def test_load_invalid_url_raises_value_error(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        with pytest.raises(ValueError):
            store.load("not-a-url")

    def test_load_missing_url_raises(self, tmp_path: Path):
        store = ImageStore(MemoryBackend(max_memory_items=10, spill_dir=tmp_path))
        with pytest.raises((FileNotFoundError, KeyError)):
            store.load("img://obs/never_existed.png")


# ---------- mock backend (duck typing) ----------

class TestImageStoreMockBackend:
    def test_works_with_any_storage_backend(self):
        mock = MockBackend()
        store = ImageStore(mock)
        img = _img(3)
        url = store.save(img, "obs")
        # save went to backend
        assert len(mock.save_calls) == 1
        # load roundtrip
        loaded = store.load(url)
        assert np.array_equal(loaded, img)
        assert len(mock.load_calls) == 1
        # exists
        assert store.exists(url) is True
        assert len(mock.exists_calls) == 1
        # list
        listing = store.list("obs")
        assert len(listing) == 1
        assert len(mock.list_calls) >= 1

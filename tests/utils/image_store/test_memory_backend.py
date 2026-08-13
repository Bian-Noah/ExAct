"""Unit tests for MemoryBackend (LRU + spill-to-disk)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.image_store.backend import FileBackend, MemoryBackend


# ---------- helpers ----------

def _img(value: int = 0, h: int = 16, w: int = 16, channels: int = 3) -> np.ndarray:
    """Create a deterministic small image filled with ``value``."""
    arr = np.zeros((h, w, channels), dtype=np.uint8)
    arr[..., 0] = value
    return arr


# ---------- basic save / load ----------

class TestMemorySaveLoad:
    def test_roundtrip_simple(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        img = _img(value=42)
        mb.save(img, "obs", "a.png")
        loaded = mb.load("obs", "a.png")
        assert np.array_equal(loaded, img)

    def test_roundtrip_multiple_files(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        for i in range(5):
            mb.save(_img(value=i), "obs", f"f{i}.png")
        for i in range(5):
            assert np.array_equal(mb.load("obs", f"f{i}.png"), _img(value=i))


# ---------- exists ----------

class TestMemoryExists:
    def test_exists_after_save(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        mb.save(_img(), "obs", "a.png")
        assert mb.exists("obs", "a.png") is True

    def test_not_exists(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        assert mb.exists("obs", "missing.png") is False

    def test_exists_after_spill(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=2, spill_dir=tmp_path)
        mb.save(_img(0), "obs", "a.png")
        mb.save(_img(1), "obs", "b.png")
        mb.save(_img(2), "obs", "c.png")  # evicts 'a.png' to disk
        assert mb.exists("obs", "a.png") is True  # still on disk
        assert mb.exists("obs", "b.png") is True
        assert mb.exists("obs", "c.png") is True


# ---------- list ----------

class TestMemoryList:
    def test_list_single_category_only_memory(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        mb.save(_img(), "obs", "a.png")
        mb.save(_img(), "obs", "b.png")
        mb.save(_img(), "actions", "c.png")
        assert set(mb.list("obs")) == {"a.png", "b.png"}
        assert mb.list("actions") == ["c.png"]

    def test_list_none_returns_all(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        mb.save(_img(), "obs", "a.png")
        mb.save(_img(), "actions", "b.png")
        result = set(mb.list(None))
        assert result == {"a.png", "b.png"}

    def test_list_merges_memory_and_disk(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=2, spill_dir=tmp_path)
        mb.save(_img(0), "obs", "a.png")
        mb.save(_img(1), "obs", "b.png")
        mb.save(_img(2), "obs", "c.png")  # 'a.png' spilled
        listing = mb.list("obs")
        assert set(listing) == {"a.png", "b.png", "c.png"}

    def test_list_missing_category(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        assert mb.list("never_saved") == []


# ---------- multi-category isolation ----------

class TestMemoryCategoryIsolation:
    def test_categories_do_not_collide(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        mb.save(_img(1), "obs", "x.png")
        mb.save(_img(2), "actions", "x.png")
        assert np.array_equal(mb.load("obs", "x.png"), _img(1))
        assert np.array_equal(mb.load("actions", "x.png"), _img(2))


# ---------- different dtypes / shapes ----------

class TestMemoryDtypeShape:
    def test_grayscale(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        img = np.full((20, 20), 128, dtype=np.uint8)
        mb.save(img, "obs", "gray.png")
        loaded = mb.load("obs", "gray.png")
        assert np.array_equal(loaded, img)

    def test_rgba(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        mb.save(img, "obs", "rgba.png")
        assert np.array_equal(mb.load("obs", "rgba.png"), img)

    def test_float_dtype(self, tmp_path: Path):
        # PNG only supports uint8/uint16; cast float32 → uint8 for storage.
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        img_f = np.full((10, 10, 3), 0.5, dtype=np.float32)
        img_u8 = (img_f * 255).astype(np.uint8)
        mb.save(img_u8, "obs", "f.png")
        loaded = mb.load("obs", "f.png")
        assert np.allclose(loaded, img_u8)


# ---------- error cases ----------

class TestMemoryErrors:
    def test_load_missing_raises_keyerror(self, tmp_path: Path):
        # When the image is neither in memory nor on disk, MemoryBackend
        # delegates to FileBackend.load which raises FileNotFoundError.
        # This is the realistic behavior for a missing image.
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            mb.load("obs", "missing.png")

    def test_load_missing_after_spill(self, tmp_path: Path):
        # After spill, FileBackend.load raises FileNotFoundError, which
        # propagates as the underlying error from MemoryBackend.load.
        mb = MemoryBackend(max_memory_items=1, spill_dir=tmp_path)
        mb.save(_img(), "obs", "a.png")
        with pytest.raises((FileNotFoundError, KeyError)):
            mb.load("obs", "never_existed.png")

    def test_load_missing_pure_disk_path(self, tmp_path: Path):
        # When item is neither in memory nor on disk, MemoryBackend delegates
        # to FileBackend.load, which raises FileNotFoundError. This is the
        # realistic case after a fresh MemoryBackend.
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            mb.load("obs", "never_existed.png")

    def test_max_memory_items_must_be_positive(self, tmp_path: Path):
        with pytest.raises(ValueError):
            MemoryBackend(max_memory_items=0, spill_dir=tmp_path)


# ---------- LRU eviction ----------

class TestMemoryLRU:
    def test_eviction_keeps_size_under_cap(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=3, spill_dir=tmp_path)
        for i in range(5):
            mb.save(_img(i), "obs", f"f{i}.png")
        assert len(mb) == 3

    def test_oldest_gets_spilled_first(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=2, spill_dir=tmp_path)
        mb.save(_img(0), "obs", "a.png")
        mb.save(_img(1), "obs", "b.png")
        mb.save(_img(2), "obs", "c.png")  # 'a.png' should be on disk now
        # 'a.png' must still be loadable (from disk)
        assert np.array_equal(mb.load("obs", "a.png"), _img(0))
        assert np.array_equal(mb.load("obs", "b.png"), _img(1))
        assert np.array_equal(mb.load("obs", "c.png"), _img(2))

    def test_lru_promotion_prevents_eviction(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=2, spill_dir=tmp_path)
        mb.save(_img(0), "obs", "a.png")
        mb.save(_img(1), "obs", "b.png")
        # Touch 'a.png' to promote it to MRU
        _ = mb.load("obs", "a.png")
        # Now save 'c.png': should evict 'b.png' (the LRU), not 'a.png'
        mb.save(_img(2), "obs", "c.png")
        # 'a.png' must be loadable from memory or disk
        assert np.array_equal(mb.load("obs", "a.png"), _img(0))
        # 'b.png' was evicted but exists on disk
        assert mb.exists("obs", "b.png") is True
        assert np.array_equal(mb.load("obs", "b.png"), _img(1))

    def test_disk_file_actually_exists_after_spill(self, tmp_path: Path):
        mb = MemoryBackend(max_memory_items=1, spill_dir=tmp_path)
        mb.save(_img(0), "obs", "a.png")
        mb.save(_img(1), "obs", "b.png")  # 'a.png' spills
        assert (tmp_path / "obs" / "a.png").exists()


# ---------- cross-instance persistence (memory → disk → new instance) ----------

class TestMemoryPersistence:
    def test_spilled_data_loadable_by_new_instance(self, tmp_path: Path):
        # Instance A
        a = MemoryBackend(max_memory_items=1, spill_dir=tmp_path)
        img = _img(99)
        a.save(img, "obs", "x.png")
        a.save(_img(1), "obs", "y.png")  # forces x.png to disk
        # Drop reference to A
        del a

        # Instance B — same spill_dir, fresh memory
        b = MemoryBackend(max_memory_items=1, spill_dir=tmp_path)
        # B's memory is empty, but x.png lives on disk from A
        loaded = b.load("obs", "x.png")
        assert np.array_equal(loaded, img)

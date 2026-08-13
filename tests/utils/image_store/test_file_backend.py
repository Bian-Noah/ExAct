"""Unit tests for FileBackend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.utils.image_store.backend import FileBackend


def _img(value: int = 0, h: int = 16, w: int = 16, channels: int = 3) -> np.ndarray:
    arr = np.zeros((h, w, channels), dtype=np.uint8)
    arr[..., 0] = value
    return arr


# ---------- save / load roundtrip ----------

class TestFileSaveLoad:
    def test_roundtrip_creates_file(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        img = _img(7)
        fb.save(img, "obs", "a.png")
        assert (tmp_path / "obs" / "a.png").exists()
        loaded = fb.load("obs", "a.png")
        assert np.array_equal(loaded, img)

    def test_overwrite_existing(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(0), "obs", "a.png")
        fb.save(_img(1), "obs", "a.png")
        assert np.array_equal(fb.load("obs", "a.png"), _img(1))

    def test_auto_creates_category_subdir(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(), "new_category", "a.png")
        assert (tmp_path / "new_category").is_dir()
        assert (tmp_path / "new_category" / "a.png").exists()

    def test_nested_subdir_creation(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(), "deep/path", "a.png")
        assert (tmp_path / "deep" / "path" / "a.png").exists()


# ---------- exists ----------

class TestFileExists:
    def test_exists_after_save(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(), "obs", "a.png")
        assert fb.exists("obs", "a.png") is True

    def test_not_exists(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        assert fb.exists("obs", "missing.png") is False

    def test_exists_after_external_delete(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(), "obs", "a.png")
        (tmp_path / "obs" / "a.png").unlink()
        assert fb.exists("obs", "a.png") is False


# ---------- list ----------

class TestFileList:
    def test_list_single_category(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(0), "obs", "a.png")
        fb.save(_img(1), "obs", "b.png")
        assert set(fb.list("obs")) == {"a.png", "b.png"}

    def test_list_filters_non_png(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(), "obs", "a.png")
        # Manually create a non-PNG file
        (tmp_path / "obs" / "junk.txt").write_text("not an image")
        (tmp_path / "obs" / "b.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        listing = fb.list("obs")
        assert listing == ["a.png"]

    def test_list_none_merges_categories(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        fb.save(_img(), "obs", "a.png")
        fb.save(_img(), "actions", "b.png")
        listing = fb.list(None)
        assert set(listing) == {"a.png", "b.png"}

    def test_list_missing_category(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        assert fb.list("never_saved") == []


# ---------- dtype / shape coverage ----------

class TestFileDtypeShape:
    def test_grayscale(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        img = np.full((20, 20), 128, dtype=np.uint8)
        fb.save(img, "obs", "g.png")
        assert np.array_equal(fb.load("obs", "g.png"), img)

    def test_rgba(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        img = np.zeros((10, 10, 4), dtype=np.uint8)
        fb.save(img, "obs", "rgba.png")
        assert np.array_equal(fb.load("obs", "rgba.png"), img)

    def test_float_image(self, tmp_path: Path):
        # PNG only supports uint8 / uint16; cast float32 to uint8 for storage.
        fb = FileBackend(tmp_path)
        img_f = np.full((10, 10, 3), 0.5, dtype=np.float32)
        img_u8 = (img_f * 255).astype(np.uint8)
        fb.save(img_u8, "obs", "f.png")
        assert np.allclose(fb.load("obs", "f.png"), img_u8)


# ---------- error cases ----------

class TestFileErrors:
    def test_load_missing_raises(self, tmp_path: Path):
        fb = FileBackend(tmp_path)
        with pytest.raises(FileNotFoundError):
            fb.load("obs", "missing.png")


# ---------- cross-instance persistence ----------

class TestFilePersistence:
    def test_data_survives_instance_recreation(self, tmp_path: Path):
        # Instance A
        a = FileBackend(tmp_path)
        img = _img(99)
        a.save(img, "obs", "x.png")
        del a

        # Instance B with same root_dir
        b = FileBackend(tmp_path)
        loaded = b.load("obs", "x.png")
        assert np.array_equal(loaded, img)

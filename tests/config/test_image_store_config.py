"""Unit tests for ImageStoreConfig integration with the YAML loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.image_store_config import ImageStoreConfig
from src.config.loader import load_config


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


# Use simple, consistent-indent YAML (no triple-quoted strings to avoid
# dedent confusion). Each test writes its own complete config.
BASE_YAML = """\
env:
  mode: direct
  renderer: auto
  camera_resolution: [640, 480]
vla:
  backend: mock
llm: {}
explore: {}
robot: {}
task: {}
agent: {}
experiment:
  enabled: true
  root: "data/experiment"
  log_to_stdout: true
"""


class TestImageStoreConfigDefaults:
    def test_image_store_section_present_in_default(self):
        cfg = load_config("configs/default.yaml")
        assert cfg.image_store == ImageStoreConfig(
            backend="memory", file_dir="data/images", max_memory_items=10
        )

    def test_missing_image_store_section_falls_back_to_defaults(self, tmp_path: Path):
        p = _write_yaml(tmp_path, BASE_YAML)
        cfg = load_config(str(p))
        assert cfg.image_store == ImageStoreConfig()

    def test_explicit_backend_file_loaded(self, tmp_path: Path):
        p = _write_yaml(
            tmp_path,
            BASE_YAML
            + """\
image_store:
  backend: file
  file_dir: "custom/dir"
  max_memory_items: 50
""",
        )
        cfg = load_config(str(p))
        assert cfg.image_store.backend == "file"
        assert cfg.image_store.file_dir == "custom/dir"
        assert cfg.image_store.max_memory_items == 50

    def test_explicit_max_memory_items_loaded(self, tmp_path: Path):
        p = _write_yaml(
            tmp_path,
            BASE_YAML
            + """\
image_store:
  backend: memory
  file_dir: "data/images"
  max_memory_items: 25
""",
        )
        cfg = load_config(str(p))
        assert cfg.image_store.max_memory_items == 25

    def test_partial_image_store_section_uses_defaults_for_missing(self, tmp_path: Path):
        p = _write_yaml(
            tmp_path,
            BASE_YAML
            + """\
image_store:
  backend: file
""",
        )
        cfg = load_config(str(p))
        assert cfg.image_store.backend == "file"
        assert cfg.image_store.file_dir == "data/images"
        assert cfg.image_store.max_memory_items == 10


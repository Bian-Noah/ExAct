"""Manual smoke test for create_image_store factory.

Run with:
    PYTHONPATH=src python scripts/smoke_factory.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

from config.image_store_config import ImageStoreConfig
from utils.image_store import create_image_store


def main() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Memory backend via factory
        cfg_mem = ImageStoreConfig(
            backend="memory", file_dir=str(tmp_path / "mem"), max_memory_items=3
        )
        store_mem = create_image_store(cfg_mem)
        url = store_mem.save(np.full((8, 8, 3), 1, dtype=np.uint8))
        loaded = store_mem.load(url)
        if np.array_equal(loaded, np.full((8, 8, 3), 1, dtype=np.uint8)):
            print(f"  ✓ factory memory backend roundtrip OK")
        else:
            print(f"  ✗ factory memory backend roundtrip FAILED")
            fails += 1

        # File backend via factory
        cfg_file = ImageStoreConfig(
            backend="file", file_dir=str(tmp_path / "file"), max_memory_items=10
        )
        store_file = create_image_store(cfg_file)
        url2 = store_file.save(np.full((8, 8, 3), 2, dtype=np.uint8), category="x")
        if np.array_equal(store_file.load(url2), np.full((8, 8, 3), 2, dtype=np.uint8)):
            print(f"  ✓ factory file backend roundtrip OK")
        else:
            print(f"  ✗ factory file backend roundtrip FAILED")
            fails += 1

        # Invalid backend
        try:
            create_image_store(ImageStoreConfig(backend="redis", file_dir=str(tmp_path)))
            print(f"  ✗ invalid backend should have raised")
            fails += 1
        except ValueError:
            print(f"  ✓ invalid backend raises ValueError")

    print(f"\n=== {('PASS' if fails == 0 else f'{fails} FAIL')} ===")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

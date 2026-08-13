"""Manual smoke test for ImageStore facade.

Run with:
    PYTHONPATH=src python scripts/smoke_image_store.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from utils.image_store.backend import FileBackend, MemoryBackend
from utils.image_store.store import ImageStore


def main() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Memory backend
        mb = MemoryBackend(max_memory_items=10, spill_dir=tmp_path)
        store = ImageStore(mb)
        img = np.full((16, 16, 3), 99, dtype=np.uint8)
        url = store.save(img, "observations")
        if np.array_equal(store.load(url), img):
            print(f"  ✓ MemoryBackend roundtrip: {url}")
        else:
            print(f"  ✗ MemoryBackend roundtrip failed")
            fails += 1

        # File backend
        del store, mb
        fb = FileBackend(tmp_path / "file")
        store = ImageStore(fb)
        img2 = np.full((16, 16, 3), 7, dtype=np.uint8)
        url2 = store.save(img2, "actions")
        if np.array_equal(store.load(url2), img2):
            print(f"  ✓ FileBackend roundtrip: {url2}")
        else:
            print(f"  ✗ FileBackend roundtrip failed")
            fails += 1
        if (tmp_path / "file" / "actions").exists():
            print(f"  ✓ FileBackend wrote to disk")
        else:
            print(f"  ✗ FileBackend did not write to disk")
            fails += 1

        # List cross-category
        store.save(np.full((8, 8, 3), 1, dtype=np.uint8), "obs")
        store.save(np.full((8, 8, 3), 2, dtype=np.uint8), "obs")
        all_urls = store.list(None)
        # 1 actions + 2 obs = 3 URLs
        if len(all_urls) == 3:
            print(f"  ✓ list(None) returned {len(all_urls)} URLs")
        else:
            print(f"  ✗ list(None) returned {len(all_urls)} URLs (expected 3)")
            fails += 1

    print(f"\n=== {('PASS' if fails == 0 else f'{fails} FAIL')} ===")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

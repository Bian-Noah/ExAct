"""Manual smoke test for MemoryBackend (LRU + spill-to-disk).

Run with:
    PYTHONPATH=src python scripts/smoke_memory_backend.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from utils.image_store.backend import MemoryBackend


def main() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mb = MemoryBackend(max_memory_items=2, spill_dir=tmp_path)
        imgs = []
        for i in range(4):
            img = np.full((8, 8, 3), i, dtype=np.uint8)
            imgs.append(img)
            mb.save(img, "obs", f"f{i}.png")

        # Cap is 2, so memory has at most 2 entries
        if len(mb) == 2:
            print(f"  ✓ LRU cap enforced: len(memory)={len(mb)}")
        else:
            print(f"  � LRU cap broken: len(memory)={len(mb)}")
            fails += 1

        # Spilled images must still be loadable
        for i in range(4):
            loaded = mb.load("obs", f"f{i}.png")
            if np.array_equal(loaded, imgs[i]):
                print(f"  ✓ f{i}.png roundtrip OK")
            else:
                print(f"  ✗ f{i}.png roundtrip FAILED")
                fails += 1

        # Spilled files exist on disk
        for i in range(2):  # f0 and f1 were spilled
            p = tmp_path / "obs" / f"f{i}.png"
            if p.exists():
                print(f"  ✓ spill file exists: {p}")
            else:
                print(f"  ✗ spill file missing: {p}")
                fails += 1

        # Cross-instance persistence
        del mb
        mb2 = MemoryBackend(max_memory_items=2, spill_dir=tmp_path)
        loaded = mb2.load("obs", "f0.png")
        if np.array_equal(loaded, imgs[0]):
            print(f"  ✓ cross-instance load OK")
        else:
            print(f"  ✗ cross-instance load FAILED")
            fails += 1

    print(f"\n=== {('PASS' if fails == 0 else f'{fails} FAIL')} ===")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

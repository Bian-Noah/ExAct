"""Manual smoke test for FileBackend.

Run with:
    PYTHONPATH=src python scripts/smoke_file_backend.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from utils.image_store.backend import FileBackend


def main() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # Write
        a = FileBackend(tmp_path)
        img = np.full((16, 16, 3), 42, dtype=np.uint8)
        a.save(img, "obs", "x.png")

        if (tmp_path / "obs" / "x.png").exists():
            print(f"  ✓ file on disk: {tmp_path}/obs/x.png")
        else:
            print(f"  ✗ file missing on disk")
            fails += 1

        # Cross-instance read
        del a
        b = FileBackend(tmp_path)
        loaded = b.load("obs", "x.png")
        if np.array_equal(loaded, img):
            print(f"  ✓ cross-instance read OK")
        else:
            print(f"  ✗ cross-instance read FAILED")
            fails += 1

        # Listing
        b.save(np.full((8, 8, 3), 7, dtype=np.uint8), "obs", "y.png")
        listing = b.list("obs")
        if set(listing) == {"x.png", "y.png"}:
            print(f"  ✓ list returns: {listing}")
        else:
            print(f"  ✗ list wrong: {listing}")
            fails += 1

        # None category
        b.save(np.full((8, 8, 3), 9, dtype=np.uint8), "actions", "z.png")
        listing_all = b.list(None)
        if set(listing_all) == {"x.png", "y.png", "z.png"}:
            print(f"  ✓ list(None) returns: {listing_all}")
        else:
            print(f"  ✗ list(None) wrong: {listing_all}")
            fails += 1

        # Missing load raises
        try:
            b.load("obs", "missing.png")
            print(f"  ✗ missing load should have raised")
            fails += 1
        except FileNotFoundError:
            print(f"  ✓ missing load raises FileNotFoundError")

    print(f"\n=== {('PASS' if fails == 0 else f'{fails} FAIL')} ===")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

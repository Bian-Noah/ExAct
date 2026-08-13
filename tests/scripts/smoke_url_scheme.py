"""Manual smoke test for src.utils.image_store.url_scheme.

Run with:
    python scripts/smoke_url_scheme.py

Exits 0 on success, non-zero on first failure.
"""

from __future__ import annotations

import sys

from utils.image_store.url_scheme import (
    build_url,
    is_valid_url,
    parse_url,
)


def main() -> int:
    cases_ok = 0
    cases_fail = 0

    # 5 valid scenarios: build + parse should roundtrip
    for cat, fn in [
        ("observations", "001.png"),
        ("actions", "frame_42.png"),
        ("explorations", "2026-08-13_001.png"),
        ("a", "b.png"),
        ("obs_v2", "snapshot.final.png"),
    ]:
        url = build_url(cat, fn)
        parsed = parse_url(url)
        if parsed == (cat, fn) and is_valid_url(url):
            print(f"  ✓ valid roundtrip: {cat}/{fn}")
            cases_ok += 1
        else:
            print(f"  ✗ valid roundtrip failed: {cat}/{fn}")
            cases_fail += 1

    # 5 invalid scenarios: build/parse must raise or return False
    invalid_cases = [
        ("build empty category", lambda: build_url("", "001.png")),
        ("build empty filename", lambda: build_url("obs", "")),
        ("build category with space", lambda: build_url("obs x", "001.png")),
        ("build filename without .png", lambda: build_url("obs", "001.jpg")),
        ("parse wrong scheme", lambda: parse_url("http://obs/001.png")),
    ]
    for name, fn in invalid_cases:
        try:
            fn()
            print(f"  ✗ {name}: should have raised")
            cases_fail += 1
        except ValueError:
            print(f"  ✓ {name}: raised ValueError")
            cases_ok += 1

    print(f"\n=== {cases_ok} PASS, {cases_fail} FAIL ===")
    return 0 if cases_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

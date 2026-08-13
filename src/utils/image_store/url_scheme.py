"""URL scheme utilities for the image store.

URL format: ``img://{category}/{filename}``

- category: ``[a-zA-Z0-9_]+`` (1-64 chars, prevents path traversal)
- filename: ``[a-zA-Z0-9_.-]+`` (must end with ``.png``, 1-128 chars)
"""

from __future__ import annotations

import re
from typing import Tuple

URL_SCHEME: str = "img"

CATEGORY_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_]{1,64}$")
FILENAME_PATTERN: re.Pattern[str] = re.compile(r"^[a-zA-Z0-9_.-]{1,128}$")
FILENAME_SUFFIX: str = ".png"

FULL_URL_PATTERN: re.Pattern[str] = re.compile(
    r"^img://(?P<category>[a-zA-Z0-9_]{1,64})/(?P<filename>[a-zA-Z0-9_.-]{1,128})$"
)


def build_url(category: str, filename: str) -> str:
    """Construct a ``img://{category}/{filename}`` URL.

    Args:
        category: The category name (must match ``CATEGORY_PATTERN``).
        filename: The filename (must match ``FILENAME_PATTERN`` and end with ``.png``).

    Returns:
        The constructed URL string.

    Raises:
        ValueError: If category or filename fails validation, with a clear message
            identifying which field and why.
    """
    if not CATEGORY_PATTERN.match(category or ""):
        raise ValueError(
            f"Invalid category {category!r}: must match {CATEGORY_PATTERN.pattern}"
        )
    if not FILENAME_PATTERN.match(filename or ""):
        raise ValueError(
            f"Invalid filename {filename!r}: must match {FILENAME_PATTERN.pattern}"
        )
    if not filename.endswith(FILENAME_SUFFIX):
        raise ValueError(
            f"Invalid filename {filename!r}: must end with {FILENAME_SUFFIX!r}"
        )
    return f"{URL_SCHEME}://{category}/{filename}"


def parse_url(url: str) -> Tuple[str, str]:
    """Parse a ``img://...`` URL into ``(category, filename)``.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple ``(category, filename)``.

    Raises:
        ValueError: If the URL is malformed (wrong scheme, missing parts,
            invalid characters, wrong filename suffix).
    """
    if not isinstance(url, str) or not url:
        raise ValueError(f"Invalid URL {url!r}: must be a non-empty string")
    match = FULL_URL_PATTERN.match(url)
    if match is None:
        if not url.startswith(f"{URL_SCHEME}://"):
            raise ValueError(
                f"Invalid URL {url!r}: must start with {URL_SCHEME + '://'!r}"
            )
        raise ValueError(
            f"Invalid URL {url!r}: must match pattern "
            f"'img://{{category}}/{{filename}}' where category matches "
            f"{CATEGORY_PATTERN.pattern} and filename matches "
            f"{FILENAME_PATTERN.pattern} and ends with {FILENAME_SUFFIX!r}"
        )
    category = match.group("category")
    filename = match.group("filename")
    if not filename.endswith(FILENAME_SUFFIX):
        raise ValueError(
            f"Invalid filename in URL {url!r}: must end with {FILENAME_SUFFIX!r}"
        )
    return category, filename


def is_valid_url(url: str) -> bool:
    """Check whether ``url`` is a valid image-store URL (does not raise).

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid, False otherwise.
    """
    if not isinstance(url, str) or not url:
        return False
    match = FULL_URL_PATTERN.match(url)
    if match is None:
        return False
    return match.group("filename").endswith(FILENAME_SUFFIX)

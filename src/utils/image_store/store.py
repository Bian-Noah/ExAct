"""ImageStore: high-level facade combining URL scheme + storage backend.

The caller uses ``save`` / ``load`` / ``exists`` / ``list`` and receives
``img://...`` URLs. They never see the underlying backend.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import List, Optional

import numpy as np

from utils.image_store.backend import StorageBackend
from utils.image_store.url_scheme import build_url, parse_url


class ImageStore:
    """High-level image store facade."""

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        # Per-instance monotonic counter for filename generation
        self._counter: int = 0
        self._counter_lock = Lock()

    # ---- filename generation ----

    def _generate_filename(self) -> str:
        """Generate a unique ``YYYYMMDD_HHMMSS_NNN.png`` filename."""
        with self._counter_lock:
            self._counter += 1
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{ts}_{self._counter:03d}.png"

    # ---- public API ----

    def save(
        self,
        image: np.ndarray,
        category: str = "observations",
    ) -> str:
        """Save an image; return an ``img://...`` URL.

        Args:
            image: ndarray to persist.
            category: logical bucket, default ``"observations"``.

        Returns:
            A URL string of the form ``img://{category}/{filename}.png``.
        """
        filename = self._generate_filename()
        self._backend.save(image, category, filename)
        return build_url(category, filename)

    def load(self, url: str) -> np.ndarray:
        """Load an image by URL.

        Raises:
            ValueError: if the URL is malformed.
            FileNotFoundError / KeyError: if the URL is well-formed but no
                such image exists.
        """
        category, filename = parse_url(url)
        return self._backend.load(category, filename)

    def exists(self, url: str) -> bool:
        """Return whether the URL points to a stored image."""
        category, filename = parse_url(url)
        return self._backend.exists(category, filename)

    def list(self, category: Optional[str] = None) -> List[str]:
        """Return URLs of stored images, optionally filtered by category.

        The output is a list of URL strings (``img://...``), one per
        stored file in the requested scope.
        """
        filenames = self._backend.list(category)
        if category is None:
            # Need to reconstruct full URLs with category info
            # The backend's list() with None returns filenames across all
            # categories — we need to inspect memory keys / disk layout
            # to recover category info. For simplicity here, delegate to
            # backend.list per-category and union.
            categories = self._discover_categories()
            results: List[str] = []
            seen: set[str] = set()
            for cat in categories:
                for fn in self._backend.list(cat):
                    url = build_url(cat, fn)
                    if url not in seen:
                        seen.add(url)
                        results.append(url)
            return results

        return [build_url(category, fn) for fn in filenames]

    def _discover_categories(self) -> List[str]:
        """Return the union of categories known to the backend."""
        if hasattr(self._backend, "_memory"):
            # MemoryBackend: scan memory keys
            cats = sorted({cat for (cat, _fn) in self._backend._memory.keys()})
            # Merge with spill categories if any
            try:
                spill_root = self._backend._spill.root_dir
                for sub in sorted(spill_root.iterdir()):
                    if sub.is_dir() and sub.name not in cats:
                        cats.append(sub.name)
            except (FileNotFoundError, OSError):
                pass
            return cats

        if hasattr(self._backend, "root_dir"):
            # FileBackend: list root_dir subdirs
            root = self._backend.root_dir
            try:
                return sorted(p.name for p in root.iterdir() if p.is_dir())
            except (FileNotFoundError, OSError):
                return []

        return []

    # ---- minimax file upload (iteration 5 fix) ----

    def upload_to_minimax(self, img_url: str) -> str:
        """把 img:// URL 对应的图片上传到 MiniMax file API，返回 mm_file://{file_id}。

        API key 从 configs/local.yaml（存在）或 configs/default.yaml（fallback）直接读取。
        不通过 config dataclass 层层传，方法自给自足。

        Args:
            img_url: img:// URL（来自 image_store.save()）。

        Returns:
            mm_file://{file_id} URL（provider 可消费）。

        Raises:
            RuntimeError: 当 base_resp.status_code != 0 时。
            requests.HTTPError: 当 HTTP 状态码非 2xx 时。
        """
        import io
        from pathlib import Path
        import yaml
        import imageio.v3 as iio
        import requests

        from utils.image_store.url_scheme import parse_url

        # 1. 读 api_key（先 local.yaml，没有再 fallback default.yaml）
        # src/utils/image_store/store.py → parents[3] 是项目根（ExAct/）
        config_root = Path(__file__).parents[3] / "configs"
        local_yaml = config_root / "local.yaml"
        default_yaml = config_root / "default.yaml"
        yaml_path = local_yaml if local_yaml.exists() else default_yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        api_key = cfg["llm"]["api_key"]

        # 2. 从 ImageStore 拿 ndarray
        category, filename = parse_url(img_url)
        image = self._backend.load(category, filename)

        # 3. ndarray → PNG bytes
        buf = io.BytesIO()
        iio.imwrite(buf, image, format="PNG")
        buf.seek(0)

        # 4. POST 到 MiniMax /v1/files/upload
        response = requests.post(
            "https://api.minimaxi.com/v1/files/upload",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": ("observation.png", buf, "image/png")},
            data={"purpose": "video_generation_input"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        # 业务层校验：HTTP 200 但 status_code != 0 视为失败
        base_resp = data.get("base_resp", {})
        if base_resp.get("status_code", -1) != 0:
            raise RuntimeError(
                f"MiniMax upload failed: status_code={base_resp.get('status_code')}, "
                f"status_msg={base_resp.get('status_msg')}"
            )
        return f"mm_file://{data['file']['file_id']}"

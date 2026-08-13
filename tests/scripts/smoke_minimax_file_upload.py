"""冒烟：ImageStore.upload_to_minimax 真实端到端（Iter 5 fix）。

目的：验证 MiniMax file upload API 在 video_generation_input purpose 下
能成功上传 640x480 PNG（真实 observe 帧大小），返回 mm_file://{file_id}。

跑法：
    PYTHONPATH=src python tests/scripts/smoke_minimax_file_upload.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np

from utils.image_store import ImageStore, MemoryBackend


def main() -> int:
    # 1. 构造与真实 PyBullet observe 同样尺寸的图（640x480）
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # 2. 构造 ImageStore 并存图
    img_dir = PROJECT_ROOT / "data" / "images"
    backend = MemoryBackend(max_memory_items=5, spill_dir=img_dir)
    store = ImageStore(backend)
    img_url = store.save(img, category="observations")
    print(f"[1] 已存图: img_url={img_url}")

    # 3. 上传到 MiniMax（直接读 yaml 拿 api_key）
    try:
        mm_url = store.upload_to_minimax(img_url)
    except Exception as e:
        print(f"[2] 上传失败: {e}")
        return 1
    print(f"[2] 上传成功: mm_url={mm_url}")

    # 4. 验证 URL 形态
    assert mm_url.startswith("mm_file://"), f"URL 形态错误: {mm_url}"
    assert mm_url != "mm_file://", "file_id 为空"
    print(f"[3] URL 格式校验通过: {mm_url}")

    # 5. 删除临时图文件（清理 spill）
    print("\n✓ 端到端通过（上传成功 + URL 格式正确）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
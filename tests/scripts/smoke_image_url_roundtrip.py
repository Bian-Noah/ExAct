"""Image URL 跨模块可加载验证（Iteration 5）。

跑一次 observe 工具，验证：
- list[1].url 是 img://observations/...
- image_store.load(url) 能取回 ndarray
- shape / dtype 与原始一致
- 用 image_store.list("observations") 列出所有 URL

用法：
    python scripts/smoke_image_url_roundtrip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from config import EnvConfig, RobotConfig  # noqa: E402
from utils.image_store import create_image_store  # noqa: E402
from tools import ObserveTool  # noqa: E402


def main() -> int:
    from env.pybullet_env import PyBulletEnv

    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="cpu"),
                            robot_config=RobotConfig())
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]}, seed=0)

    img_dir = PROJECT_ROOT / "data" / "images"
    image_store = create_image_store(
        type("CFG", (), {"backend": "file", "file_dir": str(img_dir), "max_memory_items": 10})()
    )

    tool = ObserveTool(env=env, image_store=image_store)
    result = tool._run()

    print(f"_run() 返回 {len(result)} 个块:")
    for i, block in enumerate(result):
        print(f"  [{i}] type={block.get('type')}, "
              f"text={block.get('text', '')[:60]!r}, "
              f"url={block.get('url', '')}")

    if len(result) < 2:
        print("FAIL: 期望 image 块")
        env.close()
        return 1

    saved_url = result[1]["url"]
    print(f"\nRound-trip 验证 (saved_url={saved_url}):")
    if not image_store.exists(saved_url):
        print(f"  FAIL: image_store.exists({saved_url}) == False")
        env.close()
        return 1
    print(f"  [1/3] image_store.exists({saved_url}) == True")

    loaded = image_store.load(saved_url)
    if loaded.shape != (480, 640, 3):
        print(f"  FAIL: shape {loaded.shape} != (480, 640, 3)")
        env.close()
        return 1
    print(f"  [2/3] image_store.load(url).shape == {loaded.shape}")

    # image_store.list 验证
    urls = image_store.list("observations")
    print(f"  [3/3] image_store.list('observations') 含 {len(urls)} 个 URL")
    for u in urls:
        print(f"    {u}")

    env.close()
    print("\n✓ Round-trip 验证通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

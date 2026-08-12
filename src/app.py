"""ExAct 第一步迭代手动验证脚本。

运行方式：
    PYTHONPATH=src python src/app.py
"""

import os
import sys

# PYTHONPATH fallback：确保 src 目录在搜索路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.pybullet_env import PyBulletPandaEnv
from env.base import Action7D


def main():
    print("Creating env...")
    env = PyBulletPandaEnv(use_gui=True, camera_resolution=(640, 480))

    print("Resetting...")
    obs = env.reset(
        task_spec={
            "objects": [
                {"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"},
            ]
        },
        seed=0,
    )

    print(f"obs keys: {obs.keys()}")
    print(f"ee_pos: {obs['ee_pos']}")
    print(f"render shape: {obs['rgb'].shape}, dtype: {obs['rgb'].dtype}")

    print("Stepping 50 times...")
    for i in range(50):
        action = Action7D(0.005, 0, 0, 0, 0, 0, 0.5)
        obs, reward, done, info = env.step(action)

    print("Closing env...")
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()

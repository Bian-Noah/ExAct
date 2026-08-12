"""ExAct 第二步迭代手动验证脚本。

验证 executor + MockVLA 闭环：
1. 创建 PyBulletPandaEnv（GUI 模式）
2. reset + render 打印 obs 信息
3. 实例化 MockVLA + Executor，调用 run_action
4. 打印 ExecResult 字段
5. close env

运行方式：
    PYTHONPATH=src python src/app.py
"""

import os
import sys

# PYTHONPATH fallback：确保 src 目录在搜索路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from env.pybullet_env import PyBulletPandaEnv
from executor import Executor, MockVLA


def main():
    print("Creating env...")
    env = PyBulletPandaEnv(use_gui=True, camera_resolution=(640, 480))

    # 任务场景：红色方块位置作为 target_pos
    task_spec = {
        "objects": [
            {"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"},
        ]
    }
    target_pos = tuple(task_spec["objects"][0]["pos"])

    try:
        print("Resetting...")
        obs = env.reset(task_spec=task_spec, seed=0)

        print(f"obs keys: {obs.keys()}")
        print(f"ee_pos: {obs['ee_pos']}")
        print(f"render shape: {obs['rgb'].shape}, dtype: {obs['rgb'].dtype}")
        print(f"target_pos: {target_pos}")

        print("Running executor...")
        vla = MockVLA(seed=0)
        executor = Executor(vla, max_steps=20)
        result = executor.run_action(
            env,
            "移动到红色方块上方",
            done_criteria="reached",
            target_pos=target_pos,
        )

        print(
            f"ExecResult: success={result.success}, "
            f"steps={result.steps}, message={result.message}"
        )
    finally:
        print("Closing env...")
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()

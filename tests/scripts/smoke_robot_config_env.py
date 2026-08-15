"""robot-vla-adapter 功能场景 A：RobotConfig 新结构加载 + env 拆分不回归。

验证：
  1. configs/default.yaml 新 robot 节（type + urdf_path 顶层 + panda 嵌套）能加载
  2. PyBulletEnv 构造/reset/step 与改造前行为一致（ee_pos 合理位移）

强制 EnvConfig(mode="direct")，不开 GUI 弹窗。PASS/FAIL 通过退出码表达。
运行：PYTHONPATH=src python tests/scripts/smoke_robot_config_env.py
"""

import sys

sys.path.insert(0, "src")

from env.base import Action7D
from env.pybullet_env import PyBulletEnv
from config.loader import EnvConfig, load_config


def main() -> int:
    config = load_config("configs/default.yaml")

    # 1. robot 节新结构断言
    robot = config.robot
    assert robot.type == "panda", f"type 应为 panda，得到 {robot.type}"
    assert robot.urdf_path == "franka_panda/panda.urdf"
    assert robot.panda.arm_joint_indices == (0, 1, 2, 3, 4, 5, 6)
    assert robot.panda.ee_link_index == 11
    print("[smoke] ✓ configs/default.yaml robot 节新结构加载正确")

    # 2. PyBulletEnv direct 模式：reset + step，验证 ee_pos 位移
    env = PyBulletEnv(
        env_config=EnvConfig(mode="direct", renderer="cpu"),
        robot_config=robot,
    )
    env.reset(task_spec={"objects": [dict(obj) for obj in config.task.objects]}, seed=0)

    obs_before = env.get_obs()
    ee_before = obs_before["ee_pos"]
    print(f"[smoke] ee_before={ee_before}")

    env.step(Action7D(0.01, 0, 0, 0, 0, 0, 0.5))
    obs_after = env.get_obs()
    ee_after = obs_after["ee_pos"]
    print(f"[smoke] ee_after ={ee_after}")

    dx = ee_after[0] - ee_before[0]
    assert dx > 0.005, f"dx 位移应 > 0.005，实际 {dx:.4f}"
    print(f"[smoke] ✓ step 后 ee_pos 位移 dx={dx:.4f}（合理，无回归）")

    env.close()
    print("\n=== PASS ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"\n=== FAIL: {type(e).__name__}: {e} ===")
        sys.exit(1)

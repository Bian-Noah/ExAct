"""WidowX 功能冒烟测试（DIRECT 模式，无 GUI）。

验证 PyBulletEnv 以 robot.type=widowx 加载 URDF 并执行一步动作：
  - 场景 A：正 dx 增量 → 末端 ee_pos.x 增大
  - 场景 B：gripper 开合 → 两指关节按 URDF 机械限位运动（left [0.015,0.037] /
    right [-0.037,-0.015]：开=外极限 [0.037,-0.037]，闭=内极限 [0.015,-0.015]）

前置条件：robot/widowx/wx250.urdf + meshes 存在（首次需自动下载/手动放置）。
URDF 缺失时 pytest.skip（不强制联网）。

运行：cd 项目根目录 && PYTHONPATH=src /Users/noah/miniconda3/envs/exact/bin/python -m pytest tests/functional/test_widowx_smoke.py
"""

import os

import pybullet as p
import pytest

from config.loader import EnvConfig, RobotConfig
from env.base import Action7D
from env.pybullet_env import PyBulletEnv

# 本项目 wx250.urdf 的夹爪机械限位（与 URDF 一致）
_LEFT_FINGER = (0.015, 0.037)    # (lower, upper)
_RIGHT_FINGER = (-0.037, -0.015)

URDF_PATH = "robot/widowx/wx250.urdf"


def _urdf_available() -> bool:
    """URDF 主文件存在即可跑；mesh 缺失会在 reset 时暴露，由用例报错。"""
    return os.path.isfile(URDF_PATH)


def _make_env():
    return PyBulletEnv(
        EnvConfig(mode="direct"),
        RobotConfig(type="widowx", urdf_path=URDF_PATH),
    )


def _gripper_positions(env):
    rid, cid = env._robot_id, env._client_id
    return [
        p.getJointState(rid, i, physicsClientId=cid)[0] for i in (9, 10)
    ]


@pytest.mark.skipif(not _urdf_available(), reason="widowx URDF 未下载，跳过冒烟")
def test_widowx_step_moves_ee_positive_x():
    """场景 A：一步正 dx 增量 → 末端 x 增大（DIRECT，无 GUI）。"""
    env = _make_env()
    try:
        obs = env.reset({"objects": [{"type": "cube", "pos": [0.4, 0, 0.1], "color": "red"}]})
        before = obs["ee_pos"]
        assert env.input_spec.space == "task"

        env.step(Action7D(dx=0.03, dy=0.0, dz=0.0, drx=0.0, dry=0.0, drz=0.0, gripper=0.5))
        after = env.get_obs(include_rgb=False)["ee_pos"]

        assert after[0] > before[0] + 1e-4, (
            f"正 dx 增量应使 ee_pos.x 增大: before={before[0]:.4f} after={after[0]:.4f}"
        )
    finally:
        env.close()


@pytest.mark.skipif(not _urdf_available(), reason="widowx URDF 未下载，跳过冒烟")
def test_widowx_gripper_open_close_limits():
    """场景 B：gripper 开 → 外极限；闭 → 内极限（按 URDF 机械限位）。"""
    env = _make_env()
    try:
        env.reset({"objects": [{"type": "cube", "pos": [0.4, 0, 0.1], "color": "red"}]})

        env.step(Action7D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper=1.0))
        open_pos = _gripper_positions(env)
        assert abs(open_pos[0] - _LEFT_FINGER[1]) < 1e-2, f"开：左指应达上限 {open_pos}"
        assert abs(open_pos[1] - _RIGHT_FINGER[0]) < 1e-2, f"开：右指应达下限 {open_pos}"

        env.step(Action7D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper=0.0))
        close_pos = _gripper_positions(env)
        assert abs(close_pos[0] - _LEFT_FINGER[0]) < 1e-2, f"闭：左指应达下限 {close_pos}"
        assert abs(close_pos[1] - _RIGHT_FINGER[1]) < 1e-2, f"闭：右指应达上限 {close_pos}"
    finally:
        env.close()


@pytest.mark.skipif(not _urdf_available(), reason="widowx URDF 未下载，跳过冒烟")
def test_widowx_joint_state_shape():
    """get_joint_state → (6,)：5 臂 + 1 gripper（第一指）。"""
    env = _make_env()
    try:
        env.reset({"objects": [{"type": "cube", "pos": [0.4, 0, 0.1], "color": "red"}]})
        state = env.get_joint_state()
        assert state.shape == (6,)
    finally:
        env.close()

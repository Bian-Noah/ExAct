"""WidowX 全链路测试（用户要求）：mock(task) + direct，VLA → adapter → robot 贯通。

验证目标：
  1. MockVLA（task 空间 7 维）输出 spec 与 WidowxRobot.input_spec 组合 ("task","task")
     自动命中 identity_transform（adapter 直通，适配层零改动）。
  2. Executor.run_action 走完整链路（env.reset → VLA.predict → adapter → env.step），
     多 chunk 执行后 ee_pos 有位移（Mock 增量驱动），无异常。
  3. 全程 env.mode=direct（无 GUI 窗口，不干扰用户界面）。
  4. URDF 缺失时 skip（不强制联网）——资产由首次运行/手动下载保证。

运行：cd 项目根目录 && PYTHONPATH=src /Users/noah/miniconda3/envs/exact/bin/python -m pytest tests/functional/test_widowx_full_link.py
"""

import os

import pytest

from config.loader import EnvConfig, RobotConfig
from env.base import Action7D
from env.pybullet_env import PyBulletEnv
from executor import Executor
from executor.model.mock.mock_vla import MockVLA
from utils.adapter import get_adapter

URDF_PATH = "robot/widowx/wx250.urdf"


def _urdf_available() -> bool:
    return os.path.isfile(URDF_PATH)


def _make_env_and_vla():
    env = PyBulletEnv(
        EnvConfig(mode="direct"),
        RobotConfig(type="widowx", urdf_path=URDF_PATH),
    )
    vla = MockVLA(seed=0)  # task 空间 7 维
    return env, vla


@pytest.mark.skipif(not _urdf_available(), reason="widowx URDF 未下载，跳过全链路")
def test_widowx_full_link_identity_adapter():
    """spec 组合 ("task","task") → identity_transform，适配层零改动直通。"""
    env, vla = _make_env_and_vla()
    try:
        env.reset({"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]})
        adapter = get_adapter(vla.output_spec, env.input_spec)
        # identity_transform 是 task→task 同空间直通
        assert adapter.__name__ == "identity_transform"
    finally:
        env.close()


@pytest.mark.skipif(not _urdf_available(), reason="widowx URDF 未下载，跳过全链路")
def test_widowx_full_link_executor_steps():
    """Executor.run_action 多 chunk 执行：ee_pos 有位移，全程 direct 无 GUI。"""
    env, vla = _make_env_and_vla()
    try:
        obs0 = env.reset(
            {"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]}
        )
        executor = Executor(vla)
        adapter = get_adapter(vla.output_spec, env.input_spec)

        ee_before = obs0["ee_pos"]
        total_disp = 0.0
        for _ in range(2):  # 2 个 chunk
            out = executor.run_action(
                env,
                "pick up the cube",
                done_criteria="reached",
                adapter=adapter,
            )
            assert out.steps >= 1, "每个 chunk 至少执行 1 步"
            final_ee = out.final_obs["ee_pos"]
            total_disp += sum(abs(a - b) for a, b in zip(ee_before, final_ee))
            ee_before = final_ee

        # Mock 增量驱动下末端应有可见位移（非零）
        assert total_disp > 1e-4, f"全链路执行应产生末端位移，实际累计 {total_disp}"
    finally:
        env.close()


@pytest.mark.skipif(not _urdf_available(), reason="widowx URDF 未下载，跳过全链路")
def test_widowx_full_link_uses_direct_mode():
    """env.mode=direct：连接模式为 DIRECT（无 GUI 窗口，不干扰用户界面）。"""
    import pybullet as p

    env, _ = _make_env_and_vla()
    try:
        env.reset({"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]})
        info = p.getConnectionInfo(env._client_id)
        # connectionMethod: 1=GUI, 2=DIRECT
        assert info["connectionMethod"] == 2, (
            f"期望 DIRECT(2) 连接，实际 {info.get('connectionMethod')}"
        )
    finally:
        env.close()

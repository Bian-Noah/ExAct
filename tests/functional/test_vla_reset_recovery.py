"""VLA reset 恢复功能测试（iter11-reset-multicam）P0 验证。

P0 验证目标:reset 把关节拉回 home 后,VLA 推理不受破坏——
即"同一起点(原始 home)推理的结果"与"reset 后推理的结果"一致。

M4 Mac 无真实 smolVLA 权重(需 RTX 4060),用确定性 VLA(JointMockVLA
按 joint state 哈希输出)做逻辑等价验证:
  - 场景 A(原始 home 起点):env 刚 reset(关节在 home)→ predict
  - 场景 B(reset 后):把关节推离 home → reset_arm_to_home() → predict
  - 断言 A 与 B 的 VLA 输入(observation.state)一致,推理输出一致

真实 smolVLA 的 P0 验证需在 RTX 4060 上跑(test_vla_reset_recovery_real.py 预留)。
"""
from __future__ import annotations

import numpy as np

from config.loader import EnvConfig, RobotConfig
from env.pybullet_env import PyBulletEnv
from env.robot.so101.so101_robot import SO101Robot
from executor import Executor
from executor.model.mock.mock_vla import JointMockVLA


def _predict_after_reset(seed: int):
    """在 env 上:reset → predict,推离 → reset → predict,返回两次 state + output。"""
    robot_config = RobotConfig(type="so101")
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="cpu"), robot_config=robot_config)
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]})
    vla = JointMockVLA(seed=seed)
    executor = Executor(vla)

    # 场景 A:原始 home 起点
    state_a = env.get_joint_state().copy()
    out_a = executor.run_action(env, "pick red cube", done_criteria="reached", adapter=lambda raw, e: raw)
    assert out_a.steps == 1  # JointMockVLA 返回 (1, 6)

    # 把关节推离 home
    push = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.0])
    env.robot.step_action(push, env._robot_id, env._client_id)
    state_pushed = env.get_joint_state().copy()
    assert not np.allclose(state_pushed[:5], state_a[:5], atol=1e-3)

    # 场景 B:reset 后
    env.reset_arm_to_home()
    state_b = env.get_joint_state().copy()
    out_b = executor.run_action(env, "pick red cube", done_criteria="reached", adapter=lambda raw, e: raw)
    assert out_b.steps == 1

    env.close()
    return state_a, state_b, out_a, out_b


def test_vla_reset_recovery_home_state_restored():
    """reset 后臂关节 state 与原始 home 起点一致(SO101 前 5 维臂关节)。"""
    state_a, state_b, _, _ = _predict_after_reset(seed=0)
    # 只比较臂关节部分(SO101 前 5 维,gripper 不在 reset 范围)
    np.testing.assert_allclose(state_a[:5], state_b[:5], atol=1e-6)


def test_vla_reset_recovery_inference_consistent():
    """reset 后 VLA 推理输出与原始 home 起点一致(逻辑等价验证)。

    注意:JointMockVLA 的输出依赖输入 state,若臂关节 state 恢复一致则输出一致。
    真实 smolVLA 需 RTX 4060 上跑 P0。
    """
    state_a, state_b, out_a, out_b = _predict_after_reset(seed=42)
    # 输入一致(臂关节 state 恢复)
    np.testing.assert_allclose(state_a[:5], state_b[:5], atol=1e-6)
    # 输出一致
    np.testing.assert_allclose(
        out_a.final_obs.get("ee_pos", ()),
        out_b.final_obs.get("ee_pos", ()),
        atol=1e-6,
    )
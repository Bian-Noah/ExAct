"""env.reset_arm_to_home() 单元测试（iter11-reset-multicam）。

验证 reset 把关节瞬时复位到 home,cube 位置不变,env 不重置,可重复调用。
"""
from __future__ import annotations

import numpy as np
import pytest

from config.loader import EnvConfig, RobotConfig
from env.base import BaseEnv
from env.pybullet_env import PyBulletEnv
from env.robot.panda.panda_robot import PandaRobot
from env.robot.so101.so101_robot import SO101Robot


# ----- U2 单元测试 -----

def test_base_env_reset_arm_to_home_not_implemented():
    """BaseEnv 默认 reset 抛 NotImplementedError。"""
    # 用一个最小子类来测 NotImplementedError
    class _FakeEnv(BaseEnv):
        def reset(self, task_spec, seed=0): pass
        def step(self, action): pass
        def render(self): pass
        def get_obs(self, include_rgb=True): pass
        def close(self): pass
        @property
        def input_spec(self): pass

    fake = _FakeEnv()
    with pytest.raises(NotImplementedError, match="reset_arm_to_home"):
        fake.reset_arm_to_home()


def test_reset_arm_to_home_joint_positions_so101():
    """SO101 reset 后关节状态 == SO101Robot.home_joint_positions()。"""
    env = PyBulletEnv(env_config=EnvConfig(), robot_config=RobotConfig(type="so101"))
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]})
    env.reset_arm_to_home()
    state = env.get_joint_state()
    expected = np.array(SO101Robot(RobotConfig(type="so101")).home_joint_positions())
    # 只比较前 5 维（机械臂关节）,第 6 维是 gripper,SO101Robot.home 只返回 5 维
    np.testing.assert_allclose(state[:5], expected, atol=1e-6)
    env.close()


def test_reset_arm_to_home_no_object_movement():
    """reset 后 cube 位置不变(不动 env/cube)。"""
    env = PyBulletEnv(env_config=EnvConfig(), robot_config=RobotConfig(type="so101"))
    cube_pos = [0.5, 0, 0.1]
    env.reset(task_spec={"objects": [{"type": "cube", "pos": cube_pos, "color": "red"}]})

    # reset 前记录物体位置
    obs_before = env.get_obs(include_rgb=False)
    obj_pos_before = obs_before["object_info"][0]["pos"]

    env.reset_arm_to_home()

    obs_after = env.get_obs(include_rgb=False)
    obj_pos_after = obs_after["object_info"][0]["pos"]

    np.testing.assert_allclose(obj_pos_before, obj_pos_after, atol=1e-6)
    env.close()


def test_reset_arm_to_home_idempotent():
    """reset 幂等——重复调用结果一致。"""
    env = PyBulletEnv(env_config=EnvConfig(), robot_config=RobotConfig(type="so101"))
    env.reset(task_spec={"objects": []})
    env.reset_arm_to_home()
    state1 = env.get_joint_state().copy()
    env.reset_arm_to_home()
    state2 = env.get_joint_state().copy()
    env.reset_arm_to_home()
    state3 = env.get_joint_state().copy()

    np.testing.assert_allclose(state1, state2, atol=1e-6)
    np.testing.assert_allclose(state2, state3, atol=1e-6)
    env.close()


def test_reset_arm_to_home_does_not_modify_sim_state_globally():
    """reset 不重置 env——env.get_obs 仍能返回观测。"""
    env = PyBulletEnv(env_config=EnvConfig(), robot_config=RobotConfig(type="so101"))
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.3, 0, 0.1], "color": "red"}]})
    env.reset_arm_to_home()
    # reset 后 get_obs 仍能调通,env 未被销毁
    obs = env.get_obs(include_rgb=False)
    assert "ee_pos" in obs
    assert "object_info" in obs
    assert len(obs["object_info"]) == 1
    env.close()


def test_panda_robot_home_joint_positions_length():
    """PandaRobot.home_joint_positions() 返回 7 维 tuple。"""
    home = PandaRobot(RobotConfig(type="panda")).home_joint_positions()
    assert isinstance(home, tuple)
    assert len(home) == 7
    assert all(isinstance(v, float) for v in home)


def test_so101_robot_home_joint_positions_length():
    """SO101Robot.home_joint_positions() 返回 5 维 tuple。"""
    home = SO101Robot(RobotConfig(type="so101")).home_joint_positions()
    assert isinstance(home, tuple)
    assert len(home) == 5
    assert all(isinstance(v, float) for v in home)
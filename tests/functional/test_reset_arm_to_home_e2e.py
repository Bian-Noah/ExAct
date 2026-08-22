"""reset_arm_to_home 端到端功能测试（iter11-reset-multicam）。

在真实 PyBullet SO101 环境上验证:
  1. 关节被推到极限 → env.reset_arm_to_home() → 关节回到 home
  2. ActionTool(operation="reset") 能触发复位
  3. 复位后 cube 位置不变
"""
from __future__ import annotations

import numpy as np

from config.loader import EnvConfig, RobotConfig
from env.pybullet_env import PyBulletEnv
from env.robot.so101.so101_robot import SO101Robot
from executor import Executor
from executor.model.mock.mock_vla import JointMockVLA
from tools.action import ActionTool
from utils.adapter import get_adapter


def test_reset_arm_to_home_e2e():
    """F2: reset 端到端——关节极限 → reset → 关节回 home。"""
    robot_config = RobotConfig(type="so101")
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="cpu"), robot_config=robot_config)
    env.reset(task_spec={"objects": [{"type": "cube", "pos": [0.5, 0, 0.1], "color": "red"}]})

    home = np.array(SO101Robot(robot_config).home_joint_positions())

    # 1. 把关节推到非 home 位姿(通过 step_action)
    action = np.array([0.5, 0.3, -0.4, 0.2, -0.3, 0.0])
    env.robot.step_action(action, env._robot_id, env._client_id)
    state_before = env.get_joint_state()
    assert not np.allclose(state_before[:5], home, atol=1e-3)

    # 2. reset
    env.reset_arm_to_home()

    # 3. 验证关节回到 home
    state_after = env.get_joint_state()
    np.testing.assert_allclose(state_after[:5], home, atol=1e-6)

    # 4. 验证 cube 位置不变(x/y 不动,z 因重力略微下沉,容差放宽)
    obs = env.get_obs(include_rgb=False)
    cube_pos = obs["object_info"][0]["pos"]
    np.testing.assert_allclose(cube_pos[:2], [0.5, 0], atol=1e-6)
    assert abs(cube_pos[2] - 0.1) < 0.02  # 重力下沉 < 2cm

    env.close()


def test_action_tool_reset_operation_e2e():
    """ActionTool(operation="reset") 触发 env 复位。"""
    robot_config = RobotConfig(type="so101")
    env = PyBulletEnv(env_config=EnvConfig(mode="direct", renderer="cpu"), robot_config=robot_config)
    env.reset(task_spec={"objects": []})

    vla = JointMockVLA(seed=0)
    executor = Executor(vla)
    adapter = get_adapter(vla.output_spec, env.input_spec)
    tool = ActionTool(env=env, executor=executor, adapter=adapter)

    # 推到非 home 位姿
    action = np.array([0.4, 0.2, -0.5, 0.3, -0.2, 0.0])
    env.robot.step_action(action, env._robot_id, env._client_id)

    home = np.array(SO101Robot(robot_config).home_joint_positions())
    state_before = env.get_joint_state()
    assert not np.allclose(state_before[:5], home, atol=1e-3)

    # 调用 action(operation="reset")
    result = tool._run(instruction="", operation="reset")

    # 验证关节回 home + 返回字符串含 home
    state_after = env.get_joint_state()
    np.testing.assert_allclose(state_after[:5], home, atol=1e-6)
    assert "home" in result.lower() or "复位" in result

    env.close()
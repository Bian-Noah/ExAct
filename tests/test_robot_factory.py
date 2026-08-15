"""build_robot 工厂分派单元测试（robot-vla-adapter 任务 4）。"""

import pytest

from config.loader import RobotConfig
from env.robot import build_robot
from env.robot.panda import PandaRobot


def test_build_robot_panda():
    """type="panda" → PandaRobot 实例。"""
    robot = build_robot(RobotConfig(type="panda"))
    assert isinstance(robot, PandaRobot)


def test_build_robot_default_type_is_panda():
    """未指定 type 时默认 panda。"""
    robot = build_robot(RobotConfig())
    assert isinstance(robot, PandaRobot)


def test_build_robot_so101_placeholder():
    """type="so101" → NotImplementedError（后续迭代填充）。"""
    with pytest.raises(NotImplementedError):
        build_robot(RobotConfig(type="so101"))


def test_build_robot_unknown_type():
    """type="ghost" → ValueError，列出可选类型。"""
    with pytest.raises(ValueError) as exc_info:
        build_robot(RobotConfig(type="ghost"))
    assert "panda" in str(exc_info.value)


def test_panda_robot_construct_no_pybullet():
    """PandaRobot 构造不触发 pybullet（只存索引）。"""
    robot = PandaRobot(RobotConfig())
    assert robot.arm_joint_indices == (0, 1, 2, 3, 4, 5, 6)
    assert robot.ee_link_index == 11
    assert robot.finger_joint_indices == (9, 10)


def test_panda_robot_input_spec():
    """PandaRobot.input_spec → task 7 维 / gripper_index=6。"""
    robot = PandaRobot(RobotConfig())
    spec = robot.input_spec
    assert spec.space == "task"
    assert spec.dim == 7
    assert spec.gripper_index == 6

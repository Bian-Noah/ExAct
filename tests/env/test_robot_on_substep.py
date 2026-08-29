"""robot step_action on_substep 回调单元测试（iter12-video-recording 任务 8）。

策略：monkeypatch pybullet 模块（stepSimulation / IK / 关节控制 / getJointStates），
验证 on_substep 每子步回调、默认 None 兼容。
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import env.robot.panda.panda_robot as panda_mod
import env.robot.so101.so101_robot as so101_mod
from config.loader import RobotConfig
from env.robot.panda.panda_robot import PandaRobot
from env.robot.so101.so101_robot import SO101Robot


# ============================================================================
# Fake pybullet
# ============================================================================


@pytest.fixture
def fake_pybullet(monkeypatch):
    """注入 fake pybullet：记录 stepSimulation 次数与 setJointMotorControlArray。"""
    fake = types.ModuleType("pybullet")
    fake.step_count = 0
    fake.ik_calls = 0

    def stepSimulation(physicsClientId=None):
        fake.step_count += 1

    def getLinkState(bodyUniqueId, linkIndex, physicsClientId=None):
        return ([0.3, 0.0, 0.2], [0, 0, 0, 1])

    def calculateInverseKinematics(*args, **kwargs):
        fake.ik_calls += 1
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def setJointMotorControlArray(*args, **kwargs):
        pass

    def setJointMotorControl2(*args, **kwargs):
        pass

    def getJointStates(robot_id, indices, physicsClientId=None):
        # 返回 (joint_value, velocity, reaction, ...) 元组列表——模拟已收敛
        return [(0.0, 0.0, 0.0, 0.0) for _ in indices]

    def getJointState(robot_id, joint, physicsClientId=None):
        return (0.0, 0.0, 0.0, 0.0)

    fake.stepSimulation = stepSimulation
    fake.getLinkState = getLinkState
    fake.calculateInverseKinematics = calculateInverseKinematics
    fake.setJointMotorControlArray = setJointMotorControlArray
    fake.setJointMotorControl2 = setJointMotorControl2
    fake.getJointStates = getJointStates
    fake.getJointState = getJointState
    fake.POSITION_CONTROL = 3
    monkeypatch.setitem(sys.modules, "pybullet", fake)
    # robot 模块内 `p` 在 import 时已绑定真实 pybullet，需替换模块级引用
    monkeypatch.setattr(panda_mod, "p", fake)
    monkeypatch.setattr(so101_mod, "p", fake)
    return fake


class FakeAction7D:
    """模拟 Action7D：有 dx/dy/dz/gripper 属性。"""

    def __init__(self, dx=0.01, dy=0.0, dz=0.0, gripper=1.0):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.gripper = gripper


# ============================================================================
# Panda
# ============================================================================


def test_panda_on_substep_called_10_times(fake_pybullet):
    """Panda on_substep 被调 10 次，参数 0..9。"""
    robot = PandaRobot(RobotConfig())
    calls = []
    robot.step_action(FakeAction7D(), 0, 0, on_substep=calls.append)
    assert fake_pybullet.step_count == 10
    assert calls == list(range(10))


def test_panda_no_callback_default(fake_pybullet):
    """Panda 不传 on_substep 行为不变（stepSimulation 10 次）。"""
    robot = PandaRobot(RobotConfig())
    robot.step_action(FakeAction7D(), 0, 0)
    assert fake_pybullet.step_count == 10


def test_panda_callback_none_explicit(fake_pybullet):
    """Panda on_substep=None 显式传参不报错。"""
    robot = PandaRobot(RobotConfig())
    robot.step_action(FakeAction7D(), 0, 0, on_substep=None)
    assert fake_pybullet.step_count == 10


# ============================================================================
# SO101
# ============================================================================


def test_so101_on_substep_called(fake_pybullet):
    """SO101 收敛循环内每子步回调 ≥1 次。"""
    robot = SO101Robot(RobotConfig())
    calls = []
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])  # 目标 0 → 立即收敛
    robot.step_action(action, 0, 0, on_substep=calls.append)
    assert len(calls) >= 1
    # 每步回调序号连续 0..n-1
    assert calls == list(range(len(calls)))
    # fake getJointStates 返回已收敛 → 首次检查(step 0 后)即 break
    assert fake_pybullet.step_count == len(calls)


def test_so101_no_callback_default(fake_pybullet):
    """SO101 不传 on_substep 行为不变（仍推进物理）。"""
    robot = SO101Robot(RobotConfig())
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    robot.step_action(action, 0, 0)
    assert fake_pybullet.step_count >= 1


def test_so101_callback_none_explicit(fake_pybullet):
    """SO101 on_substep=None 显式传参不报错。"""
    robot = SO101Robot(RobotConfig())
    action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    robot.step_action(action, 0, 0, on_substep=None)
    assert fake_pybullet.step_count >= 1

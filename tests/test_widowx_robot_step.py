"""WidowxRobot step_action / get_joint_state 单元测试（不触发真实 pybullet）。

策略：monkeypatch env.robot.widowx.widowx_robot.p 为 FakePybullet 假模块，
覆盖 getLinkState / calculateInverseKinematics / setJointMotorControlArray /
setJointMotorControl2 / getJointInfo / getJointStates / getJointState /
stepSimulation，并 monkeypatch ensure_urdf_downloaded 返回 True 避免任何网络请求。

收敛循环：min_steps=60 保证夹爪至少推进 60 个物理子步；随后每 5 步检查
getJointStates 误差（返回与 IK 一致 → 首次检查即 break），因此 stepSimulation
恰好执行 60 次。

gripper 目标基于 pybullet 运行时关节限位（getJointInfo lower/upper）：
  - 开 (gripper≥0.5)：左指 upper + 右指 lower → [0.037, -0.037]
  - 闭 (gripper<0.5)：左指 lower + 右指 upper → [0.015, -0.015]
（本项目 wx250.urdf 限位：left_finger [0.015, 0.037] / right_finger [-0.037, -0.015]）

运行：cd 项目根目录 && PYTHONPATH=src /Users/noah/miniconda3/envs/exact/bin/python -m pytest tests/test_widowx_robot_step.py

用例清单：
  1. test_construct_no_pybullet —— 构造零 pybullet 调用，索引与默认配置一致
  2. test_input_spec —— 7 维增量语义名 + gripper_index=6
  3. test_step_action_action7d_input —— Action7D → target=current+(dx,dy,z)、IK 调用、5 臂控制
  4. test_step_action_ndarray_input —— ndarray (7,) 与 Action7D 行为等价
  5. test_step_action_gripper_open_close —— gripper≥0.5 → 限位外极限；<0.5 → 限位内极限
  6. test_step_action_on_substep_callback —— 回调被调用且次数 > 0
  7. test_get_joint_state_shape_and_values —— (6,) shape 与确定数值
  8. test_home_joint_positions —— 5 维全零
  9. test_step_action_z_floor_clamped —— 目标 z 低于 0.01 时钳制到 0.01
"""

import types

import numpy as np
import pytest

import env.robot.widowx.widowx_robot as wr
from config.loader import RobotConfig
from env.base import Action7D
from env.robot.widowx import WidowxRobot


# ============================================================================
# Fake pybullet
# ============================================================================

# 模拟本项目 wx250.urdf 的夹爪关节限位：(joint_index) → (lower, upper)
_GRIPPER_LIMITS = {9: (0.015, 0.037), 10: (-0.037, -0.015)}


@pytest.fixture
def fake_pybullet(monkeypatch):
    """注入 fake pybullet：记录 IK 目标、关节控制调用与 stepSimulation 次数。"""
    fake = types.ModuleType("pybullet")
    fake.step_count = 0
    fake.ik_calls = 0
    fake.getJointStates_calls = 0
    fake.control_calls = []           # [(joint_indices, control_mode, kwargs), ...]
    fake.motor2_calls = []            # [(joint_idx, control_mode, kwargs), ...]
    fake.last_ik = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    fake.last_target = None           # IK 收到的目标位置
    fake.ee_pos = (0.3, 0.0, 0.15)    # getLinkState 返回的末端位置
    fake.state_values = None          # 覆盖 getJointStates 返回值（None → 按 last_ik 收敛）
    fake.gripper_state_value = 0.6    # getJointState（gripper 第一指）返回值

    def getLinkState(bodyUniqueId, linkIndex, physicsClientId=None):
        return (fake.ee_pos, (0, 0, 0, 1))

    def calculateInverseKinematics(*args, **kwargs):
        fake.ik_calls += 1
        fake.last_target = list(args[2])
        return list(fake.last_ik)

    def setJointMotorControlArray(robot_id, joint_indices, control_mode, **kwargs):
        fake.control_calls.append((tuple(joint_indices), control_mode, kwargs))

    def setJointMotorControl2(robot_id, joint_idx, control_mode, **kwargs):
        fake.motor2_calls.append((joint_idx, control_mode, kwargs))

    def getJointInfo(robot_id, joint_idx, physicsClientId=None):
        lower, upper = _GRIPPER_LIMITS.get(joint_idx, (0.0, 1.0))
        info = [0] * 16
        info[8] = lower
        info[9] = upper
        return tuple(info)

    def getJointStates(robot_id, indices, physicsClientId=None):
        fake.getJointStates_calls += 1
        if fake.state_values is not None:
            values = list(fake.state_values)
            return [(float(values[i]), 0.0, 0.0, 0.0) for i in range(len(indices))]
        # 返回与 last_ik 完全一致的关节角 → 收敛循环首次检查即 break
        return [(float(fake.last_ik[i]), 0.0, 0.0, 0.0) for i in range(len(indices))]

    def getJointState(robot_id, joint, physicsClientId=None):
        return (fake.gripper_state_value, 0.0, 0.0, 0.0)

    def stepSimulation(physicsClientId=None):
        fake.step_count += 1

    fake.getLinkState = getLinkState
    fake.calculateInverseKinematics = calculateInverseKinematics
    fake.setJointMotorControlArray = setJointMotorControlArray
    fake.setJointMotorControl2 = setJointMotorControl2
    fake.getJointInfo = getJointInfo
    fake.getJointStates = getJointStates
    fake.getJointState = getJointState
    fake.stepSimulation = stepSimulation
    fake.POSITION_CONTROL = 3

    # 替换 robot 模块内 `p` 引用 + 禁用 URDF/资产下载（零网络）
    monkeypatch.setattr(wr, "p", fake)
    monkeypatch.setattr(wr, "ensure_urdf_downloaded", lambda *a, **k: True)
    monkeypatch.setattr(wr, "ensure_assets_downloaded", lambda *a, **k: 0)
    return fake


def _control_for(fake, joint_indices):
    """取回对指定关节索引组发出的 setJointMotorControlArray 调用 kwargs。"""
    for ji, _mode, kwargs in fake.control_calls:
        if ji == joint_indices:
            return kwargs
    raise AssertionError(
        f"未找到 joint_indices={joint_indices} 的控制调用，实际记录：{fake.control_calls}"
    )


def _gripper_targets(fake):
    """取回两指 setJointMotorControl2 的 targetPosition 列表（按关节顺序）。"""
    targets = {}
    for joint_idx, _mode, kwargs in fake.motor2_calls:
        targets[joint_idx] = kwargs["targetPosition"]
    return [targets[j] for j in (9, 10)]


# ============================================================================
# 构造 / spec / home
# ============================================================================


def test_construct_no_pybullet(fake_pybullet):
    """构造不触发任何 pybullet 调用，且索引与默认配置一致。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    assert robot.arm_joint_indices == (0, 1, 2, 3, 4)
    assert robot.ee_link_index == 11
    assert robot.gripper_joint_indices == (9, 10)
    assert fake_pybullet.step_count == 0
    assert fake_pybullet.ik_calls == 0
    assert fake_pybullet.control_calls == []


def test_input_spec(fake_pybullet):
    """input_spec → task 空间 7 维增量语义名，gripper_index=6。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    spec = robot.input_spec
    assert spec.space == "task"
    assert spec.components == ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
    assert spec.dim == 7
    assert spec.gripper_index == 6


def test_home_joint_positions(fake_pybullet):
    """home_joint_positions → 5 维全零。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    assert robot.home_joint_positions() == (0.0, 0.0, 0.0, 0.0, 0.0)


# ============================================================================
# step_action
# ============================================================================


def test_step_action_action7d_input(fake_pybullet):
    """Action7D：target ee = current + (dx,dy,z)，IK 被调用，5 臂 POSITION_CONTROL。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    robot.step_action(
        Action7D(dx=0.02, dy=0.0, dz=0.0, drx=0.0, dry=0.0, drz=0.0, gripper=1.0),
        robot_id=7,
        client_id=3,
    )
    # current ee = (0.3, 0.0, 0.15) → target = (0.32, 0.0, 0.15)
    assert fake_pybullet.last_target == [0.32, 0.0, 0.15]
    assert fake_pybullet.ik_calls == 1
    # 5 臂控制：targetPositions = IK 结果前 5 个关节角
    arm_kwargs = _control_for(fake_pybullet, (0, 1, 2, 3, 4))
    assert arm_kwargs["targetPositions"] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_step_action_ndarray_input(fake_pybullet):
    """np.ndarray (7,)：行为与 Action7D 等价。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    robot.step_action(
        np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float),
        robot_id=7,
        client_id=3,
    )
    assert fake_pybullet.last_target == [0.32, 0.0, 0.15]
    assert fake_pybullet.ik_calls == 1
    arm_kwargs = _control_for(fake_pybullet, (0, 1, 2, 3, 4))
    assert arm_kwargs["targetPositions"] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_step_action_gripper_open_close(fake_pybullet):
    """gripper≥0.5 → 两指外极限（左 upper + 右 lower）；<0.5 → 内极限（左 lower + 右 upper）。

    基于 pybullet getJointInfo 运行时限位：left [0.015,0.037] / right [-0.037,-0.015]。
    """
    robot = WidowxRobot(RobotConfig(type="widowx"))
    robot.step_action(
        Action7D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper=1.0), robot_id=0, client_id=0
    )
    assert _gripper_targets(fake_pybullet) == [0.037, -0.037]

    fake_pybullet.motor2_calls.clear()
    robot.step_action(
        Action7D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper=0.0), robot_id=0, client_id=0
    )
    assert _gripper_targets(fake_pybullet) == [0.015, -0.015]


def test_step_action_on_substep_callback(fake_pybullet):
    """on_substep 回调被调用且次数 > 0，序号从 0 连续。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    calls = []
    robot.step_action(
        Action7D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper=0.0),
        robot_id=0,
        client_id=0,
        on_substep=calls.append,
    )
    assert len(calls) >= 1
    assert calls == list(range(len(calls)))


def test_step_action_z_floor_clamped(fake_pybullet):
    """目标 z = max(current_z + dz, 0.01)，低于下限时钳制到 0.01。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    # current z=0.15，dz=-0.5 → 0.15-0.5=-0.35 → 钳制为 0.01
    robot.step_action(
        Action7D(0.0, 0.0, -0.5, 0.0, 0.0, 0.0, gripper=0.0), robot_id=0, client_id=0
    )
    assert fake_pybullet.last_target[:2] == [0.3, 0.0]
    assert fake_pybullet.last_target[2] == 0.01


# ============================================================================
# get_joint_state
# ============================================================================


def test_get_joint_state_shape_and_values(fake_pybullet):
    """getJointStates/getJointState 返回确定值 → shape (6,) 且数值正确。"""
    robot = WidowxRobot(RobotConfig(type="widowx"))
    fake_pybullet.state_values = [0.1, 0.2, 0.3, 0.4, 0.5]
    fake_pybullet.gripper_state_value = 0.6
    state = robot.get_joint_state(robot_id=0, client_id=0)
    assert state.shape == (6,)
    assert np.allclose(state, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert fake_pybullet.getJointStates_calls == 1

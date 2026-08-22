"""Panda 机器人特化：7 臂关节 IK + 2 夹爪控制。

从 pybullet_env.py 的 step() 迁入，逻辑保持不变：
  - 通过逆运动学（calculateInverseKinematics）把 task 空间位移转为关节角
  - POSITION_CONTROL 驱动 7 臂关节 + 2 夹爪
  - stepSimulation 推进物理
"""

import numpy as np
import pybullet as p

from config.loader import RobotConfig
from env.base import ActionSpec


class PandaRobot:
    """Panda 特化机器人（由 build_robot 实例化，PyBulletEnv 持有）。

    Args:
        robot_config: 机器人配置，读取 robot_config.panda 特化字段。
    """

    def __init__(self, robot_config: RobotConfig):
        self.arm_joint_indices = tuple(robot_config.panda.arm_joint_indices)
        self.ee_link_index = robot_config.panda.ee_link_index
        self.finger_joint_indices = tuple(robot_config.panda.finger_joint_indices)

    def home_joint_positions(self) -> tuple[float, ...]:
        """iter11-reset-multicam:Panda 7 关节 home pose。

        典型 ready pose：上方抬起,夹爪水平,关节在可达空间中央。
        """
        return (0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854)

    @property
    def input_spec(self) -> ActionSpec:
        """Panda 消费 task 空间 7 维动作（Action7D 语义）。"""
        return ActionSpec(
            "task", ("dx", "dy", "dz", "drx", "dry", "drz", "gripper")
        )

    def step_action(self, action, robot_id: int, client_id: int) -> None:
        """执行一步动作：IK + 关节驱动 + 夹爪控制 + 推进物理。

        Args:
            action: task 空间动作（Action7D，含 dx/dy/dz 位移、gripper 开合）。
            robot_id: pybullet 中该机器人的 body id。
            client_id: pybullet 连接 id。
        """
        # 获取当前末端位置
        link_state = p.getLinkState(
            robot_id,
            self.ee_link_index,
            physicsClientId=client_id,
        )
        current_ee = link_state[0]

        # 计算目标位置（当前位置 + 位移增量）
        target_ee = [
            current_ee[0] + action.dx,
            current_ee[1] + action.dy,
            current_ee[2] + action.dz,
        ]

        # 逆运动学求解关节角度
        joint_angles = p.calculateInverseKinematics(
            robot_id,
            self.ee_link_index,
            target_ee,
            physicsClientId=client_id,
        )

        # 控制机械臂关节
        p.setJointMotorControlArray(
            robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=joint_angles[:len(self.arm_joint_indices)],
            physicsClientId=client_id,
        )

        # 控制夹爪（gripper: 0=闭, 1=开 → 关节角度 0~0.04）
        gripper_pos = 0.04 * action.gripper
        p.setJointMotorControlArray(
            robot_id,
            self.finger_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=[gripper_pos, gripper_pos],
            physicsClientId=client_id,
        )

        # 推进物理仿真
        for _ in range(10):
            p.stepSimulation(physicsClientId=client_id)

    def get_joint_state(self, robot_id: int, client_id: int) -> np.ndarray:
        """读取 Panda 当前关节角（仅臂，finger 不在 VLA state 里）作为 VLA 的 state 输入。

        Returns:
            np.ndarray, shape=(7,): 7 臂关节，按 URDF 关节索引顺序。
        """
        states = p.getJointStates(
            robot_id, self.arm_joint_indices, physicsClientId=client_id
        )
        return np.array([s[0] for s in states], dtype=float)

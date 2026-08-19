"""SO101 机器人特化：单臂 5 自由度 + 1 夹爪。

真实 URDF（so101_new_calib.urdf）关节布局：
  joint[0-4]  shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll（臂关节）
  joint[5]    gripper_frame_joint（固定，不可动）
  joint[6]    gripper（夹爪铰链）

SO101 消费 joint 空间 6 维动作（adapter 已把 VLA 输出转换为 env 原生动作）：
  - 前 5 维 → POSITION_CONTROL 驱动 5 臂关节
  - 第 6 维 → 驱动 gripper 关节
  - stepSimulation 推进物理
"""

import numpy as np
import pybullet as p

from config.loader import RobotConfig
from env.base import ActionSpec


class SO101Robot:
    """SO101 特化机器人（由 build_robot 实例化，PyBulletEnv 持有）。

    Args:
        robot_config: 机器人配置，读取 robot_config.so101 特化字段。
    """

    def __init__(self, robot_config: RobotConfig):
        self.arm_joint_indices = tuple(robot_config.so101.arm_joint_indices)
        self.ee_link_index = robot_config.so101.ee_link_index
        self.gripper_joint_index = robot_config.so101.gripper_joint_index

    @property
    def input_spec(self) -> ActionSpec:
        """SO101 消费 joint 空间 6 维动作（5 臂 + gripper）。"""
        return ActionSpec("joint", ("joint",) * 6)

    def step_action(self, action, robot_id: int, client_id: int) -> None:
        """执行一步动作：POSITION_CONTROL 驱动 5 臂关节 + gripper + 推进物理。

        Args:
            action: joint 空间动作（np.ndarray 或类似可索引容器，前 5 维为**关节角增量**
                rad，第 6 维为 gripper 开合 0~1）。
                语义对齐 VLA 输出："当前关节角 + action = 目标位置"（增量控制），
                而非直接把 action 当作目标位置。
            robot_id: pybullet 中该机器人的 body id。
            client_id: pybullet 连接 id。
        """
        deltas = list(action)[: len(self.arm_joint_indices)]
        # 读取当前关节角，target = current + delta（增量控制）
        current_states = p.getJointStates(
            robot_id, self.arm_joint_indices, physicsClientId=client_id,
        )
        target_positions = [
            cur + d for cur, d in zip((s[0] for s in current_states), deltas)
        ]

        p.setJointMotorControlArray(
            robot_id,
            self.arm_joint_indices,
            p.POSITION_CONTROL,
            targetPositions=target_positions,
            physicsClientId=client_id,
        )

        # gripper 关节（绝对位置控制；第 6 维 0~1 直接当目标）
        if len(action) > len(self.arm_joint_indices):
            gripper_pos = float(action[len(self.arm_joint_indices)])
            p.setJointMotorControl2(
                robot_id,
                self.gripper_joint_index,
                p.POSITION_CONTROL,
                targetPosition=gripper_pos,
                physicsClientId=client_id,
            )

        for _ in range(10):
            p.stepSimulation(physicsClientId=client_id)

    def get_joint_state(self, robot_id: int, client_id: int) -> np.ndarray:
        """读取 SO101 当前关节角（含 gripper）作为 VLA 的 state 输入。

        Returns:
            np.ndarray, shape=(6,): 5 臂关节 + 1 gripper，按 URDF 关节索引顺序。
        """
        states = p.getJointStates(
            robot_id, self.arm_joint_indices, physicsClientId=client_id
        )
        arm = [s[0] for s in states]
        gripper_state = p.getJointState(
            robot_id, self.gripper_joint_index, physicsClientId=client_id
        )
        return np.array(list(arm) + [gripper_state[0]], dtype=float)
